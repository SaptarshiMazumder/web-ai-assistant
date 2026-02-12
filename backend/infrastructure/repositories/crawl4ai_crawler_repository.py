import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from urllib.parse import urlparse, urldefrag

from domain.entities import Document
from domain.repositories import CrawlerRepository

from infrastructure.rag.crawl_service import (
    _best_text,
    _clean_markdown_for_rag,
    _get_str,
    _is_url_under_root_path,
    _len_attr,
    _meta_attr,
    CRAWL_MAX_CONCURRENCY,
    CRAWL_MAX_DEPTH,
    CRAWL_WAIT_FOR_CONTENT,
    HEADLESS,
)
from infrastructure.rag.error_handling import safe_execute, safe_execute_async
from infrastructure.rag.robots_policy import robots_policy

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


_MAX_INDIVIDUAL_RETRIES = 5
_BACKOFF_BASE = 1.5


class _FailedResult:
    """Result placeholder for a URL that failed to crawl."""
    def __init__(self, url: str, error: str):
        self.url = url
        self.success = False
        self.error = error
        self.links = {}


async def _crawl_urls_individually(crawler: AsyncWebCrawler, urls: List[str], run_config) -> List[Any]:
    """Fallback: crawl URLs individually with retries. Every URL gets a result (success or FailedResult)."""
    results = []
    for url in urls:
        last_error = None
        for attempt in range(_MAX_INDIVIDUAL_RETRIES):
            try:
                result = await crawler.arun(url=url, config=run_config)
                if result:
                    results.append(result)
                    break
            except Exception as e:
                last_error = e
                logger.debug(f"Individual crawl attempt {attempt + 1}/{_MAX_INDIVIDUAL_RETRIES} failed for {url}: {type(e).__name__}")
                if attempt < _MAX_INDIVIDUAL_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
        else:
            results.append(_FailedResult(url, str(last_error)[:200] if last_error else "unknown"))
    return results


class Crawl4AICrawlerRepository(CrawlerRepository):
    async def crawl_urls_bfs(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        stop_event: Optional[asyncio.Event] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        headless: Optional[bool] = None,
    ) -> List[Document]:
        browser_config = BrowserConfig(headless=HEADLESS if headless is None else headless, verbose=False)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            wait_for=CRAWL_WAIT_FOR_CONTENT,
        )
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )

        parsed_root = urlparse(root_url)
        root_netloc = parsed_root.netloc
        visited = set()
        current_urls = set([_normalize_url(root_url)])
        all_docs: List[Document] = []
        rp = robots_policy()

        def is_internal(url: str) -> bool:
            return _is_url_under_root_path(url, root_url)

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for depth in range(max_depth):
                    if stop_event and stop_event.is_set():
                        break
                    urls_to_crawl = [u for u in current_urls if u not in visited]
                    if urls_to_crawl:
                        urls_to_crawl = await rp.filter_urls(urls_to_crawl)
                    if not urls_to_crawl:
                        break

                    # Retry batch crawl with exponential backoff on connection errors
                    results = None
                    for retry_attempt in range(3):
                        try:
                            results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                            break  # Success, exit retry loop
                        except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError) as e:
                            if retry_attempt < 2:
                                wait_time = 2.0 * (2 ** retry_attempt)
                                logger.debug(f"Retry {retry_attempt + 1}/3 for batch crawl after {type(e).__name__}, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.warning(f"Batch crawl failed after 3 retries: {type(e).__name__}")
                                # Try individual URLs as fallback
                                results = await _crawl_urls_individually(crawler, urls_to_crawl, run_config)
                                break
                        except asyncio.CancelledError:
                            return all_docs
                        except Exception as e:
                            # Other errors - try individual crawl as fallback
                            logger.debug(f"Batch crawl error {type(e).__name__}, trying individual URLs")
                            results = await _crawl_urls_individually(crawler, urls_to_crawl, run_config)
                            break

                    if not results:
                        # If all crawling failed, continue to next depth or return what we have
                        break

                    next_level_urls = set()

                    # Process results - continue even if some failed
                    for result in results:
                        try:
                            result_url = getattr(result, "url", None) or ""
                            norm = _normalize_url(result_url) if result_url else ""
                            
                            if norm:
                                visited.add(norm)
                            
                            # Process successful results
                            if getattr(result, "success", False):
                                content, src = safe_execute(
                                    lambda: _best_text(result),
                                    (None, None),
                                )
                                if content:
                                    content = _clean_markdown_for_rag(content)
                                    all_docs.append(
                                        Document(
                                            url=result_url,
                                            content=f"Source URL: {result_url}\n\n{content}",
                                            metadata={"source": src or "unknown", "depth": depth},
                                        )
                                    )
                                
                                # Update progress
                                if progress_cb:
                                    safe_execute(
                                        lambda: progress_cb({
                                            "type": "page_crawled",
                                            "count": len(all_docs),
                                            "url": result_url,
                                            "depth": depth,
                                        }),
                                        None,
                                    )
                                
                                # Extract links even if page had minor issues
                                links = safe_execute(
                                    lambda: getattr(result, "links", None) or {},
                                    {},
                                )
                                for link in links.get("internal", []):
                                    href = safe_execute(
                                        lambda: _normalize_url(link.get("href", "")),
                                        "",
                                    )
                                    if href and href not in visited and is_internal(href):
                                        if await rp.is_allowed(href):
                                            next_level_urls.add(href)
                            else:
                                # Failed result - still report progress
                                if progress_cb:
                                    safe_execute(
                                        lambda: progress_cb({
                                            "type": "fetch",
                                            "url": result_url,
                                            "success": False,
                                            "status_code": _meta_attr(result, "status_code")
                                            or _meta_attr(result, "http_status")
                                            or _meta_attr(result, "status"),
                                            "error": _meta_attr(result, "error")
                                            or _meta_attr(result, "error_message")
                                            or _meta_attr(result, "message")
                                            or "Unknown error",
                                        }),
                                        None,
                                    )
                        except Exception as e:
                            # Continue processing other results even if one fails completely
                            logger.debug(f"Error processing crawl result: {type(e).__name__}: {str(e)[:100]}")
                            continue
                    
                    current_urls = next_level_urls
        except asyncio.CancelledError:
            return all_docs
        except Exception as e:
            # Return whatever we collected so far - never return empty due to errors
            logger.warning(f"Crawl error in BFS: {type(e).__name__}: {str(e)[:100]}, returning {len(all_docs)} docs collected so far")
            return all_docs

        return all_docs

    async def crawl_urls_list(
        self,
        urls: List[str],
        *,
        max_concurrent: int,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        headless: Optional[bool] = None,
    ) -> List[Document]:
        """Crawl a list of URLs. Reports progress after each URL completes (not per chunk)."""
        if not urls:
            return []

        rp = robots_policy()
        filtered_urls = await rp.filter_urls(urls)
        if not filtered_urls:
            return []

        browser_config = BrowserConfig(headless=HEADLESS if headless is None else headless, verbose=False)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            wait_for=CRAWL_WAIT_FOR_CONTENT,
        )
        sem = asyncio.Semaphore(max_concurrent)
        results_list: List[tuple] = []  # (url, result) in completion order

        docs: List[Document] = []
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:

                async def crawl_one(url: str):
                    async with sem:
                        try:
                            result = await crawler.arun(url=url, config=run_config)
                            return (url, result)
                        except Exception as e:
                            logger.debug(f"Single URL crawl failed for {url}: {type(e).__name__}")
                            return (url, _FailedResult(url, str(e)[:200]))

                async def process_one(url: str):
                    url_ret, result = await crawl_one(url)
                    results_list.append((url_ret, result))
                    n = len(results_list)
                    if progress_cb:
                        # Always report page_crawled so job.pages_crawled increments (progress bar moves)
                        safe_execute(
                            lambda: progress_cb({
                                "type": "page_crawled",
                                "count": n,
                                "url": url_ret,
                                "depth": 0,
                            }),
                            None,
                        )
                        if not getattr(result, "success", False):
                            safe_execute(
                                lambda: progress_cb({
                                    "type": "fetch",
                                    "url": url_ret,
                                    "success": False,
                                    "status_code": _meta_attr(result, "status_code")
                                    or _meta_attr(result, "http_status")
                                    or _meta_attr(result, "status"),
                                    "error": getattr(result, "error", None)
                                    or _meta_attr(result, "error_message")
                                    or _meta_attr(result, "message")
                                    or "Unknown error",
                                }),
                                None,
                            )
                    return (url_ret, result)

                await asyncio.gather(*[process_one(u) for u in filtered_urls])

            for result_url, result in results_list:
                try:
                    if getattr(result, "success", False):
                        content, src = safe_execute(
                            lambda r=result: _best_text(r),
                            (None, None),
                        )
                        if content:
                            content = _clean_markdown_for_rag(content)
                            docs.append(
                                Document(
                                    url=result_url,
                                    content=f"Source URL: {result_url}\n\n{content}",
                                    metadata={"source": src or "unknown"},
                                )
                            )
                    # progress already reported in process_one
                except Exception as e:
                    logger.debug(f"Error processing result: {type(e).__name__}: {str(e)[:100]}")
        except Exception as e:
            logger.warning(f"Crawler initialization failed: {type(e).__name__}: {str(e)[:100]}")
            return docs if docs else []

        return docs

    async def discover_internal_urls(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        max_urls: int = 2000,
        headless: Optional[bool] = None,
    ) -> List[str]:
        browser_config = BrowserConfig(headless=HEADLESS if headless is None else headless, verbose=False)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            wait_for=CRAWL_WAIT_FOR_CONTENT,
        )
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )

        parsed_root = urlparse(root_url)
        root_netloc = parsed_root.netloc
        visited = set()
        current_urls = set([_normalize_url(root_url)])
        discovered: List[str] = []
        rp = robots_policy()

        def is_internal(url: str) -> bool:
            return _is_url_under_root_path(url, root_url)

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for _depth in range(max_depth):
                    urls_to_crawl = [u for u in current_urls if u not in visited]
                    if not urls_to_crawl:
                        break
                    if len(discovered) >= max_urls:
                        break

                    urls_to_crawl = await rp.filter_urls(urls_to_crawl)
                    if not urls_to_crawl:
                        break
                    
                    try:
                        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                    except Exception as e:
                        # If batch crawl fails completely, try to continue with individual URLs
                        # or return what we have so far
                        if not discovered:
                            # If we have no URLs yet, try crawling root URL individually
                            try:
                                single_result = await crawler.arun(url=root_url, config=run_config)
                                if single_result and single_result.success:
                                    norm = _normalize_url(single_result.url)
                                    if norm and is_internal(norm) and await rp.is_allowed(norm):
                                        discovered.append(norm)
                                    if hasattr(single_result, "links") and single_result.links:
                                        for link in single_result.links.get("internal", []):
                                            href = _normalize_url(link.get("href", ""))
                                            if href and is_internal(href) and await rp.is_allowed(href):
                                                current_urls.add(href)
                            except Exception:
                                pass
                        # Return what we have so far instead of empty
                        break
                    
                    next_level_urls = set()
                    # Process results even if some failed - crawl4ai returns partial results
                    if results:
                        for result in results:
                            try:
                                # Even failed results might have a URL we can record
                                result_url = getattr(result, "url", None) or ""
                                norm = _normalize_url(result_url) if result_url else ""
                                
                                if norm:
                                    visited.add(norm)
                                    if norm not in discovered and is_internal(norm) and await rp.is_allowed(norm):
                                        discovered.append(norm)
                                        if len(discovered) >= max_urls:
                                            break
                                
                                # Extract links even if page had errors (result might still have links)
                                if hasattr(result, "links") and result.links:
                                    for link in result.links.get("internal", []):
                                        href = _normalize_url(link.get("href", ""))
                                        if href and href not in visited and is_internal(href) and await rp.is_allowed(href):
                                            next_level_urls.add(href)
                            except Exception:
                                # Continue processing other results even if one fails
                                continue
                    current_urls = next_level_urls
        except Exception:
            # Return whatever we discovered so far, don't return empty
            pass

        return discovered
