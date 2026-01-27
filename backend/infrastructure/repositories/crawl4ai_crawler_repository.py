import asyncio
from typing import Any, Callable, Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from urllib.parse import urlparse, urldefrag

from domain.entities import Document
from domain.repositories import CrawlerRepository

from infrastructure.rag.crawl_service import (
    _best_text,
    _get_str,
    _len_attr,
    _meta_attr,
    CRAWL_MAX_CONCURRENCY,
    CRAWL_MAX_DEPTH,
    HEADLESS,
)


def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


class Crawl4AICrawlerRepository(CrawlerRepository):
    async def crawl_urls_bfs(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        stop_event: Optional[asyncio.Event] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Document]:
        browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
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

        def is_internal(url: str) -> bool:
            return urlparse(url).netloc == root_netloc

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for depth in range(max_depth):
                    if stop_event and stop_event.is_set():
                        break
                    urls_to_crawl = [u for u in current_urls if u not in visited]
                    if not urls_to_crawl:
                        break

                    try:
                        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                    except asyncio.CancelledError:
                        return all_docs

                    next_level_urls = set()

                    for result in results:
                        norm = _normalize_url(result.url)
                        visited.add(norm)
                        if result.success:
                            content, src = _best_text(result)
                            if content:
                                all_docs.append(
                                    Document(
                                        url=result.url,
                                        content=f"Source URL: {result.url}\n\n{content}",
                                        metadata={"source": src, "depth": depth},
                                    )
                                )
                            if progress_cb:
                                try:
                                    progress_cb(
                                        {
                                            "type": "page_crawled",
                                            "count": len(all_docs),
                                            "url": result.url,
                                            "depth": depth,
                                        }
                                    )
                                except Exception:
                                    pass
                            for link in result.links.get("internal", []):
                                href = _normalize_url(link.get("href", ""))
                                if href and href not in visited and is_internal(href):
                                    next_level_urls.add(href)
                        else:
                            if progress_cb:
                                try:
                                    progress_cb(
                                        {
                                            "type": "fetch",
                                            "url": getattr(result, "url", ""),
                                            "success": False,
                                            "status_code": _meta_attr(result, "status_code")
                                            or _meta_attr(result, "http_status")
                                            or _meta_attr(result, "status"),
                                            "error": _meta_attr(result, "error")
                                            or _meta_attr(result, "error_message")
                                            or _meta_attr(result, "message"),
                                        }
                                    )
                                except Exception:
                                    pass
                    current_urls = next_level_urls
        except asyncio.CancelledError:
            return all_docs

        return all_docs

    async def crawl_urls_list(
        self,
        urls: List[str],
        *,
        max_concurrent: int,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Document]:
        if not urls:
            return []

        browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
        except Exception:
            return []

        docs: List[Document] = []
        for result in results:
            norm = _normalize_url(result.url)
            if result.success:
                content, src = _best_text(result)
                if content:
                    docs.append(
                        Document(
                        url=result.url,
                        content=f"Source URL: {result.url}\n\n{content}",
                        metadata={"source": src},
                    )
                    )
                if progress_cb:
                    try:
                        progress_cb(
                            {
                                "type": "page_crawled",
                                "count": len(docs),
                                "url": result.url,
                                "depth": 0,
                            }
                        )
                    except Exception:
                        pass
            else:
                if progress_cb:
                    try:
                        progress_cb(
                            {
                                "type": "fetch",
                                "url": norm,
                                "success": False,
                                "status_code": _meta_attr(result, "status_code")
                                or _meta_attr(result, "http_status")
                                or _meta_attr(result, "status"),
                                "error": _meta_attr(result, "error")
                                or _meta_attr(result, "error_message")
                                or _meta_attr(result, "message"),
                            }
                        )
                    except Exception:
                        pass

        return docs

    async def discover_internal_urls(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        max_urls: int = 2000,
    ) -> List[str]:
        browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
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

        def is_internal(url: str) -> bool:
            return urlparse(url).netloc == root_netloc

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for _depth in range(max_depth):
                    urls_to_crawl = [u for u in current_urls if u not in visited]
                    if not urls_to_crawl:
                        break
                    if len(discovered) >= max_urls:
                        break
                    results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                    next_level_urls = set()
                    for result in results:
                        norm = _normalize_url(result.url)
                        visited.add(norm)
                        if norm and norm not in discovered and is_internal(norm):
                            discovered.append(norm)
                            if len(discovered) >= max_urls:
                                break
                        for link in result.links.get("internal", []):
                            href = _normalize_url(link.get("href", ""))
                            if href and href not in visited and is_internal(href):
                                next_level_urls.add(href)
                    current_urls = next_level_urls
        except Exception:
            return discovered

        return discovered
