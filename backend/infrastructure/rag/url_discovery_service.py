import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import requests

from infrastructure.rag.crawl_service import discover_internal_urls
from infrastructure.rag.error_handling import (
    is_bot_detected,
    is_captcha_page,
    retry_with_backoff,
    safe_execute,
    safe_execute_async,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # Increased timeout for large sitemaps
_MAX_SITEMAP_URLS = 5000
_MAX_DISCOVERY_URLS = 2000
_MAX_SITEMAP_DEPTH = 6


def _ensure_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        return f"https://{value}"
    return value


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_same_domain(url: str, root: str) -> bool:
    return urlparse(url).netloc == urlparse(root).netloc


def _normalize_url(url: str) -> str:
    return url.split("#", 1)[0].strip()


def _get_robots_parser(root_url: str) -> Optional[RobotFileParser]:
    """Get a RobotFileParser for the given root URL. Never fails - returns None on error."""
    return safe_execute(
        lambda: _get_robots_parser_impl(root_url),
        None,
    )


def _get_robots_parser_impl(root_url: str) -> Optional[RobotFileParser]:
    """Internal implementation of robots parser."""
    try:
        robots_url = urljoin(_origin(root_url), "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception as e:
        logger.debug(f"Failed to get robots.txt for {root_url}: {str(e)[:100]}")
        return None


def _parse_robots_sitemaps(root_url: str) -> List[str]:
    """Parse robots.txt to extract sitemap URLs. Never fails - returns empty list on error."""
    return safe_execute(
        lambda: _parse_robots_sitemaps_impl(root_url),
        [],
    )


def _parse_robots_sitemaps_impl(root_url: str) -> List[str]:
    """Internal implementation of robots sitemap parsing."""
    try:
        robots_url = urljoin(_origin(root_url), "/robots.txt")
        resp = requests.get(robots_url, timeout=_DEFAULT_TIMEOUT, allow_redirects=True)
        if resp.status_code >= 400:
            return []
        
        # Check for bot detection/CAPTCHA
        if is_bot_detected(resp.text) or is_captcha_page(resp.text):
            logger.debug(f"robots.txt appears to be protected for {root_url}")
            return []
        
        sitemaps = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[-1].strip()
                if sitemap_url:
                    sitemaps.append(_normalize_url(sitemap_url))
        return sitemaps
    except Exception as e:
        logger.debug(f"Failed to parse robots.txt for {root_url}: {str(e)[:100]}")
        return []


def _is_url_allowed_by_robots(url: str, robots_parser: Optional[RobotFileParser], user_agent: str = "*") -> bool:
    """Check if a URL is allowed by robots.txt rules."""
    if not robots_parser:
        return True  # If no robots.txt, allow by default
    try:
        return robots_parser.can_fetch(user_agent, url)
    except Exception:
        return True  # On error, allow by default


def _is_valid_xml(text: str) -> bool:
    """Check if text is valid XML (not HTML/CAPTCHA page)."""
    text_lower = text.lower().strip()
    # Common indicators of non-XML content
    if not text_lower.startswith("<?xml") and not text_lower.startswith("<"):
        return False
    # Check for HTML/CAPTCHA indicators
    html_indicators = ["<html", "<body", "<div", "cloudflare", "captcha", "challenge", "verify you are human"]
    if any(indicator in text_lower for indicator in html_indicators):
        return False
    return True


def _parse_sitemap_urls(xml_text: str) -> List[str]:
    """Parse sitemap XML and extract URLs."""
    if not _is_valid_xml(xml_text):
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    urls = []
    for node in root.iter():
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag.endswith("loc") and node.text:
            urls.append(_normalize_url(node.text))
    return urls


async def _fetch_sitemap_with_crawl4ai(sitemap_url: str) -> Optional[str]:
    """Fetch sitemap using crawl4ai (handles JS/CAPTCHA). Never fails - returns None on error."""
    return await safe_execute_async(
        lambda: _fetch_sitemap_with_crawl4ai_impl(sitemap_url),
        None,
    )


async def _fetch_sitemap_with_crawl4ai_impl(sitemap_url: str) -> Optional[str]:
    """Internal implementation of crawl4ai sitemap fetching."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        from infrastructure.rag.crawl_service import HEADLESS
        
        browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False, wait_for="domcontentloaded")
        
        # Retry with backoff for connection errors
        result = await retry_with_backoff(
            lambda: _crawl_sitemap_single(sitemap_url, browser_config, run_config),
            max_retries=2,
            initial_delay=2.0,
        )
        
        if not result:
            return None
        
        # Try to extract XML from HTML/markdown
        content = result.get("html") or result.get("markdown") or ""
        if not content:
            return None
        
        # Check for bot detection/CAPTCHA
        if is_bot_detected(content) or is_captcha_page(content):
            logger.debug(f"Sitemap {sitemap_url} appears to be protected")
            return None
        
        # Look for XML content
        if "<?xml" in content or "<urlset" in content.lower() or "<sitemapindex" in content.lower():
            # Extract XML portion
            xml_start = content.find("<?xml")
            if xml_start >= 0:
                # Find the end tag
                urlset_end = content.find("</urlset>", xml_start)
                sitemapindex_end = content.find("</sitemapindex>", xml_start)
                
                if urlset_end > xml_start:
                    return content[xml_start:urlset_end + len("</urlset>")]
                elif sitemapindex_end > xml_start:
                    return content[xml_start:sitemapindex_end + len("</sitemapindex>")]
                else:
                    # Try to find end by looking for closing tag
                    return content[xml_start:]
            
            # If no <?xml but has urlset/sitemapindex, try to extract
            urlset_start = content.lower().find("<urlset")
            sitemapindex_start = content.lower().find("<sitemapindex")
            start_idx = max(urlset_start, sitemapindex_start) if urlset_start >= 0 or sitemapindex_start >= 0 else -1
            
            if start_idx >= 0:
                urlset_end = content.find("</urlset>", start_idx)
                sitemapindex_end = content.find("</sitemapindex>", start_idx)
                end_idx = max(urlset_end, sitemapindex_end) if urlset_end >= 0 or sitemapindex_end >= 0 else -1
                if end_idx > start_idx:
                    return content[start_idx:end_idx + (len("</urlset>") if urlset_end >= 0 else len("</sitemapindex>"))]
        
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch sitemap with crawl4ai {sitemap_url}: {str(e)[:100]}")
        return None


async def _crawl_sitemap_single(sitemap_url: str, browser_config, run_config) -> Optional[Dict[str, Any]]:
    """Single crawl attempt for sitemap."""
    from crawl4ai import AsyncWebCrawler
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=sitemap_url, config=run_config)
            if result and result.success:
                return {"html": result.html, "markdown": result.markdown}
    except Exception as e:
        logger.debug(f"Crawl attempt failed for {sitemap_url}: {str(e)[:100]}")
    return None


def _fetch_sitemap_urls(sitemap_url: str, depth: int = 0) -> Tuple[List[str], bool]:
    """
    Fetch sitemap URLs using requests (fast, but may fail on protected sites).
    Returns (urls, is_valid_xml) tuple. Never fails - returns empty on error.
    """
    if depth > _MAX_SITEMAP_DEPTH:
        return [], False
    
    return safe_execute(
        lambda: _fetch_sitemap_urls_impl(sitemap_url, depth),
        ([], False),
    )


def _fetch_sitemap_urls_impl(sitemap_url: str, depth: int = 0) -> Tuple[List[str], bool]:
    """Internal implementation of sitemap fetching."""
    try:
        resp = requests.get(
            sitemap_url,
            timeout=_DEFAULT_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebBot/1.0)"},
        )
        if resp.status_code >= 400:
            logger.warning(f"Sitemap {sitemap_url} returned status {resp.status_code}")
            return [], False
        
        # Check for bot detection/CAPTCHA
        if is_bot_detected(resp.text) or is_captcha_page(resp.text):
            logger.warning(f"Sitemap {sitemap_url} appears to be protected (CAPTCHA/bot detection)")
            return [], False
        
        # Check if response is valid XML (not HTML/CAPTCHA)
        is_valid = _is_valid_xml(resp.text)
        if not is_valid:
            logger.warning(f"Sitemap {sitemap_url} returned invalid XML (might be HTML/CAPTCHA)")
            return [], False
        
        urls = _parse_sitemap_urls(resp.text)
        if not urls:
            logger.warning(f"Sitemap {sitemap_url} parsed successfully but returned no URLs")
            return [], True  # Valid XML but empty
        
        logger.debug(f"Sitemap {sitemap_url} parsed successfully: {len(urls)} URLs found")
        
        # If this looks like a sitemap index, recurse into each sitemap URL.
        if any(url.endswith(".xml") for url in urls):
            logger.debug(f"Sitemap {sitemap_url} is a sitemap index with {len(urls)} child sitemaps")
            all_urls = []
            for child in urls:
                child_urls, _ = _fetch_sitemap_urls(child, depth + 1)
                all_urls.extend(child_urls)
            logger.info(f"Sitemap index {sitemap_url} returned {len(all_urls)} total URLs from {len(urls)} children")
            return all_urls, True
        return urls, True
    except Exception as e:
        logger.error(f"Exception fetching sitemap {sitemap_url}: {type(e).__name__}: {str(e)[:200]}")
        return [], False


async def _fetch_sitemap_urls_async(sitemap_url: str, depth: int = 0) -> List[str]:
    """Fetch sitemap URLs with fallback to crawl4ai if requests fails or gets CAPTCHA. Never fails completely."""
    if depth > _MAX_SITEMAP_DEPTH:
        logger.warning(f"Sitemap depth limit reached for {sitemap_url}")
        return []
    
    urls: List[str] = []
    
    # Try requests first (fast)
    urls, is_valid_xml = _fetch_sitemap_urls(sitemap_url, depth)
    
    # Log what we got
    if urls:
        logger.debug(f"Fetched {len(urls)} URLs from {sitemap_url} via requests (valid_xml={is_valid_xml})")
    elif not is_valid_xml:
        logger.debug(f"Invalid XML from {sitemap_url}, trying crawl4ai fallback")
    
    # If requests failed, returned invalid content (HTML/CAPTCHA), or got empty result, try crawl4ai
    if not urls or not is_valid_xml:
        logger.debug(f"Trying crawl4ai fallback for {sitemap_url}")
        xml_content = await _fetch_sitemap_with_crawl4ai(sitemap_url)
        if xml_content:
            parsed_urls = safe_execute(
                lambda: _parse_sitemap_urls(xml_content),
                [],
            )
            if parsed_urls:
                logger.debug(f"Fetched {len(parsed_urls)} URLs from {sitemap_url} via crawl4ai")
                urls = parsed_urls
            else:
                logger.warning(f"Failed to parse XML content from {sitemap_url} via crawl4ai")
        else:
            logger.warning(f"Failed to fetch {sitemap_url} via both requests and crawl4ai")
    
    # If this looks like a sitemap index, recurse into each sitemap URL.
    # Continue even if some child sitemaps fail - collect partial results
    if any(url.endswith(".xml") for url in urls):
        logger.debug(f"{sitemap_url} is a sitemap index with {len(urls)} child sitemaps")
        all_urls = []
        for child in urls:
            child_urls = await safe_execute_async(
                lambda: _fetch_sitemap_urls_async(child, depth + 1),
                [],
            )
            if child_urls:
                logger.debug(f"Child sitemap {child} returned {len(child_urls)} URLs")
                all_urls.extend(child_urls)
            else:
                logger.warning(f"Child sitemap {child} returned no URLs")
        logger.info(f"Sitemap index {sitemap_url} returned {len(all_urls)} total URLs from {len(urls)} child sitemaps")
        return all_urls
    
    return urls


def _dedupe_and_filter(
    urls: List[str],
    root_url: str,
    limit: int,
    robots_parser: Optional[RobotFileParser] = None,  # Ignored - kept for API compatibility
) -> List[str]:
    """Deduplicate and filter URLs by domain. No robots.txt filtering."""
    seen: Set[str] = set()
    filtered = []
    for url in urls:
        norm = _normalize_url(url)
        if not norm or norm in seen:
            continue
        if not _is_same_domain(norm, root_url):
            continue
        # No robots.txt filtering - we don't care about robots.txt rules
        seen.add(norm)
        filtered.append(norm)
        if len(filtered) >= limit:
            break
    return filtered


async def discover_urls_from_sitemap(root_url: str) -> List[str]:
    """
    Discover URLs from sitemap only (via robots.txt).
    Returns empty if no sitemap found or all sitemaps fail.
    """
    root_url = _ensure_url(root_url)
    if not root_url:
        return []

    # Try to get URLs from sitemap (via robots.txt - we only use robots.txt to find sitemap URLs)
    sitemap_urls = _parse_robots_sitemaps(root_url)
    sitemap_candidates = []
    
    if sitemap_urls:
        logger.info(f"Found {len(sitemap_urls)} sitemap(s) for {root_url}: {sitemap_urls}")
        # Try fetching sitemaps with async method (has crawl4ai fallback)
        # Process ALL sitemaps - don't stop early, collect all URLs
        for sitemap_url in sitemap_urls:
            logger.debug(f"Fetching sitemap: {sitemap_url}")
            fetched_urls = await safe_execute_async(
                lambda: _fetch_sitemap_urls_async(sitemap_url),
                [],
            )
            if fetched_urls:
                logger.info(f"Found {len(fetched_urls)} URLs from sitemap {sitemap_url}")
                sitemap_candidates.extend(fetched_urls)
            else:
                logger.warning(f"No URLs found from sitemap {sitemap_url} - it may be empty, protected, or invalid")
            # Don't break early - process all sitemaps even if we hit the limit
            # We'll limit after combining all results

    if sitemap_candidates:
        logger.info(f"Total URLs from all sitemaps: {len(sitemap_candidates)}")
        sitemap_candidates = _dedupe_and_filter(sitemap_candidates, root_url, _MAX_DISCOVERY_URLS, None)
        logger.info(f"After deduplication and filtering: {len(sitemap_candidates)} URLs")
        return sitemap_candidates
    
    logger.warning(f"No URLs found from sitemap for {root_url}")
    return []


async def discover_urls_auto(root_url: str) -> List[str]:
    """
    Discover URLs automatically using crawl4ai (default method).
    NEVER returns empty unless the site is truly uncrawlable.
    Handles all errors gracefully and always returns partial results.
    """
    root_url = _ensure_url(root_url)
    if not root_url:
        return []

    # Use crawl4ai discovery of internal links (handles JS/CAPTCHA)
    # This should always work
    discovered = []
    discovered = await safe_execute_async(
        lambda: discover_internal_urls(
            root_url,
            max_depth=3,
            max_concurrent=10,
            max_urls=_MAX_DISCOVERY_URLS,
        ),
        [],
    )
    
    if discovered:
        discovered = _dedupe_and_filter(discovered, root_url, _MAX_DISCOVERY_URLS, None)
        if discovered:
            logger.info(f"Discovered {len(discovered)} URLs via crawl4ai for {root_url}")
            return discovered
    
    # Last resort - return at least the root URL
    # This ensures we NEVER return empty
    logger.info(f"Returning root URL only for {root_url} (crawl4ai discovery found nothing)")
    return [root_url]


async def discover_urls(root_url: str) -> List[str]:
    """
    Legacy function - defaults to auto discovery.
    Use discover_urls_auto() or discover_urls_from_sitemap() instead.
    """
    return await discover_urls_auto(root_url)


def create_robots_filter(root_url: str) -> Callable[[str], bool]:
    """Create a filter function that always allows URLs (robots.txt filtering disabled)."""
    # Always return True - we don't care about robots.txt rules
    def filter_url(url: str) -> bool:
        return True
    
    return filter_url
