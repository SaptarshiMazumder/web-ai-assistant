"""
HTTP-based URL discovery: BFS with plain HTTP + HTML parsing (no browser).
Fast, deterministic, same event shapes as browser-based discovery for drop-in replacement.
"""
import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup

from infrastructure.rag.crawl_service import _is_probably_page_url, _is_url_under_root_path, _normalize_url

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
_MAX_URLS_DEFAULT = 2000
_FETCH_RETRIES = 3
_FETCH_RETRY_BACKOFF = (0.5, 1.0, 2.0)  # seconds between retries


def _extract_links_from_html(html: str, base_url: str, root_netloc: str) -> Set[str]:
    """Parse HTML and return same-domain, normalized, probably-page URLs from <a href>."""
    out: Set[str] = set()
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            try:
                absolute = urljoin(base_url, href)
                absolute = urldefrag(absolute)[0]
                parsed = urlparse(absolute)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    continue
                if parsed.netloc != root_netloc:
                    continue
                norm = _normalize_url(absolute)
                if norm and _is_probably_page_url(norm, root_netloc=root_netloc):
                    out.add(norm)
            except Exception:
                continue
    except Exception as e:
        logger.debug("HTTP discovery: failed to parse HTML for %s: %s", base_url, e)
    return out


async def _fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    """GET url and return response text, or None on failure."""
    try:
        r = await client.get(url, follow_redirects=True, timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        return (r.text or "") if r.headers.get("content-type", "").lower().startswith("text/") else None
    except Exception as e:
        logger.debug("HTTP discovery: fetch failed for %s: %s", url, type(e).__name__)
        return None


async def _fetch_html_with_retries(client: httpx.AsyncClient, url: str) -> str | None:
    """GET url with retries so transient failures don't shrink the candidate set (more stable counts)."""
    last_err: Exception | None = None
    for attempt in range(_FETCH_RETRIES):
        html = await _fetch_html(client, url)
        if html is not None:
            return html
        if attempt < _FETCH_RETRIES - 1 and attempt < len(_FETCH_RETRY_BACKOFF):
            await asyncio.sleep(_FETCH_RETRY_BACKOFF[attempt])
    return None


async def discover_internal_urls_http(
    root_url: str,
    *,
    max_depth: int = 10,
    max_concurrent: int = 50,
    max_urls: int = _MAX_URLS_DEFAULT,
) -> List[str]:
    """
    BFS discovery using HTTP + HTML parsing. Deterministic, fast.
    Returns sorted list of same-domain URLs (no browser).
    """
    root_url = (root_url or "").strip()
    if not root_url or not root_url.startswith(("http://", "https://")):
        return []
    try:
        parsed_root = urlparse(root_url)
        root_netloc = parsed_root.netloc
    except Exception:
        return []

    def is_internal(url: str) -> bool:
        return _is_url_under_root_path(url, root_url)

    discovered: Set[str] = set()
    visited: Set[str] = set()
    current_level: Set[str] = {_normalize_url(root_url)}
    sem = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_DEFAULT_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (compatible; WebAIBot/1.0)"},
    ) as client:
        for depth in range(max_depth):
            if not current_level or len(discovered) >= max_urls:
                break
            to_fetch = sorted(
                u for u in current_level if u not in visited and _is_probably_page_url(u, root_netloc=root_netloc)
            )
            if not to_fetch:
                break
            next_level: Set[str] = set()
            level_norms: Set[str] = set()

            async def fetch_one(url: str) -> None:
                async with sem:
                    norm = _normalize_url(url)
                    if norm in visited:
                        return
                    visited.add(norm)
                    html = await _fetch_html_with_retries(client, url)
                    if not html:
                        return
                    if _is_probably_page_url(norm, root_netloc=root_netloc):
                        level_norms.add(norm)
                    for link in _extract_links_from_html(html, url, root_netloc):
                        if link not in visited and is_internal(link):
                            next_level.add(link)

            await asyncio.gather(*[fetch_one(u) for u in to_fetch])

            candidates = discovered | level_norms | next_level
            discovered = set(sorted(candidates)[:max_urls])
            current_level = discovered - visited

    if not discovered:
        root_norm = _normalize_url(root_url)
        if root_norm:
            discovered.add(root_norm)
    return sorted(discovered)


async def discover_internal_urls_http_stream(
    root_url: str,
    *,
    max_depth: int = 10,
    max_concurrent: int = 50,
    max_urls: int = _MAX_URLS_DEFAULT,
    max_duration_sec: Optional[int] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Same as discover_internal_urls_http but yields events for UI (same shapes as crawl4ai stream).
    Enforces max_duration_sec timeout on the backend to prevent infinite discovery processes.
    """
    root_url = (root_url or "").strip()
    if not root_url or not root_url.startswith(("http://", "https://")):
        yield {"type": "error", "message": "Invalid root URL"}
        yield {"type": "done", "urls": []}
        return

    try:
        parsed_root = urlparse(root_url)
        root_netloc = parsed_root.netloc
    except Exception:
        yield {"type": "error", "message": "Invalid root URL"}
        yield {"type": "done", "urls": []}
        return

    def is_internal(url: str) -> bool:
        return _is_url_under_root_path(url, root_url)

    yield {"type": "start", "root_url": root_url, "max_depth": max_depth, "max_urls": max_urls}

    discovered_set: Set[str] = set()
    visited: Set[str] = set()
    current_level: Set[str] = {_normalize_url(root_url)}
    sem = asyncio.Semaphore(max_concurrent)
    yielded: Set[str] = set()
    timed_out = False
    start_time = time.monotonic()

    async def _discovery_logic() -> None:
        nonlocal discovered_set, visited, current_level, yielded, timed_out
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_DEFAULT_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebAIBot/1.0)"},
        ) as client:
            for depth in range(max_depth):
                # Check timeout at start of each depth level
                if max_duration_sec is not None and (time.monotonic() - start_time) >= max_duration_sec:
                    logger.warning(
                        "HTTP discovery TIMEOUT for %s after %.1fs (depth=%d, discovered=%d). Returning partial results.",
                        root_url, time.monotonic() - start_time, depth, len(discovered_set)
                    )
                    timed_out = True
                    break
                    
                if not current_level or len(discovered_set) >= max_urls:
                    break
                to_fetch = sorted(
                    u for u in current_level if u not in visited and _is_probably_page_url(u, root_netloc=root_netloc)
                )
                if not to_fetch:
                    break
                next_level: Set[str] = set()
                level_norms: Set[str] = set()

                async def fetch_one(url: str) -> None:
                    async with sem:
                        norm = _normalize_url(url)
                        if norm in visited:
                            return
                        visited.add(norm)
                        html = await _fetch_html_with_retries(client, url)
                        if not html:
                            return
                        if _is_probably_page_url(norm, root_netloc=root_netloc):
                            level_norms.add(norm)
                        for link in _extract_links_from_html(html, url, root_netloc):
                            if link not in visited and is_internal(link):
                                next_level.add(link)

                await asyncio.gather(*[fetch_one(u) for u in to_fetch])

                candidates = discovered_set | level_norms | next_level
                discovered_set = set(sorted(candidates)[:max_urls])
                for u in sorted(discovered_set - yielded):
                    yielded.add(u)
                    yield {"type": "discovered", "url": u, "count": len(discovered_set), "depth": depth}
                current_level = discovered_set - visited

    try:
        if max_duration_sec is not None:
            # CRITICAL: Hard timeout enforced on backend to prevent runaway discovery
            try:
                async for item in _discovery_logic():
                    yield item
            except asyncio.CancelledError:
                logger.warning(
                    "HTTP discovery CANCELLED for %s after %.1fs (discovered=%d). Task was externally cancelled.",
                    root_url, time.monotonic() - start_time, len(discovered_set)
                )
                timed_out = True
                raise
        else:
            # No timeout specified, run to completion
            async for item in _discovery_logic():
                yield item

        if not discovered_set:
            root_norm = _normalize_url(root_url)
            if root_norm:
                discovered_set.add(root_norm)
                if root_norm not in yielded:
                    yield {"type": "discovered", "url": root_norm, "count": 1, "depth": 0}
        
        elapsed = time.monotonic() - start_time
        discovered_sorted = sorted(discovered_set)
        
        if timed_out:
            logger.info(
                "HTTP discovery COMPLETED with timeout for %s: discovered=%d URLs in %.1fs",
                root_url, len(discovered_sorted), elapsed
            )
        else:
            logger.info(
                "HTTP discovery COMPLETED successfully for %s: discovered=%d URLs in %.1fs",
                root_url, len(discovered_sorted), elapsed
            )
        
        yield {"type": "done", "urls": discovered_sorted, "timed_out": timed_out}
        
    except asyncio.CancelledError:
        # Task was cancelled (user reload, server shutdown, etc)
        elapsed = time.monotonic() - start_time
        logger.error(
            "HTTP discovery CANCELLED/ABANDONED for %s after %.1fs (discovered=%d URLs). Frontend may have disconnected.",
            root_url, elapsed, len(discovered_set)
        )
        discovered_sorted = sorted(discovered_set) if discovered_set else []
        yield {"type": "done", "urls": discovered_sorted, "timed_out": True, "cancelled": True}
        raise
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.exception(
            "HTTP discovery FAILED for %s after %.1fs (discovered=%d URLs): %s",
            root_url, elapsed, len(discovered_set), e
        )
        yield {"type": "error", "message": f"{type(e).__name__}: {str(e)}"}
        yield {"type": "done", "urls": sorted(discovered_set) if discovered_set else [], "timed_out": False}
