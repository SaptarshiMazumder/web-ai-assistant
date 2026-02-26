"""
URL discovery adapter: implements domain UrlDiscoveryPort.
AutoUrlDiscoveryAdapter: combine multiple strategies for maximum coverage.
"""
import asyncio
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

from domain.platform_profiles import normalize_url_for_crawl, resolve_platform_profile, should_allow_url
from domain.repositories import UrlDiscoveryPort

from infrastructure.rag.crawl_service import _is_url_under_root_path, discover_internal_urls_stream
from infrastructure.rag.http_discovery import discover_internal_urls_http, discover_internal_urls_http_stream
from infrastructure.rag.url_discovery_service import discover_urls_auto, discover_urls_from_sitemap
from infrastructure.rag.urlfinder_discovery import discover_urls_urlfinder
from infrastructure.rag.robots_policy import robots_policy

logger = logging.getLogger(__name__)


def _ensure_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _normalize_discovery_path(url: str) -> str:
    """
    Normalize URL path for discovery dedupe:
    - remove fragment
    - keep root as "/"
    - for path-like pages (no file extension), prefer trailing slash
    """
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            return url
        path = p.path or "/"
        if not path:
            path = "/"
        if path != "/" and not path.endswith("/"):
            leaf = path.rsplit("/", 1)[-1]
            # Keep likely file resources unchanged (e.g., .pdf, .xml).
            if not re.search(r"\.[a-z0-9]{1,8}$", leaf, re.IGNORECASE):
                path = f"{path}/"
        return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))
    except Exception:
        return url


def _canonical_discovery_url(url: str) -> str:
    """
    Canonicalize URL for discovery output/dedup.
    For platforms that mark strip_query_params=True (e.g. Hotpepper),
    remove query/fragment so UI-state variants collapse into one URL.
    """
    u = (url or "").strip()
    if not u:
        return ""
    profile, _ = resolve_platform_profile(u)
    if profile is not None and profile.strip_query_params:
        u = normalize_url_for_crawl(u)
    return _normalize_discovery_path(u)


def _passes_discovery_policy(url: str, root_url: str) -> bool:
    """Single URL gate for discovery outputs (scope + platform profile rules)."""
    canonical = _canonical_discovery_url(url)
    if not canonical:
        return False
    if not _is_url_under_root_path(canonical, root_url):
        return False
    return should_allow_url(canonical)


async def _filter_urls_for_discovery(
    urls: List[str],
    *,
    root_url: str,
    check_robots: bool,
) -> Tuple[List[str], int, int]:
    """
    Filter URLs by scope/profile policy and optionally robots.txt.

    Returns:
      (allowed_urls, blocked_by_profile_count, blocked_by_robots_count)
    """
    allowed: List[str] = []
    seen: Set[str] = set()
    blocked_by_profile = 0
    blocked_by_robots = 0
    rp = robots_policy() if check_robots else None

    for raw in urls:
        if not isinstance(raw, str):
            continue
        url = _canonical_discovery_url(raw)
        if not url or url in seen:
            continue
        if not _passes_discovery_policy(url, root_url):
            blocked_by_profile += 1
            continue
        if rp is not None and not await rp.is_allowed(url):
            blocked_by_robots += 1
            continue
        seen.add(url)
        allowed.append(url)

    return allowed, blocked_by_profile, blocked_by_robots


class HttpUrlDiscoveryAdapter(UrlDiscoveryPort):
    """HTTP + HTML parsing discovery (fast, deterministic). Used for 'auto'. Sitemap for 'sitemap'."""

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
        else:
            urls = await discover_internal_urls_http(
                root_url, max_depth=10, max_concurrent=50, max_urls=2000
            )
        allowed, _, _ = await _filter_urls_for_discovery(
            urls,
            root_url=root_url,
            check_robots=True,
        )
        return allowed

    def discover_stream(
        self,
        root_url: str,
        method: str = "auto",
        *,
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        return self._discover_stream_impl(
            root_url,
            method=method,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        )

    async def _discover_stream_impl(
        self,
        root_url: str,
        *,
        method: str = "auto",
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        rp = robots_policy()
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
            if not urls:
                yield {"type": "error", "message": "No URLs found from sitemap.", "failure_reason": "sitemap_empty"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "sitemap_empty"}
                return
            allowed, blocked_by_profile, blocked_by_robots = await _filter_urls_for_discovery(
                urls,
                root_url=root_url,
                check_robots=True,
            )
            if not allowed and blocked_by_robots > 0 and blocked_by_profile == 0:
                yield {"type": "error", "message": f"All {blocked_by_robots} URLs from sitemap are blocked by robots.txt", "failure_reason": "robots_blocked"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "robots_blocked"}
                return
            for i, u in enumerate(allowed, start=1):
                yield {"type": "discovered", "url": u, "count": i, "depth": 0, "method_used": "sitemap"}
            yield {"type": "done", "urls": allowed, "method_used": "sitemap"}
            return
        async for evt in discover_internal_urls_http_stream(
            root_url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        ):
            if evt.get("type") == "discovered" and isinstance(evt.get("url"), str):
                u = _canonical_discovery_url(evt["url"])
                if _passes_discovery_policy(u, root_url) and await rp.is_allowed(u):
                    evt = dict(evt)
                    evt["url"] = u
                    yield evt
                continue
            if evt.get("type") == "done":
                urls = list(evt.get("urls") or [])
                allowed, blocked_by_profile, blocked_by_robots = await _filter_urls_for_discovery(
                    urls,
                    root_url=root_url,
                    check_robots=True,
                )
                evt = dict(evt)
                evt["urls"] = allowed
                if not allowed and urls and blocked_by_robots > 0 and blocked_by_profile == 0:
                    evt["failure_reason"] = "robots_blocked"
                    yield {"type": "error", "message": f"All {len(urls)} discovered URLs are blocked by robots.txt", "failure_reason": "robots_blocked"}
                yield evt
                continue
            yield evt


class Crawl4AIUrlDiscoveryAdapter(UrlDiscoveryPort):
    """Implements URL discovery using crawl4ai (auto) and sitemap. Replace this class to swap discovery."""

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
        else:
            urls = await discover_urls_auto(root_url)
        allowed, _, _ = await _filter_urls_for_discovery(
            urls,
            root_url=root_url,
            check_robots=True,
        )
        return allowed

    def discover_stream(
        self,
        root_url: str,
        method: str = "auto",
        *,
        max_depth: int = 10,
        max_concurrent: int = 10,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Returns an async generator; use async for evt in port.discover_stream(...)."""
        return self._discover_stream_impl(
            root_url,
            method=method,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        )

    async def _discover_stream_impl(
        self,
        root_url: str,
        *,
        method: str = "auto",
        max_depth: int = 10,
        max_concurrent: int = 10,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        rp = robots_policy()
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
            if not urls:
                yield {"type": "error", "message": "No URLs found from sitemap.", "failure_reason": "sitemap_empty"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "sitemap_empty"}
                return
            allowed, blocked_by_profile, blocked_by_robots = await _filter_urls_for_discovery(
                urls,
                root_url=root_url,
                check_robots=True,
            )
            if not allowed and blocked_by_robots > 0 and blocked_by_profile == 0:
                yield {"type": "error", "message": f"All {blocked_by_robots} URLs from sitemap are blocked by robots.txt", "failure_reason": "robots_blocked"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "robots_blocked"}
                return
            for i, u in enumerate(allowed, start=1):
                yield {"type": "discovered", "url": u, "count": i, "depth": 0, "method_used": "sitemap"}
            yield {"type": "done", "urls": allowed, "method_used": "sitemap"}
            return
        async for evt in discover_internal_urls_stream(
            root_url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        ):
            if evt.get("type") == "discovered" and isinstance(evt.get("url"), str):
                u = _canonical_discovery_url(evt["url"])
                if _passes_discovery_policy(u, root_url) and await rp.is_allowed(u):
                    evt = dict(evt)
                    evt["url"] = u
                    yield evt
                continue
            if evt.get("type") == "done":
                urls = list(evt.get("urls") or [])
                allowed, blocked_by_profile, blocked_by_robots = await _filter_urls_for_discovery(
                    urls,
                    root_url=root_url,
                    check_robots=True,
                )
                evt = dict(evt)
                evt["urls"] = allowed
                if not allowed and urls and blocked_by_robots > 0 and blocked_by_profile == 0:
                    evt["failure_reason"] = "robots_blocked"
                    yield {"type": "error", "message": f"All {len(urls)} discovered URLs are blocked by robots.txt", "failure_reason": "robots_blocked"}
                yield evt
                continue
            yield evt


class AutoUrlDiscoveryAdapter(UrlDiscoveryPort):
    """
    Auto: combine URLFinder (if available), sitemap, HTTP crawl, and crawl4ai
    to maximize coverage within the time budget.
    """

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
            allowed, _, _ = await _filter_urls_for_discovery(
                urls,
                root_url=root_url,
                check_robots=True,
            )
            return allowed
        urls: List[str] = []
        async for evt in self._discover_stream_impl(
            root_url,
            method=method,
            max_depth=10,
            max_concurrent=50,
            max_urls=2000,
            max_duration_sec=60,
        ):
            if evt.get("type") == "done":
                final = list(evt.get("urls") or [])
                allowed, _, _ = await _filter_urls_for_discovery(
                    final,
                    root_url=root_url,
                    check_robots=True,
                )
                return allowed
            if evt.get("type") == "discovered" and isinstance(evt.get("url"), str):
                urls.append(evt["url"])
        allowed, _, _ = await _filter_urls_for_discovery(
            urls,
            root_url=root_url,
            check_robots=True,
        )
        return allowed

    def discover_stream(
        self,
        root_url: str,
        method: str = "auto",
        *,
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        return self._discover_stream_impl(
            root_url,
            method=method,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        )

    async def _discover_stream_impl(
        self,
        root_url: str,
        *,
        method: str = "auto",
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
        max_duration_sec: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        method = (method or "auto").lower()
        root_url = _ensure_url(root_url or "")
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
            if not urls:
                yield {"type": "error", "message": "No URLs found from sitemap.", "failure_reason": "sitemap_empty"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "sitemap_empty"}
                return
            allowed, blocked_by_profile, blocked_by_robots = await _filter_urls_for_discovery(
                urls,
                root_url=root_url,
                check_robots=True,
            )
            if not allowed and blocked_by_robots > 0 and blocked_by_profile == 0:
                yield {"type": "error", "message": f"All {blocked_by_robots} URLs from sitemap are blocked by robots.txt", "failure_reason": "robots_blocked"}
                yield {"type": "done", "urls": [], "method_used": "sitemap", "failure_reason": "robots_blocked"}
                return
            for i, u in enumerate(allowed, start=1):
                yield {"type": "discovered", "url": u, "count": i, "depth": 0, "method_used": "sitemap"}
            yield {"type": "done", "urls": allowed, "method_used": "sitemap"}
            return
        async for evt in self._discover_stream_auto(
            root_url=root_url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            max_urls=max_urls,
            max_duration_sec=max_duration_sec,
        ):
            yield evt

    async def _discover_stream_auto(
        self,
        *,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        max_urls: int,
        max_duration_sec: Optional[int],
    ) -> AsyncIterator[Dict[str, Any]]:
        if not root_url:
            yield {"type": "error", "message": "Invalid root URL", "failure_reason": "invalid_url"}
            yield {"type": "done", "urls": [], "method_used": "auto", "failure_reason": "invalid_url"}
            return

        start = time.monotonic()
        deadline = start + max_duration_sec if max_duration_sec else None
        queue: asyncio.Queue = asyncio.Queue()
        done_marker = object()
        seen: set = set()
        collected: List[str] = []
        sources_with_hits: set = set()
        no_results_warning_sent = False
        NO_RESULTS_WARNING_THRESHOLD = 45.0  # Warn after 45 seconds with 0 results
        rp = robots_policy()

        async def emit_list(source: str, urls: List[str]) -> None:
            for u in urls:
                await queue.put((source, {"type": "discovered", "url": u}))
            await queue.put((source, done_marker))

        async def run_sitemap() -> None:
            try:
                urls = await discover_urls_from_sitemap(root_url)
            except Exception as exc:
                logger.debug("Sitemap discovery failed for %s: %s", root_url, exc)
                urls = []
            await emit_list("sitemap", urls)

        async def run_urlfinder() -> None:
            try:
                kwargs = {"max_urls": max_urls}
                if max_duration_sec is not None:
                    timeout_sec = max(5, int(min(30, max_duration_sec * 0.6)))
                    max_time_min = max(1, int((timeout_sec + 59) / 60))
                    kwargs["timeout_sec"] = timeout_sec
                    kwargs["max_time_min"] = max_time_min
                urls = await discover_urls_urlfinder(root_url, **kwargs)
            except Exception as exc:
                logger.debug("URLFinder discovery failed for %s: %s", root_url, exc)
                urls = []
            await emit_list("urlfinder", urls)

        async def run_http() -> None:
            try:
                async for evt in discover_internal_urls_http_stream(
                    root_url,
                    max_depth=max_depth,
                    max_concurrent=max_concurrent,
                    max_urls=max_urls,
                    max_duration_sec=max_duration_sec,
                ):
                    await queue.put(("http", evt))
            except Exception as exc:
                logger.debug("HTTP discovery stream failed for %s: %s", root_url, exc)
            finally:
                await queue.put(("http", done_marker))

        async def run_crawl4ai() -> None:
            try:
                async for evt in discover_internal_urls_stream(
                    root_url,
                    max_depth=max_depth,
                    max_concurrent=10,
                    max_urls=max_urls,
                    max_duration_sec=max_duration_sec,
                ):
                    await queue.put(("crawl4ai", evt))
            except Exception as exc:
                logger.debug("crawl4ai discovery stream failed for %s: %s", root_url, exc)
            finally:
                await queue.put(("crawl4ai", done_marker))

        tasks = [
            asyncio.create_task(run_urlfinder()),
            asyncio.create_task(run_sitemap()),
            asyncio.create_task(run_http()),
            asyncio.create_task(run_crawl4ai()),
        ]
        done_sources: set = set()
        timed_out = False
        try:
            while len(done_sources) < len(tasks):
                # Check if we should send "no results" warning
                elapsed = time.monotonic() - start
                if not no_results_warning_sent and elapsed >= NO_RESULTS_WARNING_THRESHOLD and not collected:
                    no_results_warning_sent = True
                    yield {
                        "type": "warning",
                        "message": f"Still finding URLs... No results after {int(NO_RESULTS_WARNING_THRESHOLD)}s. Site may be blocking crawlers or have no discoverable links.",
                        "failure_reason": "slow_no_results"
                    }

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        source, evt = await asyncio.wait_for(queue.get(), timeout=min(remaining, 2.0))
                    except asyncio.TimeoutError:
                        if remaining <= 0:
                            timed_out = True
                            break
                        continue  # Check warning threshold and try again
                else:
                    try:
                        source, evt = await asyncio.wait_for(queue.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue  # Check warning threshold and try again

                if evt is done_marker:
                    done_sources.add(source)
                    continue

                if not isinstance(evt, dict):
                    continue

                if evt.get("type") == "discovered":
                    raw_url = evt.get("url")
                    url = _canonical_discovery_url(raw_url) if isinstance(raw_url, str) else ""
                    if isinstance(url, str) and url:
                        if url in seen:
                            continue
                        if not _passes_discovery_policy(url, root_url):
                            continue
                        if not await rp.is_allowed(url):
                            continue
                        seen.add(url)
                        collected.append(url)
                        sources_with_hits.add(source)
                        yield {"type": "discovered", "url": url, "count": len(collected), "source": source}
                        if len(collected) >= max_urls:
                            break
                elif evt.get("type") == "error":
                    message = evt.get("message")
                    if isinstance(message, str) and message:
                        logger.debug("Discovery %s error for %s: %s", source, root_url, message)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        fallback_only = False
        if not collected and root_url:
            if root_url not in seen and _passes_discovery_policy(root_url, root_url) and await rp.is_allowed(root_url):
                collected.append(root_url)
                fallback_only = True
                yield {"type": "discovered", "url": root_url, "count": len(collected), "source": "fallback"}

        done_evt = {
            "type": "done",
            "urls": collected,
            "method_used": "auto",
            "timed_out": timed_out,
            "sources": sorted(sources_with_hits),
        }
        # Report no_results if we only have fallback URL, or if nothing found after warning threshold.
        if fallback_only or (not collected and elapsed >= NO_RESULTS_WARNING_THRESHOLD):
            done_evt["failure_reason"] = "no_results"
            done_evt["fallback_only"] = fallback_only
        yield done_evt
