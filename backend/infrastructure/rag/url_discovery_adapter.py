"""
URL discovery adapter: implements domain UrlDiscoveryPort.
AutoUrlDiscoveryAdapter: combine multiple strategies for maximum coverage.
"""
import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

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
        return await robots_policy().filter_urls(urls)

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
                yield {"type": "error", "message": "No URLs found from sitemap."}
                yield {"type": "done", "urls": [], "method_used": "sitemap"}
                return
            allowed: List[str] = []
            for u in urls:
                if await rp.is_allowed(u):
                    allowed.append(u)
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
                u = evt["url"]
                if await rp.is_allowed(u):
                    yield evt
                continue
            if evt.get("type") == "done":
                urls = list(evt.get("urls") or [])
                allowed = await rp.filter_urls(urls)
                evt = dict(evt)
                evt["urls"] = allowed
                yield evt
                continue
            yield evt


class Crawl4AIUrlDiscoveryAdapter(UrlDiscoveryPort):
    """Implements URL discovery using crawl4ai (auto) and sitemap. Replace this class to swap discovery."""

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        method = (method or "auto").lower()
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
        else:
            urls = await discover_urls_auto(root_url)
        return await robots_policy().filter_urls(urls)

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
        rp = robots_policy()
        if method == "sitemap":
            urls = await discover_urls_from_sitemap(root_url)
            if not urls:
                yield {"type": "error", "message": "No URLs found from sitemap."}
                yield {"type": "done", "urls": [], "method_used": "sitemap"}
                return
            allowed: List[str] = []
            for u in urls:
                if await rp.is_allowed(u):
                    allowed.append(u)
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
                u = evt["url"]
                if await rp.is_allowed(u):
                    yield evt
                continue
            if evt.get("type") == "done":
                urls = list(evt.get("urls") or [])
                allowed = await rp.filter_urls(urls)
                evt = dict(evt)
                evt["urls"] = allowed
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
            return await robots_policy().filter_urls(urls)
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
                return await robots_policy().filter_urls(final)
            if evt.get("type") == "discovered" and isinstance(evt.get("url"), str):
                urls.append(evt["url"])
        return await robots_policy().filter_urls(urls)

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
                yield {"type": "error", "message": "No URLs found from sitemap."}
                yield {"type": "done", "urls": [], "method_used": "sitemap"}
                return
            for i, u in enumerate(urls, start=1):
                yield {"type": "discovered", "url": u, "count": i, "depth": 0, "method_used": "sitemap"}
            yield {"type": "done", "urls": urls, "method_used": "sitemap"}
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
            yield {"type": "error", "message": "Invalid root URL"}
            yield {"type": "done", "urls": [], "method_used": "auto"}
            return

        start = time.monotonic()
        deadline = start + max_duration_sec if max_duration_sec else None
        queue: asyncio.Queue = asyncio.Queue()
        done_marker = object()
        seen: set = set()
        collected: List[str] = []
        sources_with_hits: set = set()

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
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        source, evt = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        timed_out = True
                        break
                else:
                    source, evt = await queue.get()

                if evt is done_marker:
                    done_sources.add(source)
                    continue

                if not isinstance(evt, dict):
                    continue

                if evt.get("type") == "discovered":
                    url = evt.get("url")
                    if isinstance(url, str) and url:
                        if url in seen:
                            continue
                        if not _is_url_under_root_path(url, root_url):
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

        if not collected and root_url:
            if root_url not in seen:
                collected.append(root_url)
                yield {"type": "discovered", "url": root_url, "count": len(collected), "source": "fallback"}

        yield {
            "type": "done",
            "urls": collected,
            "method_used": "auto",
            "timed_out": timed_out,
            "sources": sorted(sources_with_hits),
        }
