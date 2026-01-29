"""
URL discovery adapter: implements domain UrlDiscoveryPort.
Default is HttpUrlDiscoveryAdapter (HTTP + HTML, fast). Crawl4AIUrlDiscoveryAdapter (browser) kept for fallback.
"""
from typing import Any, AsyncIterator, Dict, List

from domain.repositories import UrlDiscoveryPort

from infrastructure.rag.crawl_service import discover_internal_urls_stream
from infrastructure.rag.http_discovery import discover_internal_urls_http, discover_internal_urls_http_stream
from infrastructure.rag.url_discovery_service import discover_urls_auto, discover_urls_from_sitemap


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
            return await discover_urls_from_sitemap(root_url)
        return await discover_internal_urls_http(
            root_url, max_depth=10, max_concurrent=50, max_urls=2000
        )

    def discover_stream(
        self,
        root_url: str,
        method: str = "auto",
        *,
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
    ) -> AsyncIterator[Dict[str, Any]]:
        return self._discover_stream_impl(
            root_url, method=method, max_depth=max_depth, max_concurrent=max_concurrent, max_urls=max_urls
        )

    async def _discover_stream_impl(
        self,
        root_url: str,
        *,
        method: str = "auto",
        max_depth: int = 10,
        max_concurrent: int = 50,
        max_urls: int = 2000,
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
        async for evt in discover_internal_urls_http_stream(
            root_url, max_depth=max_depth, max_concurrent=max_concurrent, max_urls=max_urls
        ):
            yield evt


class Crawl4AIUrlDiscoveryAdapter(UrlDiscoveryPort):
    """Implements URL discovery using crawl4ai (auto) and sitemap. Replace this class to swap discovery."""

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        method = (method or "auto").lower()
        if method == "sitemap":
            return await discover_urls_from_sitemap(root_url)
        return await discover_urls_auto(root_url)

    def discover_stream(
        self,
        root_url: str,
        method: str = "auto",
        *,
        max_depth: int = 10,
        max_concurrent: int = 10,
        max_urls: int = 2000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Returns an async generator; use async for evt in port.discover_stream(...)."""
        return self._discover_stream_impl(
            root_url, method=method, max_depth=max_depth, max_concurrent=max_concurrent, max_urls=max_urls
        )

    async def _discover_stream_impl(
        self,
        root_url: str,
        *,
        method: str = "auto",
        max_depth: int = 10,
        max_concurrent: int = 10,
        max_urls: int = 2000,
    ) -> AsyncIterator[Dict[str, Any]]:
        method = (method or "auto").lower()
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
        async for evt in discover_internal_urls_stream(
            root_url, max_depth=max_depth, max_concurrent=max_concurrent, max_urls=max_urls
        ):
            yield evt
