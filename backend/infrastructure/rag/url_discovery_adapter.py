"""
URL discovery adapter: implements domain UrlDiscoveryPort using crawl4ai (and sitemap).
Swap this implementation to use a different discovery service without touching API or domain.
"""
from typing import Any, AsyncIterator, Dict, List

from domain.repositories import UrlDiscoveryPort

from infrastructure.rag.crawl_service import discover_internal_urls_stream
from infrastructure.rag.url_discovery_service import discover_urls_auto, discover_urls_from_sitemap


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
