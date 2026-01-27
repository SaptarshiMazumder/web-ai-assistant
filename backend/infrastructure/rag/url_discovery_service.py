from typing import List, Set
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree as ET

import requests

from infrastructure.rag.crawl_service import discover_internal_urls

_DEFAULT_TIMEOUT = 8
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


def _parse_robots_sitemaps(root_url: str) -> List[str]:
    try:
        robots_url = urljoin(_origin(root_url), "/robots.txt")
        resp = requests.get(robots_url, timeout=_DEFAULT_TIMEOUT)
        if resp.status_code >= 400:
            return []
        sitemaps = []
        for line in resp.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[-1].strip()
                if sitemap_url:
                    sitemaps.append(_normalize_url(sitemap_url))
        return sitemaps
    except Exception:
        return []


def _parse_sitemap_urls(xml_text: str) -> List[str]:
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


def _fetch_sitemap_urls(sitemap_url: str, depth: int = 0) -> List[str]:
    if depth > _MAX_SITEMAP_DEPTH:
        return []
    try:
        resp = requests.get(sitemap_url, timeout=_DEFAULT_TIMEOUT)
        if resp.status_code >= 400:
            return []
        urls = _parse_sitemap_urls(resp.text)
        # If this looks like a sitemap index, recurse into each sitemap URL.
        if any(url.endswith(".xml") for url in urls):
            all_urls = []
            for child in urls:
                all_urls.extend(_fetch_sitemap_urls(child, depth + 1))
            return all_urls
        return urls
    except Exception:
        return []


def _dedupe_and_filter(urls: List[str], root_url: str, limit: int) -> List[str]:
    seen: Set[str] = set()
    filtered = []
    for url in urls:
        norm = _normalize_url(url)
        if not norm or norm in seen:
            continue
        if not _is_same_domain(norm, root_url):
            continue
        seen.add(norm)
        filtered.append(norm)
        if len(filtered) >= limit:
            break
    return filtered


async def discover_urls(root_url: str) -> List[str]:
    root_url = _ensure_url(root_url)
    if not root_url:
        return []

    sitemap_urls = _parse_robots_sitemaps(root_url)
    sitemap_candidates = []
    for sitemap_url in sitemap_urls:
        sitemap_candidates.extend(_fetch_sitemap_urls(sitemap_url))
        if len(sitemap_candidates) >= _MAX_SITEMAP_URLS:
            break

    sitemap_candidates = _dedupe_and_filter(sitemap_candidates, root_url, _MAX_DISCOVERY_URLS)
    if sitemap_candidates:
        return sitemap_candidates

    # Fallback: crawl4ai discovery of internal links.
    discovered = await discover_internal_urls(
        root_url,
        max_depth=3,
        max_concurrent=10,
        max_urls=_MAX_DISCOVERY_URLS,
    )
    return _dedupe_and_filter(discovered, root_url, _MAX_DISCOVERY_URLS)
