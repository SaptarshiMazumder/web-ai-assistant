"""
URL discovery via ProjectDiscovery URLFinder (passive sources: Wayback, Common Crawl, etc.).
Runs the urlfinder binary; fast for sites already indexed. No browser required.
Install: go install -v github.com/projectdiscovery/urlfinder/cmd/urlfinder@latest
"""
import asyncio
import json
import logging
import shutil
from typing import List, Set
from urllib.parse import urlparse, urldefrag, urlunparse

logger = logging.getLogger(__name__)

URLFINDER_BINARY = "urlfinder"
_URLFINDER_TIMEOUT_SEC = 120  # max time for urlfinder run
_URLFINDER_MAX_TIME_MIN = 5   # -max-time passed to binary (minutes)


def _domain_from_url(root_url: str) -> str:
    """Extract host (domain) from URL for urlfinder -d flag."""
    root_url = (root_url or "").strip()
    if not root_url.startswith(("http://", "https://")):
        root_url = "https://" + root_url
    parsed = urlparse(root_url)
    return (parsed.netloc or "").strip() or ""


def _normalize_url(url: str) -> str:
    """Strip fragment; keep scheme, netloc, path, query."""
    url = urldefrag(url)[0]
    try:
        p = urlparse(url)
        path = (p.path or "/").strip() or "/"
        return urlunparse((p.scheme, p.netloc, path, "", p.query, ""))
    except Exception:
        return url


def _is_same_netloc(url: str, root_url: str) -> bool:
    return urlparse(url).netloc == urlparse(root_url).netloc


def _urlfinder_available() -> bool:
    """Return True if urlfinder binary is on PATH."""
    return shutil.which(URLFINDER_BINARY) is not None


async def discover_urls_urlfinder(
    root_url: str,
    *,
    max_urls: int = 2000,
    timeout_sec: int = _URLFINDER_TIMEOUT_SEC,
    max_time_min: int = _URLFINDER_MAX_TIME_MIN,
) -> List[str]:
    """
    Run urlfinder -d <domain> -j and parse JSONL stdout.
    Returns same-host URLs, deduped and normalized, capped at max_urls.
    Returns [] if binary missing, timeout, or parse error.
    """
    root_url = (root_url or "").strip()
    if not root_url.startswith(("http://", "https://")):
        root_url = "https://" + root_url
    domain = _domain_from_url(root_url)
    if not domain:
        return []

    if not _urlfinder_available():
        logger.debug("URLFinder binary not found on PATH; skipping passive discovery")
        return []

    cmd = [
        URLFINDER_BINARY,
        "-d", domain,
        "-j",
        "-timeout", "30",
        "-max-time", str(max_time_min),
        "-silent",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("URLFinder timed out for %s after %s s", domain, timeout_sec)
        return []
    except FileNotFoundError:
        logger.debug("URLFinder binary not found")
        return []
    except Exception as e:
        logger.warning("URLFinder failed for %s: %s", domain, e)
        return []

    if proc.returncode != 0 and stderr:
        logger.debug("URLFinder stderr for %s: %s", domain, stderr.decode("utf-8", errors="replace")[:300])

    seen: Set[str] = set()
    urls: List[str] = []
    for line in (stdout or b"").decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            u = (obj.get("url") or "").strip()
        except (json.JSONDecodeError, TypeError):
            continue
        if not u or not u.startswith(("http://", "https://")):
            continue
        if not _is_same_netloc(u, root_url):
            continue
        norm = _normalize_url(u)
        if norm in seen:
            continue
        seen.add(norm)
        urls.append(norm)
        if len(urls) >= max_urls:
            break

    if urls:
        logger.info("URLFinder found %d URLs for %s (domain %s)", len(urls), root_url, domain)
    return urls
