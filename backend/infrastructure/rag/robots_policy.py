import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


def _origin(url: str) -> str:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


@dataclass(frozen=True)
class RobotsConfig:
    user_agent: str = ""
    timeout_sec: float = 0.0
    cache_ttl_sec: int = 0
    max_bytes: int = 512 * 1024  # 512KB safety cap

    @staticmethod
    def from_env() -> "RobotsConfig":
        ua = (os.environ.get("ROBOTS_USER_AGENT") or "").strip() or "WebAIbot"
        try:
            timeout_sec = float(os.environ.get("ROBOTS_TIMEOUT_SEC", "10"))
        except Exception:
            timeout_sec = 10.0
        try:
            ttl = int(os.environ.get("ROBOTS_CACHE_TTL_SEC", "3600"))
        except Exception:
            ttl = 3600
        ttl = max(30, ttl)
        timeout_sec = max(1.0, timeout_sec)
        cfg = RobotsConfig(
            user_agent=ua,
            timeout_sec=timeout_sec,
            cache_ttl_sec=ttl,
        )
        return cfg


@dataclass
class _CacheEntry:
    expires_at: float
    parser: Optional[RobotFileParser]  # None means \"allow all\" per policy
    fetched_at: float
    status_code: Optional[int]
    error: Optional[str]


class RobotsPolicy:
    """
    Central robots.txt policy with caching.

    Policy:
    - If robots.txt exists and can be parsed -> enforce can_fetch(user_agent, url)
    - If robots.txt is missing/unreadable -> allow all (keep current behavior)
    """

    def __init__(self, config: Optional[RobotsConfig] = None) -> None:
        self._cfg = config or RobotsConfig.from_env()
        self._cache: Dict[str, _CacheEntry] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, origin: str) -> asyncio.Lock:
        lock = self._locks.get(origin)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[origin] = lock
        return lock

    async def _fetch_robots_text(self, robots_url: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        headers = {
            "User-Agent": self._cfg.user_agent,
            "Accept": "text/plain,*/*;q=0.8",
        }
        timeout = httpx.Timeout(self._cfg.timeout_sec)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                resp = await client.get(robots_url)
                status = int(resp.status_code)
                # Treat 404/410 as \"missing\" -> allow all
                if status in (404, 410):
                    return None, status, None
                # Any other >=400 is \"unreadable\" -> allow all (but log)
                if status >= 400:
                    return None, status, f"http_{status}"
                raw = resp.content or b""
                if not raw:
                    return "", status, None
                if len(raw) > self._cfg.max_bytes:
                    raw = raw[: self._cfg.max_bytes]
                # robots.txt is ASCII-ish, but allow utf-8
                text = raw.decode("utf-8", errors="replace")
                return text, status, None
        except Exception as e:
            return None, None, f"{type(e).__name__}: {str(e)[:200]}"

    async def get_parser(self, root_or_any_url: str) -> Optional[RobotFileParser]:
        """
        Return a RobotFileParser if robots.txt is readable+parseable, else None.
        None means allow-all (per configured policy).
        """
        url = (root_or_any_url or "").strip()
        origin = _origin(url)
        if not origin:
            return None

        now = time.time()
        cached = self._cache.get(origin)
        if cached and cached.expires_at > now:
            return cached.parser

        async with self._lock_for(origin):
            # Re-check inside lock
            now = time.time()
            cached = self._cache.get(origin)
            if cached and cached.expires_at > now:
                return cached.parser

            robots_url = urljoin(origin, "/robots.txt")
            text, status, err = await self._fetch_robots_text(robots_url)

            parser: Optional[RobotFileParser] = None
            if text is None:
                # Missing or unreadable -> allow-all per policy
                parser = None
                if err:
                    logger.info("robots_unreadable origin=%s status=%s err=%s", origin, status, err)
            else:
                try:
                    rp = RobotFileParser()
                    # Use parse(lines) so we control fetching and UA.
                    rp.parse(text.splitlines())
                    parser = rp
                except Exception as e:
                    parser = None
                    logger.info(
                        "robots_parse_error origin=%s status=%s err=%s",
                        origin,
                        status,
                        f"{type(e).__name__}: {str(e)[:200]}",
                    )

            entry = _CacheEntry(
                expires_at=now + max(30, int(self._cfg.cache_ttl_sec)),
                parser=parser,
                fetched_at=now,
                status_code=status,
                error=err,
            )
            self._cache[origin] = entry
            return parser

    async def is_allowed(self, url: str) -> bool:
        u = (url or "").strip()
        if not u:
            return False
        rp = await self.get_parser(u)
        if rp is None:
            return True
        try:
            return bool(rp.can_fetch(self._cfg.user_agent, u))
        except Exception:
            # If parser misbehaves, keep allow-all default (but log)
            logger.debug("robots_can_fetch_error url=%s", u)
            return True

    async def filter_urls(self, urls: Iterable[str]) -> List[str]:
        out: List[str] = []
        for u in urls:
            if not u:
                continue
            if await self.is_allowed(u):
                out.append(u)
            else:
                logger.info("robots_blocked url=%s ua=%s", u, self._cfg.user_agent)
        return out


# Module-level singleton (shared per process).
_DEFAULT_POLICY = RobotsPolicy()


def robots_policy() -> RobotsPolicy:
    return _DEFAULT_POLICY

