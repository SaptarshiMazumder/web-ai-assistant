"""
Platform-specific crawl profiles for restaurant booking platforms and other domains.

A PlatformProfile defines how to crawl a specific domain (e.g., hotpepper.jp):
- Which URL paths to include/exclude
- How deep to follow links
- Custom extraction rules for menus, images, etc.

New platforms can be added by creating a profile and registering it in PLATFORM_PROFILES.
Profiles are applied during URL crawling; if no profile matches, default behavior is used.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PlatformProfile:
    """Defines crawling and extraction strategy for a specific domain."""

    domain_pattern: str
    """Regex pattern to match domain (e.g., 'hotpepper\\.jp', 'tabelog\\.com')."""

    include_paths: List[str] = field(default_factory=list)
    """List of regex patterns for URL paths to include in crawl.
    If empty, all paths on the domain are included (subject to exclude_paths)."""

    exclude_paths: List[str] = field(default_factory=list)
    """List of regex patterns for URL paths to exclude from crawl."""

    max_depth: int = 2
    """Maximum link-following depth from seed URL. 0 = don't follow links."""

    priority: int = 0
    """Tiebreaker if multiple profiles match the same domain."""

    strip_query_params: bool = False
    """If True, strip query parameters and fragments before URL matching and deduplication.
    Use for platforms where query params represent UI state (e.g., ?RDT=20260226 on HotPepper),
    not different content. Defaults to False (unknown domains keep all query params)."""

    # ─── Menu Extraction (future) ─────────────────────────────────────────
    menu_url_patterns: List[str] = field(default_factory=list)
    """Regex patterns for URLs containing menu/course data.
    To be used by menu extraction pipeline (not yet implemented)."""

    menu_extraction_rules: Optional[Dict[str, Any]] = None
    """Custom rules for extracting menu items, prices, descriptions.
    Format TBD based on extraction requirements.
    If None, default extraction logic applies."""

    # ─── Image Extraction (future) ────────────────────────────────────────
    image_extraction_enabled: bool = True
    """Whether to extract images from this platform."""

    image_url_patterns: List[str] = field(default_factory=list)
    """Regex patterns for URLs containing images (e.g., /photo/, /gallery/).
    If empty, images from all included URLs are extracted."""

    image_extraction_rules: Optional[Dict[str, Any]] = None
    """Custom rules for filtering/processing images.
    Can specify: min resolution, skip certain image types, alt text extraction, etc.
    If None, default image extraction logic applies."""

    # ─── Metadata ─────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata dict for any platform-specific data."""


# ═══════════════════════════════════════════════════════════════════════════
# RESTAURANT PLATFORMS
# ═══════════════════════════════════════════════════════════════════════════

HOTPEPPER_PROFILE = PlatformProfile(
    domain_pattern=r"hotpepper\.jp",
    include_paths=[],
    exclude_paths=[
        r"/report",
        r"/favorite",
    ],
    priority=10,
    strip_query_params=True,  # ?RDT=YYYYMMDD is UI state, not separate pages
    menu_url_patterns=[
        r"/course",
    ],
    metadata={
        "platform_type": "restaurant_reservation",
        "country": "JP",
        "service_name": "HotPepper Gourmet",
    },
)

TABELOG_PROFILE = PlatformProfile(
    domain_pattern=r"tabelog\.com",
    include_paths=[],
    exclude_paths=[
        r"/peripheral_map(?:/|$)",
        r"/dtlphotolst(?:/|$)",
        r"/dtlrvwlst(?:/|$)",
        r"/dtlmap(?:/|$)",
    ],
    priority=10,
    strip_query_params=True,  # query params are UI state, not separate pages
    menu_url_patterns=[
        r"/party(?:/|$)",
        r"/dtlmenu(?:/|$)",
    ],
    menu_extraction_rules={
        "enabled": True,
        "mode": "deterministic",
        "extractor": "tabelog_v1",
        "allowed_path_patterns": [
            r"^/(?:[a-z]{2}(?:-[a-z]{2})?/)?[A-Za-z0-9._-]+/A\d+/A\d+/\d+/?$",
            r"^/(?:[a-z]{2}(?:-[a-z]{2})?/)?[A-Za-z0-9._-]+/A\d+/A\d+/\d+/(?:party|dtlmenu)(?:/|$)",
        ],
        "path_category_patterns": [
            {"pattern": r"/party(?:/|$)", "category": "course"},
            {"pattern": r"/dtlmenu/drink(?:/|$)", "category": "drink"},
            {"pattern": r"/dtlmenu/lunch(?:/|$)", "category": "lunch"},
            {"pattern": r"/dtlmenu(?:/|$)", "category": "dish"},
        ],
    },
    metadata={
        "platform_type": "restaurant_reservation",
        "country": "JP",
        "service_name": "Tabelog",
        "reservation": {
            "enabled": True,
            "link_label": {
                "en": "Tabelog Online Reservation",
                "ja": "食べログ ネット予約",
            },
            "intent_keywords": {
                "en": [
                    "reservation",
                    "reserve",
                    "booking",
                    "book a table",
                    "book table",
                    "online reservation",
                ],
                "ja": [
                    "予約",
                    "ネット予約",
                    "オンライン予約",
                    "席予約",
                    "予約したい",
                ],
            },
            "response_templates": {
                "en": "For online reservations, please use [{label}]({url}).",
                "ja": "オンライン予約はこちらをご利用ください: [{label}]({url})",
            },
        },
        "suggested_messages": [
            {
                "id": "suggest_menu",
                "type": "ai_response",
                "label": {"en": "Menu", "ja": "メニュー"},
                "prompt": {"en": "Menu", "ja": "メニュー"},
            }
        ],
    },
)

TABLECHECK_PROFILE = PlatformProfile(
    domain_pattern=r"tablecheck\.com",
    include_paths=[],
    exclude_paths=[],
    priority=10,
    metadata={
        "platform_type": "restaurant_reservation",
        "country": "JP",
        "service_name": "TableCheck",
    },
)


# ═══════════════════════════════════════════════════════════════════════════
# PLATFORM REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

PLATFORM_PROFILES = {
    "hotpepper.jp": HOTPEPPER_PROFILE,
    "tabelog.com": TABELOG_PROFILE,
    "tablecheck.com": TABLECHECK_PROFILE,
}
"""
Central registry mapping domain patterns to platform profiles.
Used to look up crawl/extraction rules when processing a URL.
"""


# ═══════════════════════════════════════════════════════════════════════════
# RESOLVER + URL FILTERS
# Imported by both infrastructure (discovery) and application (crawl pipeline)
# so filtering rules are defined once and applied everywhere.
# ═══════════════════════════════════════════════════════════════════════════

import re
from typing import Tuple
from urllib.parse import unquote, urlparse


def resolve_platform_profile(url: str) -> Tuple[Optional[PlatformProfile], Optional[str]]:
    """
    Find a matching platform profile for the given URL.

    Args:
        url: The URL to match against registered profiles

    Returns:
        Tuple of (profile, domain_key) where:
        - profile: The matching PlatformProfile, or None if no match
        - domain_key: The key in PLATFORM_PROFILES (e.g. 'hotpepper.jp'), or None

    Behavior:
        - Matches domain_pattern as a regex against the URL
        - If multiple profiles match, returns the one with highest priority
        - If no profile matches, returns (None, None) → default crawl behavior applies
    """
    matches: List[Tuple[PlatformProfile, str, int]] = []

    for domain_key, profile in PLATFORM_PROFILES.items():
        if re.search(profile.domain_pattern, url):
            matches.append((profile, domain_key, profile.priority))

    if not matches:
        return None, None

    # Return highest priority match (or first if tied)
    profile, domain_key, _ = max(matches, key=lambda x: x[2])
    return profile, domain_key


def normalize_url_for_crawl(url: str) -> str:
    """Remove query parameters and fragments from a URL for pattern matching."""
    if '#' in url:
        url = url.split('#')[0]
    if '?' in url:
        url = url.split('?')[0]
    return url


def _decoded_path_segments(path: str) -> List[str]:
    """
    Return path segments decoded up to two rounds to handle mixed/double encoding.
    """
    out: List[str] = []
    for raw in (path or "").split("/"):
        seg = (raw or "").strip()
        if not seg:
            continue
        decoded = seg
        for _ in range(2):
            next_decoded = unquote(decoded)
            if next_decoded == decoded:
                break
            decoded = next_decoded
        out.append(decoded)
    return out


def is_junk_url(url: str) -> bool:
    """
    Detect obviously broken crawler artifacts that should never be discovered or crawled.

    Catches redirect URLs embedded in paths, malformed joined URLs, and CSS/tracking
    artifacts that get scraped as links. Applied globally regardless of platform.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
    except Exception:
        return True

    # URL-encoded URL in path (e.g., /strJ000.../https%3A%2F%2Fwww.hotpepper.jp%2F...)
    if "%3A%2F%2F" in path or "%3a%2f%2f" in path:
        return True

    # Path starts with /http:// or /https:// — malformed joined URL
    lower_path = path.lower()
    if lower_path.startswith("/http://") or lower_path.startswith("/https://"):
        return True

    # Comma in path segment — CSS media query or tracking artifact (e.g. /o,i.media=)
    if re.search(r"/[^/]*,[^/]*", path):
        return True

    # Segments that decode to quoted, space-containing payload text are usually
    # crawler artifacts (e.g., encoded UI labels), not real pages.
    for seg in _decoded_path_segments(path):
        s = seg.strip()
        has_double_quote = '"' in s or "“" in s or "”" in s
        if has_double_quote and (
            any(ch.isspace() for ch in s)
            or s.startswith(('"', "“", "”"))
            or s.endswith(('"', "“", "”"))
        ):
            return True

    return False


def should_allow_url(url: str) -> bool:
    """
    Single gate for both URL discovery and crawl indexing.

    Returns True if the URL should be:
    - Added to the discovery list shown in the dashboard
    - Followed during BFS link traversal
    - Indexed and crawled

    Applies in order:
    1. Global junk detection (broken artifacts, malformed paths)
    2. Platform-specific include/exclude rules from the matching profile
       (using normalized URL if profile.strip_query_params is True)

    Unknown domains (no matching profile) pass through by default.
    """
    if is_junk_url(url):
        return False

    profile, _ = resolve_platform_profile(url)
    if profile is None:
        return True  # unknown domain: allow

    # Normalize for matching only when the profile requests it
    check_url = normalize_url_for_crawl(url) if profile.strip_query_params else url

    # Exclude patterns take priority
    for pattern in profile.exclude_paths:
        if re.search(pattern, check_url):
            return False

    # If include patterns are defined, URL must match at least one
    if profile.include_paths:
        return any(re.search(p, check_url) for p in profile.include_paths)

    return True
