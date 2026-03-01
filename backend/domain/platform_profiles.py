"""
Platform-specific profiles for restaurant booking platforms and other domains.

A PlatformProfile defines behavior for a domain (e.g., tabelog.com). The same config
drives all channels (web widget, Line, Instagram); only the presentation layer differs.

Tabelog profile defines:
- reservation: URL, instruction template, link labels (EN/JA)
- suggested_messages: Menu, Reservation, Ask a question
- asset_instructions: when to show menu assets vs reservation link
- menu_extraction_rules: enabled, paths, categories

Helpers (get_reservation_config_from_widget, get_suggested_messages_for_widget, etc.)
read from the active profile. Web, Line, and Instagram all use these; each channel
renders the result in its own UI (quick replies, flex buttons, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
        "reservation": {
            "enabled": True,
            "link_label": {"en": "HotPepper Gourmet", "ja": "ホットペッパーグルメ"},
            "instruction_template": {
                "en": "When the customer asks about reservations or booking, include this exact link: {url}. Answer naturally based on the evidence; use this URL whenever reservation is relevant.",
                "ja": "お客様が予約・ご予約についてお問い合わせの際は、このリンクを含めてください: {url}。証拠に基づいて自然に回答し、予約に関連する場合はこのURLを使用してください。",
            },
        },
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
            "instruction_template": {
                "en": "When the customer asks about reservations or booking, include this exact link: {url}. Answer naturally based on the evidence; use this URL whenever reservation is relevant.",
                "ja": "お客様が予約・ご予約についてお問い合わせの際は、このリンクを含めてください: {url}。証拠に基づいて自然に回答し、予約に関連する場合はこのURLを使用してください。",
            },
        },
        "suggested_messages": [
            {"id": "suggest_menu", "type": "ai_response", "label": {"en": "Menu", "ja": "メニュー"}, "prompt": {"en": "Menu", "ja": "メニュー"}},
            {"id": "suggest_reservation", "type": "ai_response", "label": {"en": "Reservation", "ja": "予約"}, "prompt": {"en": "Reservation", "ja": "予約"}},
            {"id": "suggest_question", "type": "ai_response", "label": {"en": "Ask a question", "ja": "質問する"}, "prompt": {"en": "Ask a question", "ja": "質問する"}},
        ],
        "asset_instructions": {
            "en": (
                "ASSET USAGE RULES (menu items). You decide when to use assets based on the user's intent:\n"
                "- Reservation/booking intent: Answer with reservation information (link, hours, policies, how to reserve, etc.). Do NOT use {{asset:ID}}. Do NOT mention menu, dishes, courses, or products. Keep the response focused on reservation only.\n"
                "- Menu/dish/course intent: Use {{asset:ID}} only for items that directly match their question. Be precise: e.g. 'butter chicken curry' means cite only the butter chicken curry item, not chicken curry.\n"
                "- Other questions: Use {{asset:ID}} only when the asset is directly relevant to the answer."
            ),
            "ja": (
                "アセット使用ルール（メニュー項目）。ユーザーの意図に応じて判断してください:\n"
                "- 予約・ご予約の意図: 予約情報（リンク、営業時間、ポリシー、予約方法など）で回答してください。{{asset:ID}}は使用しないでください。メニュー、料理、コース、商品には触れないでください。予約にのみ焦点を当ててください。\n"
                "- メニュー・料理・コースの意図: 質問に直接一致する項目のみ{{asset:ID}}を使用してください。正確に: 例「バターチキンカレー」はバターチキンカレーの項目のみを引用し、チキンカレーは含めない。\n"
                "- その他の質問: アセットが回答に直接関連する場合のみ{{asset:ID}}を使用してください。"
            ),
        },
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
        "reservation": {
            "enabled": True,
            "link_label": {"en": "TableCheck Reservation", "ja": "TableCheck予約"},
            "instruction_template": {
                "en": "When the customer asks about reservations or booking, include this exact link: {url}. Answer naturally based on the evidence; use this URL whenever reservation is relevant.",
                "ja": "お客様が予約・ご予約についてお問い合わせの際は、このリンクを含めてください: {url}。証拠に基づいて自然に回答し、予約に関連する場合はこのURLを使用してください。",
            },
        },
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

# One profile per restaurant agent. Maps platform id -> (widget_config URL key, domain_key)
RESERVATION_PLATFORM_CONFIG: Dict[str, Tuple[str, str]] = {
    "tabelog": ("tabelogUrl", "tabelog.com"),
    "hotpepper": ("hotPepperUrl", "hotpepper.jp"),
    "tablecheck": ("tableCheckUrl", "tablecheck.com"),
}

# Default suggested messages when no profile defines them. Used across Line, Instagram, web widget.
DEFAULT_SUGGESTED_MESSAGES: List[Dict[str, Any]] = [
    {"id": "suggest_1", "type": "ai_response", "label": {"en": "What can you do?", "ja": "何ができますか？"}, "prompt": {"en": "What can you do?", "ja": "何ができますか？"}},
    {"id": "suggest_2", "type": "ai_response", "label": {"en": "Ask a question", "ja": "質問する"}, "prompt": {"en": "Ask a question", "ja": "質問する"}},
]


def get_reservation_config_from_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Get reservation config from widget_config and platform profiles.
    Fully config-driven: presence of reservationPlatform + URL is the only signal.

    Returns:
        Dict with url, instruction, domain_key, link_label, platform_id; or None if not applicable.
    """
    if not isinstance(widget_config, dict):
        return None

    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    # One profile per agent: use reservationPlatform if set, else infer from first URL
    platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
    if not platform_id:
        for pid, (config_key, _) in RESERVATION_PLATFORM_CONFIG.items():
            if (widget_config.get(config_key) or "").strip():
                platform_id = pid
                break
    if not platform_id or platform_id not in RESERVATION_PLATFORM_CONFIG:
        return None

    config_key, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    raw_url = (widget_config.get(config_key) or "").strip()
    if not raw_url:
        return None
    if not raw_url.startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"

    profile, resolved_domain = resolve_platform_profile(raw_url)
    if profile is None or resolved_domain != domain_key:
        return None

    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation")
    if not isinstance(reservation, dict):
        return None
    if not reservation.get("enabled", True):
        return None

    # Optional override from widget_config; else use platform profile template
    custom = (widget_config.get("reservationInstruction") or "").strip()
    if custom:
        instruction = custom
    else:
        templates = reservation.get("instruction_template")
        if isinstance(templates, dict):
            instruction = str(templates.get(lang) or templates.get("en") or "").strip()
        else:
            instruction = "When the customer asks about reservations or booking, include this exact link: {url}. Answer naturally based on the evidence."
    if not instruction:
        return None

    try:
        instruction = instruction.format(url=raw_url)
    except (KeyError, ValueError):
        instruction = f"{instruction} {raw_url}"

    labels = reservation.get("link_label")
    if isinstance(labels, dict):
        link_label = str(labels.get(lang) or labels.get("en") or "Online Reservation").strip()
    else:
        link_label = "Online Reservation"

    return {
        "url": raw_url,
        "instruction": instruction,
        "domain_key": domain_key,
        "link_label": link_label,
        "platform_id": platform_id,
    }


def get_platform_features_from_widget(widget_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get platform features (menu, suggested_messages) from widget config.
    Fully config-driven: reads from the active platform profile.
    """
    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return None
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    menu_rules = getattr(profile, "menu_extraction_rules", None)
    menu_enabled = (
        isinstance(menu_rules, dict)
        and menu_rules.get("enabled", True)
    )
    suggested = metadata.get("suggested_messages")
    if not isinstance(suggested, list):
        suggested = None
    return {
        "menu_extraction_enabled": menu_enabled,
        "suggested_messages": suggested,
    }


def _resolve_label_or_prompt(raw: Any, lang: str) -> str:
    """Resolve label/prompt from string or {en, ja} dict."""
    if isinstance(raw, dict):
        return str(raw.get(lang) or raw.get("en") or "").strip()
    return str(raw or "").strip()


def get_suggested_messages_for_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Get suggested messages for a bot from its platform profile or default.
    Used across Line, Instagram, web widget. Fully config-driven.

    - If the active profile has suggested_messages → use it (resolved by lang).
    - Else → use DEFAULT_SUGGESTED_MESSAGES (2 items: What can you do?, Ask a question).

    Returns list of dicts with id, label (string), prompt (string), type.
    """
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    def resolve_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            label = _resolve_label_or_prompt(raw.get("label"), lang)
            if not label:
                continue
            prompt = _resolve_label_or_prompt(raw.get("prompt"), lang) or label
            out.append({
                "id": str(raw.get("id") or "").strip() or f"suggest_{len(out) + 1}",
                "label": label,
                "prompt": prompt,
                "type": str(raw.get("type") or "ai_response").strip() or "ai_response",
            })
        return out

    features = get_platform_features_from_widget(widget_config)
    if features and features.get("suggested_messages"):
        return resolve_items(features["suggested_messages"])
    return resolve_items(DEFAULT_SUGGESTED_MESSAGES)


def get_platform_asset_instructions(widget_config: Dict[str, Any], *, lang: str = "en") -> Optional[str]:
    """
    Get asset usage instructions from the active platform profile.
    Config-driven: no platform-specific logic in callers.
    """
    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return None
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    instructions = metadata.get("asset_instructions")
    if not isinstance(instructions, dict):
        return None
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"
    text = str(instructions.get(lang) or instructions.get("en") or "").strip()
    return text if text else None


def ensure_canonical_reservation_url_in_text(text: str, canonical_url: str, domain_key: str) -> str:
    """Replace any URLs from the given domain in text with the canonical URL."""
    import re
    if not text or not canonical_url or not domain_key:
        return text
    escaped = re.escape(domain_key)
    pattern = rf"https?://[^\s\)\]\"\']*{escaped}[^\s\)\]\"\']*"
    return re.sub(pattern, canonical_url, text, flags=re.IGNORECASE)


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
