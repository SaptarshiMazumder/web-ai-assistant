"""
Platform-specific profiles for restaurant booking platforms and other domains.

All platform config (Tabelog, HotPepper, TableCheck, etc.) is loaded from
config/platform_profiles.yml. Edit that file to add or change platforms -
no code changes needed.

Helpers (get_reservation_config_from_widget, get_suggested_messages_for_widget, etc.)
read from the active profile. Web, Line, and Instagram all use these; each channel
renders the result in its own UI (quick replies, flex buttons, etc.).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Path to config file (backend/config/platform_profiles.yml)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "platform_profiles.yml"


def _load_platform_config() -> Dict[str, Any]:
    """Load platform profiles from YAML. Returns empty dict if file missing or invalid."""
    if not _CONFIG_PATH.exists():
        logger.warning("Platform config not found: %s", _CONFIG_PATH)
        return {}
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception("Failed to load platform config from %s: %s", _CONFIG_PATH, e)
        return {}


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
# LOAD FROM CONFIG (config/platform_profiles.yml)
# ═══════════════════════════════════════════════════════════════════════════


def _dict_to_platform_profile(domain_key: str, data: Dict[str, Any]) -> PlatformProfile:
    """Build PlatformProfile from YAML dict."""
    raw = data or {}
    return PlatformProfile(
        domain_pattern=str(raw.get("domain_pattern") or domain_key.replace(".", r"\.")),
        include_paths=list(raw.get("include_paths") or []),
        exclude_paths=list(raw.get("exclude_paths") or []),
        max_depth=int(raw.get("max_depth", 2)),
        priority=int(raw.get("priority", 0)),
        strip_query_params=bool(raw.get("strip_query_params", False)),
        menu_url_patterns=list(raw.get("menu_url_patterns") or []),
        menu_extraction_rules=raw.get("menu_extraction_rules") if isinstance(raw.get("menu_extraction_rules"), dict) else None,
        image_extraction_enabled=bool(raw.get("image_extraction_enabled", True)),
        image_url_patterns=list(raw.get("image_url_patterns") or []),
        image_extraction_rules=raw.get("image_extraction_rules") if isinstance(raw.get("image_extraction_rules"), dict) else None,
        metadata=dict(raw.get("metadata") or {}),
    )


def _build_platform_registry() -> tuple[
    Dict[str, PlatformProfile],
    Dict[str, Tuple[str, str]],
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, List[str]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[str],
    List[str],
    str,
    str,
    str,
    str,
]:
    """Load config and build PLATFORM_PROFILES, RESERVATION_PLATFORM_CONFIG, DEFAULT_SUGGESTED_MESSAGES, DEFAULT_ASSET_RULES, DEFAULT_ASSET_TERM_CONFIG, DEFAULT_JSON_RESPONSE_FORMAT, DEFAULT_RAG_INSTRUCTION, DEFAULT_MENU_TEXTS."""
    cfg = _load_platform_config()
    profiles: Dict[str, PlatformProfile] = {}
    reservation_config: Dict[str, Tuple[str, str]] = {}
    default_suggested: List[Dict[str, Any]] = []
    default_asset_rules: Dict[str, str] = {}
    default_asset_term_config: Dict[str, List[str]] = {}

    if isinstance(cfg.get("default_asset_rules"), dict):
        dar = cfg["default_asset_rules"]
        marker = str(dar.get("marker_rule") or "").strip()
        evidence = str(dar.get("evidence_template") or "").strip()
        if marker:
            default_asset_rules["marker_rule"] = marker
        if evidence:
            default_asset_rules["evidence_template"] = evidence

    if isinstance(cfg.get("default_asset_term_config"), dict):
        datc = cfg["default_asset_term_config"]
        for key in ("generic_tokens", "asset_intent_terms", "visual_request_terms", "visual_request_many_terms", "visual_suppress_terms"):
            val = datc.get(key)
            if isinstance(val, list):
                default_asset_term_config[key] = [str(v).strip() for v in val if str(v).strip()]

    # Reservation platform mapping
    rpc = cfg.get("reservation_platform_config") or {}
    for pid, entry in rpc.items():
        if isinstance(entry, dict):
            wk = str(entry.get("widget_key") or "").strip()
            dk = str(entry.get("domain_key") or "").strip()
            if wk and dk:
                reservation_config[str(pid).strip().lower()] = (wk, dk)

    # Default suggested messages
    dsm = cfg.get("default_suggested_messages")
    if isinstance(dsm, list):
        default_suggested = [m for m in dsm if isinstance(m, dict)]

    # Platform profiles
    platforms = cfg.get("platforms") or {}
    if isinstance(platforms, dict):
        for domain_key, pdata in platforms.items():
            if isinstance(pdata, dict) and domain_key:
                profiles[str(domain_key).strip()] = _dict_to_platform_profile(domain_key, pdata)

    default_json = cfg.get("default_json_response_format") if isinstance(cfg.get("default_json_response_format"), dict) else None
    default_rag = cfg.get("default_rag_instruction") if isinstance(cfg.get("default_rag_instruction"), dict) else None
    default_menu_texts = cfg.get("default_menu_texts") if isinstance(cfg.get("default_menu_texts"), dict) else None
    default_menu_keywords = cfg.get("default_menu_keywords")
    default_menu_category_order = cfg.get("default_menu_category_order")
    default_menu_request_pattern = str(cfg.get("default_menu_request_pattern") or "").strip()
    line_menu_payload = str(cfg.get("line_menu_quick_payload") or "SHOW_FULL_MENU").strip()
    line_menu_prefix = str(cfg.get("line_menu_page_payload_prefix") or "SHOW_MENU_PAGE:").strip()
    ig_menu_payload = str(cfg.get("instagram_menu_quick_payload") or "SHOW_FULL_MENU").strip()
    ig_menu_prefix = str(cfg.get("instagram_menu_page_payload_prefix") or "SHOW_MENU_PAGE:").strip()

    return (
        profiles,
        reservation_config,
        default_suggested,
        default_asset_rules,
        default_asset_term_config,
        default_json,
        default_rag,
        default_menu_texts,
        default_menu_keywords if isinstance(default_menu_keywords, list) else [],
        default_menu_category_order if isinstance(default_menu_category_order, list) else ["course", "dish", "drink", "lunch", "menu"],
        default_menu_request_pattern,
        line_menu_payload,
        line_menu_prefix,
        ig_menu_payload,
        ig_menu_prefix,
    )


(
    PLATFORM_PROFILES,
    RESERVATION_PLATFORM_CONFIG,
    DEFAULT_SUGGESTED_MESSAGES,
    DEFAULT_ASSET_RULES,
    DEFAULT_ASSET_TERM_CONFIG,
    DEFAULT_JSON_RESPONSE_FORMAT,
    DEFAULT_RAG_INSTRUCTION,
    DEFAULT_MENU_TEXTS,
    DEFAULT_MENU_KEYWORDS,
    DEFAULT_MENU_CATEGORY_ORDER,
    DEFAULT_MENU_REQUEST_PATTERN,
    LINE_MENU_QUICK_PAYLOAD,
    LINE_MENU_PAGE_PAYLOAD_PREFIX,
    INSTAGRAM_MENU_QUICK_PAYLOAD,
    INSTAGRAM_MENU_PAGE_PAYLOAD_PREFIX,
) = _build_platform_registry()


def get_reservation_platforms_list(*, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Return list of reservation platforms from config (for dashboard dropdowns).
    No hardcoding: add platforms in platform_profiles.yml only.
    """
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"
    cfg = _load_platform_config()
    rpc = cfg.get("reservation_platform_config") or {}
    out: List[Dict[str, Any]] = []
    for pid, (widget_key, domain_key) in RESERVATION_PLATFORM_CONFIG.items():
        label = pid  # fallback
        if domain_key in PLATFORM_PROFILES:
            profile = PLATFORM_PROFILES[domain_key]
            metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
            sn = metadata.get("service_name")
            if isinstance(sn, dict):
                label = str(sn.get(lang) or sn.get("en") or pid).strip() or pid
            elif isinstance(sn, str) and sn.strip():
                label = sn.strip()
        entry = rpc.get(pid) if isinstance(rpc, dict) else {}
        url_placeholder = str(entry.get("url_placeholder") or "").strip() if isinstance(entry, dict) else ""
        out.append({
            "id": pid,
            "widget_key": widget_key,
            "domain_key": domain_key,
            "label": label,
            "url_placeholder": url_placeholder,
        })
    return out


def get_asset_rules_from_widget(widget_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get asset rules and term config from platform profile or default.
    Config-driven: no hardcoding in asset_resolver.
    Returns dict with marker_rule, evidence_template, asset_term_config.
    """
    out: Dict[str, Any] = dict(DEFAULT_ASSET_RULES)
    out["asset_term_config"] = dict(DEFAULT_ASSET_TERM_CONFIG)

    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return out
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return out
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    rules = metadata.get("asset_rules")
    if isinstance(rules, dict):
        marker = str(rules.get("marker_rule") or "").strip()
        evidence = str(rules.get("evidence_template") or "").strip()
        if marker:
            out["marker_rule"] = marker
        if evidence:
            out["evidence_template"] = evidence
        term_cfg = rules.get("asset_term_config")
        if isinstance(term_cfg, dict):
            for key in ("generic_tokens", "asset_intent_terms", "visual_request_terms", "visual_request_many_terms", "visual_suppress_terms"):
                val = term_cfg.get(key)
                if isinstance(val, list):
                    out["asset_term_config"][key] = [str(v).strip() for v in val if str(v).strip()]
    return out


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


def get_knowledge_tabs_for_widget(widget_config: Dict[str, Any]) -> List[str]:
    """
    Get which knowledge tabs (image-assets, menu-list) to show in the bot dashboard.
    Config-driven via reservation_platform_config.knowledge_tabs in platform_profiles.yml.

    - tabelog, hotpepper: ["menu"] — Menu tab only
    - tablecheck, others: ["image"] — Image tab only (default)
    """
    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return ["image"]
    platform_id = cfg.get("platform_id")
    if not platform_id:
        return ["image"]
    raw = _load_platform_config()
    rpc = raw.get("reservation_platform_config") or {}
    entry = rpc.get(platform_id) if isinstance(rpc, dict) else {}
    if not isinstance(entry, dict):
        return ["image"]
    tabs = entry.get("knowledge_tabs")
    if isinstance(tabs, list) and tabs:
        out = [str(t).strip().lower() for t in tabs if str(t).strip()]
        if out:
            return out
    return ["image"]


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
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    menu_rules = getattr(profile, "menu_extraction_rules", None)
    menu_enabled = (
        isinstance(menu_rules, dict)
        and menu_rules.get("enabled", True)
    )
    suggested = reservation.get("suggested_messages")
    if not isinstance(suggested, list):
        suggested = metadata.get("suggested_messages")  # fallback: top-level
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


_VALID_SUGGESTED_TYPES = ("ai_response", "show_menu", "escalate")


def get_available_suggested_message_types(platform_id: Optional[str] = None) -> List[str]:
    """
    Return the types of suggested messages available for a platform (or default).
    Config-driven: only types present in platform's suggested_messages are allowed.
    Default (no platform): only ai_response.
    """
    if not platform_id or not str(platform_id).strip():
        # Default: only types from default_suggested_messages
        types_seen: set = set()
        for m in DEFAULT_SUGGESTED_MESSAGES:
            if isinstance(m, dict):
                t = str(m.get("type") or "ai_response").strip() or "ai_response"
                if t in _VALID_SUGGESTED_TYPES:
                    types_seen.add(t)
        return list(types_seen) if types_seen else ["ai_response"]

    platform_id = str(platform_id).strip().lower()
    if platform_id not in RESERVATION_PLATFORM_CONFIG:
        return ["ai_response"]
    _, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    if domain_key not in PLATFORM_PROFILES:
        return ["ai_response"]
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    raw = reservation.get("suggested_messages") or metadata.get("suggested_messages")
    if not isinstance(raw, list) or not raw:
        return ["ai_response"]
    types_seen = set()
    for r in raw:
        if isinstance(r, dict):
            t = str(r.get("type") or "ai_response").strip() or "ai_response"
            if t in _VALID_SUGGESTED_TYPES:
                types_seen.add(t)
    return list(types_seen) if types_seen else ["ai_response"]


def get_suggested_messages_for_platform(
    platform_id: str,
    *,
    lang: str = "en",
) -> Optional[List[Dict[str, Any]]]:
    """
    Get platform default suggested messages (for create-bot initial load).
    Returns list of {id, label, type, prompt} or None if platform has no suggested_messages.
    """
    platform_id = str(platform_id or "").strip().lower()
    if platform_id not in RESERVATION_PLATFORM_CONFIG:
        return None
    _, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    if domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    raw = reservation.get("suggested_messages") or metadata.get("suggested_messages")
    if not isinstance(raw, list) or not raw:
        return None
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    def resolve_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            label = _resolve_label_or_prompt(r.get("label"), lang)
            if not label:
                continue
            prompt = _resolve_label_or_prompt(r.get("prompt"), lang) or label
            raw_type = str(r.get("type") or "ai_response").strip() or "ai_response"
            if raw_type not in ("ai_response", "show_menu", "escalate"):
                raw_type = "ai_response"
            out.append({
                "id": str(r.get("id") or "").strip() or f"suggest_{len(out) + 1}",
                "label": label,
                "prompt": prompt,
                "type": raw_type,
            })
        return out

    resolved = resolve_items(raw)
    return resolved if resolved else None


def get_suggested_messages_for_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Get suggested messages for a bot. Used across Line, Instagram, web widget.

    Priority: DB first, then platform fallback (for first default), then platform_profiles default.
    - If widget_config has suggestedMessages (saved in DB): use those (edits persist)
    - Else: use platform profile (Tabelog, HotPepper, TableCheck) as initial default
    - Else: use default_suggested_messages
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
            raw_type = str(raw.get("type") or "ai_response").strip() or "ai_response"
            if raw_type not in ("ai_response", "show_menu", "escalate"):
                raw_type = "ai_response"
            out.append({
                "id": str(raw.get("id") or "").strip() or f"suggest_{len(out) + 1}",
                "label": label,
                "prompt": prompt,
                "type": raw_type,
            })
        return out

    # 1. DB first: widget_config.suggestedMessages (saved from dashboard; edits persist)
    #    Exception: platform bot with generic defaults (Ask a question/質問する) -> use platform
    wc_suggested = widget_config.get("suggestedMessages")
    features = get_platform_features_from_widget(widget_config)
    platform_suggested = features.get("suggested_messages") if features else None

    if isinstance(wc_suggested, list) and wc_suggested:
        resolved = resolve_items(wc_suggested)
        if resolved:
            # If platform has config and DB has generic default (2nd item = Ask question), use platform
            generic_second = ("ask a question", "質問する")
            if platform_suggested and len(resolved) >= 2:
                second_label = (resolved[1].get("label") or "").strip().lower()
                if second_label in generic_second:
                    platform_resolved = resolve_items(platform_suggested)
                    if platform_resolved and len(platform_resolved) >= 2:
                        plat_second = (platform_resolved[1].get("label") or "").strip().lower()
                        if plat_second not in generic_second:  # platform has 予約/Reservation
                            return platform_resolved
            return resolved

    # 2. Fallback: platform profile (Tabelog, HotPepper, TableCheck) as initial default
    if platform_suggested:
        return resolve_items(platform_suggested)
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


def _build_json_instruction_from_jrf(jrf: Dict[str, Any], lang: str) -> Optional[str]:
    """Build JSON response instruction from a jrf dict (default or platform)."""
    if not isinstance(jrf, dict):
        return None
    schema_dict = jrf.get("schema")
    if not isinstance(schema_dict, dict):
        return None
    if "show_assets" in schema_dict:
        true_rules = jrf.get("show_assets_true_when")
        false_rules = jrf.get("show_assets_false_when")
        true_txt = str((true_rules or {}).get("en") or "").strip() if isinstance(true_rules, dict) else ""
        false_txt = str((false_rules or {}).get("en") or "").strip() if isinstance(false_rules, dict) else ""
        if not true_txt or not false_txt:
            return None

    schema_items = [f'"{k}": {v}' for k, v in schema_dict.items() if isinstance(v, str) and v.strip()]
    if not schema_items:
        return None
    schema_line = "{" + ", ".join(schema_items) + "}"
    parts: List[str] = [
        "\n\nRESPONSE FORMAT (CRITICAL): You MUST respond with valid JSON only. No other text before or after.\n"
        f"Schema: {schema_line}\n"
    ]
    for key, desc in schema_dict.items():
        if isinstance(desc, str) and desc.strip():
            parts.append(f"- {key}: {desc}\n")

    true_rules = jrf.get("show_assets_true_when")
    false_rules = jrf.get("show_assets_false_when")
    true_txt = str((true_rules or {}).get(lang) or (true_rules or {}).get("en") or "").strip() if isinstance(true_rules, dict) else ""
    false_txt = str((false_rules or {}).get(lang) or (false_rules or {}).get("en") or "").strip() if isinstance(false_rules, dict) else ""
    if true_txt and false_txt:
        parts.append(
            "- show_assets: boolean. Follow these rules exactly:\n"
            f"  WHEN TRUE: {true_txt}\n"
            f"  WHEN FALSE: {false_txt}\n"
        )

    intent_when = jrf.get("intent_when")
    if isinstance(intent_when, dict):
        intent_txt = str(intent_when.get(lang) or intent_when.get("en") or "").strip()
        if intent_txt:
            parts.append(f"- intent: Follow these rules:\n{intent_txt}\n")

    parts.append("Output ONLY the JSON object, no markdown code fences.")
    return "".join(parts)


def get_platform_json_response_instruction(widget_config: Dict[str, Any], *, lang: str = "en") -> Optional[str]:
    """
    Get JSON response format instruction. Uses default for all bots; platforms (Tabelog, HotPepper)
    add extra keys like intent via their json_response_format config.
    """
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    jrf: Optional[Dict[str, Any]] = None
    cfg = get_reservation_config_from_widget(widget_config)
    if cfg:
        domain_key = cfg.get("domain_key")
        if domain_key and domain_key in PLATFORM_PROFILES:
            profile = PLATFORM_PROFILES[domain_key]
            metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
            platform_jrf = metadata.get("json_response_format")
            if isinstance(platform_jrf, dict) and platform_jrf.get("enabled"):
                jrf = platform_jrf

    if jrf is None and DEFAULT_JSON_RESPONSE_FORMAT:
        jrf = DEFAULT_JSON_RESPONSE_FORMAT

    return _build_json_instruction_from_jrf(jrf, lang) if jrf else None


def get_platform_json_response_enabled(widget_config: Dict[str, Any]) -> bool:
    """Return True when JSON response format is configured (default or platform)."""
    return get_platform_json_response_instruction(widget_config, lang="en") is not None


def get_default_rag_instruction() -> Dict[str, Any]:
    """Return default RAG/LLM instruction config (default_system, grounding_suffix, etc.)."""
    return dict(DEFAULT_RAG_INSTRUCTION) if DEFAULT_RAG_INSTRUCTION else {}


def get_default_marker_rule() -> str:
    """Return default marker rule for asset bank (from config)."""
    return str(DEFAULT_ASSET_RULES.get("marker_rule") or "").strip()


def get_menu_texts(lang: str, *, widget_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get menu flow texts for the given lang. Platform can override via metadata.menu_texts."""
    lang = "ja" if (lang or "").strip().lower() in ("ja", "jp") else "en"
    base = (DEFAULT_MENU_TEXTS or {}).get(lang) or (DEFAULT_MENU_TEXTS or {}).get("en") or {}
    if widget_config:
        cfg = get_reservation_config_from_widget(widget_config)
        if cfg:
            domain_key = cfg.get("domain_key")
            if domain_key and domain_key in PLATFORM_PROFILES:
                profile = PLATFORM_PROFILES[domain_key]
                metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
                platform_texts = metadata.get("menu_texts")
                if isinstance(platform_texts, dict):
                    platform_lang = (platform_texts.get(lang) or platform_texts.get("en") or {})
                    if isinstance(platform_lang, dict):
                        base = dict(base)
                        base.update(platform_lang)
    return base


def get_menu_keywords() -> List[str]:
    """Get menu request keywords from config."""
    return list(DEFAULT_MENU_KEYWORDS) if DEFAULT_MENU_KEYWORDS else []


def get_menu_category_order() -> Tuple[str, ...]:
    """Get menu category order from config."""
    return tuple(DEFAULT_MENU_CATEGORY_ORDER) if DEFAULT_MENU_CATEGORY_ORDER else ("course", "dish", "drink", "lunch", "menu")


def get_menu_request_pattern() -> Optional[str]:
    """Get menu request regex pattern from config."""
    return (DEFAULT_MENU_REQUEST_PATTERN or "").strip() or None


def get_line_menu_quick_payload() -> str:
    """Get LINE menu quick reply payload from config."""
    return LINE_MENU_QUICK_PAYLOAD or "SHOW_FULL_MENU"


def get_line_menu_page_payload_prefix() -> str:
    """Get LINE menu page payload prefix from config."""
    return LINE_MENU_PAGE_PAYLOAD_PREFIX or "SHOW_MENU_PAGE:"


def get_instagram_menu_quick_payload() -> str:
    """Get Instagram menu quick reply payload from config."""
    return INSTAGRAM_MENU_QUICK_PAYLOAD or "SHOW_FULL_MENU"


def get_instagram_menu_page_payload_prefix() -> str:
    """Get Instagram menu page payload prefix from config."""
    return INSTAGRAM_MENU_PAGE_PAYLOAD_PREFIX or "SHOW_MENU_PAGE:"


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
