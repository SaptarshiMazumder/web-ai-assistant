"""
Post-answer business asset resolver.

Design:
- The LLM answer is generated without business assets injected into the prompt.
- After the answer is generated, this module matches assets against the final
  answer text (URL/name/keyword overlap) and returns relevant cards.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from common.config import config
from domain.entities import BotAsset
from domain.platform_profiles import (
    get_asset_base_stopwords,
    get_default_asset_term_config,
    get_default_marker_rule,
)
from infrastructure.db.repositories import PostgresBotAssetRepository
from redis import Redis

logger = logging.getLogger(__name__)

# When set, duplicate [AssetMenuTrace] to stdout so it shows in terminal (webhook often has no visible logs)
_TRACE_STDOUT = os.environ.get("ASSET_MENU_TRACE_STDOUT", "").strip().lower() in ("1", "true", "yes")


def _trace_stdout(msg: str) -> None:
    if _TRACE_STDOUT:
        print(msg, flush=True)


_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_GLOBAL_ASSET_CARD_LIMIT = 6
_MAX_ASSET_CARDS_PER_ANSWER = max(
    1,
    min(int(os.environ.get("ASSET_MAX_CARDS_PER_ANSWER", "3")), _GLOBAL_ASSET_CARD_LIMIT),
)
_MAX_ASSET_CARDS_EXPLICIT_REQUEST = max(
    _MAX_ASSET_CARDS_PER_ANSWER,
    min(int(os.environ.get("ASSET_MAX_CARDS_EXPLICIT_REQUEST", "6")), _GLOBAL_ASSET_CARD_LIMIT),
)
_ASSET_MATCH_CANDIDATE_POOL = max(
    _MAX_ASSET_CARDS_EXPLICIT_REQUEST * 5,
    _MAX_ASSET_CARDS_PER_ANSWER * 5,
)
_ASSET_SESSION_DEDUPE_TTL_SECONDS = max(
    300,
    min(int(os.environ.get("ASSET_SESSION_DEDUPE_TTL_SECONDS", "43200")), 604800),
)
# Base stopwords are config-driven.
_BASE_STOPWORDS = set(get_asset_base_stopwords())
_SPECIAL_SHORT_TOKENS = {"xl", "xxl", "xs"}
def _resolve_term_config(asset_term_config: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build term sets from config without in-code vocabulary defaults."""
    merged = get_default_asset_term_config()
    if isinstance(asset_term_config, dict):
        for key in (
            "generic_tokens",
            "asset_intent_terms",
            "visual_request_terms",
            "visual_request_many_terms",
            "visual_suppress_terms",
        ):
            if key in asset_term_config and isinstance(asset_term_config.get(key), list):
                merged[key] = [str(v).strip() for v in asset_term_config.get(key) if str(v).strip()]

    generic = set(merged.get("generic_tokens") or [])
    intent = set(merged.get("asset_intent_terms") or [])
    visual = set(merged.get("visual_request_terms") or [])
    visual_many = set(merged.get("visual_request_many_terms") or [])
    suppress = set(merged.get("visual_suppress_terms") or [])
    stopwords = _BASE_STOPWORDS | generic
    return {
        "generic_tokens": generic,
        "asset_intent_terms": intent,
        "visual_request_terms": visual,
        "visual_request_many_terms": visual_many,
        "visual_suppress_terms": suppress,
        "stopwords": stopwords,
    }


_QUALIFIER_GROUPS = (
    {"single", "double", "triple", "quad"},
    {"king", "queen", "twin", "full"},
    {"male", "man", "men", "female", "woman", "women", "kid", "kids", "child", "children"},
    {"small", "medium", "large", "xl", "xxl", "xs"},
)

_repo: Optional[PostgresBotAssetRepository] = None
_redis: Optional[Redis] = None


def _get_repo() -> PostgresBotAssetRepository:
    global _repo
    if _repo is None:
        _repo = PostgresBotAssetRepository()
    return _repo


def _get_redis() -> Optional[Redis]:
    global _redis
    if _redis is not None:
        return _redis
    try:
        _redis = Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)
    except Exception:
        _redis = None
    return _redis


def _asset_to_card(asset: BotAsset) -> Dict[str, str]:
    return {
        "asset_id": asset.asset_id,
        "name": asset.name,
        "image_url": asset.image_public_url,
        "link_url": asset.link_url or "",
        "description": asset.description or "",
        "asset_type": getattr(asset, "asset_type", "image"),
    }


def _menu_fields(asset: BotAsset) -> tuple[str, str, str]:
    metadata = asset.metadata if isinstance(getattr(asset, "metadata", None), dict) else {}
    price_text = str(metadata.get("price_text") or "").strip()
    details = str(metadata.get("details") or "").strip()
    category = str(metadata.get("category") or "").strip()
    if not price_text and isinstance(metadata.get("price"), dict):
        price_text = str((metadata.get("price") or {}).get("text") or "").strip()
    if not details:
        details = (asset.description or "").strip()
    return price_text, details, category


def _asset_session_key(bot_id: str, session_id: str) -> str:
    return f"webai:asset_cards:sent:{bot_id}:{session_id}"


def _asset_session_image_key(bot_id: str, session_id: str) -> str:
    return f"webai:asset_cards:sent_images:{bot_id}:{session_id}"


def _normalize_card_image_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    # Keep host/path only so query-string variations do not bypass dedupe.
    lowered = raw.lower()
    cut = min(
        (idx for idx in (lowered.find("?"), lowered.find("#")) if idx >= 0),
        default=len(raw),
    )
    return lowered[:cut]


_CJK_RE = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _contains_any_term(text_norm: str, terms: Set[str]) -> bool:
    if not text_norm:
        return False
    for term in terms:
        t = (term or "").strip()
        if not t:
            continue
        if " " in t:
            if t in text_norm:
                return True
            continue
        # CJK characters don't have word boundaries, use substring match
        if _is_cjk(t):
            if t in text_norm:
                return True
            continue
        if re.search(rf"\b{re.escape(t)}\b", text_norm):
            return True
    return False


def _query_matches_cards(query_norm: str, cards: List[Dict[str, str]], stopwords: Set[str]) -> bool:
    q_tokens = _tokenize(query_norm, stopwords)
    if not q_tokens:
        return False
    for card in cards:
        name_tokens = _tokenize(card.get("name", ""), stopwords)
        if q_tokens & name_tokens:
            return True
    return False


def _max_cards_for_query(
    answer: Optional[str],
    cards: List[Dict[str, str]],
    term_sets: Dict[str, Set[str]],
) -> int:
    """Decide max cards based on LLM answer content, not user query."""
    answer_norm = _normalize_text(answer or "")
    if not answer_norm:
        return 0
    if _contains_any_term(answer_norm, term_sets["visual_suppress_terms"]):
        return 0

    explicit_visual = _contains_any_term(answer_norm, term_sets["visual_request_terms"])
    intent_in_answer = _contains_any_term(answer_norm, term_sets["asset_intent_terms"])
    card_match = _query_matches_cards(answer_norm, cards, term_sets["stopwords"])

    if not (explicit_visual or intent_in_answer or card_match):
        msg = f"[AssetMenuTrace] _max_cards_for_query=0: explicit_visual={explicit_visual} intent_in_answer={intent_in_answer} card_match={card_match} answer_preview={(answer_norm or '')[:120]!r}"
        logger.info(msg)
        _trace_stdout(msg)
        return 0

    explicit_many = explicit_visual and (
        _contains_any_term(answer_norm, term_sets["visual_request_many_terms"])
    )
    if explicit_many:
        return _MAX_ASSET_CARDS_EXPLICIT_REQUEST
    return _MAX_ASSET_CARDS_PER_ANSWER


def _normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\bbed\s+room(s)?\b", r"bedroom\1", t)
    # Preserve CJK characters (Japanese, Chinese, Korean) alongside ASCII alphanumerics
    t = re.sub(r"[^a-z0-9\s\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]+", " ", t)
    # Normalize fullwidth/ideographic spaces to regular spaces
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _stem_token(token: str) -> str:
    t = token.strip().lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _tokenize(text: str, stopwords: Optional[Set[str]] = None) -> Set[str]:
    sw = stopwords if stopwords is not None else _BASE_STOPWORDS
    norm = _normalize_text(text)
    if not norm:
        return set()
    out: Set[str] = set()
    for raw in norm.split():
        if _is_cjk(raw):
            # Keep CJK tokens as-is (no stemming, no length filter)
            out.add(raw)
            # Also add individual CJK characters as tokens for partial matching
            for ch in raw:
                if _CJK_RE.match(ch):
                    out.add(ch)
            continue
        tok = _stem_token(raw)
        if len(tok) < 3 and tok not in _SPECIAL_SHORT_TOKENS:
            continue
        if tok in sw:
            continue
        out.add(tok)
    return out


def _contains_phrase(text_norm: str, phrase: str) -> bool:
    p = _normalize_text(phrase)
    if not p or len(p) < 3:
        return False
    return p in text_norm


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    try:
        u = urlparse(raw)
    except Exception:
        return ""
    host = (u.netloc or "").lower()
    path = (u.path or "").rstrip("/")
    if path == "/":
        path = ""
    if not host:
        return ""
    return f"{host}{path}"


def _extract_normalized_urls(text: str) -> Set[str]:
    out: Set[str] = set()
    for m in _URL_RE.findall(text or ""):
        # Trim common trailing punctuation.
        candidate = m.rstrip(".,;:!?)\"]}'")
        n = _normalize_url(candidate)
        if n:
            out.add(n)
    return out


def _qualifier_conflict(answer_tokens: Set[str], asset_tokens: Set[str]) -> bool:
    for group in _QUALIFIER_GROUPS:
        q = answer_tokens & group
        a = asset_tokens & group
        if q and a and q.isdisjoint(a):
            return True
    return False


def _asset_answer_score(
    asset: BotAsset,
    answer_norm: str,
    answer_tokens: Set[str],
    answer_urls: Set[str],
    *,
    generic_tokens: Set[str],
    stopwords: Set[str],
) -> int:
    text = " ".join([asset.name or "", asset.description or "", " ".join(asset.keywords or [])]).strip()
    asset_tokens = _tokenize(text, stopwords)
    if not asset_tokens:
        return 0
    if _qualifier_conflict(answer_tokens, asset_tokens):
        return 0

    score = 0
    url_hit = False
    link_norm = _normalize_url(asset.link_url or "")
    if link_norm and link_norm in answer_urls:
        url_hit = True
        score += 20

    name_phrase = _contains_phrase(answer_norm, asset.name or "")
    if name_phrase:
        score += 10

    keyword_phrase = False
    for kw in asset.keywords or []:
        if _contains_phrase(answer_norm, kw):
            keyword_phrase = True
            score += 6
            break

    overlap = answer_tokens & asset_tokens
    overlap_count = len(overlap)
    if overlap_count >= 2:
        score += 4 + overlap_count
    elif overlap_count == 1 and any(t not in generic_tokens for t in overlap):
        score += 1

    # Require strong evidence from answer text.
    if not (url_hit or name_phrase or keyword_phrase or overlap_count >= 2):
        return 0
    return score


def _match_assets_from_answer(
    answer: str,
    assets: List[BotAsset],
    *,
    max_cards: int = _MAX_ASSET_CARDS_PER_ANSWER,
    term_sets: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict[str, str]]:
    ts = term_sets or _resolve_term_config(None)
    answer_norm = _normalize_text(answer)
    answer_tokens = _tokenize(answer, ts["stopwords"])
    answer_urls = _extract_normalized_urls(answer)

    if not answer_norm:
        return []

    scored: List[Tuple[int, str, BotAsset]] = []
    for a in assets:
        s = _asset_answer_score(
            a, answer_norm, answer_tokens, answer_urls,
            generic_tokens=ts["generic_tokens"], stopwords=ts["stopwords"],
        )
        if s <= 0:
            continue
        scored.append((s, a.asset_id, a))

    # Stable deterministic ordering by score desc then asset_id asc.
    scored.sort(key=lambda it: (-it[0], it[1]))
    out: List[Dict[str, str]] = []
    seen_image_urls: Set[str] = set()
    for _, _, a in scored:
        card = _asset_to_card(a)
        image_key = _normalize_card_image_url(card.get("image_url", ""))
        if image_key and image_key in seen_image_urls:
            continue
        if image_key:
            seen_image_urls.add(image_key)
        out.append(card)
        if len(out) >= max_cards:
            break
    return out


def _asset_query_score(
    asset: BotAsset,
    query_norm: str,
    query_tokens: Set[str],
    *,
    generic_tokens: Set[str],
    stopwords: Set[str],
    intent_terms: Set[str],
) -> int:
    if not query_norm:
        return 0
    parts = [asset.name or "", asset.description or "", " ".join(asset.keywords or [])]
    if getattr(asset, "asset_type", "image") == "menu_item":
        _, _, category = _menu_fields(asset)
        if category:
            parts.append(category)
    text = " ".join(parts).strip()
    asset_tokens = _tokenize(text, stopwords)
    if not asset_tokens:
        return 0

    score = 0
    name_phrase = _contains_phrase(query_norm, asset.name or "")
    if name_phrase:
        score += 12

    keyword_phrase = False
    for kw in asset.keywords or []:
        if _contains_phrase(query_norm, kw):
            keyword_phrase = True
            score += 6
            break

    overlap = query_tokens & asset_tokens
    non_generic_overlap = [t for t in overlap if t not in generic_tokens]
    if non_generic_overlap:
        score += 3 + len(non_generic_overlap)

    # Menu-aware boost to improve restaurant recall when user asks for menu details.
    if getattr(asset, "asset_type", "image") == "menu_item":
        if _contains_any_term(query_norm, intent_terms):
            score += 3
        _, _, category = _menu_fields(asset)
        if category and category in query_norm:
            score += 4

    if not (name_phrase or keyword_phrase or non_generic_overlap):
        return 0
    return score


def _match_assets_from_query(
    user_query: Optional[str],
    assets: List[BotAsset],
    *,
    max_cards: int = _ASSET_MATCH_CANDIDATE_POOL,
    term_sets: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict[str, str]]:
    ts = term_sets or _resolve_term_config(None)
    query_norm = _normalize_text(user_query or "")
    query_tokens = _tokenize(query_norm, ts["stopwords"])
    if not query_norm:
        return []

    scored: List[Tuple[int, str, BotAsset]] = []
    for a in assets:
        s = _asset_query_score(
            a, query_norm, query_tokens,
            generic_tokens=ts["generic_tokens"],
            stopwords=ts["stopwords"],
            intent_terms=ts["asset_intent_terms"],
        )
        if s <= 0:
            continue
        scored.append((s, a.asset_id, a))

    scored.sort(key=lambda it: (-it[0], it[1]))
    out: List[Dict[str, str]] = []
    seen_image_urls: Set[str] = set()
    for _, _, a in scored:
        card = _asset_to_card(a)
        image_key = _normalize_card_image_url(card.get("image_url", ""))
        if image_key and image_key in seen_image_urls:
            continue
        if image_key:
            seen_image_urls.add(image_key)
        out.append(card)
        if len(out) >= max_cards:
            break
    return out


def _merge_cards_unique(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen_asset_ids: Set[str] = set()
    seen_images: Set[str] = set()
    for group in groups:
        for card in group or []:
            aid = str(card.get("asset_id") or "").strip()
            image_key = _normalize_card_image_url(card.get("image_url", ""))
            if aid and aid in seen_asset_ids:
                continue
            if image_key and image_key in seen_images:
                continue
            if aid:
                seen_asset_ids.add(aid)
            if image_key:
                seen_images.add(image_key)
            out.append(card)
    return out


_ASSET_BANK_LIMIT = max(
    50,
    min(int(os.environ.get("ASSET_BANK_LIMIT", "150")), 200),
)


def build_asset_bank(
    bot_id: str,
    *,
    asset_rules: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a compact asset reference (id, name, price, type, category) for the LLM.
    Up to 150 assets. URLs are resolved server-side from asset IDs after generation.
    marker_rule from asset_rules (config) controls the instruction text.
    """
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return ""

    marker_rule = (
        (asset_rules or {}).get("marker_rule") or get_default_marker_rule()
    ).strip()

    # Take up to limit (storage order)
    selected = assets[:_ASSET_BANK_LIMIT]
    lines: list[str] = []
    for a in selected:
        price_text, _, category = _menu_fields(a)
        atype = str(getattr(a, "asset_type", "image") or "image")
        parts = [a.asset_id, a.name or "", price_text or "-", atype, category or "-"]
        lines.append(" | ".join(str(p).strip() for p in parts))

    header = f"ASSET BANK (id | name | price | type | category) — {len(selected)} items"
    if len(assets) > len(selected):
        header += f" (showing first {len(selected)} of {len(assets)})"
    bank = header + "\n" + "\n".join(lines)
    return f"\n\n{bank}\n\nASSET RULES (CRITICAL): {marker_rule}\n"


def build_asset_instruction(
    bot_id: str,
    *,
    asset_rules: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build system instruction with the asset bank (id, name, price, type, category).
    Up to 150 assets. URLs are resolved server-side from {{asset_ID}} markers.
    marker_rule from asset_rules (config) controls the instruction text.
    """
    return build_asset_bank(bot_id, asset_rules=asset_rules)


def _filter_and_record_session_assets(
    cards: List[Dict[str, str]], 
    bot_id: str, 
    session_id: Optional[str]
) -> List[Dict[str, str]]:
    sid = (session_id or "").strip()
    if not sid or not cards:
        return cards
        
    r = _get_redis()
    if r is None:
        return cards

    key = _asset_session_key(bot_id, sid)
    image_key = _asset_session_image_key(bot_id, sid)
    
    filtered: List[Dict[str, str]] = []
    for card in cards:
        aid = (card.get("asset_id") or "").strip()
        img = _normalize_card_image_url(card.get("image_url", ""))
        
        # Check if already sent
        try:
            already_sent = False
            if aid and r.sismember(key, aid):
                already_sent = True
            elif img and r.sismember(image_key, img):
                already_sent = True
        except Exception:
            already_sent = False
            
        if already_sent:
            continue
        filtered.append(card)
        
    # Record new
    try:
        new_aids = [c.get("asset_id") for c in filtered if c.get("asset_id")]
        new_imgs = [_normalize_card_image_url(c.get("image_url", "")) for c in filtered]
        new_imgs = [u for u in new_imgs if u]
        
        if new_aids:
            r.sadd(key, *new_aids)
            r.expire(key, _ASSET_SESSION_DEDUPE_TTL_SECONDS)
        if new_imgs:
            r.sadd(image_key, *new_imgs)
            r.expire(image_key, _ASSET_SESSION_DEDUPE_TTL_SECONDS)
    except Exception:
        pass
        
    return filtered


def sanitize_answer_for_display(answer: str) -> str:
    """
    Clean the answer before showing to user:
    - Strip any remaining asset_id / asset marker syntax (never show to user)
    - Ensure URLs are on their own line (avoids concatenation with random text)
    """
    if not (answer or "").strip():
        return answer or ""
    text = answer.strip()

    # 1. Strip any remaining asset markers (never show asset_id to user)
    text = re.sub(r"\s*\{\{asset[:_][a-zA-Z0-9_\-]+\}\}", "", text)
    text = re.sub(r"asset_[a-f0-9]{12,}", "", text)

    # 1b. Strip malformed {{...}} (LLM sometimes outputs {{| name}} or {{ name}} instead of {{asset_ID}})
    def _unwrap_malformed_braces(m: re.Match) -> str:
        inner = (m.group(1) or "").strip()
        inner = re.sub(r"^\|\s*", "", inner)  # remove leading "| " if present
        return inner if inner else ""

    text = re.sub(r"\s*\{\{([^}]*)\}\}", _unwrap_malformed_braces, text)

    # 2. Ensure standalone URLs are on their own line (prevents concatenation with following text)
    # Use ASCII-only URL chars so we stop before Japanese/CJK (e.g. から) that can run into the URL
    _URL_PATTERN = re.compile(
        r"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+",
        re.IGNORECASE,
    )

    def _url_on_newline(m: re.Match) -> str:
        url = m.group(0)
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        # Skip URLs inside markdown links [text](url)
        if before == "(" and after == ")":
            return url
        prefix = "\n" if before and before not in "\n" else ""
        suffix = "\n" if after and after not in "\n)" else ""
        return f"{prefix}{url}{suffix}"

    text = _URL_PATTERN.sub(_url_on_newline, text)

    # Clean up excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


def resolve_asset_markers(
    answer: str,
    bot_id: str,
    session_id: Optional[str] = None,
    allowed_asset_types: Optional[Set[str]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Parse {{asset_ID}} markers from the LLM answer.
    Returns:
        - Cleaned answer text (markers removed)
        - List of asset cards referenced by the markers (deduplicated per session)
    """
    # Match {{asset_ID}} only
    pattern = r"\s*\{\{asset_([a-zA-Z0-9_\-]+)\}\}"
    matches = list(re.finditer(pattern, answer))
    if not matches:
        return sanitize_answer_for_display(answer), []

    cards = []
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    normalized_allowed = {
        str(t or "").strip().lower()
        for t in (allowed_asset_types or set())
        if str(t or "").strip()
    }
    if normalized_allowed:
        assets = [
            a
            for a in assets
            if str(getattr(a, "asset_type", "image") or "image").strip().lower() in normalized_allowed
        ]
    asset_map = {a.asset_id: a for a in assets}

    cleaned_answer = answer
    for m in reversed(matches):
        start, end = m.span()
        raw_id = m.group(1)
        cleaned_answer = cleaned_answer[:start] + cleaned_answer[end:]

        asset_id = "asset_" + raw_id if "asset_" + raw_id in asset_map else (raw_id if raw_id in asset_map else None)
        if asset_id:
            cards.append(_asset_to_card(asset_map[asset_id]))

    # Deduplicate cards within this single answer first
    unique_cards = []
    seen = set()
    # matches were processed backwards, so cards are in reverse order
    for c in reversed(cards):
        aid = c["asset_id"]
        if aid not in seen:
            seen.add(aid)
            unique_cards.append(c)
            
    # Apply session-based deduplication
    final_cards = _filter_and_record_session_assets(unique_cards, bot_id, session_id)

    return sanitize_answer_for_display(cleaned_answer), final_cards[:_GLOBAL_ASSET_CARD_LIMIT]


def process_answer_assets(
    answer: str,
    bot_id: str,
    user_query: Optional[str] = None,
    session_id: Optional[str] = None,
    allowed_asset_types: Optional[Set[str]] = None,
    asset_term_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Match assets after answer generation.
    - Marker cards from {{asset_ID}} in the answer.
    - Name-based fallback: assets mentioned by name but without markers.
    - Only return cards that have images.
    """
    cleaned, marker_cards = resolve_asset_markers(
        answer,
        bot_id,
        session_id,
        allowed_asset_types=allowed_asset_types,
    )
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    normalized_allowed = {
        str(t or "").strip().lower()
        for t in (allowed_asset_types or set())
        if str(t or "").strip()
    }
    if normalized_allowed:
        assets = [
            a
            for a in assets
            if str(getattr(a, "asset_type", "image") or "image").strip().lower() in normalized_allowed
        ]
    if not assets:
        logger.info("[AssetResolver] No active assets for bot %s", bot_id)
        return cleaned, []

    marker_count = len(marker_cards)
    msg = f"[AssetMenuTrace] process_answer_assets bot={bot_id} user_query={user_query!r}: marker_cards={marker_count} assets_total={len(assets)}"
    logger.info(msg)
    _trace_stdout(msg)

    # Merge: marker cards + name-matched cards (assets mentioned in text but without markers)
    term_sets = _resolve_term_config(asset_term_config)
    marker_ids = {c["asset_id"] for c in marker_cards}
    name_matched = _match_assets_from_answer(
        cleaned,
        assets,
        max_cards=_ASSET_BANK_LIMIT,
        term_sets=term_sets,
    )
    for c in name_matched:
        if c["asset_id"] not in marker_ids:
            marker_cards.append(c)
            marker_ids.add(c["asset_id"])

    cards = marker_cards if marker_cards else name_matched
    if not cards:
        return cleaned, []

    # Only return cards that have images (skip broken/empty cards)
    cards = [c for c in cards if (c.get("image_url") or "").strip()]
    if not cards:
        return sanitize_answer_for_display(cleaned), []

    max_cards_for_query = _max_cards_for_query(cleaned, cards, term_sets)
    if max_cards_for_query <= 0:
        msg = f"[AssetMenuTrace] process_answer_assets bot={bot_id}: max_cards_for_query={max_cards_for_query} -> returning 0 cards (had {len(cards)} before filter)"
        logger.info(msg)
        _trace_stdout(msg)
        return sanitize_answer_for_display(cleaned), []

    sid = (session_id or "").strip()
    if sid and cards:
        cards = _filter_and_record_session_assets(cards, bot_id, sid)

    limit = min(max_cards_for_query, _ASSET_BANK_LIMIT, _GLOBAL_ASSET_CARD_LIMIT)
    cards = cards[:limit]
    final_count = len(cards)
    msg = (
        f"[AssetMenuTrace] process_answer_assets bot={bot_id} user_query={user_query!r}: "
        f"final_cards={final_count} limit={limit} (marker={marker_count} name_matched={len(name_matched)})"
    )
    logger.info(msg)
    _trace_stdout(msg)
    return sanitize_answer_for_display(cleaned), cards
