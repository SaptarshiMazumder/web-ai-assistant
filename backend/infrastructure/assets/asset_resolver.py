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
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from common.config import config
from domain.entities import BotAsset
from infrastructure.db.repositories import PostgresBotAssetRepository
from redis import Redis

logger = logging.getLogger(__name__)


_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_MAX_ASSET_CARDS_PER_ANSWER = max(1, min(int(os.environ.get("ASSET_MAX_CARDS_PER_ANSWER", "3")), 5))
_MAX_ASSET_CARDS_EXPLICIT_REQUEST = max(
    _MAX_ASSET_CARDS_PER_ANSWER,
    min(int(os.environ.get("ASSET_MAX_CARDS_EXPLICIT_REQUEST", "6")), 8),
)
_ASSET_MATCH_CANDIDATE_POOL = max(
    _MAX_ASSET_CARDS_EXPLICIT_REQUEST * 5,
    _MAX_ASSET_CARDS_PER_ANSWER * 5,
)
_ASSET_SESSION_DEDUPE_TTL_SECONDS = max(
    300,
    min(int(os.environ.get("ASSET_SESSION_DEDUPE_TTL_SECONDS", "43200")), 604800),
)
_VISUAL_REQUEST_TERMS = {
    "photo",
    "photos",
    "image",
    "images",
    "picture",
    "pictures",
    "pic",
    "pics",
    "gallery",
    "show me",
    "send me",
    "share",
    "let me see",
    "what it looks like",
    # Japanese
    "写真",
    "画像",
    "見せて",
    "見たい",
    "見せてください",
    "見たいです",
}
_VISUAL_REQUEST_MANY_TERMS = {
    "all",
    "more",
    "many",
    "several",
    "multiple",
    "full menu",
    "whole menu",
    "entire menu",
    "more photos",
    "more images",
    # Japanese
    "全部",
    "もっと",
    "全メニュー",
    "メニュー全部",
    "一覧",
}
_VISUAL_SUPPRESS_TERMS = {
    "no image",
    "no images",
    "no photo",
    "no photos",
    "no picture",
    "no pictures",
    "without image",
    "without images",
    "without photo",
    "without photos",
    # Japanese
    "画像なし",
    "写真なし",
    "画像不要",
    "写真不要",
}
_ASSET_INTENT_TERMS = {
    "menu",
    "dish",
    "dishes",
    "food",
    "drink",
    "drinks",
    "beverage",
    "beverages",
    "product",
    "products",
    "service",
    "services",
    "package",
    "packages",
    "plan",
    "plans",
    "room",
    "rooms",
    "suite",
    "suites",
    "facility",
    "facilities",
    "amenity",
    "amenities",
    "location",
    "locations",
    "map",
    "branch",
    "branches",
    "store",
    "stores",
    "item",
    "items",
    "option",
    "options",
    "catalog",
    "collection",
    # Japanese
    "メニュー",
    "料理",
    "コース",
    "食べ物",
    "飲み物",
    "ドリンク",
    "商品",
    "サービス",
    "プラン",
    "部屋",
    "施設",
    "おすすめ",
    "人気",
    "定番",
    "ランチ",
    "ディナー",
    "デザート",
    "前菜",
    "刺身",
    "寿司",
    "焼肉",
    "セット",
}
_SPECIAL_SHORT_TOKENS = {"xl", "xxl", "xs"}
_GENERIC_TOKENS = {
    "about",
    "available",
    "booking",
    "cost",
    "detail",
    "details",
    "info",
    "information",
    "item",
    "items",
    "option",
    "options",
    "price",
    "product",
    "products",
    "service",
    "services",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "we",
    "with",
    "you",
    "your",
} | _GENERIC_TOKENS
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


def _query_matches_cards(query_norm: str, cards: List[Dict[str, str]]) -> bool:
    q_tokens = _tokenize(query_norm)
    if not q_tokens:
        return False
    for card in cards:
        name_tokens = _tokenize(card.get("name", ""))
        if q_tokens & name_tokens:
            return True
    return False


def _max_cards_for_query(user_query: Optional[str], cards: List[Dict[str, str]]) -> int:
    query_norm = _normalize_text(user_query or "")
    if not query_norm:
        return 0
    if _contains_any_term(query_norm, _VISUAL_SUPPRESS_TERMS):
        return 0

    explicit_visual = _contains_any_term(query_norm, _VISUAL_REQUEST_TERMS) or bool(
        re.search(
            r"\b(show|send|share|see|view)\b.*\b(photo|image|picture|pic|gallery|menu|product|service|room|suite|location|map)\b",
            query_norm,
        )
    )
    intent_query = _contains_any_term(query_norm, _ASSET_INTENT_TERMS)
    card_match = _query_matches_cards(query_norm, cards)

    if not (explicit_visual or intent_query or card_match):
        return 0

    explicit_many = explicit_visual and (
        _contains_any_term(query_norm, _VISUAL_REQUEST_MANY_TERMS)
        or bool(re.search(r"\b(all|more|many|several|multiple)\b", query_norm))
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


def _tokenize(text: str) -> Set[str]:
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
        if tok in _STOPWORDS:
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
) -> int:
    text = " ".join([asset.name or "", asset.description or "", " ".join(asset.keywords or [])]).strip()
    asset_tokens = _tokenize(text)
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
    elif overlap_count == 1 and any(t not in _GENERIC_TOKENS for t in overlap):
        score += 1

    # Require strong evidence from answer text.
    if not (url_hit or name_phrase or keyword_phrase or overlap_count >= 2):
        # logger.debug(f"Asset mismatch {asset.asset_id}: url={url_hit} name={name_phrase} kw={keyword_phrase} overlap={overlap_count}")
        return 0

    # logger.debug(f"Asset match {asset.asset_id}: score={score} (url={url_hit} name={name_phrase} kw={keyword_phrase} overlap={overlap_count})")
    return score


def _match_assets_from_answer(
    answer: str,
    assets: List[BotAsset],
    *,
    max_cards: int = _MAX_ASSET_CARDS_PER_ANSWER,
) -> List[Dict[str, str]]:
    answer_norm = _normalize_text(answer)
    answer_tokens = _tokenize(answer)
    answer_urls = _extract_normalized_urls(answer)

    if not answer_norm:
        return []

    scored: List[Tuple[int, str, BotAsset]] = []
    for a in assets:
        s = _asset_answer_score(a, answer_norm, answer_tokens, answer_urls)
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
) -> int:
    if not query_norm:
        return 0
    parts = [asset.name or "", asset.description or "", " ".join(asset.keywords or [])]
    if getattr(asset, "asset_type", "image") == "menu_item":
        _, _, category = _menu_fields(asset)
        if category:
            parts.append(category)
    text = " ".join(parts).strip()
    asset_tokens = _tokenize(text)
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
    non_generic_overlap = [t for t in overlap if t not in _GENERIC_TOKENS]
    if non_generic_overlap:
        score += 3 + len(non_generic_overlap)

    # Menu-aware boost to improve restaurant recall when user asks for menu details.
    if getattr(asset, "asset_type", "image") == "menu_item":
        if any(tok in query_norm for tok in ("menu", "dish", "drink", "lunch", "course", "party")):
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
) -> List[Dict[str, str]]:
    query_norm = _normalize_text(user_query or "")
    query_tokens = _tokenize(query_norm)
    if not query_norm:
        return []

    scored: List[Tuple[int, str, BotAsset]] = []
    for a in assets:
        s = _asset_query_score(a, query_norm, query_tokens)
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


def build_asset_bank(bot_id: str) -> str:
    """
    Build a compact asset reference (id, name, price, type, category) for the LLM.
    Up to 150 assets. URLs are resolved server-side from asset IDs after generation.
    """
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return ""

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
    return (
        f"\n\n{bank}\n\n"
        "ASSET RULES (CRITICAL): Include {{asset_ASSET_ID}} for EVERY product/service you mention by name. "
        "One marker per item — do not skip any. URLs and images are resolved server-side from the ID.\n"
    )


def build_asset_instruction(bot_id: str) -> str:
    """
    Build system instruction with the asset bank (id, name, price, type, category).
    Up to 150 assets. URLs are resolved server-side from {{asset_ID}} markers.
    """
    return build_asset_bank(bot_id)


_MAX_ASSETS_IN_EVIDENCE = max(
    5,
    min(int(os.environ.get("ASSET_MAX_IN_EVIDENCE", "20")), 50),
)


def _asset_to_evidence_snippet(a: BotAsset) -> dict:
    """Build a single evidence dict for an asset. _skip_rerank ensures it is not dropped by reranking."""
    desc = (a.description or "").strip()
    kw = ", ".join(a.keywords or [])
    label = "Menu Item" if getattr(a, "asset_type", "image") == "menu_item" else "Product/Service"
    snippet_parts = [f"{label}: {a.name}."]
    if getattr(a, "asset_type", "image") == "menu_item":
        price_text, details, category = _menu_fields(a)
        if category:
            snippet_parts.append(f"Category: {category}.")
        if price_text:
            snippet_parts.append(f"Price: {price_text}.")
        if details:
            snippet_parts.append(f"Details: {details}.")
    if desc:
        snippet_parts.append(f"Description: {desc}.")
    if kw:
        snippet_parts.append(f"Related keywords: {kw}.")
    snippet_parts.append(
        f"To show this product's image in the response, include {{{{asset_{a.asset_id}}}}} in your answer."
    )
    return {
        "url": a.link_url or f"asset:{a.asset_id}",
        "snippet": " ".join(snippet_parts),
        "_skip_rerank": True,
    }


def build_asset_evidence(bot_id: str, query: Optional[str] = None) -> list[dict]:
    """
    Convert active business assets into evidence snippets that can be
    injected into the RAG pipeline via extra_evidence.

    Each asset becomes an evidence snippet so the LLM can discover and
    reference products/services it wouldn't otherwise know about.
    Capped to avoid overwhelming the RAG context.

    When query is provided, assets are ranked by relevance (name, keywords,
    description overlap) so that menu questions like "what wagyu dishes?"
    include ALL matching items instead of the first N in storage order.
    """
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return []

    # Query-aware selection: rank by relevance so "wagyu" returns all wagyu items
    query_norm = _normalize_text(query or "")
    query_tokens = _tokenize(query_norm)
    if query_norm:
        scored: List[Tuple[int, str, BotAsset]] = []
        for a in assets:
            s = _asset_query_score(a, query_norm, query_tokens)
            scored.append((s, a.asset_id or "", a))
        scored.sort(key=lambda it: (-it[0], it[1]))
        selected = [a for _, _, a in scored[:_MAX_ASSETS_IN_EVIDENCE]]
    else:
        selected = assets[:_MAX_ASSETS_IN_EVIDENCE]

    evidence: list[dict] = []
    for a in selected:
        evidence.append(_asset_to_evidence_snippet(a))
    return evidence


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
        return answer, []

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

    return cleaned_answer.strip(), final_cards


def process_answer_assets(
    answer: str,
    bot_id: str,
    user_query: Optional[str] = None,
    session_id: Optional[str] = None,
    allowed_asset_types: Optional[Set[str]] = None,
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

    # Merge: marker cards + name-matched cards (assets mentioned in text but without markers)
    marker_ids = {c["asset_id"] for c in marker_cards}
    name_matched = _match_assets_from_answer(
        cleaned,
        assets,
        max_cards=_ASSET_BANK_LIMIT,
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
        return cleaned, []

    max_cards_for_query = _max_cards_for_query(user_query, cards)
    if max_cards_for_query <= 0:
        return cleaned, []

    sid = (session_id or "").strip()
    if sid and cards:
        cards = _filter_and_record_session_assets(cards, bot_id, sid)
    return cleaned, cards[:_ASSET_BANK_LIMIT]
