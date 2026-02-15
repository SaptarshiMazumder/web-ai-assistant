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

_ASSET_MARKER_RE = re.compile(r"\{\{asset:([a-zA-Z0-9_]+)\}\}")
_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_MAX_ASSET_CARDS_PER_ANSWER = max(1, min(int(os.environ.get("ASSET_MAX_CARDS_PER_ANSWER", "2")), 5))
_ASSET_SESSION_DEDUPE_TTL_SECONDS = max(
    300,
    min(int(os.environ.get("ASSET_SESSION_DEDUPE_TTL_SECONDS", "43200")), 604800),
)
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
    }


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


def _normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\bbed\s+room(s)?\b", r"bedroom\1", t)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
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
        return 0

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


def build_asset_instruction(bot_id: str) -> str:
    """
    Disabled by design.
    Assets are intentionally not injected before generation.
    """
    return ""


def resolve_asset_markers(
    answer: str,
    bot_id: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Parse {{asset:ID}} markers from answer text.
    Kept for backward compatibility; markers are stripped from final answer.
    """
    marker_ids = _ASSET_MARKER_RE.findall(answer)
    if not marker_ids:
        return answer, []

    seen = set()
    unique_ids = []
    for mid in marker_ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    repo = _get_repo()
    cards: List[Dict[str, str]] = []
    for aid in unique_ids:
        asset = repo.get_asset(aid)
        if asset and asset.bot_id == bot_id and asset.is_active:
            cards.append(_asset_to_card(asset))

    cleaned = _ASSET_MARKER_RE.sub("", answer).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, cards


def process_answer_assets(
    answer: str,
    bot_id: str,
    user_query: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Match assets after answer generation.
    - No pre-generation asset influence.
    - Cards are selected only from answer content matches.
    """
    cleaned, _ = resolve_asset_markers(answer, bot_id)
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return cleaned, []
    cards = _match_assets_from_answer(cleaned, assets, max_cards=_MAX_ASSET_CARDS_PER_ANSWER)

    sid = (session_id or "").strip()
    if sid and cards:
        r = _get_redis()
        if r is not None:
            key = _asset_session_key(bot_id, sid)
            image_key = _asset_session_image_key(bot_id, sid)
            filtered: List[Dict[str, str]] = []
            for card in cards:
                aid = (card.get("asset_id") or "").strip()
                if not aid:
                    continue
                img = _normalize_card_image_url(card.get("image_url", ""))
                try:
                    already_sent = bool(r.sismember(key, aid)) or (bool(img) and bool(r.sismember(image_key, img)))
                except Exception:
                    already_sent = False
                if already_sent:
                    continue
                filtered.append(card)
            cards = filtered
            if cards:
                try:
                    asset_ids = [(c.get("asset_id") or "").strip() for c in cards if (c.get("asset_id") or "").strip()]
                    image_urls = [_normalize_card_image_url(c.get("image_url", "")) for c in cards]
                    image_urls = [u for u in image_urls if u]
                    if asset_ids:
                        r.sadd(key, *asset_ids)
                    if image_urls:
                        r.sadd(image_key, *image_urls)
                    r.expire(key, _ASSET_SESSION_DEDUPE_TTL_SECONDS)
                    r.expire(image_key, _ASSET_SESSION_DEDUPE_TTL_SECONDS)
                except Exception:
                    pass
    return cleaned, cards
