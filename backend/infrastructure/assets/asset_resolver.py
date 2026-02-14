"""
Asset resolver – injects asset descriptions into the system prompt and resolves
{{asset:ID}} markers from the LLM answer back into structured AssetCard objects.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from domain.entities import BotAsset
from infrastructure.db.repositories import PostgresBotAssetRepository

logger = logging.getLogger(__name__)

_ASSET_MARKER_RE = re.compile(r"\{\{asset:([a-zA-Z0-9_]+)\}\}")

_repo: Optional[PostgresBotAssetRepository] = None


def _get_repo() -> PostgresBotAssetRepository:
    global _repo
    if _repo is None:
        _repo = PostgresBotAssetRepository()
    return _repo


# ---------------------------------------------------------------------------
# 1. Build the asset section to append to the system instruction
# ---------------------------------------------------------------------------


def build_asset_instruction(bot_id: str) -> str:
    """Return a block of text to append to the system prompt (empty string if no assets)."""
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return ""
    lines = []
    for a in assets:
        desc = (a.description or a.name).strip()
        lines.append(f"- [{a.asset_id}] \"{a.name}\" – {desc}")
    return (
        "\n\n--- Business Assets ---\n"
        "You have access to the following business assets (images/cards). "
        "When relevant to the user's question, include the asset by writing exactly: {{asset:ASSET_ID}} on its own line.\n"
        "Only include assets when they directly relate to what the user is asking about.\n\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# 2. Resolve {{asset:ID}} markers in the answer
# ---------------------------------------------------------------------------


def resolve_asset_markers(
    answer: str,
    bot_id: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Parse {{asset:ID}} markers from the answer text.

    Returns:
        (cleaned_answer, asset_cards)  where asset_cards is a list of dicts
        with keys: asset_id, name, image_url, link_url (nullable).
    """
    marker_ids = _ASSET_MARKER_RE.findall(answer)
    if not marker_ids:
        return answer, []

    # Dedupe while preserving order
    seen = set()
    unique_ids = []
    for mid in marker_ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    # Fetch assets
    repo = _get_repo()
    cards: List[Dict[str, str]] = []
    for aid in unique_ids:
        asset = repo.get_asset(aid)
        if asset and asset.bot_id == bot_id and asset.is_active:
            cards.append({
                "asset_id": asset.asset_id,
                "name": asset.name,
                "image_url": asset.image_public_url,
                "link_url": asset.link_url or "",
            })

    # Strip markers from answer text
    cleaned = _ASSET_MARKER_RE.sub("", answer).strip()
    # Clean up extra blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned, cards


# ---------------------------------------------------------------------------
# 3. Keyword fallback – append assets whose keywords appear in the answer
# ---------------------------------------------------------------------------


def keyword_fallback(
    answer: str,
    bot_id: str,
    already_ids: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Scan the answer for keyword matches and return additional asset cards."""
    assets = _get_repo().list_assets_for_bot(bot_id, active_only=True)
    if not assets:
        return []

    already = already_ids or set()
    answer_lower = answer.lower()
    extra: List[Dict[str, str]] = []

    for a in assets:
        if a.asset_id in already:
            continue
        if not a.keywords:
            continue
        for kw in a.keywords:
            if kw.lower() in answer_lower:
                extra.append({
                    "asset_id": a.asset_id,
                    "name": a.name,
                    "image_url": a.image_public_url,
                    "link_url": a.link_url or "",
                })
                already.add(a.asset_id)
                break  # one match is enough

    return extra


# ---------------------------------------------------------------------------
# Combined convenience
# ---------------------------------------------------------------------------


def process_answer_assets(
    answer: str,
    bot_id: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Resolve markers + keyword fallback in one call."""
    cleaned, cards = resolve_asset_markers(answer, bot_id)
    already = {c["asset_id"] for c in cards}
    extra = keyword_fallback(cleaned, bot_id, already_ids=already)
    return cleaned, cards + extra
