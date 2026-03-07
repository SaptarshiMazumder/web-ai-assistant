from __future__ import annotations

from typing import Any, Dict, List, Optional

from infrastructure.assets.asset_resolver import process_answer_assets, resolve_asset_markers


class ConfigAssetPolicy:
    """Asset decision policy that is independent from route/controller code."""

    def resolve_assets(
        self,
        *,
        answer: str,
        bot_id: str,
        user_query: Optional[str],
        session_id: Optional[str],
        allowed_asset_types: Optional[set[str]],
        show_assets: Optional[bool],
        asset_term_config: Optional[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, str]]]:
        if show_assets is False:
            return answer, []
        cleaned, marker_cards = resolve_asset_markers(
            answer,
            bot_id,
            session_id,
            allowed_asset_types=allowed_asset_types,
        )
        if marker_cards:
            return cleaned, marker_cards
        return process_answer_assets(
            cleaned,
            bot_id,
            user_query=user_query,
            session_id=session_id,
            allowed_asset_types=allowed_asset_types,
            asset_term_config=asset_term_config,
        )
