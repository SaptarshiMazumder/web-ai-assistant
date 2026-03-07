from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from domain.interfaces import ReservationContext
from domain.platform_profiles import (
    get_menu_category_order,
    get_menu_category_aliases,
    get_menu_view_all_url_tokens,
    get_post_crawl_jobs_for_widget,
    get_reservation_config_from_widget,
    normalize_reservation_links,
)


def _menu_item_source_url(item: Any) -> str:
    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
    u = str(metadata.get("source_url") or getattr(item, "link_url", "") or "").strip()
    return u if u.startswith("http") else ""


@dataclass(frozen=True)
class ConfigBackedPlatformStrategy:
    platform_id: str

    def build_reservation_context(self, widget_config: Dict[str, Any], *, lang: str) -> Optional[ReservationContext]:
        cfg = get_reservation_config_from_widget(widget_config, lang=lang)
        if not cfg:
            return None
        return ReservationContext(
            platform_id=str(cfg.get("platform_id") or "").strip(),
            domain_key=str(cfg.get("domain_key") or "").strip(),
            url=str(cfg.get("url") or "").strip(),
            instruction=str(cfg.get("instruction") or "").strip(),
            link_label=str(cfg.get("link_label") or "").strip(),
        )

    def menu_link(self, category: str, items: List[Any], widget_config: Optional[Dict[str, Any]] = None) -> str:
        return build_menu_view_all_url_for_category(category, items, widget_config=widget_config)

    def post_crawl_jobs(self, widget_config: Dict[str, Any]) -> List[str]:
        return get_post_crawl_jobs_for_widget(widget_config)


class PlatformStrategyRegistry:
    """Registry keyed by platform_id; strategies are config-backed and extensible."""

    def __init__(self) -> None:
        self._strategies: Dict[str, ConfigBackedPlatformStrategy] = {}

    def get(self, platform_id: str) -> ConfigBackedPlatformStrategy:
        pid = str(platform_id or "").strip().lower()
        if pid not in self._strategies:
            self._strategies[pid] = ConfigBackedPlatformStrategy(platform_id=pid)
        return self._strategies[pid]

    def resolve_for_widget(self, widget_config: Dict[str, Any]) -> Optional[ConfigBackedPlatformStrategy]:
        links = normalize_reservation_links(widget_config)
        if not links:
            return None
        platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
        if platform_id and platform_id in links:
            return self.get(platform_id)
        first_platform = next(iter(links.keys()), "")
        return self.get(first_platform) if first_platform else None


def normalize_menu_category(raw_category: str, widget_config: Optional[Dict[str, Any]] = None) -> str:
    default_category = next((c for c in get_menu_category_order() if c), "")
    category = str(raw_category or "").strip().lower()
    aliases = get_menu_category_aliases(widget_config=widget_config)
    normalized = aliases.get(category, category)
    return normalized or default_category


def build_menu_view_all_url_for_category(
    category: str,
    items: List[Any],
    *,
    widget_config: Optional[Dict[str, Any]] = None,
) -> str:
    tokens_map = get_menu_view_all_url_tokens(widget_config=widget_config)
    urls = list({_menu_item_source_url(i) for i in items if _menu_item_source_url(i)})
    for u in urls:
        low = u.lower()
        for token in tokens_map.get(category, []):
            if str(token).lower() in low:
                return u
    return urls[0] if urls else ""
