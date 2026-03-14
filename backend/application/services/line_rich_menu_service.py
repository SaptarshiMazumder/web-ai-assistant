import asyncio
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from common.language_utils import normalize_lang
from domain.platform_profiles import (
    build_line_design_effective,
    get_line_rich_menu_definition,
    get_line_rich_menu_labels_from_suggested_messages,
    normalize_line_design_overrides,
    get_reservation_config_from_widget,
    has_support_suggested_message_for_widget,
)
from infrastructure.clients.line_client import (
    clear_default_rich_menu,
    create_rich_menu,
    delete_rich_menu,
    link_user_rich_menu,
    set_default_rich_menu,
    unlink_user_rich_menu,
    upload_rich_menu_image,
)
from infrastructure.db.repositories import (
    PostgresBotAssetRepository,
    PostgresBotRepository,
    PostgresConversationRepository,
    PostgresLineChannelRepository,
    PostgresLineDesignConfigRepository,
    PostgresLineRichMenuStateRepository,
)

_PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL")
    or os.environ.get("BACKEND_PUBLIC_BASE_URL")
    or os.environ.get("API_BASE_URL")
    or os.environ.get("EXTERNAL_BASE_URL")
    or os.environ.get("PUBLIC_API_BASE_URL")
    or ""
).strip()

_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
# Bump to force re-sync of existing rich menus so `selected: false` is re-applied
# for already-linked LINE channels/users.
_RICH_MENU_RENDER_VERSION = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_dict(raw: Optional[str]) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_color(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if not raw.startswith("#") and len(raw) in {3, 6}:
        raw = f"#{raw}"
    return raw if _HEX_COLOR_RE.fullmatch(raw) else fallback


def _font_candidates(*, lang: str = "en") -> List[str]:
    configured = str(os.environ.get("LINE_RICH_MENU_FONT_PATH") or "").strip()
    configured_candidates = [item.strip() for item in configured.split(os.pathsep) if item.strip()] if configured else []
    latin_candidates = [
        "arial.ttf",
        "segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    cjk_candidates = [
        "meiryo.ttc",
        "YuGothM.ttc",
        "msgothic.ttc",
        "C:\\Windows\\Fonts\\meiryo.ttc",
        "C:\\Windows\\Fonts\\YuGothM.ttc",
        "C:\\Windows\\Fonts\\msgothic.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKJP-Regular.otf",
        "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    ordered: List[str] = []
    seen = set()
    lang_first = cjk_candidates + latin_candidates if normalize_lang(lang) == "ja" else latin_candidates + cjk_candidates
    for candidate in configured_candidates + lang_first:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _load_font(size: int, *, lang: str = "en") -> ImageFont.ImageFont:
    for path in _font_candidates(lang=lang):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(0, bbox[2] - bbox[0])


class LineRichMenuService:
    def __init__(
        self,
        *,
        bot_repo: Optional[PostgresBotRepository] = None,
        line_channel_repo: Optional[PostgresLineChannelRepository] = None,
        line_design_repo: Optional[PostgresLineDesignConfigRepository] = None,
        state_repo: Optional[PostgresLineRichMenuStateRepository] = None,
        conversation_repo: Optional[PostgresConversationRepository] = None,
        asset_repo: Optional[PostgresBotAssetRepository] = None,
    ):
        self._bot_repo = bot_repo or PostgresBotRepository()
        self._line_channel_repo = line_channel_repo or PostgresLineChannelRepository()
        self._line_design_repo = line_design_repo or PostgresLineDesignConfigRepository()
        self._state_repo = state_repo or PostgresLineRichMenuStateRepository()
        self._conversation_repo = conversation_repo or PostgresConversationRepository()
        self._asset_repo = asset_repo or PostgresBotAssetRepository()
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, bot_id: str) -> asyncio.Lock:
        bid = (bot_id or "").strip()
        lock = self._locks.get(bid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[bid] = lock
        return lock

    def get_state(self, bot_id: str):
        return self._state_repo.get_by_bot_id(bot_id)

    def _get_widget_config(self, bot) -> Dict[str, Any]:
        return _parse_json_dict(getattr(bot, "widget_config", None))

    def _get_line_design_overrides(self, bot_id: str) -> Dict[str, Any]:
        row = self._line_design_repo.get_by_bot_id(bot_id)
        payload = _parse_json_dict(getattr(row, "config_json", None) if row else None)
        return normalize_line_design_overrides(payload)

    def _get_line_design_effective(self, bot_id: str) -> Dict[str, Any]:
        return build_line_design_effective(self._get_line_design_overrides(bot_id))

    def _get_bot_lang(self, widget_config: Dict[str, Any]) -> str:
        return normalize_lang(widget_config.get("language") or widget_config.get("botLanguage") or "en")

    def _get_menu_page_url(self, publishable_key: Optional[str]) -> Optional[str]:
        pk = str(publishable_key or "").strip()
        if not pk or not _PUBLIC_BASE_URL:
            return None
        base = _PUBLIC_BASE_URL.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"https://{base}"
        url = f"{base.rstrip('/')}/v1/pk/{pk}/menu"
        return url if url.startswith("https://") else None

    def _get_menu_count(self, bot_id: str) -> int:
        _, total_count, _ = self._asset_repo.list_assets_for_bot_paginated(
            bot_id,
            active_only=True,
            asset_type="menu_item",
            page_size=1,
            offset=0,
        )
        return total_count

    def _resolve_capabilities(self, bot) -> Dict[str, Any]:
        widget_config = self._get_widget_config(bot)
        bot_lang = self._get_bot_lang(widget_config)
        theme_color = _normalize_color(widget_config.get("color"), "#06c755")
        reservation_cfg = get_reservation_config_from_widget(widget_config, lang=bot_lang) or {}
        reservation_url = str(reservation_cfg.get("url") or "").strip() or None
        menu_count = self._get_menu_count(bot.bot_id)
        menu_page_url = self._get_menu_page_url(getattr(bot, "publishable_key", None)) if menu_count > 0 else None
        support_enabled = has_support_suggested_message_for_widget(widget_config)
        return {
            "bot_lang": bot_lang,
            "theme_color": theme_color,
            "reservation_url": reservation_url,
            "menu_count": menu_count,
            "menu_page_url": menu_page_url,
            "menu_postback": "lineux:menu" if menu_count > 0 else None,
            "support_enabled": support_enabled,
            "support_postback": "lineux:support" if support_enabled else None,
            "back_to_ai_postback": "lineux:back_to_ai",
            "widget_config": widget_config,
        }

    def _resolve_action_target(self, action_def: Dict[str, Any], capabilities: Dict[str, Any]) -> Optional[Dict[str, str]]:
        capability = str(action_def.get("capability") or action_def.get("id") or "").strip()
        label = str(action_def.get("label") or "").strip()
        if not capability or not label:
            return None
        if capability == "support":
            postback_data = capabilities.get("support_postback")
            if not postback_data:
                return None
            return {"type": "postback", "label": label, "data": str(postback_data)}
        if capability == "back_to_ai":
            postback_data = capabilities.get("back_to_ai_postback")
            if not postback_data:
                return None
            return {"type": "postback", "label": label, "data": str(postback_data)}

        fallbacks: List[str] = []
        for key in ("uri_fallback_order", "fallback_order"):
            value = action_def.get(key)
            if isinstance(value, list):
                fallbacks.extend(str(item).strip() for item in value if str(item).strip())

        if capability == "reserve":
            for fallback in fallbacks or ["reservation_url"]:
                if fallback == "reservation_url" and capabilities.get("reservation_url"):
                    return {
                        "type": "uri",
                        "label": label,
                        "uri": str(capabilities["reservation_url"]),
                    }
            return None

        if capability == "menu":
            for fallback in fallbacks or ["menu_page_url", "menu_postback"]:
                if fallback == "menu_page_url" and capabilities.get("menu_page_url"):
                    return {
                        "type": "uri",
                        "label": label,
                        "uri": str(capabilities["menu_page_url"]),
                    }
                if fallback == "menu_postback" and capabilities.get("menu_postback"):
                    return {
                        "type": "postback",
                        "label": label,
                        "data": str(capabilities["menu_postback"]),
                    }
            return None
        return None

    def _build_variant_specs(self, bot) -> Dict[str, Any]:
        capabilities = self._resolve_capabilities(bot)
        widget_config = capabilities.get("widget_config") if isinstance(capabilities.get("widget_config"), dict) else {}
        line_design_effective = self._get_line_design_effective(bot.bot_id)
        rich_design = (
            line_design_effective.get("rich_menu")
            if isinstance(line_design_effective.get("rich_menu"), dict)
            else {}
        )
        design_styles = rich_design.get("styles") if isinstance(rich_design.get("styles"), dict) else {}
        design_actions_list = rich_design.get("actions") if isinstance(rich_design.get("actions"), list) else []
        design_layouts = rich_design.get("layouts") if isinstance(rich_design.get("layouts"), dict) else {}
        design_action_map: Dict[str, Dict[str, Any]] = {}
        for item in design_actions_list:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip()
            if aid:
                design_action_map[aid] = item
        suggested_label_map_by_lang = {
            lang_key: get_line_rich_menu_labels_from_suggested_messages(widget_config, lang=lang_key)
            for lang_key in ("en", "ja")
        }

        variants: Dict[str, Any] = {}
        for state_name in ("normal", "support"):
            for lang in ("en", "ja"):
                definition = get_line_rich_menu_definition(state=state_name, lang=lang)
                definition_actions = [item for item in (definition.get("actions") or []) if isinstance(item, dict)]
                if not definition_actions:
                    continue
                definition_action_map = {
                    str(item.get("id") or "").strip(): item
                    for item in definition_actions
                    if str(item.get("id") or "").strip()
                }
                definition_order = [aid for aid in definition_action_map.keys()]
                override_order = [
                    str(aid).strip()
                    for aid in (
                        design_layouts.get(state_name)
                        if isinstance(design_layouts.get(state_name), list)
                        else []
                    )
                    if str(aid).strip() in definition_action_map
                ]
                ordered_ids: List[str] = []
                for aid in override_order + definition_order:
                    if aid and aid not in ordered_ids:
                        ordered_ids.append(aid)

                resolved_actions = []
                for action_id in ordered_ids:
                    action_def = definition_action_map.get(action_id)
                    if not action_def:
                        continue
                    target = self._resolve_action_target(action_def, capabilities)
                    if not target:
                        continue
                    design_override = design_action_map.get(action_id, {})
                    if design_override.get("enabled") is False:
                        continue
                    labels = design_override.get("labels") if isinstance(design_override.get("labels"), dict) else {}
                    capability = str(action_def.get("capability") or action_id).strip().lower()
                    suggested_label = suggested_label_map_by_lang.get(lang, {}).get(action_id) or (
                        suggested_label_map_by_lang.get(lang, {}).get(capability)
                    )
                    label = str(labels.get(lang) or suggested_label or action_def.get("label") or "").strip()
                    if not label:
                        continue
                    icon = str(design_override.get("icon") or action_def.get("icon") or action_id).strip() or action_id
                    resolved_actions.append(
                        {
                            "id": action_id,
                            "label": label,
                            "icon": icon,
                            "target": target,
                        }
                    )
                if not resolved_actions:
                    continue

                base_styles = definition.get("styles") if isinstance(definition.get("styles"), dict) else {}
                state_style_override = (
                    design_styles.get(state_name)
                    if isinstance(design_styles.get(state_name), dict)
                    else {}
                )
                merged_styles = dict(base_styles)
                if state_style_override:
                    merged_styles["button_border"] = state_style_override.get(
                        "border",
                        merged_styles.get("button_border"),
                    )
                    if state_name == "support":
                        merged_styles["support_background"] = state_style_override.get(
                            "background",
                            merged_styles.get("support_background"),
                        )
                        merged_styles["support_text"] = state_style_override.get(
                            "text",
                            merged_styles.get("support_text"),
                        )
                        merged_styles["support_muted_text"] = state_style_override.get(
                            "muted_text",
                            merged_styles.get("support_muted_text"),
                        )
                        merged_styles["support_button_background"] = state_style_override.get(
                            "button_background",
                            merged_styles.get("support_button_background"),
                        )
                        merged_styles["support_button_text"] = state_style_override.get(
                            "button_text",
                            merged_styles.get("support_button_text"),
                        )
                        merged_styles["support_accent"] = state_style_override.get(
                            "accent",
                            merged_styles.get("support_accent"),
                        )
                    else:
                        merged_styles["background"] = state_style_override.get(
                            "background",
                            merged_styles.get("background"),
                        )
                        merged_styles["text"] = state_style_override.get(
                            "text",
                            merged_styles.get("text"),
                        )
                        merged_styles["muted_text"] = state_style_override.get(
                            "muted_text",
                            merged_styles.get("muted_text"),
                        )
                        merged_styles["button_background"] = state_style_override.get(
                            "button_background",
                            merged_styles.get("button_background"),
                        )
                        merged_styles["button_text"] = state_style_override.get(
                            "button_text",
                            merged_styles.get("button_text"),
                        )
                        merged_styles["accent"] = state_style_override.get(
                            "accent",
                            merged_styles.get("accent"),
                        )
                variant_key = f"{state_name}_{lang}"
                variants[variant_key] = {
                    "state": state_name,
                    "lang": lang,
                    "chat_bar_text": str(definition.get("chat_bar_text") or "").strip() or "Quick actions",
                    "styles": merged_styles,
                    "size": definition.get("size") if isinstance(definition.get("size"), dict) else {},
                    "actions": resolved_actions,
                }
        return {
            "bot_id": bot.bot_id,
            "bot_name": (getattr(bot, "display_name", None) or "Bot").strip() or "Bot",
            "base_lang": capabilities["bot_lang"],
            "theme_color": capabilities["theme_color"],
            "render_version": _RICH_MENU_RENDER_VERSION,
            "variants": variants,
        }

    def _hash_spec(self, spec: Dict[str, Any]) -> str:
        payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _variant_to_rich_menu_object(self, bot_name: str, variant_key: str, variant: Dict[str, Any]) -> Dict[str, Any]:
        size_cfg = variant.get("size") if isinstance(variant.get("size"), dict) else {}
        width = int(size_cfg.get("width") or 2500)
        height = int(size_cfg.get("height") or 843)
        actions = variant["actions"]
        action_count = max(1, len(actions))
        segment_width = width // action_count
        areas = []
        current_x = 0
        for idx, action in enumerate(actions):
            action_width = width - current_x if idx == action_count - 1 else segment_width
            target = action["target"]
            rich_action: Dict[str, Any] = {"type": target["type"], "label": action["label"][:20]}
            if target["type"] == "uri":
                rich_action["uri"] = target["uri"]
            else:
                rich_action["data"] = target["data"]
            areas.append(
                {
                    "bounds": {"x": current_x, "y": 0, "width": action_width, "height": height},
                    "action": rich_action,
                }
            )
            current_x += action_width
        return {
            "size": {"width": width, "height": height},
            "selected": False,
            "name": f"{bot_name} {variant_key}"[:300],
            "chatBarText": variant["chat_bar_text"][:14],
            "areas": areas,
        }

    def _draw_icon(self, draw: ImageDraw.ImageDraw, *, icon: str, center: Tuple[int, int], size: int, color: str) -> None:
        cx, cy = center
        half = size // 2
        if icon == "reserve":
            draw.rounded_rectangle((cx - half, cy - half, cx + half, cy + half), radius=18, outline=color, width=8)
            draw.line((cx - half, cy - half + 28, cx + half, cy - half + 28), fill=color, width=8)
            draw.line((cx - half + 36, cy - half - 6, cx - half + 36, cy - half + 22), fill=color, width=8)
            draw.line((cx + half - 36, cy - half - 6, cx + half - 36, cy - half + 22), fill=color, width=8)
        elif icon == "menu":
            for offset in (-28, 0, 28):
                draw.rounded_rectangle(
                    (cx - half, cy + offset - 10, cx + half, cy + offset + 10),
                    radius=10,
                    fill=color,
                )
        elif icon == "support":
            draw.ellipse((cx - half, cy - half, cx + half, cy + half), outline=color, width=8)
            draw.arc((cx - half + 24, cy - half + 24, cx + half - 24, cy + half - 24), 200, 340, fill=color, width=8)
            draw.line((cx - half + 10, cy + half - 26, cx - half + 56, cy + half - 26), fill=color, width=8)
            draw.line((cx + half - 10, cy + half - 26, cx + half - 56, cy + half - 26), fill=color, width=8)
        else:
            draw.rounded_rectangle((cx - half, cy - half, cx + half, cy + half), radius=18, outline=color, width=8)
            draw.polygon(
                [
                    (cx + half - 24, cy + half - 12),
                    (cx + half + 18, cy + half + 18),
                    (cx + half - 10, cy + half - 26),
                ],
                fill=color,
            )
            draw.arc((cx - half + 22, cy - half + 22, cx + half - 22, cy + half - 22), 40, 260, fill=color, width=8)

    def _truncate_to_width(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if _measure_text(draw, raw, font) <= max_width:
            return raw
        ellipsis = "..."
        current = raw
        while current:
            candidate = f"{current}{ellipsis}"
            if _measure_text(draw, candidate, font) <= max_width:
                return candidate
            current = current[:-1]
        return ellipsis

    def _wrap_label(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        separator = " " if " " in raw else ""
        tokens = raw.split() if separator else list(raw)
        if not tokens:
            return []
        lines: List[str] = []
        current = tokens[0]
        for token in tokens[1:]:
            candidate = f"{current}{separator}{token}" if separator else f"{current}{token}"
            if _measure_text(draw, candidate, font) <= max_width:
                current = candidate
                continue
            lines.append(current)
            current = str(token)
            if len(lines) == 1 and _measure_text(draw, current, font) > max_width:
                current = self._truncate_to_width(draw, current, font, max_width)
                break
            if len(lines) >= 2:
                break
        if len(lines) < 2 and current:
            lines.append(current)
        if len(lines) > 2:
            lines = lines[:2]
        if lines:
            lines[-1] = self._truncate_to_width(draw, lines[-1], font, max_width)
        return [line for line in lines if line]

    def _fit_label_font(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        text: str,
        lang: str,
        max_width: int,
        max_height: int,
    ) -> Tuple[ImageFont.ImageFont, List[str]]:
        for size in (48, 44, 40, 36, 32, 28):
            font = _load_font(size, lang=lang)
            lines = self._wrap_label(draw, text, font, max_width)
            if not lines:
                continue
            line_height = size + 10
            if len(lines) * line_height <= max_height:
                return font, lines
        font = _load_font(28, lang=lang)
        return font, self._wrap_label(draw, text, font, max_width)

    def _render_variant_image(
        self,
        *,
        bot_name: str,
        variant: Dict[str, Any],
        theme_color: str,
    ) -> bytes:
        width = int((variant.get("size") or {}).get("width") or 2500)
        height = int((variant.get("size") or {}).get("height") or 843)
        state_name = variant["state"]
        styles = variant.get("styles") if isinstance(variant.get("styles"), dict) else {}
        background = _normalize_color(
            styles.get("support_background" if state_name == "support" else "background"),
            "#0f172a" if state_name == "support" else "#f5f7fb",
        )
        foreground = _normalize_color(
            styles.get("support_text" if state_name == "support" else "text"),
            "#f8fafc" if state_name == "support" else "#0f172a",
        )
        muted = _normalize_color(
            styles.get("support_muted_text" if state_name == "support" else "muted_text"),
            "#cbd5e1" if state_name == "support" else "#475569",
        )
        button_fill = _normalize_color(
            styles.get("support_button_background" if state_name == "support" else "button_background"),
            "#fef3c7" if state_name == "support" else "#ffffff",
        )
        button_text = _normalize_color(
            styles.get("support_button_text" if state_name == "support" else "button_text"),
            "#92400e" if state_name == "support" else "#0f172a",
        )
        border_color = _normalize_color(styles.get("button_border"), "#d7dde7")
        accent = _normalize_color(
            styles.get("support_accent" if state_name == "support" else "accent"),
            "#f59e0b" if state_name == "support" else theme_color,
        )
        image = Image.new("RGBA", (width, height), background)
        draw = ImageDraw.Draw(image)
        variant_lang = normalize_lang(variant.get("lang") or "en")
        title_font = _load_font(54, lang=variant_lang)
        subtitle_font = _load_font(26, lang=variant_lang)

        title = bot_name[:40]
        subtitle = str(variant.get("chat_bar_text") or "").strip() or bot_name[:24]
        draw.text((110, 86), title, font=title_font, fill=foreground)
        draw.text((110, 156), subtitle, font=subtitle_font, fill=muted)

        actions = variant["actions"]
        action_count = max(1, len(actions))
        outer_margin_x = 90
        gap = 40
        top = 250
        bottom = height - 90
        available_width = width - (outer_margin_x * 2) - (gap * (action_count - 1))
        card_width = available_width // action_count if action_count else available_width
        card_height = bottom - top
        current_x = outer_margin_x
        for idx, action in enumerate(actions):
            next_width = available_width - ((card_width + gap) * idx) if idx == action_count - 1 else card_width
            left = current_x
            right = current_x + next_width
            draw.rounded_rectangle((left, top, right, bottom), radius=44, fill=button_fill, outline=border_color, width=4)
            icon_center = ((left + right) // 2, top + 132)
            self._draw_icon(draw, icon=action.get("icon") or action["id"], center=icon_center, size=116, color=accent)
            label_box_top = bottom - 164
            label_box_bottom = bottom - 40
            label_box_left = left + 30
            label_box_right = right - 30
            draw.rounded_rectangle(
                (label_box_left, label_box_top, label_box_right, label_box_bottom),
                radius=28,
                fill=button_fill,
                outline=border_color,
                width=3,
            )
            label_font, lines = self._fit_label_font(
                draw,
                text=action["label"],
                lang=variant_lang,
                max_width=max(120, next_width - 140),
                max_height=label_box_bottom - label_box_top - 24,
            )
            line_height = getattr(label_font, "size", 28) + 8
            text_block_height = len(lines) * line_height
            text_y = label_box_top + ((label_box_bottom - label_box_top - text_block_height) / 2)
            for line in lines:
                line_width = _measure_text(draw, line, label_font)
                draw.text((left + (next_width - line_width) / 2, text_y), line, font=label_font, fill=button_text)
                text_y += line_height
            current_x = right + gap

        out = io.BytesIO()
        image.convert("RGB").save(out, format="PNG")
        return out.getvalue()

    async def sync_for_bot(self, bot_id: str, *, force: bool = False):
        bid = (bot_id or "").strip()
        if not bid:
            return None
        async with self._lock_for(bid):
            bot = self._bot_repo.get_bot_record(bid)
            channel = self._line_channel_repo.get_by_bot_id(bid)
            existing_state = self._state_repo.get_by_bot_id(bid)
            if not bot or not channel:
                return existing_state
            if not channel.is_active:
                return self._state_repo.upsert(
                    channel_id=channel.channel_id,
                    bot_id=bid,
                    config_hash=existing_state.config_hash if existing_state else None,
                    default_variant=existing_state.default_variant if existing_state else None,
                    rich_menu_variants=existing_state.rich_menu_variants if existing_state else {},
                    sync_status="inactive",
                    last_synced_at=existing_state.last_synced_at if existing_state else None,
                    last_error=None,
                )

            spec = self._build_variant_specs(bot)
            config_hash = self._hash_spec(spec)
            variants = spec["variants"]
            base_lang = spec["base_lang"]
            default_variant = f"normal_{base_lang}" if f"normal_{base_lang}" in variants else next(iter(variants.keys()), None)
            previous_variant_ids = dict(existing_state.rich_menu_variants or {}) if existing_state else {}

            if not variants:
                completed_at = _utc_now()
                try:
                    await clear_default_rich_menu(channel.line_channel_access_token)
                except Exception:
                    pass
                for rich_menu_id in previous_variant_ids.values():
                    if not rich_menu_id:
                        continue
                    try:
                        await delete_rich_menu(rich_menu_id, channel.line_channel_access_token)
                    except Exception:
                        pass
                return self._state_repo.upsert(
                    channel_id=channel.channel_id,
                    bot_id=bid,
                    config_hash=config_hash,
                    default_variant=None,
                    rich_menu_variants={},
                    sync_status="no_actions",
                    last_synced_at=completed_at,
                    last_error=None,
                )

            if (
                not force
                and existing_state
                and existing_state.config_hash == config_hash
                and existing_state.sync_status == "synced"
                and all(existing_state.rich_menu_variants.get(key) for key in variants.keys())
            ):
                return existing_state

            self._state_repo.upsert(
                channel_id=channel.channel_id,
                bot_id=bid,
                config_hash=config_hash,
                default_variant=default_variant,
                rich_menu_variants=existing_state.rich_menu_variants if existing_state else {},
                sync_status="syncing",
                last_synced_at=existing_state.last_synced_at if existing_state else None,
                last_error=None,
            )

            created_variants: Dict[str, str] = {}
            try:
                bot_name = spec["bot_name"]
                for variant_key, variant_spec in variants.items():
                    rich_menu_object = self._variant_to_rich_menu_object(bot_name, variant_key, variant_spec)
                    rich_menu_id = await create_rich_menu(rich_menu_object, channel.line_channel_access_token)
                    image_bytes = self._render_variant_image(
                        bot_name=bot_name,
                        variant=variant_spec,
                        theme_color=spec["theme_color"],
                    )
                    await upload_rich_menu_image(rich_menu_id, image_bytes, channel.line_channel_access_token)
                    created_variants[variant_key] = rich_menu_id
                if default_variant and created_variants.get(default_variant):
                    await set_default_rich_menu(created_variants[default_variant], channel.line_channel_access_token)
                for variant_key, rich_menu_id in previous_variant_ids.items():
                    if not rich_menu_id:
                        continue
                    if created_variants.get(variant_key) == rich_menu_id:
                        continue
                    try:
                        await delete_rich_menu(rich_menu_id, channel.line_channel_access_token)
                    except Exception:
                        pass
                completed_at = _utc_now()
                return self._state_repo.upsert(
                    channel_id=channel.channel_id,
                    bot_id=bid,
                    config_hash=config_hash,
                    default_variant=default_variant,
                    rich_menu_variants=created_variants,
                    sync_status="synced",
                    last_synced_at=completed_at,
                    last_error=None,
                )
            except Exception as exc:
                for rich_menu_id in created_variants.values():
                    if not rich_menu_id:
                        continue
                    try:
                        await delete_rich_menu(rich_menu_id, channel.line_channel_access_token)
                    except Exception:
                        pass
                return self._state_repo.upsert(
                    channel_id=channel.channel_id,
                    bot_id=bid,
                    config_hash=config_hash,
                    default_variant=existing_state.default_variant if existing_state else default_variant,
                    rich_menu_variants=existing_state.rich_menu_variants if existing_state else {},
                    sync_status="error",
                    last_synced_at=existing_state.last_synced_at if existing_state else None,
                    last_error=str(exc),
                )

    async def ensure_user_menu(
        self,
        *,
        bot_id: str,
        line_user_id: str,
        lang: str,
        assistant_state: str,
    ) -> Optional[str]:
        bid = (bot_id or "").strip()
        uid = (line_user_id or "").strip()
        if not bid or not uid:
            return None
        state = await self.sync_for_bot(bid, force=False)
        channel = self._line_channel_repo.get_by_bot_id(bid)
        if not state or not channel or not channel.is_active:
            return None
        normalized_lang = normalize_lang(lang)
        target_state = "support" if assistant_state in {"awaiting_support_details", "human_handoff"} else "normal"
        target_variant = f"{target_state}_{normalized_lang}"
        contact = self._conversation_repo.get_channel_contact(
            bot_id=bid,
            channel="line",
            external_user_id=uid,
        )
        metadata = dict(contact.metadata or {}) if contact else {}
        if state.sync_status == "no_actions" or target_variant not in state.rich_menu_variants:
            if metadata.get("line_rich_menu_variant"):
                try:
                    await unlink_user_rich_menu(uid, channel.line_channel_access_token)
                except Exception:
                    pass
            self._conversation_repo.upsert_channel_contact(
                bot_id=bid,
                channel="line",
                external_user_id=uid,
                metadata={
                    "preferred_lang": normalized_lang,
                    "line_rich_menu_variant": "",
                    "line_rich_menu_hash": state.config_hash or "",
                },
            )
            return None

        current_variant = str(metadata.get("line_rich_menu_variant") or "").strip()
        current_hash = str(metadata.get("line_rich_menu_hash") or "").strip()
        if current_variant == target_variant and current_hash == str(state.config_hash or ""):
            if metadata.get("preferred_lang") != normalized_lang:
                self._conversation_repo.upsert_channel_contact(
                    bot_id=bid,
                    channel="line",
                    external_user_id=uid,
                    metadata={"preferred_lang": normalized_lang},
                )
            return target_variant

        await link_user_rich_menu(uid, state.rich_menu_variants[target_variant], channel.line_channel_access_token)
        self._conversation_repo.upsert_channel_contact(
            bot_id=bid,
            channel="line",
            external_user_id=uid,
            metadata={
                "preferred_lang": normalized_lang,
                "line_rich_menu_variant": target_variant,
                "line_rich_menu_hash": state.config_hash or "",
            },
        )
        return target_variant

    async def cleanup_for_bot(self, bot_id: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            return
        channel = self._line_channel_repo.get_by_bot_id(bid)
        state = self._state_repo.get_by_bot_id(bid)
        if channel and state:
            try:
                await clear_default_rich_menu(channel.line_channel_access_token)
            except Exception:
                pass
            for rich_menu_id in (state.rich_menu_variants or {}).values():
                if not rich_menu_id:
                    continue
                try:
                    await delete_rich_menu(rich_menu_id, channel.line_channel_access_token)
                except Exception:
                    pass
        self._state_repo.delete_by_bot_id(bid)
