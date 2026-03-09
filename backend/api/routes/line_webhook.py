"""LINE Messaging API webhook handler and LINE channel config CRUD.

Endpoints:
  POST /webhooks/line/{bot_id}           -- LINE webhook (public, signature-verified)
  GET  /v1/org/bots/{bot_id}/line-channel -- Get LINE config (authenticated)
  PUT  /v1/org/bots/{bot_id}/line-channel -- Create/update LINE config (authenticated)
  DELETE /v1/org/bots/{bot_id}/line-channel -- Remove LINE integration (authenticated)
"""

import json
import logging
import os
import re
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps.auth import get_current_user
from api.schemas import (
    LineChannelUpsertRequest,
    LineChannelResponse,
    LineChannelDeleteResponse,
    LineChannelTestResponse,
)
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from application.services.answer_normalization_service import (
    RENDER_TARGET_PLAIN_TEXT_CHANNEL,
    build_answer_link_candidates,
    normalize_answer_links,
)
from application.services.asset_policy import ConfigAssetPolicy
from application.services.channel_adapters import LineChannelAdapter
from application.services.function_registry import FunctionRegistry
from application.services.prompt_provider import ConfigPromptProvider
from application.services.platform_strategy import (
    build_menu_view_all_url_for_category,
    normalize_menu_category,
)
from common.language_utils import detect_user_language, normalize_lang
from common.di.container import asset_repo, bot_service, conversation_service, line_rich_menu_service
from infrastructure.clients.line_client import (
    verify_signature,
    reply_message,
    reply_with_messages,
    build_suggested_flex,
    build_buttons_template,
    build_text_with_quick_replies,
    get_profile as get_line_profile,
    show_typing as line_show_typing,
)
from infrastructure.clients.rag_client import run_vertex_rag, is_quota_exhausted_error
from domain.platform_profiles import (
    get_asset_rules_from_widget,
    get_line_menu_page_payload_prefix,
    get_line_menu_quick_payload,
    get_menu_category_order,
    get_menu_keywords,
    get_menu_request_pattern,
    get_menu_texts,
    get_platform_asset_instructions,
    get_platform_features_from_widget,
    get_platform_json_response_enabled,
    get_platform_json_response_instruction,
    get_line_cancel_keywords,
    get_line_support_messages,
    get_reservation_config_from_widget,
    get_suggested_messages_for_widget,
)
from infrastructure.assets.asset_resolver import (
    build_asset_instruction,
)
from infrastructure.db.repositories import (
    PostgresLineChannelRepository,
    PostgresLineUserSessionRepository,
)
from infrastructure.services.indexing_service import ensure_bot_corpus

logger = logging.getLogger(__name__)

router = APIRouter()

_line_channel_repo = PostgresLineChannelRepository()
_line_user_session_repo = PostgresLineUserSessionRepository()
_line_adapter = LineChannelAdapter()
_prompt_provider = ConfigPromptProvider()
_asset_policy = ConfigAssetPolicy()
_function_registry = FunctionRegistry()

# ── Rate limiting (mirrors saas.py pattern) ──────────────────────────

_rl_state: Dict[str, tuple] = {}
_rl_window_s = 60
_rl_max_per_window = 120  # higher limit for LINE (one user = many messages)


def _rate_limit(bot_id: str) -> None:
    now = time.time()
    start, count = _rl_state.get(bot_id, (now, 0))
    if now - start > _rl_window_s:
        start, count = now, 0
    count += 1
    _rl_state[bot_id] = (start, count)
    if count > _rl_max_per_window:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


_LINE_EMPTY_ANSWER_MESSAGES = {
    "en": "I'm sorry, I couldn't find an answer to that. Could you try rephrasing?",
    "ja": "申し訳ありません。その質問への答えが見つかりませんでした。言い方を変えてもう一度お試しください。",
}
_LINE_QUOTA_MESSAGES = {
    "en": "We're experiencing high demand right now. Please try again in about a minute.",
    "ja": "現在アクセスが集中しています。1分ほど待ってからもう一度お試しください。",
}
_LINE_ERROR_MESSAGES = {
    "en": "I'm sorry, something went wrong. Please try again in a moment.",
    "ja": "申し訳ありません。問題が発生しました。少し待ってからもう一度お試しください。",
}

# Cancel escalation (config-driven)
_CANCEL_KEYWORDS = {kw.lower() for kw in get_line_cancel_keywords()}


def _is_escalate_quick_reply(text: str, suggested_messages: list) -> bool:
    """True if text matches an escalate-type suggested message label or prompt."""
    if not text or not suggested_messages:
        return False
    t = (text or "").strip()
    for sm in suggested_messages:
        if str(sm.get("type") or "").strip() != "escalate":
            continue
        label = (sm.get("label") or "").strip()
        prompt = (sm.get("prompt") or "").strip()
        if t == label or t == prompt:
            return True
    return False


def _resolve_suggested_type(text: str, suggested_messages: list) -> Optional[str]:
    if not text or not suggested_messages:
        return None
    t = (text or "").strip()
    for sm in suggested_messages:
        if not isinstance(sm, dict):
            continue
        label = (sm.get("label") or "").strip()
        prompt = (sm.get("prompt") or "").strip()
        if t == label or t == prompt:
            raw_type = str(sm.get("type") or "").strip()
            return raw_type or None
    return None


def _wants_cancel_escalation(text: str) -> bool:
    """True if user wants to cancel escalation and stay with AI (matches Instagram)."""
    lower = (text or "").strip().lower()
    return any(lower == kw or lower.startswith(kw) for kw in _CANCEL_KEYWORDS)


def _line_localized_message(messages: Dict[str, str], lang: str) -> str:
    normalized_lang = normalize_lang(lang)
    return str(messages.get(normalized_lang) or messages["en"])


def _parse_line_postback_action(postback_data: Optional[str]) -> Optional[str]:
    raw = str(postback_data or "").strip().lower()
    if not raw:
        return None
    if raw == "lineux:support":
        return "support"
    if raw == "lineux:back_to_ai":
        return "back_to_ai"
    if raw == "lineux:menu":
        return "menu"
    return None


def _schedule_line_rich_menu_update(*, bot_id: str, line_user_id: str, lang: str, assistant_state: str) -> None:
    async def _runner() -> None:
        try:
            await line_rich_menu_service().ensure_user_menu(
                bot_id=bot_id,
                line_user_id=line_user_id,
                lang=lang,
                assistant_state=assistant_state,
            )
        except Exception:
            logger.exception("LINE rich menu user sync failed bot_id=%s user_id=%s", bot_id, line_user_id)

    asyncio.create_task(_runner())


def _build_line_channel_response(channel, *, rich_menu_state=None) -> LineChannelResponse:
    state = rich_menu_state or line_rich_menu_service().get_state(channel.bot_id)
    return LineChannelResponse(
        channel_id=channel.channel_id,
        bot_id=channel.bot_id,
        org_id=channel.org_id,
        line_channel_id=channel.line_channel_id,
        is_active=channel.is_active,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
        managed_rich_menu_enabled=True,
        rich_menu_sync_status=getattr(state, "sync_status", None),
        rich_menu_last_synced_at=getattr(state, "last_synced_at", None),
        rich_menu_last_error=getattr(state, "last_error", None),
        rich_menu_variants=dict(getattr(state, "rich_menu_variants", None) or {}),
    )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


# ── Menu flow (matches Instagram) ───────────────────────────────────

def _safe_int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or str(default)).strip())
    except (TypeError, ValueError):
        return default


_LINE_MENU_PAGE_SIZE = max(1, min(_safe_int_env("LINE_MENU_PAGE_SIZE", 500), 500))


def _get_menu_request_exact() -> set:
    return {k.strip().lower() for k in get_menu_keywords() if k.strip()}


def _get_menu_request_pattern_compiled():
    pat = get_menu_request_pattern()
    return re.compile(pat, re.IGNORECASE) if pat else None
_PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL")
    or os.environ.get("BACKEND_PUBLIC_BASE_URL")
    or os.environ.get("API_BASE_URL")
    or os.environ.get("EXTERNAL_BASE_URL")
    or os.environ.get("PUBLIC_API_BASE_URL")
    or ""
).strip()


def _get_menu_items_for_bot(bot_id: str) -> List[Any]:
    try:
        return asset_repo().list_assets_for_bot(bot_id, active_only=True, asset_type="menu_item")
    except Exception:
        logger.exception("Failed to load menu items for bot_id=%s", bot_id)
        return []


def _menu_category_for_item(item: Any, *, widget_config: Optional[Dict[str, Any]] = None) -> str:
    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
    raw = str(metadata.get("category") or "").strip().lower()
    category = normalize_menu_category(raw, widget_config=widget_config)
    order = get_menu_category_order()
    default_category = order[-1] if order else category
    return category if category in order else default_category


def _is_full_menu_request(text: str, payload: Optional[str]) -> bool:
    # LINE sends text directly (no separate payload); quick reply taps send "SHOW_FULL_MENU"
    raw_text = (text or "").strip().upper()
    if raw_text == get_line_menu_quick_payload():
        return True
    raw_payload = (payload or "").strip().upper()
    if raw_payload == get_line_menu_quick_payload():
        return True
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if normalized in _get_menu_request_exact():
        return True
    pat = _get_menu_request_pattern_compiled()
    return bool(pat and pat.match(normalized))


def _parse_menu_page_payload(payload: Optional[str]) -> Optional[Tuple[str, int]]:
    raw = (payload or "").strip()
    if not raw.upper().startswith(get_line_menu_page_payload_prefix()):
        return None
    rest = raw[len(get_line_menu_page_payload_prefix()):]
    parts = rest.split(":", 1)
    if len(parts) != 2:
        return None
    category = parts[0].strip().lower()
    if category not in get_menu_category_order():
        return None
    try:
        offset = int(parts[1].strip())
    except ValueError:
        return None
    return (category, offset) if offset >= 0 else None


def _menu_text(key: str, lang: str, widget_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
    table = get_menu_texts(lang, widget_config=widget_config)
    template = str(table.get(key) or "").strip()
    return template.format(**kwargs) if template and kwargs else template


def _menu_category_label(category: str, lang: str, widget_config: Optional[Dict[str, Any]] = None) -> str:
    table = get_menu_texts(lang, widget_config=widget_config)
    labels = table.get("category_labels") or {}
    return str(labels.get(category) or category.title())


def _view_more_title(category: str, lang: str, widget_config: Optional[Dict[str, Any]] = None) -> str:
    table = get_menu_texts(lang, widget_config=widget_config)
    titles = table.get("view_more_titles") or {}
    return str(titles.get(category) or ("もっと見る" if lang == "ja" else "View more"))


def _menu_price_and_details(item: Any) -> Tuple[str, str]:
    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
    price_text = str(metadata.get("price_text") or "").strip()
    if not price_text and isinstance(metadata.get("price"), dict):
        price_text = str((metadata.get("price") or {}).get("text") or "").strip()
    details = str(metadata.get("details") or (item.description or "")).strip()
    return price_text, details


def _menu_item_source_url(item: Any) -> str:
    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
    u = str(metadata.get("source_url") or getattr(item, "link_url", "") or "").strip()
    return u if u.startswith("http") else ""


def _menu_view_all_url_for_category(
    category: str,
    items: List[Any],
    *,
    widget_config: Optional[Dict[str, Any]] = None,
) -> str:
    return build_menu_view_all_url_for_category(category, items, widget_config=widget_config)


def _resolve_public_base_url(request: Optional[Request] = None) -> str:
    if _PUBLIC_BASE_URL:
        return _PUBLIC_BASE_URL.rstrip("/") if _PUBLIC_BASE_URL.startswith(("http://", "https://")) else f"https://{_PUBLIC_BASE_URL}".rstrip("/")
    if not request:
        return ""
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip()
    proto = (request.headers.get("x-forwarded-proto") or "https").split(",")[0].strip()
    return f"{proto}://{host}".rstrip("/") if host else ""


def _menu_page_url(publishable_key: str, request: Optional[Request] = None) -> str:
    base = _resolve_public_base_url(request)
    return f"{base}/v1/pk/{publishable_key}/menu" if base and publishable_key else ""


def _build_line_text_chunks(lines: List[str], max_chars: int = 4500) -> List[str]:
    chunks, current = [], ""
    for line in (str(l or "") for l in lines):
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), max_chars):
                c = line[i:i + max_chars]
                if c:
                    chunks.append(c)
            continue
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _make_menu_page_payload(category: str, offset: int) -> str:
    return f"{get_line_menu_page_payload_prefix()}{category}:{offset}"


async def _send_menu_by_category_line(
    *,
    bot_id: str,
    line_user_id: str,
    reply_token: str,
    access_token: str,
    suggested_flex: Optional[dict],
    page_category: Optional[str] = None,
    page_offset: int = 0,
    lang: str = "en",
    publishable_key: Optional[str] = None,
    request: Optional[Request] = None,
    widget_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Send menu by category (matches Instagram flow, adapted for LINE)."""
    items = _get_menu_items_for_bot(bot_id)
    logger.info("LINE menu flow bot_id=%s items=%d page_category=%s", bot_id, len(items), page_category)
    if not items:
        await reply_message(
            reply_token,
            [_menu_text("menu_not_ready", lang)],
            access_token,
            suggested_flex=suggested_flex,
        )
        return

    menu_page_url = _menu_page_url(publishable_key, request) if publishable_key else None
    is_initial_menu = page_category is None and page_offset == 0
    if publishable_key and (not menu_page_url or not menu_page_url.startswith("https://")):
        logger.info(
            "LINE menu: menu_page_url empty or invalid (set PUBLIC_BASE_URL?). bot_id=%s pk=%s url=%s",
            bot_id, publishable_key, menu_page_url or "(empty)",
        )

    # Button to view full menu in-app (matches Instagram) — send button ONLY, never text list
    if menu_page_url and is_initial_menu and menu_page_url.lower().startswith("https://") and len(menu_page_url) > 12:
        prompt = _menu_text("view_menu_button_prompt", lang)
        btn_title = _menu_text("view_menu_button_title", lang)
        tmpl = build_buttons_template(
            "View full menu",
            prompt,
            [{"type": "uri", "label": btn_title[:20], "uri": menu_page_url}],
        )
        # Send button only (no suggested_flex — combo caused LINE 400)
        ok = await reply_with_messages(reply_token, [tmpl], access_token)
        if ok:
            return
        # Fallback: plain text with clickable URL (LINE renders URLs as links)
        fallback = f"{prompt}\n\n{menu_page_url}"
        ok2 = await reply_message(reply_token, [fallback], access_token, suggested_flex=suggested_flex)
        if ok2:
            return
        logger.warning("LINE menu button and fallback failed bot_id=%s", bot_id)
        raise RuntimeError("LINE menu send failed")

    order = get_menu_category_order()
    grouped: Dict[str, List[Any]] = {k: [] for k in order}
    for item in items:
        grouped[_menu_category_for_item(item, widget_config=widget_config)].append(item)

    categories = [page_category] if page_category and page_category in order else list(order)
    pending_more_qr: List[dict] = []
    text_parts: List[str] = []

    for category in categories:
        category_items = grouped.get(category) or []
        if not category_items:
            continue
        label = _menu_category_label(category, lang)
        start = page_offset if page_category else 0
        page_items = category_items[start:start + _LINE_MENU_PAGE_SIZE]
        if not page_items:
            continue

        view_url = menu_page_url or _menu_view_all_url_for_category(
            category,
            category_items,
            widget_config=widget_config,
        )
        text_parts.append(label)
        if view_url:
            text_parts.append(_menu_text("view_full_menu", lang, url=view_url))
        for idx, item in enumerate(page_items, start=start + 1):
            name = str(item.name or "Menu item").strip()
            price_text, _ = _menu_price_and_details(item)
            text_parts.append(f"{idx}. {name}" + (f" - {price_text}" if price_text else ""))

        next_offset = start + _LINE_MENU_PAGE_SIZE
        if next_offset < len(category_items):
            payload = _make_menu_page_payload(category, next_offset)
            pending_more_qr.append({"label": _view_more_title(category, lang), "text": payload})
        if page_category:
            break

    if not text_parts:
        await reply_message(
            reply_token,
            [_menu_text("render_error", lang)],
            access_token,
            suggested_flex=suggested_flex,
        )
        return

    chunks = _build_line_text_chunks(text_parts)
    if not chunks:
        await reply_message(
            reply_token,
            [_menu_text("render_error", lang)],
            access_token,
            suggested_flex=suggested_flex,
        )
        return

    messages: List[dict] = []
    for i, chunk in enumerate(chunks[:-1]):
        messages.append({"type": "text", "text": chunk})
    last_chunk = chunks[-1]
    if pending_more_qr:
        msg_with_qr = build_text_with_quick_replies(last_chunk, pending_more_qr)
        messages.append(msg_with_qr)
        prompt = _menu_text("more_items_prompt", lang) if len(pending_more_qr) > 1 else _menu_text("tap_view_more", lang)
        messages.append({"type": "text", "text": prompt})
    else:
        messages.append({"type": "text", "text": last_chunk})
    if suggested_flex and len(messages) < 5:
        messages.append(suggested_flex)
    ok = await reply_with_messages(reply_token, messages[:5], access_token)
    if not ok:
        logger.warning("LINE menu text list send failed bot_id=%s, falling back to RAG", bot_id)
        raise RuntimeError("LINE menu text list send failed (400 or network error)")


# ── Helper: format conversation context ──────────────────────────────

_CONVERSATION_CONTEXT_MAX_CHARS = 2000


def _format_conversation_context(messages: list) -> str:
    lines = []
    for m in messages:
        content = (m.content or "").strip()
        if len(content) > _CONVERSATION_CONTEXT_MAX_CHARS:
            content = content[:_CONVERSATION_CONTEXT_MAX_CHARS] + "..."
        role_label = "Assistant" if (m.role or "").lower() == "bot" else "User"
        lines.append(f"{role_label}: {content}")
    return "\n\n".join(lines) if lines else ""


# ── Helper: resolve org for auth (mirrors saas.py pattern) ───────────

def _resolve_org_id(user_ctx, org_id: Optional[str] = None) -> str:
    from application.auth.jwt_auth import is_super_admin
    if org_id:
        if org_id in user_ctx.org_ids or is_super_admin(user_ctx.claims):
            return org_id
        raise HTTPException(status_code=403, detail="Org membership required")
    if len(user_ctx.org_ids) == 1:
        return user_ctx.org_ids[0]
    if is_super_admin(user_ctx.claims):
        raise HTTPException(status_code=400, detail="org_id is required for admin")
    raise HTTPException(status_code=403, detail="Org membership required")


def _assert_bot_org(bot_id: str, org_id: str) -> None:
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    if bot.org_id != org_id:
        raise HTTPException(status_code=403, detail="Bot does not belong to this org")


# ══════════════════════════════════════════════════════════════════════
# LINE Webhook
# ══════════════════════════════════════════════════════════════════════


@router.post("/webhooks/line/{bot_id}")
async def line_webhook(bot_id: str, request: Request):
    """Receive and handle LINE webhook events."""
    # 1. Load LINE channel config
    channel = _line_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        logger.warning("LINE webhook 404: no channel for bot_id=%s", bot_id)
        raise HTTPException(status_code=404, detail="LINE channel not configured for this bot")
    if not channel.is_active:
        logger.warning("LINE webhook 404: channel inactive for bot_id=%s", bot_id)
        raise HTTPException(status_code=404, detail="LINE channel is inactive")

    # 2. Read raw body and verify signature
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")
    if not signature or not verify_signature(body, signature, channel.line_channel_secret):
        raise HTTPException(status_code=403, detail="Invalid LINE signature")

    # 3. Parse events
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    events = payload.get("events", [])

    # 4. Rate limit
    _rate_limit(bot_id)

    # 5. Load bot
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")

    # Process each event through the channel adapter.
    for event in events:
        parsed = _line_adapter.parse_event(event)
        if not parsed:
            continue

        await _handle_line_event(
            bot=bot,
            channel=channel,
            event=parsed,
            request=request,
        )

    return {"ok": True}


async def _handle_line_event(
    *,
    bot,
    channel,
    event,
    request: Request,
) -> None:
    """Core handler for a single LINE text or postback event."""
    line_user_id = event.user_id
    text = event.text
    reply_token = event.reply_token
    postback_action = _parse_line_postback_action(getattr(event, "postback_data", None))
    access_token = channel.line_channel_access_token

    # Load suggested messages and session BEFORE typing — we must not show typing when
    # user is escalated and we won't reply (avoids typing bubble stuck until timeout)
    suggested_flex = None
    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    bot_lang = normalize_lang(widget_config.get("language") or widget_config.get("botLanguage") or "en")
    existing_contact = conversation_service().get_channel_contact(
        bot_id=bot.bot_id,
        channel="line",
        external_user_id=line_user_id,
    )
    stored_lang = normalize_lang(
        (getattr(existing_contact, "metadata", None) or {}).get("preferred_lang") if existing_contact else bot_lang,
        fallback=bot_lang,
    )
    if event.event_type == "follow":
        lang = bot_lang
    elif event.event_type == "message":
        lang = detect_user_language(text, fallback=stored_lang)
    else:
        lang = stored_lang
    suggested_messages = get_suggested_messages_for_widget(widget_config, lang=lang)
    if suggested_messages:
        suggested_flex = build_suggested_flex(suggested_messages)
    suggested_type = _resolve_suggested_type(text, suggested_messages or []) if event.event_type == "message" else None
    suggested_function_id = _function_registry.resolve_from_suggested_type(suggested_type or "")

    # Mapping rows only keep the current routed session and display name.
    mapping = _line_user_session_repo.get(line_user_id=line_user_id, bot_id=bot.bot_id)
    line_display_name = (getattr(mapping, "display_name", None) or "").strip() or None
    if not line_display_name:
        try:
            profile = await get_line_profile(line_user_id, access_token)
        except Exception as exc:
            logger.debug("LINE profile lookup failed bot_id=%s user_id=%s err=%s", bot.bot_id, line_user_id, exc)
            profile = None
        fetched_name = str((profile or {}).get("displayName") or "").strip()
        if fetched_name:
            line_display_name = fetched_name
            _line_user_session_repo.set_display_name(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                display_name=line_display_name,
            )
    if event.event_type == "follow":
        if mapping and line_display_name and line_display_name != (getattr(mapping, "display_name", None) or "").strip():
            _line_user_session_repo.set_display_name(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                display_name=line_display_name,
            )
        conversation_service().upsert_channel_contact(
            bot_id=bot.bot_id,
            channel="line",
            external_user_id=line_user_id,
            current_session_id=getattr(mapping, "session_id", None),
            display_name=line_display_name,
            metadata={"preferred_lang": bot_lang},
        )
        _schedule_line_rich_menu_update(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            lang=bot_lang,
            assistant_state="bot",
        )
        return
    session, contact, _session_changed = conversation_service().resolve_or_create_channel_session(
        bot_id=bot.bot_id,
        org_id=bot.org_id,
        channel="line",
        external_user_id=line_user_id,
        current_session_id=getattr(mapping, "session_id", None),
        display_name=line_display_name,
        metadata={"preferred_lang": lang},
        site_url=None,
        site_title=None,
        user_agent="LINE",
        ip=None,
    )
    if mapping is None:
        mapping = _line_user_session_repo.get_or_create(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            session_id=session.session_id,
            display_name=line_display_name,
        )
    else:
        if mapping.session_id != session.session_id:
            _line_user_session_repo.update_session_id(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                session_id=session.session_id,
            )
        if line_display_name and line_display_name != (getattr(mapping, "display_name", None) or "").strip():
            _line_user_session_repo.set_display_name(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                display_name=line_display_name,
            )
        mapping = _line_user_session_repo.get(line_user_id=line_user_id, bot_id=bot.bot_id) or mapping
    if line_display_name:
        conversation_service().set_session_title(session.session_id, line_display_name)
    assistant_state = getattr(session, "assistant_state", "bot") or "bot"
    msgs = get_line_support_messages(lang=lang)
    page_req = _parse_menu_page_payload(text) if event.event_type == "message" else None
    is_menu_request_event = bool(
        postback_action == "menu"
        or page_req
        or (event.event_type == "message" and _is_full_menu_request(text, None))
    )

    # ── Escalated: log message only, no bot reply — return BEFORE typing ───
    if assistant_state == "human_handoff" and not (_wants_cancel_escalation(text) or postback_action in {"back_to_ai", "menu"}):
        if event.event_type == "message" and text:
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text,
                sender_name=line_display_name,
            )
        _schedule_line_rich_menu_update(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            lang=lang,
            assistant_state="human_handoff",
        )
        return

    # Show typing only when we will send a reply (avoids stuck typing when escalated)
    if event.event_type == "message" and postback_action not in {"support", "back_to_ai", "menu"} and not is_menu_request_event:
        await line_show_typing(line_user_id, access_token)

    # ── De-escalation check (cancel only, matches Instagram) ───────────
    if assistant_state == "human_handoff" and (_wants_cancel_escalation(text) or postback_action == "back_to_ai"):
        conversation_service().cancel_handoff(
            bot_id=bot.bot_id,
            session_id=session.session_id,
        )
        if event.event_type == "message" and text:
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text,
                sender_name=line_display_name,
            )
        cancel_msg = msgs["cancel_ack"]
        await reply_message(reply_token, [cancel_msg], access_token, suggested_flex=suggested_flex)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=cancel_msg,
        )
        _schedule_line_rich_menu_update(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            lang=lang,
            assistant_state="bot",
        )
        return

    # ── Awaiting optional escalation message ─────────────────────────
    if assistant_state == "awaiting_support_details":
        if _wants_cancel_escalation(text) or postback_action == "back_to_ai":
            conversation_service().release_handoff(
                bot_id=bot.bot_id,
                session_id=session.session_id,
                ended_reason="user_canceled",
            )
            if event.event_type == "message" and text:
                conversation_service().add_message(
                    session_id=session.session_id,
                    bot_id=bot.bot_id,
                    role="user",
                    content=text,
                    sender_name=line_display_name,
                )
            cancel_msg = msgs["cancel_ack"]
            await reply_message(reply_token, [cancel_msg], access_token, suggested_flex=suggested_flex)
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="bot",
                content=cancel_msg,
            )
            _schedule_line_rich_menu_update(
                bot_id=bot.bot_id,
                line_user_id=line_user_id,
                lang=lang,
                assistant_state="bot",
            )
            return
        if postback_action != "menu" and event.event_type != "message":
            _schedule_line_rich_menu_update(
                bot_id=bot.bot_id,
                line_user_id=line_user_id,
                lang=lang,
                assistant_state="awaiting_support_details",
            )
            return
        if postback_action == "menu":
            assistant_state = "awaiting_support_details"
        else:
            user_msg = text.strip()
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text,
                sender_name=line_display_name,
            )
            details = user_msg if user_msg else msgs["email_details_no_message"]
            conversation_service().request_support(
                bot_id=bot.bot_id,
                session_id=session.session_id,
                visitor_email=f"line:{line_user_id}",
                details=details,
                contact_id=getattr(contact, "contact_id", None),
                source_channel="line",
            )
            from infrastructure.email import maybe_send_escalation_email

            maybe_send_escalation_email(
                bot,
                session_id=session.session_id,
                channel="line",
                visitor_email=f"line:{line_user_id}",
                details=details,
            )
            ack_msg = msgs["escalation_ack"]
            await reply_message(reply_token, [ack_msg], access_token)
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="bot",
                content=ack_msg,
            )
            _schedule_line_rich_menu_update(
                bot_id=bot.bot_id,
                line_user_id=line_user_id,
                lang=lang,
                assistant_state="human_handoff",
            )
            return

    # ── Escalation request ───────────────────────────────────────────
    if postback_action == "support" or suggested_function_id == "escalate" or _is_escalate_quick_reply(text, suggested_messages or []):
        conversation_service().set_awaiting_support_details(
            bot_id=bot.bot_id,
            session_id=session.session_id,
            contact_id=getattr(contact, "contact_id", None),
            source_channel="line",
        )
        if event.event_type == "message" and text:
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text,
                sender_name=line_display_name,
            )
        prompt_msg = msgs["prompt"]
        await reply_message(reply_token, [prompt_msg], access_token)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=prompt_msg,
        )
        _schedule_line_rich_menu_update(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            lang=lang,
            assistant_state="awaiting_support_details",
        )
        return

    # ── Menu flow (matches Instagram) ─────────────────────────────────
    platform_features = get_platform_features_from_widget(widget_config)
    menu_extraction_enabled = platform_features.get("menu_extraction_enabled") if platform_features else False
    quick_payload = None  # LINE only has text; show_menu sends SHOW_FULL_MENU as text
    effective_text = get_line_menu_quick_payload() if postback_action == "menu" else text
    menu_fallback_skip_assets = False  # set True when menu flow fails and we fall back to RAG

    if page_req:
        category, offset = page_req
        if event.event_type == "message" and text:
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text or "View more menu",
                sender_name=line_display_name,
            )
        await _send_menu_by_category_line(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            reply_token=reply_token,
            access_token=access_token,
            suggested_flex=suggested_flex,
            page_category=category,
            page_offset=offset,
            lang=lang,
            publishable_key=getattr(bot, "publishable_key", None),
            request=request,
            widget_config=widget_config,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=f"Shared more {category} menu items.",
        )
        _schedule_line_rich_menu_update(
            bot_id=bot.bot_id,
            line_user_id=line_user_id,
            lang=lang,
            assistant_state=assistant_state,
        )
        return

    # Run menu flow when user explicitly requests menu (Menu/SHOW_FULL_MENU),
    # regardless of platform — ensures Menu button always responds
    if postback_action == "menu" or suggested_function_id == "show_menu" or _is_full_menu_request(effective_text, quick_payload):
        try:
            await _send_menu_by_category_line(
                bot_id=bot.bot_id,
                line_user_id=line_user_id,
                reply_token=reply_token,
                access_token=access_token,
                suggested_flex=suggested_flex,
                lang=lang,
                publishable_key=getattr(bot, "publishable_key", None),
                request=request,
                widget_config=widget_config,
            )
            if event.event_type == "message" and effective_text:
                conversation_service().add_message(
                    session_id=session.session_id,
                    bot_id=bot.bot_id,
                    role="user",
                    content=effective_text,
                    sender_name=line_display_name,
                )
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="bot",
                content="Shared the categorized menu.",
            )
            _schedule_line_rich_menu_update(
                bot_id=bot.bot_id,
                line_user_id=line_user_id,
                lang=lang,
                assistant_state=assistant_state,
            )
            return
        except Exception as e:
            logger.warning("LINE menu flow failed for bot_id=%s, falling back to RAG: %s", bot.bot_id, e)
            menu_fallback_skip_assets = True  # skip asset carousel to avoid invalid hero/url errors
            # Fall through to normal AI flow with "Menu" as query
            pass

    # ── Normal AI flow ───────────────────────────────────────────────
    # Save user message
    if event.event_type == "message" and text:
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
            sender_name=line_display_name,
        )

    # Get conversation context
    recent = conversation_service().list_recent_messages(
        session.session_id, limit=CONVERSATION_HISTORY_MESSAGES
    )
    conversation_context = _format_conversation_context(recent)

    # Load agent config
    agent_config: Dict[str, Any] = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = _prompt_provider.resolve_system_prompt(
        agent_config=agent_config,
        lang=lang,
        bot_name=(bot.display_name or "").strip(),
        widget_config=widget_config,
    )
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    # Run RAG
    asset_cards: list = []

    # Inject asset bank as system instruction (up to 150 items; URLs resolved server-side)
    extra_evidence: list[dict[str, str]] = []
    asset_rules = get_asset_rules_from_widget(widget_config)
    asset_instruction = build_asset_instruction(bot.bot_id, asset_rules=asset_rules)
    if asset_instruction:
        system_instruction = f"{system_instruction}\n\n{asset_instruction}" if system_instruction else asset_instruction
    platform_asset_instruction = get_platform_asset_instructions(widget_config, lang=lang)
    if platform_asset_instruction:
        system_instruction = f"{system_instruction}\n\n{platform_asset_instruction}" if system_instruction else platform_asset_instruction
    json_response_instruction = get_platform_json_response_instruction(widget_config, lang=lang)
    if json_response_instruction:
        system_instruction = f"{system_instruction}\n\n{json_response_instruction}" if system_instruction else json_response_instruction

    # Reservation: config-driven from platform profiles (Tabelog, HotPepper, TableCheck)
    reservation_config = get_reservation_config_from_widget(widget_config, lang=lang)
    if reservation_config:
        extra_evidence.append({
            "url": reservation_config["url"],
            "snippet": f"Official online reservation page: {reservation_config['url']}",
        })
        system_instruction = (
            f"{system_instruction}\n\n{reservation_config['instruction']}"
            if system_instruction
            else reservation_config["instruction"]
        )

    platform_features = get_platform_features_from_widget(widget_config)
    menu_extraction_enabled = platform_features.get("menu_extraction_enabled") if platform_features else False
    allowed_asset_types = {"menu_item"} if menu_extraction_enabled else None

    ai_query = effective_text or text
    if effective_text == get_line_menu_quick_payload() or _is_full_menu_request(effective_text, None):
        ai_query = "Menu"

    try:
        corpus = ensure_bot_corpus(bot.bot_id)
        result = run_vertex_rag(
            ai_query,
            rag_corpus=corpus,
            allowed_host=None,
            debug_cb=None,
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            conversation_context=conversation_context or None,
            extra_evidence=extra_evidence if extra_evidence else None,
            parse_json_response=get_platform_json_response_enabled(widget_config),
        )
        sources = result.get("sources") or []
        answer = str(result.get("answer") or "").strip()
        if not answer:
            answer = _line_localized_message(_LINE_EMPTY_ANSWER_MESSAGES, lang)
        answer = normalize_answer_links(
            answer,
            candidates=build_answer_link_candidates(
                reservation_config=reservation_config,
                sources=sources,
            ),
            render_target=RENDER_TARGET_PLAIN_TEXT_CHANNEL,
        ).text

        answer, asset_cards = _asset_policy.resolve_assets(
            answer=answer,
            bot_id=bot.bot_id,
            user_query=ai_query,
            session_id=session.session_id,
            allowed_asset_types=allowed_asset_types,
            show_assets=result.get("show_assets"),
            asset_term_config=asset_rules.get("asset_term_config"),
        )
    except Exception as e:
        if is_quota_exhausted_error(e):
            logger.warning("RAG quota exhausted for LINE message bot_id=%s", bot.bot_id)
            answer = _line_localized_message(_LINE_QUOTA_MESSAGES, lang)
        else:
            logger.exception("RAG error for LINE message bot_id=%s", bot.bot_id)
            answer = _line_localized_message(_LINE_ERROR_MESSAGES, lang)

    # Save bot response
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
    )

    # Build absolute image URLs for LINE (server-side fetch)
    base = f"https://{request.headers.get('host', 'localhost')}"
    for card in asset_cards:
        img = card.get("image_url", "")
        if img and not img.startswith("http"):
            card["image_url"] = f"{base}{img}"

    # Debug logging
    logger.info(f"LINE bot_id={bot.bot_id} asset_cards count: {len(asset_cards)}")
    if asset_cards:
        logger.info(f"LINE bot_id={bot.bot_id} asset_cards: {asset_cards}")

    # LINE has a 5000 char limit per message; split if needed
    # Skip asset carousel when falling back from menu flow (avoids invalid hero/url errors)
    reply_asset_cards = None if menu_fallback_skip_assets else asset_cards
    if len(answer) > 5000:
        chunks = [answer[i:i + 5000] for i in range(0, len(answer), 5000)]
        await reply_message(reply_token, chunks[:5], access_token, asset_cards=reply_asset_cards, suggested_flex=suggested_flex)
    else:
        await reply_message(reply_token, [answer], access_token, asset_cards=reply_asset_cards, suggested_flex=suggested_flex)
    _schedule_line_rich_menu_update(
        bot_id=bot.bot_id,
        line_user_id=line_user_id,
        lang=lang,
        assistant_state=assistant_state,
    )


# ══════════════════════════════════════════════════════════════════════
# LINE Channel Config CRUD (authenticated)
# ══════════════════════════════════════════════════════════════════════


@router.get("/v1/org/bots/{bot_id}/line-channel", response_model=LineChannelResponse)
async def v1_org_get_line_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    channel = _line_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        raise HTTPException(status_code=404, detail="No LINE channel configured for this bot")
    return _build_line_channel_response(channel)


@router.put("/v1/org/bots/{bot_id}/line-channel", response_model=LineChannelResponse)
async def v1_org_upsert_line_channel(
    bot_id: str,
    payload: LineChannelUpsertRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    # Resolve secrets: if blank on update, keep existing values
    secret = (payload.line_channel_secret or "").strip()
    access_token = (payload.line_channel_access_token or "").strip()
    existing = _line_channel_repo.get_by_bot_id(bot_id)
    if existing:
        if not secret:
            secret = existing.line_channel_secret
        if not access_token:
            access_token = existing.line_channel_access_token
    else:
        if not secret or not access_token:
            raise HTTPException(
                status_code=400,
                detail="Channel secret and access token are required for initial setup",
            )

    channel = _line_channel_repo.upsert(
        bot_id=bot_id,
        org_id=resolved_org,
        line_channel_id=payload.line_channel_id,
        line_channel_secret=secret,
        line_channel_access_token=access_token,
        is_active=payload.is_active,
    )
    rich_menu_state = None
    try:
        rich_menu_state = await line_rich_menu_service().sync_for_bot(bot_id, force=True)
    except Exception:
        logger.exception("Managed LINE rich-menu sync failed during channel save bot_id=%s", bot_id)
    return _build_line_channel_response(channel, rich_menu_state=rich_menu_state)


@router.delete("/v1/org/bots/{bot_id}/line-channel", response_model=LineChannelDeleteResponse)
async def v1_org_delete_line_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    existing = _line_channel_repo.get_by_bot_id(bot_id)
    if not existing:
        raise HTTPException(status_code=404, detail="No LINE channel configured for this bot")
    try:
        await line_rich_menu_service().cleanup_for_bot(bot_id)
    except Exception:
        logger.exception("Managed LINE rich-menu cleanup failed bot_id=%s", bot_id)
    deleted = _line_channel_repo.delete_by_bot_id(bot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No LINE channel configured for this bot")
    return LineChannelDeleteResponse(ok=True, bot_id=bot_id)


@router.post("/v1/org/bots/{bot_id}/line-channel/rich-menu/resync", response_model=LineChannelResponse)
async def v1_org_resync_line_rich_menu(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    channel = _line_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        raise HTTPException(status_code=404, detail="No LINE channel configured for this bot")
    rich_menu_state = await line_rich_menu_service().sync_for_bot(bot_id, force=True)
    return _build_line_channel_response(channel, rich_menu_state=rich_menu_state)


@router.post("/v1/org/bots/{bot_id}/line-channel/test", response_model=LineChannelTestResponse)
async def v1_org_test_line_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Test that the saved LINE channel credentials are valid by calling the LINE Bot Info API."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    channel = _line_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        return LineChannelTestResponse(
            ok=False,
            message="No LINE channel configured. Save your credentials first.",
        )

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.line.me/v2/bot/info",
                headers={"Authorization": f"Bearer {channel.line_channel_access_token}"},
            )
        if resp.status_code == 200:
            info = resp.json()
            display_name = str(info.get("displayName") or "").strip() or None
            basic_id = str(info.get("basicId") or "").strip() or None
            picture_url = str(info.get("pictureUrl") or "").strip() or None
            user_id = str(info.get("userId") or "").strip() or None
            bot_name = display_name or basic_id or "your bot"
            return LineChannelTestResponse(
                ok=True,
                message=f"Connection successful! LINE bot: {bot_name}",
                display_name=display_name,
                basic_id=basic_id,
                picture_url=picture_url,
                user_id=user_id,
            )
        elif resp.status_code == 401:
            return LineChannelTestResponse(
                ok=False,
                message="Invalid access token. Please check your Channel Access Token and try again.",
            )
        else:
            return LineChannelTestResponse(
                ok=False,
                message=f"LINE API returned status {resp.status_code}. Check your credentials.",
            )
    except Exception as exc:
        return LineChannelTestResponse(
            ok=False,
            message=f"Could not reach LINE API: {str(exc)}",
        )
