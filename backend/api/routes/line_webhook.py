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
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps.auth import get_current_user
from api.schemas import (
    LineChannelUpsertRequest,
    LineChannelResponse,
    LineChannelDeleteResponse,
)
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from application.services.asset_policy import ConfigAssetPolicy
from application.services.channel_adapters import LineChannelAdapter
from application.services.function_registry import FunctionRegistry
from application.services.prompt_provider import ConfigPromptProvider
from application.services.platform_strategy import (
    build_menu_view_all_url_for_category,
    normalize_menu_category,
)
from common.di.container import asset_repo, bot_service, conversation_service
from infrastructure.clients.line_client import (
    verify_signature,
    reply_message,
    reply_with_messages,
    build_suggested_flex,
    build_buttons_template,
    build_text_with_quick_replies,
    show_typing as line_show_typing,
)
from infrastructure.clients.rag_client import run_vertex_rag, is_quota_exhausted_error
from domain.platform_profiles import (
    ensure_canonical_reservation_url_in_text,
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


# ── Escalation keyword detection (matches Instagram) ───────────────────

_ESCALATION_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "prompt": "I'll connect you with our staff right away.\n\n"
        "When you send your message, we'll forward it to our team. The next reply you receive will be from our staff — please wait for them to respond. From here on, the AI will not reply; our team will take over.\n\n"
        "If you wish to cancel and return to the AI assistant, say \"cancel\" at any time.",
        "cancel_ack": "Cancelled. How can I help you?",
        "escalation_ack": "We've notified our team. Someone will reply shortly — please wait for our staff to respond.",
        "takeover_ack": "A team member is now assisting you. Please wait for their reply.",
        "email_details_no_message": "User requested human assistance via LINE.",
    },
    "ja": {
        "prompt": "スタッフにおつなぎいたします。\n\n"
        "送信いただいた内容はスタッフに転送されます。次の返信はスタッフからお届けしますので、お待ちください。このあとはAIではなくスタッフがお返事いたします。\n\n"
        "AIアシスタントに戻りたい場合はいつでも「キャンセル」と送信してください。",
        "cancel_ack": "キャンセルしました。何かお手伝いできますか？",
        "escalation_ack": "スタッフに通知しました。まもなく返信いたしますので、お待ちください。",
        "takeover_ack": "スタッフが対応いたします。お返事をお待ちください。",
        "email_details_no_message": "LINE経由でサポートを依頼されました。",
    },
}

# Cancel escalation (matches Instagram — no skip, no back-to-bot)
_CANCEL_KEYWORDS = {"cancel", "キャンセル"}


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

        await _handle_text_message(
            bot=bot,
            channel=channel,
            line_user_id=parsed.user_id,
            text=parsed.text,
            reply_token=parsed.reply_token,
            request=request,
        )

    return {"ok": True}


async def _handle_text_message(
    *,
    bot,
    channel,
    line_user_id: str,
    text: str,
    reply_token: str,
    request: Request,
) -> None:
    """Core handler for a single text message from LINE."""
    access_token = channel.line_channel_access_token

    # ── Early menu path: send button BEFORE session/typing (reply token expires in ~30s) ──
    if _is_full_menu_request(text, None):
        try:
            menu_url = _menu_page_url(getattr(bot, "publishable_key", None), request)
            if menu_url and menu_url.lower().startswith("https://") and len(menu_url) > 12:
                items = _get_menu_items_for_bot(bot.bot_id)
                if items:
                    wc = {}
                    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
                        try:
                            wc = json.loads(bot.widget_config)
                        except (TypeError, ValueError):
                            pass
                    lang = "ja" if str(wc.get("language") or "").lower() in ("ja", "jp") else "en"
                    prompt = _menu_text("view_menu_button_prompt", lang)
                    btn_title = _menu_text("view_menu_button_title", lang)
                    tmpl = build_buttons_template(
                        "View full menu",
                        prompt,
                        [{"type": "uri", "label": btn_title[:20], "uri": menu_url}],
                    )
                    ok = await reply_with_messages(reply_token, [tmpl], access_token)
                    if ok:
                        logger.info("LINE menu button sent early bot_id=%s", bot.bot_id)
                        return
                    logger.warning("LINE menu button failed early bot_id=%s (see LINE reply failed log)", bot.bot_id)
        except Exception as e:
            logger.warning("LINE menu early path failed bot_id=%s: %s", bot.bot_id, e)

    # Load suggested messages and session BEFORE typing — we must not show typing when
    # user is escalated and we won't reply (avoids typing bubble stuck until timeout)
    suggested_flex = None
    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    lang = str(widget_config.get("language") or widget_config.get("botLanguage") or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"
    suggested_messages = get_suggested_messages_for_widget(widget_config, lang=lang)
    if suggested_messages:
        suggested_flex = build_suggested_flex(suggested_messages)
    suggested_type = _resolve_suggested_type(text, suggested_messages or [])
    suggested_function_id = _function_registry.resolve_from_suggested_type(suggested_type or "")

    # Get or create session mapping
    mapping = _line_user_session_repo.get(line_user_id=line_user_id, bot_id=bot.bot_id)

    if mapping is None:
        # Create a new conversation session
        session = conversation_service().get_or_create_session(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            channel="line",
            session_id=None,
            site_url=None,
            site_title=None,
            user_agent="LINE",
            ip=None,
        )
        mapping = _line_user_session_repo.get_or_create(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            session_id=session.session_id,
        )
    else:
        # Refresh session (may create new if expired)
        session = conversation_service().get_or_create_session(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            channel="line",
            session_id=mapping.session_id,
            site_url=None,
            site_title=None,
            user_agent="LINE",
            ip=None,
        )
        # If session was expired and a new one was created, update mapping
        if session.session_id != mapping.session_id:
            _line_user_session_repo.update_session_id(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                session_id=session.session_id,
            )
            # Reset escalation state for new session
            _line_user_session_repo.set_escalated(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                escalated=False,
            )
            _line_user_session_repo.set_awaiting_escalation_msg(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                awaiting=False,
            )
            mapping = _line_user_session_repo.get(line_user_id=line_user_id, bot_id=bot.bot_id)

    # ── Escalated: log message only, no bot reply — return BEFORE typing ───
    if mapping and mapping.is_escalated and not _wants_cancel_escalation(text):
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        return

    # Show typing only when we will send a reply (avoids stuck typing when escalated)
    await line_show_typing(line_user_id, access_token)

    # ── De-escalation check (cancel only, matches Instagram) ───────────
    if mapping and mapping.is_escalated and _wants_cancel_escalation(text):
        _line_user_session_repo.set_escalated(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            escalated=False,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        msgs = _ESCALATION_MESSAGES.get(lang, _ESCALATION_MESSAGES["en"])
        cancel_msg = msgs["cancel_ack"]
        await reply_message(reply_token, [cancel_msg], access_token, suggested_flex=suggested_flex)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=cancel_msg,
        )
        return

    # ── Awaiting optional escalation message ─────────────────────────
    if mapping and mapping.awaiting_escalation_msg:
        if _wants_cancel_escalation(text):
            _line_user_session_repo.set_awaiting_escalation_msg(
                line_user_id=line_user_id,
                bot_id=bot.bot_id,
                awaiting=False,
            )
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=text,
            )
            msgs = _ESCALATION_MESSAGES.get(lang, _ESCALATION_MESSAGES["en"])
            cancel_msg = msgs["cancel_ack"]
            await reply_message(reply_token, [cancel_msg], access_token, suggested_flex=suggested_flex)
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="bot",
                content=cancel_msg,
            )
            return
        user_msg = text.strip()
        _line_user_session_repo.set_awaiting_escalation_msg(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            awaiting=False,
        )
        _line_user_session_repo.set_escalated(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            escalated=True,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        msgs = _ESCALATION_MESSAGES.get(lang, _ESCALATION_MESSAGES["en"])
        details = user_msg if user_msg else msgs["email_details_no_message"]
        conversation_service().create_escalation(
            bot_id=bot.bot_id,
            session_id=session.session_id,
            visitor_email=f"line:{line_user_id}",
            details=details,
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
        _line_user_session_repo.set_escalated(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            escalated=True,
        )
        return

    # ── Escalation request ───────────────────────────────────────────
    if suggested_function_id == "escalate" or _is_escalate_quick_reply(text, suggested_messages or []):
        _line_user_session_repo.set_awaiting_escalation_msg(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            awaiting=True,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        prompt_msg = _ESCALATION_MESSAGES.get(lang, _ESCALATION_MESSAGES["en"])["prompt"]
        await reply_message(reply_token, [prompt_msg], access_token, suggested_flex=suggested_flex)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=prompt_msg,
        )
        return

    # ── Menu flow (matches Instagram) ─────────────────────────────────
    platform_features = get_platform_features_from_widget(widget_config)
    menu_extraction_enabled = platform_features.get("menu_extraction_enabled") if platform_features else False
    quick_payload = None  # LINE only has text; show_menu sends SHOW_FULL_MENU as text
    effective_text = text
    menu_fallback_skip_assets = False  # set True when menu flow fails and we fall back to RAG

    # Parse pagination payload (SHOW_MENU_PAGE:category:offset) whenever sent
    page_req = _parse_menu_page_payload(effective_text)
    if page_req:
        category, offset = page_req
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text or "View more menu",
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
        return

    # Run menu flow when user explicitly requests menu (Menu/SHOW_FULL_MENU),
    # regardless of platform — ensures Menu button always responds
    if suggested_function_id == "show_menu" or _is_full_menu_request(effective_text, quick_payload):
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
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="user",
                content=effective_text,
            )
            conversation_service().add_message(
                session_id=session.session_id,
                bot_id=bot.bot_id,
                role="bot",
                content="Shared the categorized menu.",
            )
            return
        except Exception as e:
            logger.warning("LINE menu flow failed for bot_id=%s, falling back to RAG: %s", bot.bot_id, e)
            menu_fallback_skip_assets = True  # skip asset carousel to avoid invalid hero/url errors
            # Fall through to normal AI flow with "Menu" as query
            pass

    # ── Normal AI flow ───────────────────────────────────────────────
    # Save user message
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=text,
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
    lang = str(widget_config.get("language") or widget_config.get("botLanguage") or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"
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

    ai_query = text
    if text == get_line_menu_quick_payload() or _is_full_menu_request(text, None):
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
        answer = str(result.get("answer") or "").strip()
        if not answer:
            answer = "I'm sorry, I couldn't find an answer to that. Could you try rephrasing?"

        if reservation_config:
            answer = ensure_canonical_reservation_url_in_text(
                answer, reservation_config["url"], reservation_config["domain_key"]
            )

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
            answer = "We're experiencing high demand right now. Please try again in about a minute."
        else:
            logger.exception("RAG error for LINE message bot_id=%s", bot.bot_id)
            answer = "I'm sorry, something went wrong. Please try again in a moment."

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
    return LineChannelResponse(
        channel_id=channel.channel_id,
        bot_id=channel.bot_id,
        org_id=channel.org_id,
        line_channel_id=channel.line_channel_id,
        is_active=channel.is_active,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
    )


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
    return LineChannelResponse(
        channel_id=channel.channel_id,
        bot_id=channel.bot_id,
        org_id=channel.org_id,
        line_channel_id=channel.line_channel_id,
        is_active=channel.is_active,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
    )


@router.delete("/v1/org/bots/{bot_id}/line-channel", response_model=LineChannelDeleteResponse)
async def v1_org_delete_line_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    deleted = _line_channel_repo.delete_by_bot_id(bot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No LINE channel configured for this bot")
    return LineChannelDeleteResponse(ok=True, bot_id=bot_id)


@router.post("/v1/org/bots/{bot_id}/line-channel/test")
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
        return {"ok": False, "message": "No LINE channel configured. Save your credentials first."}

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.line.me/v2/bot/info",
                headers={"Authorization": f"Bearer {channel.line_channel_access_token}"},
            )
        if resp.status_code == 200:
            info = resp.json()
            bot_name = info.get("displayName") or info.get("basicId") or "your bot"
            return {"ok": True, "message": f"Connection successful! LINE bot: {bot_name}"}
        elif resp.status_code == 401:
            return {"ok": False, "message": "Invalid access token. Please check your Channel Access Token and try again."}
        else:
            return {"ok": False, "message": f"LINE API returned status {resp.status_code}. Check your credentials."}
    except Exception as exc:
        return {"ok": False, "message": f"Could not reach LINE API: {str(exc)}"}
