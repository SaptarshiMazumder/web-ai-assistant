"""Instagram Messaging API webhook handler and Instagram channel config CRUD.

Endpoints:
  GET  /webhooks/instagram/{bot_id}               -- Meta verification handshake (legacy per-bot)
  POST /webhooks/instagram/{bot_id}               -- Receive DMs (legacy per-bot, signature-verified)
  GET  /webhooks/instagram                        -- Meta verification handshake (global, for OAuth)
  POST /webhooks/instagram                        -- Receive DMs (global, routes by recipient.id)
  GET  /v1/org/bots/{bot_id}/instagram-channel    -- Get config (authenticated)
  PUT  /v1/org/bots/{bot_id}/instagram-channel    -- Create/update config (authenticated)
  DELETE /v1/org/bots/{bot_id}/instagram-channel   -- Remove integration (authenticated)
  POST /v1/org/bots/{bot_id}/instagram-channel/test -- Test connection (authenticated)
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from api.deps.auth import get_current_user
from api.schemas import (
    InstagramChannelUpsertRequest,
    InstagramChannelResponse,
    InstagramChannelDeleteResponse,
)
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from common.di.container import bot_service, conversation_service
from infrastructure.clients.instagram_client import (
    INSTAGRAM_APP_SECRET,
    verify_signature,
    send_message,
    send_image,
    send_generic_template,
    build_ig_quick_replies,
    show_typing as ig_show_typing,
)
from infrastructure.clients.rag_client import run_vertex_rag
from infrastructure.assets.asset_resolver import process_answer_assets, build_asset_evidence, build_asset_instruction, resolve_asset_markers
from infrastructure.db.repositories import (
    PostgresInstagramChannelRepository,
    PostgresInstagramUserSessionRepository,
)
from infrastructure.services.indexing_service import ensure_bot_corpus

logger = logging.getLogger(__name__)

router = APIRouter()

_ig_channel_repo = PostgresInstagramChannelRepository()
_ig_user_session_repo = PostgresInstagramUserSessionRepository()

# Global webhook verify token (set in Meta App Dashboard, stored in env)
_GLOBAL_WEBHOOK_VERIFY_TOKEN = os.environ.get("INSTAGRAM_WEBHOOK_VERIFY_TOKEN", "").strip()

# ── Rate limiting ─────────────────────────────────────────────────────

_rl_state: Dict[str, tuple] = {}
_rl_window_s = 60
_rl_max_per_window = 120


def _rate_limit(bot_id: str) -> None:
    now = time.time()
    start, count = _rl_state.get(bot_id, (now, 0))
    if now - start > _rl_window_s:
        start, count = now, 0
    count += 1
    _rl_state[bot_id] = (start, count)
    if count > _rl_max_per_window:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ── Escalation keyword detection ─────────────────────────────────────

ESCALATION_KEYWORDS = {
    "human", "agent", "staff", "real person", "operator",
    "talk to someone", "スタッフ", "人間", "担当者",
}

SKIP_KEYWORDS = {
    "skip", "s", "/skip", "no", "nope", "never mind", "cancel",
}

# Matches any intent to return to the AI/bot, case-insensitive
_DE_ESCALATION_PATTERN = re.compile(
    r"\b(back|return|switch|go back|exit|leave|end|stop|quit)\b.{0,20}\b(bot|ai|assistant|robot|auto)\b"
    r"|\b(bot|ai|assistant|robot)\b.{0,20}\b(mode|again|please|now)\b"
    r"|\b(back to (bot|ai|assistant|auto|robot))\b"
    r"|\b(ボット|戻る|ai に戻る|botに戻る)\b",
    re.IGNORECASE,
)


def _wants_escalation(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in ESCALATION_KEYWORDS)


def _wants_de_escalation(text: str) -> bool:
    return bool(_DE_ESCALATION_PATTERN.search(text))


def _wants_skip_message(text: str) -> bool:
    lower = text.lower().strip()
    return any(lower == kw or lower.startswith(kw) for kw in SKIP_KEYWORDS)


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
# Global Webhook (for OAuth-connected channels)
# ══════════════════════════════════════════════════════════════════════


@router.get("/webhooks/instagram")
async def instagram_webhook_verify_global(request: Request):
    """Handle Meta's webhook verification challenge for the global (OAuth) endpoint."""
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")

    if mode != "subscribe":
        raise HTTPException(status_code=403, detail="Invalid hub.mode")

    if not _GLOBAL_WEBHOOK_VERIFY_TOKEN:
        raise HTTPException(status_code=500, detail="Global webhook verify token not configured")

    if token != _GLOBAL_WEBHOOK_VERIFY_TOKEN:
        raise HTTPException(status_code=403, detail="Verify token mismatch")

    logger.info("Instagram global webhook verified")
    return PlainTextResponse(content=challenge)


@router.post("/webhooks/instagram")
async def instagram_webhook_global(request: Request):
    """Receive Instagram DM webhooks and route to the correct bot by recipient.id.

    This is the global endpoint used by OAuth-connected channels.
    Meta sends events here for all channels connected via our app.
    """
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")

    # Verify with our app secret (global, not per-client)
    if not signature:
        logger.warning("Instagram global webhook: No X-Hub-Signature-256 header")
        raise HTTPException(status_code=403, detail="Missing signature header")

    app_secret = INSTAGRAM_APP_SECRET
    if not app_secret:
        raise HTTPException(status_code=500, detail="Instagram app secret not configured")

    if not verify_signature(body, signature, app_secret):
        logger.warning("Instagram global webhook: Signature mismatch")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    entries = payload.get("entry", [])
    for entry in entries:
        messaging_events = entry.get("messaging", [])
        for event in messaging_events:
            message = event.get("message", {})
            text = (message.get("text") or "").strip()
            if not text:
                continue
            if message.get("is_echo"):
                continue

            sender = event.get("sender", {})
            ig_sender_id = str(sender.get("id", ""))
            if not ig_sender_id:
                continue

            # The recipient is the IG professional account that received the DM
            recipient = event.get("recipient", {})
            ig_recipient_id = str(recipient.get("id", ""))
            if not ig_recipient_id:
                continue

            # Look up which bot owns this IG account
            channel = _ig_channel_repo.get_by_ig_user_id(ig_recipient_id)
            if not channel:
                # Fallback: try ig_page_id (covers both OAuth and manual)
                channel = _ig_channel_repo.get_by_page_id(ig_recipient_id)
            if not channel:
                # Last resort: the webhook IGSID may differ from the stored
                # app-scoped ID.  Try all active OAuth channels and resolve.
                channel = _ig_channel_repo.resolve_by_webhook_id(ig_recipient_id)
            if not channel or not channel.is_active:
                logger.warning("Instagram global webhook: no active channel for recipient %s", ig_recipient_id)
                continue

            # For OAuth channels, use the global app secret for signature verification
            # (already verified above), but use the channel's token for sending
            bot = bot_service().get_bot_record(channel.bot_id)
            if not bot:
                logger.warning("Instagram global webhook: unknown bot_id %s", channel.bot_id)
                continue

            _rate_limit(channel.bot_id)

            await _handle_text_message(
                bot=bot,
                channel=channel,
                ig_user_id=ig_sender_id,
                text=text,
                request=request,
            )

    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════
# Legacy Per-Bot Webhook (for manually-configured channels)
# ══════════════════════════════════════════════════════════════════════


@router.get("/webhooks/instagram/{bot_id}")
async def instagram_webhook_verify(bot_id: str, request: Request):
    """Handle Meta's webhook verification challenge.

    Meta sends a GET with hub.mode, hub.verify_token, and hub.challenge.
    We must echo hub.challenge if the verify_token matches.
    """
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")

    if mode != "subscribe":
        raise HTTPException(status_code=403, detail="Invalid hub.mode")

    channel = _ig_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Instagram channel not configured")

    if token != channel.verify_token:
        raise HTTPException(status_code=403, detail="Verify token mismatch")

    logger.info("Instagram webhook verified for bot_id=%s", bot_id)
    return PlainTextResponse(content=challenge)


# ══════════════════════════════════════════════════════════════════════
# Instagram Webhook (POST)
# ══════════════════════════════════════════════════════════════════════


@router.post("/webhooks/instagram/{bot_id}")
async def instagram_webhook(bot_id: str, request: Request):
    """Receive and handle Instagram messaging webhook events."""
    # 1. Load channel config
    channel = _ig_channel_repo.get_by_bot_id(bot_id)
    if not channel or not channel.is_active:
        raise HTTPException(status_code=404, detail="Instagram channel not configured or inactive")

    # 2. Read raw body and verify signature
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")
    # For OAuth channels, use the global app secret; for manual channels, use per-channel secret
    effective_secret = (
        INSTAGRAM_APP_SECRET if channel.connection_method == "oauth" else channel.app_secret
    )
    logger.info("Instagram webhook: signature=%s, body_len=%d, method=%s",
                signature[:20] if signature else "NONE", len(body),
                channel.connection_method or "manual")
    if not signature:
        logger.warning("Instagram webhook: No X-Hub-Signature-256 header present")
        raise HTTPException(status_code=403, detail="Missing signature header")
    if not verify_signature(body, signature, effective_secret):
        logger.warning("Instagram webhook: Signature mismatch")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # 3. Parse payload
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 4. Rate limit
    _rate_limit(bot_id)

    # 5. Load bot
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")

    # 6. Process messaging events
    # Meta sends: { "object": "instagram", "entry": [ { "messaging": [...] } ] }
    entries = payload.get("entry", [])
    for entry in entries:
        messaging_events = entry.get("messaging", [])
        for event in messaging_events:
            message = event.get("message", {})
            text = (message.get("text") or "").strip()
            if not text:
                continue  # Skip non-text (images, stickers, etc.)

            # Ignore echo messages (messages sent by the page itself)
            if message.get("is_echo"):
                continue

            sender = event.get("sender", {})
            ig_user_id = str(sender.get("id", ""))
            if not ig_user_id:
                continue

            await _handle_text_message(
                bot=bot,
                channel=channel,
                ig_user_id=ig_user_id,
                text=text,
                request=request,
            )

    return {"ok": True}


async def _handle_text_message(
    *,
    bot,
    channel,
    ig_user_id: str,
    text: str,
    request: Request,
) -> None:
    """Core handler for a single text DM from Instagram."""
    access_token = channel.page_access_token

    # Show typing indicator immediately while processing
    await ig_show_typing(ig_user_id, access_token)

    # Load suggested messages from widget config for Instagram quick replies
    ig_quick_replies = None
    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    suggested_messages_enabled = widget_config.get("suggestedMessagesEnabled", False) if isinstance(widget_config, dict) else False
    suggested_messages = widget_config.get("suggestedMessages") if isinstance(widget_config, dict) and suggested_messages_enabled else None
    if suggested_messages:
        ig_quick_replies = build_ig_quick_replies(suggested_messages)

    # Get or create session mapping
    mapping = _ig_user_session_repo.get(ig_user_id=ig_user_id, bot_id=bot.bot_id)

    if mapping is None:
        session = conversation_service().get_or_create_session(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            channel="instagram",
            session_id=None,
            site_url=None,
            site_title=None,
            user_agent="Instagram",
            ip=None,
        )
        mapping = _ig_user_session_repo.get_or_create(
            ig_user_id=ig_user_id,
            bot_id=bot.bot_id,
            session_id=session.session_id,
        )
    else:
        session = conversation_service().get_or_create_session(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            channel="instagram",
            session_id=mapping.session_id,
            site_url=None,
            site_title=None,
            user_agent="Instagram",
            ip=None,
        )
        if session.session_id != mapping.session_id:
            _ig_user_session_repo.update_session_id(
                ig_user_id=ig_user_id,
                bot_id=bot.bot_id,
                session_id=session.session_id,
            )
            _ig_user_session_repo.set_escalated(
                ig_user_id=ig_user_id,
                bot_id=bot.bot_id,
                escalated=False,
            )
            mapping = _ig_user_session_repo.get(ig_user_id=ig_user_id, bot_id=bot.bot_id)

    # ── De-escalation check ──────────────────────────────────────────
    if mapping and mapping.is_escalated and _wants_de_escalation(text):
        _ig_user_session_repo.set_escalated(
            ig_user_id=ig_user_id,
            bot_id=bot.bot_id,
            escalated=False,
        )
        de_esc_msg = "You're now back with our AI assistant. How can I help you?"
        await send_message(ig_user_id, de_esc_msg, access_token, quick_replies=ig_quick_replies)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=de_esc_msg,
        )
        return

    # ── Escalated: log message + throttled acknowledgment ────────────
    if mapping and mapping.is_escalated:
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        # Only send the full ack once; after that send a brief "✓ Sent." to
        # avoid spamming the user with the same long message on every reply.
        recent = conversation_service().list_recent_messages(session.session_id, limit=20)
        already_acked = any(
            (m.role or "").lower() == "bot" and "back to bot" in (m.content or "")
            for m in recent
        )
        if already_acked:
            ack = "✓ Sent. (Say \"back to bot\" to return to the AI assistant.)"
        else:
            ack = (
                "✓ Message received by our team. They'll reply here shortly.\n\n"
                "To return to the AI assistant, just say \"back to bot\"."
            )
        await send_message(ig_user_id, ack, access_token)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=ack,
        )
        return

    # ── Awaiting optional escalation message ─────────────────────────
    if mapping and mapping.awaiting_escalation_msg:
        user_msg = None if _wants_skip_message(text) else text.strip()
        _ig_user_session_repo.set_awaiting_escalation_msg(
            ig_user_id=ig_user_id,
            bot_id=bot.bot_id,
            awaiting=False,
        )
        _ig_user_session_repo.set_escalated(
            ig_user_id=ig_user_id,
            bot_id=bot.bot_id,
            escalated=True,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        details = f"Message from user: {user_msg}" if user_msg else "User requested human assistance via Instagram."
        conversation_service().create_escalation(
            bot_id=bot.bot_id,
            session_id=session.session_id,
            visitor_email=f"instagram:{ig_user_id}",
            details=details,
        )
        confirm_msg = (
            "Our team has been notified and will reply to you here shortly.\n\n"
            'Reply "back to bot" anytime to return to the AI assistant.'
        )
        await send_message(ig_user_id, confirm_msg, access_token)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=confirm_msg,
        )
        return

    # ── Escalation request ───────────────────────────────────────────
    if _wants_escalation(text):
        _ig_user_session_repo.set_awaiting_escalation_msg(
            ig_user_id=ig_user_id,
            bot_id=bot.bot_id,
            awaiting=True,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        prompt_msg = (
            "I'll connect you with our team right away.\n\n"
            "Would you like to leave a message for them? Type your message below, or reply \"skip\" to connect immediately."
        )
        await send_message(ig_user_id, prompt_msg, access_token)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=prompt_msg,
        )
        return

    # ── Normal AI flow ───────────────────────────────────────────────
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=text,
    )

    recent = conversation_service().list_recent_messages(
        session.session_id, limit=CONVERSATION_HISTORY_MESSAGES
    )
    conversation_context = _format_conversation_context(recent)

    agent_config: Dict[str, Any] = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = agent_config.get("instructions") if agent_config else None
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    asset_cards: list = []

    # Inject business assets as evidence and system instruction
    extra_evidence: list[dict[str, str]] = []
    asset_evidence = build_asset_evidence(bot.bot_id)
    if asset_evidence:
        extra_evidence.extend(asset_evidence)
    asset_instruction = build_asset_instruction(bot.bot_id)
    if asset_instruction:
        system_instruction = f"{system_instruction}\n\n{asset_instruction}" if system_instruction else asset_instruction

    try:
        corpus = ensure_bot_corpus(bot.bot_id)
        result = run_vertex_rag(
            text,
            rag_corpus=corpus,
            allowed_host=None,
            debug_cb=None,
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            conversation_context=conversation_context or None,
            extra_evidence=extra_evidence if extra_evidence else None,
        )
        answer = str(result.get("answer") or "").strip()
        if not answer:
            answer = "I'm sorry, I couldn't find an answer to that. Could you try rephrasing?"
        else:
            # Extract {{asset:ID}} markers from the LLM answer first
            answer, marker_cards = resolve_asset_markers(answer, bot.bot_id, session.session_id)
            if marker_cards:
                asset_cards = marker_cards
            else:
                # Fallback to keyword-based matching
                answer, asset_cards = process_answer_assets(
                    answer,
                    bot.bot_id,
                    user_query=text,
                    session_id=session.session_id,
                )
    except Exception:
        logger.exception("RAG error for Instagram message bot_id=%s", bot.bot_id)
        answer = "I'm sorry, something went wrong. Please try again in a moment."

    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
    )

    # Instagram text limit is 1000 chars; split if needed
    if len(answer) > 1000:
        chunks = [answer[i:i + 1000] for i in range(0, len(answer), 1000)]
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            await send_message(ig_user_id, chunk, access_token, quick_replies=ig_quick_replies if is_last else None)
    else:
        await send_message(ig_user_id, answer, access_token, quick_replies=ig_quick_replies)

    # Send asset images as a Generic Template carousel (clickable cards)
    if asset_cards:
        elements = []
        for card in asset_cards[:10]:  # Instagram limit: max 10 elements
            img_url = card.get("image_url", "")
            if not img_url:
                continue
            abs_url = img_url if img_url.startswith("http") else f"https://{request.headers.get('host', 'localhost')}{img_url}"
            element: dict = {
                "title": (card.get("name") or "Image")[:80],  # IG title limit: 80 chars
                "image_url": abs_url,
            }
            link_url = card.get("link_url")
            if link_url:
                element["default_action"] = {
                    "type": "web_url",
                    "url": link_url,
                }
                element["buttons"] = [
                    {
                        "type": "web_url",
                        "url": link_url,
                        "title": "View",
                    }
                ]
            elements.append(element)
        if elements:
            try:
                await send_generic_template(ig_user_id, elements, access_token)
            except Exception:
                logger.warning("Failed to send asset carousel to Instagram user %s", ig_user_id)



# ══════════════════════════════════════════════════════════════════════
# Instagram Channel Config CRUD (authenticated)
# ══════════════════════════════════════════════════════════════════════


@router.get("/v1/org/bots/{bot_id}/instagram-channel", response_model=InstagramChannelResponse)
async def v1_org_get_instagram_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    channel = _ig_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        raise HTTPException(status_code=404, detail="No Instagram channel configured for this bot")
    return InstagramChannelResponse(
        channel_id=channel.channel_id,
        bot_id=channel.bot_id,
        org_id=channel.org_id,
        ig_page_id=channel.ig_page_id,
        verify_token=channel.verify_token,
        is_active=channel.is_active,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
        ig_user_id=channel.ig_user_id,
        ig_username=channel.ig_username,
        token_expires_at=channel.token_expires_at,
        connection_method=channel.connection_method,
    )


@router.put("/v1/org/bots/{bot_id}/instagram-channel", response_model=InstagramChannelResponse)
async def v1_org_upsert_instagram_channel(
    bot_id: str,
    payload: InstagramChannelUpsertRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    app_secret = (payload.app_secret or "").strip()
    page_access_token = (payload.page_access_token or "").strip()
    existing = _ig_channel_repo.get_by_bot_id(bot_id)
    if existing:
        if not app_secret:
            app_secret = existing.app_secret
        if not page_access_token:
            page_access_token = existing.page_access_token
    else:
        if not app_secret or not page_access_token:
            raise HTTPException(
                status_code=400,
                detail="App Secret and Page Access Token are required for initial setup",
            )

    channel = _ig_channel_repo.upsert(
        bot_id=bot_id,
        org_id=resolved_org,
        ig_page_id=payload.ig_page_id,
        app_secret=app_secret,
        page_access_token=page_access_token,
        is_active=payload.is_active,
    )
    return InstagramChannelResponse(
        channel_id=channel.channel_id,
        bot_id=channel.bot_id,
        org_id=channel.org_id,
        ig_page_id=channel.ig_page_id,
        verify_token=channel.verify_token,
        is_active=channel.is_active,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
        ig_user_id=channel.ig_user_id,
        ig_username=channel.ig_username,
        token_expires_at=channel.token_expires_at,
        connection_method=channel.connection_method,
    )


@router.delete("/v1/org/bots/{bot_id}/instagram-channel", response_model=InstagramChannelDeleteResponse)
async def v1_org_delete_instagram_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    deleted = _ig_channel_repo.delete_by_bot_id(bot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No Instagram channel configured for this bot")
    return InstagramChannelDeleteResponse(ok=True, bot_id=bot_id)


@router.post("/v1/org/bots/{bot_id}/instagram-channel/test")
async def v1_org_test_instagram_channel(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Test that the saved Instagram channel credentials are valid by calling the Graph API."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    channel = _ig_channel_repo.get_by_bot_id(bot_id)
    if not channel:
        return {"ok": False, "message": "No Instagram channel configured. Save your credentials first."}

    from infrastructure.clients.instagram_client import get_page_info
    try:
        info = await get_page_info(channel.page_access_token)
        if info:
            display = info.get("username") or info.get("name") or info.get("id") or info.get("user_id") or "your account"
            return {"ok": True, "message": f"Connection successful! Account: {display}"}
        return {"ok": False, "message": "Invalid access token. Please generate a new token from the Meta App Dashboard and try again."}
    except Exception as exc:
        return {"ok": False, "message": f"Could not reach Meta Graph API: {str(exc)}"}
