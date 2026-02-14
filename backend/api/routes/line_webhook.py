"""LINE Messaging API webhook handler and LINE channel config CRUD.

Endpoints:
  POST /webhooks/line/{bot_id}           -- LINE webhook (public, signature-verified)
  GET  /v1/org/bots/{bot_id}/line-channel -- Get LINE config (authenticated)
  PUT  /v1/org/bots/{bot_id}/line-channel -- Create/update LINE config (authenticated)
  DELETE /v1/org/bots/{bot_id}/line-channel -- Remove LINE integration (authenticated)
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps.auth import get_current_user
from api.schemas import (
    LineChannelUpsertRequest,
    LineChannelResponse,
    LineChannelDeleteResponse,
)
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from common.di.container import bot_service, conversation_service
from infrastructure.clients.line_client import (
    verify_signature,
    reply_message,
)
from infrastructure.clients.rag_client import run_vertex_rag
from infrastructure.assets.asset_resolver import build_asset_instruction, process_answer_assets
from infrastructure.db.repositories import (
    PostgresLineChannelRepository,
    PostgresLineUserSessionRepository,
)
from infrastructure.services.indexing_service import ensure_bot_corpus

logger = logging.getLogger(__name__)

router = APIRouter()

_line_channel_repo = PostgresLineChannelRepository()
_line_user_session_repo = PostgresLineUserSessionRepository()

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


# ── Escalation keyword detection ─────────────────────────────────────

ESCALATION_KEYWORDS = {
    "human", "agent", "staff", "real person", "operator",
    "talk to someone", "スタッフ", "人間", "担当者",
}

DE_ESCALATION_KEYWORDS = {
    "back to bot", "bot", "ai", "ボット", "戻る",
}


def _wants_escalation(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in ESCALATION_KEYWORDS)


def _wants_de_escalation(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in DE_ESCALATION_KEYWORDS)


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
    if not channel or not channel.is_active:
        raise HTTPException(status_code=404, detail="LINE channel not configured or inactive")

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

    # Process each event
    for event in events:
        event_type = event.get("type")
        if event_type != "message":
            continue
        message = event.get("message", {})
        if message.get("type") != "text":
            continue

        text = (message.get("text") or "").strip()
        if not text:
            continue

        reply_token = event.get("replyToken", "")
        source = event.get("source", {})
        line_user_id = source.get("userId", "")
        if not line_user_id:
            continue

        await _handle_text_message(
            bot=bot,
            channel=channel,
            line_user_id=line_user_id,
            text=text,
            reply_token=reply_token,
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

    # Get or create session mapping
    # First check if mapping exists
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
            mapping = _line_user_session_repo.get(line_user_id=line_user_id, bot_id=bot.bot_id)

    # ── De-escalation check ──────────────────────────────────────────
    if mapping and mapping.is_escalated and _wants_de_escalation(text):
        _line_user_session_repo.set_escalated(
            line_user_id=line_user_id,
            bot_id=bot.bot_id,
            escalated=False,
        )
        await reply_message(
            reply_token,
            ["You're now back with our AI assistant. How can I help you?"],
            access_token,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content="You're now back with our AI assistant. How can I help you?",
        )
        return

    # ── Escalated: log message but don't auto-reply ──────────────────
    if mapping and mapping.is_escalated:
        # Save user message for history/analytics, but don't reply -- staff handles it in LINE OA Manager
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="user",
            content=text,
        )
        return

    # ── Escalation request ───────────────────────────────────────────
    if _wants_escalation(text):
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
        # Create escalation record (use line_user_id as visitor_email substitute)
        conversation_service().create_escalation(
            bot_id=bot.bot_id,
            session_id=session.session_id,
            visitor_email=f"line:{line_user_id}",
            details="User requested human assistance via LINE",
        )
        escalation_msg = (
            "I'm connecting you with our staff. They'll reply to you shortly here in LINE.\n\n"
            "When you're done, just say \"back to bot\" to return to the AI assistant."
        )
        await reply_message(reply_token, [escalation_msg], access_token)
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=escalation_msg,
        )
        return

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
    system_instruction = agent_config.get("instructions") if agent_config else None
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    # Inject business asset descriptions into system prompt
    asset_instruction = build_asset_instruction(bot.bot_id)
    if asset_instruction:
        system_instruction = f"{system_instruction}\n{asset_instruction}" if system_instruction else asset_instruction

    # Run RAG
    asset_cards: list = []
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
        )
        answer = str(result.get("answer") or "").strip()
        if not answer:
            answer = "I'm sorry, I couldn't find an answer to that. Could you try rephrasing?"
        else:
            # Resolve asset markers + keyword fallback
            answer, asset_cards = process_answer_assets(answer, bot.bot_id)
    except Exception:
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

    # LINE has a 5000 char limit per message; split if needed
    if len(answer) > 5000:
        chunks = [answer[i:i + 5000] for i in range(0, len(answer), 5000)]
        await reply_message(reply_token, chunks[:5], access_token, asset_cards=asset_cards)
    else:
        await reply_message(reply_token, [answer], access_token, asset_cards=asset_cards)


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
