"""Instagram OAuth flow endpoints (Business Login for Instagram).

Endpoints:
  GET  /v1/org/bots/{bot_id}/instagram/auth-url   -- Generate OAuth URL (authenticated)
  GET  /v1/auth/instagram/callback                 -- OAuth callback from Meta (public)
  POST /v1/org/bots/{bot_id}/instagram/disconnect  -- Disconnect OAuth channel (authenticated)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

from api.deps.auth import get_current_user
from api.schemas import InstagramOAuthUrlResponse
from infrastructure.clients.instagram_client import (
    INSTAGRAM_APP_ID,
    INSTAGRAM_APP_SECRET,
    INSTAGRAM_REDIRECT_URI,
    build_instagram_auth_url,
    exchange_code_for_token,
    exchange_for_long_lived_token,
    subscribe_webhooks,
    get_ig_account_info,
    get_ig_webhook_igsid,
)
from infrastructure.db.repositories import PostgresInstagramChannelRepository

logger = logging.getLogger(__name__)

router = APIRouter()

_ig_channel_repo = PostgresInstagramChannelRepository()

# Dashboard URL for redirect after OAuth callback
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").strip()

# ── State encryption (simple HMAC-based signed JSON) ─────────────────

_STATE_SECRET = (INSTAGRAM_APP_SECRET or "fallback-secret-key").encode("utf-8")


def _sign_state(payload: dict) -> str:
    """Create a signed state parameter (JSON + HMAC)."""
    payload["ts"] = int(time.time())
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(_STATE_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:16]
    encoded = base64.urlsafe_b64encode(raw.encode()).decode()
    return f"{encoded}.{sig}"


def _verify_state(state: str) -> Optional[dict]:
    """Verify and decode a signed state parameter. Returns None if invalid/expired."""
    try:
        parts = state.split(".", 1)
        if len(parts) != 2:
            return None
        encoded, sig = parts
        raw = base64.urlsafe_b64decode(encoded).decode()
        expected_sig = hmac.new(_STATE_SECRET, raw.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(sig, expected_sig):
            logger.warning("Instagram OAuth state: signature mismatch")
            return None
        payload = json.loads(raw)
        # Expire after 10 minutes
        if time.time() - payload.get("ts", 0) > 600:
            logger.warning("Instagram OAuth state: expired")
            return None
        return payload
    except Exception:
        logger.exception("Instagram OAuth state: decode error")
        return None


# ── Helper: resolve org for auth (mirrors instagram_webhook.py) ──────

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
    from common.di.container import bot_service
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    if bot.org_id != org_id:
        raise HTTPException(status_code=403, detail="Bot does not belong to this org")


# ══════════════════════════════════════════════════════════════════════
# 1. Generate OAuth URL
# ══════════════════════════════════════════════════════════════════════


@router.get(
    "/v1/org/bots/{bot_id}/instagram/auth-url",
    response_model=InstagramOAuthUrlResponse,
)
async def get_instagram_auth_url(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Generate the Instagram OAuth authorization URL for this bot."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    if not INSTAGRAM_APP_ID or not INSTAGRAM_REDIRECT_URI:
        raise HTTPException(
            status_code=500,
            detail="Instagram OAuth not configured. Set INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET, and INSTAGRAM_REDIRECT_URI.",
        )

    state = _sign_state({"bot_id": bot_id, "org_id": resolved_org})
    auth_url = build_instagram_auth_url(state)
    return InstagramOAuthUrlResponse(auth_url=auth_url)


# ══════════════════════════════════════════════════════════════════════
# 2. OAuth Callback (Meta redirects here after user approves)
# ══════════════════════════════════════════════════════════════════════


@router.get("/v1/auth/instagram/callback")
async def instagram_oauth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_reason: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """Handle the OAuth redirect from Instagram.

    On success: exchanges code for tokens, subscribes webhooks, saves channel, redirects to dashboard.
    On cancel/error: redirects to dashboard with error info.
    """
    dashboard_base = DASHBOARD_URL or "/"

    # Handle user-cancelled or error cases
    if error:
        logger.warning("Instagram OAuth cancelled/error: %s / %s / %s", error, error_reason, error_description)
        return RedirectResponse(
            f"{dashboard_base}?ig_error={quote(error_description or error_reason or error)}"
        )

    if not code or not state:
        return RedirectResponse(f"{dashboard_base}?ig_error=missing_code_or_state")

    # Verify state
    payload = _verify_state(state)
    if not payload:
        return RedirectResponse(f"{dashboard_base}?ig_error=invalid_state")

    bot_id = payload.get("bot_id")
    org_id = payload.get("org_id")
    if not bot_id or not org_id:
        return RedirectResponse(f"{dashboard_base}?ig_error=invalid_state_payload")

    try:
        # Step 1: Exchange code for short-lived token
        # Strip trailing #_ that Instagram appends
        clean_code = code.rstrip("#_")
        short_token, ig_user_id = await exchange_code_for_token(clean_code)
        logger.info("Instagram OAuth: got short-lived token for ig_user_id=%s", ig_user_id)

        # Step 2: Exchange for long-lived token (60 days)
        long_token, expires_in = await exchange_for_long_lived_token(short_token)
        token_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        logger.info("Instagram OAuth: got long-lived token, expires_in=%d", expires_in)

        # Step 3: Get account info
        account_info = await get_ig_account_info(long_token)
        ig_username = account_info.get("username", "")
        logger.info("Instagram OAuth: account=%s (@%s)", ig_user_id, ig_username)

        # Step 3b: Resolve the IGSID (webhook recipient ID) which differs
        # from the app-scoped user ID returned by the token exchange.
        ig_webhook_id = await get_ig_webhook_igsid(long_token)
        if ig_webhook_id and ig_webhook_id != ig_user_id:
            logger.info("Instagram OAuth: webhook IGSID=%s (differs from app-scoped %s)", ig_webhook_id, ig_user_id)
        else:
            ig_webhook_id = None  # same or unresolved, no separate storage needed

        # Step 4: Subscribe webhooks programmatically
        webhook_ok = await subscribe_webhooks(ig_user_id, long_token)
        if not webhook_ok:
            logger.warning("Instagram OAuth: webhook subscription returned false for %s (may still work)", ig_user_id)

        # Step 5: Save to database
        _ig_channel_repo.upsert_oauth(
            bot_id=bot_id,
            org_id=org_id,
            ig_user_id=ig_user_id,
            ig_username=ig_username,
            access_token=long_token,
            token_expires_at=token_expires_at,
            ig_webhook_id=ig_webhook_id,
        )
        logger.info("Instagram OAuth: channel saved for bot_id=%s, ig_user_id=%s", bot_id, ig_user_id)

        # Redirect back to dashboard with success
        return RedirectResponse(
            f"{dashboard_base}/bots/{bot_id}/instagram?connected=true&username={quote(ig_username)}"
        )

    except Exception as exc:
        logger.exception("Instagram OAuth callback error")
        return RedirectResponse(
            f"{dashboard_base}/bots/{bot_id}/instagram?ig_error={quote(str(exc)[:200])}"
        )


# ══════════════════════════════════════════════════════════════════════
# 3. Disconnect
# ══════════════════════════════════════════════════════════════════════


@router.post("/v1/org/bots/{bot_id}/instagram/disconnect")
async def instagram_disconnect(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Disconnect the Instagram OAuth channel for this bot."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    deleted = _ig_channel_repo.delete_by_bot_id(bot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No Instagram channel configured for this bot")
    return {"ok": True, "bot_id": bot_id}
