"""Thin wrapper around Meta's Graph API for Instagram Messaging.

Uses raw httpx calls -- no facebook-sdk dependency needed.
"""

import hashlib
import hmac
import logging
import os
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE_FB = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
GRAPH_API_BASE_IG = f"https://graph.instagram.com/{GRAPH_API_VERSION}"

# ── App-level Instagram credentials (from env) ───────────────────────
INSTAGRAM_APP_ID = os.environ.get("INSTAGRAM_APP_ID", "").strip()
INSTAGRAM_APP_SECRET = os.environ.get("INSTAGRAM_APP_SECRET", "").strip()
INSTAGRAM_REDIRECT_URI = os.environ.get("INSTAGRAM_REDIRECT_URI", "").strip()


def _api_base(access_token: str) -> str:
    """Pick the right Graph API host based on token type.

    IGAA... tokens come from Instagram API with Instagram Login -> graph.instagram.com
    EAA...  tokens come from Facebook Page tokens            -> graph.facebook.com
    """
    if (access_token or "").startswith("IGAA"):
        return GRAPH_API_BASE_IG
    return GRAPH_API_BASE_FB

# ── Signature verification ────────────────────────────────────────────


def verify_signature(body: bytes, signature: str, app_secret: str) -> bool:
    """Verify Meta webhook signature (X-Hub-Signature-256: sha256=<hex>)."""
    if not signature.startswith("sha256="):
        logger.warning("Instagram verify_signature: signature does not start with sha256=")
        return False
    expected_hex = hmac.new(
        app_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    received_hex = signature[len("sha256="):]
    match = hmac.compare_digest(expected_hex, received_hex)
    if not match:
        logger.warning("Instagram verify_signature: expected=%s... received=%s... secret_len=%d",
                       expected_hex[:12], received_hex[:12], len(app_secret))
    return match


# ── Sending messages ──────────────────────────────────────────────────


async def send_message(
    recipient_id: str,
    text: str,
    page_access_token: str,
) -> bool:
    """Send a text DM to an Instagram user via the Graph API."""
    base = _api_base(page_access_token)
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base}/me/messages",
            json=payload,
            params={"access_token": page_access_token},
        )
    if resp.status_code != 200:
        logger.error("Instagram send_message failed: %s %s", resp.status_code, resp.text)
        return False
    return True


async def send_image(
    recipient_id: str,
    image_url: str,
    page_access_token: str,
) -> bool:
    """Send an image attachment DM to an Instagram user."""
    base = _api_base(page_access_token)
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "image",
                "payload": {"url": image_url, "is_reusable": True},
            }
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base}/me/messages",
            json=payload,
            params={"access_token": page_access_token},
        )
    if resp.status_code != 200:
        logger.error("Instagram send_image failed: %s %s", resp.status_code, resp.text)
        return False
    return True


async def send_generic_template(
    recipient_id: str,
    elements: list,
    page_access_token: str,
) -> bool:
    """Send a Generic Template with image cards via the Graph API."""
    base = _api_base(page_access_token)
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "generic",
                    "elements": elements,
                },
            }
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base}/me/messages",
            json=payload,
            params={"access_token": page_access_token},
        )
    if resp.status_code != 200:
        logger.error("Instagram send_generic_template failed: %s %s", resp.status_code, resp.text)
        return False
    return True


# ── Page info (connection test) ───────────────────────────────────────


async def get_page_info(page_access_token: str) -> Optional[dict]:
    """Get basic info to verify the token works.

    Uses graph.instagram.com for IGAA tokens, graph.facebook.com for EAA tokens.
    """
    base = _api_base(page_access_token)
    async with httpx.AsyncClient(timeout=10) as client:
        # Try Instagram fields first
        resp = await client.get(
            f"{base}/me",
            params={
                "fields": "user_id,username",
                "access_token": page_access_token,
            },
        )
        if resp.status_code == 200:
            return resp.json()

        # Fallback: Facebook Page fields
        resp = await client.get(
            f"{base}/me",
            params={
                "fields": "name,id",
                "access_token": page_access_token,
            },
        )
        if resp.status_code == 200:
            return resp.json()

        # Last resort: no fields
        resp = await client.get(
            f"{base}/me",
            params={"access_token": page_access_token},
        )
        if resp.status_code == 200:
            return resp.json()

    logger.warning("Instagram get_page_info failed: %s %s", resp.status_code, resp.text)
    return None


# ── OAuth helpers (Business Login for Instagram) ─────────────────────


def build_instagram_auth_url(state: str) -> str:
    """Build the Instagram OAuth authorization URL.

    The user will be redirected here to grant permissions.
    Scopes: instagram_business_basic + instagram_business_manage_messages
    """
    return (
        "https://www.instagram.com/oauth/authorize"
        f"?client_id={INSTAGRAM_APP_ID}"
        f"&redirect_uri={INSTAGRAM_REDIRECT_URI}"
        "&response_type=code"
        "&scope=instagram_business_basic,instagram_business_manage_messages"
        f"&state={state}"
    )


async def exchange_code_for_token(code: str) -> Tuple[str, str]:
    """Exchange authorization code for a short-lived Instagram User access token.

    Returns (access_token, ig_user_id).
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.instagram.com/oauth/access_token",
            data={
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "grant_type": "authorization_code",
                "redirect_uri": INSTAGRAM_REDIRECT_URI,
                "code": code,
            },
        )
    if resp.status_code != 200:
        logger.error("exchange_code_for_token failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"Token exchange failed: {resp.text}")
    data = resp.json()
    # Response may be {"data": [{"access_token": ..., "user_id": ...}]}
    # or {"access_token": ..., "user_id": ...}
    if "data" in data and isinstance(data["data"], list) and data["data"]:
        entry = data["data"][0]
    else:
        entry = data
    return entry["access_token"], str(entry["user_id"])


async def exchange_for_long_lived_token(short_token: str) -> Tuple[str, int]:
    """Exchange a short-lived token for a 60-day long-lived token.

    Returns (long_lived_token, expires_in_seconds).
    """
    exchange_candidates = [
        (
            "GET",
            "https://graph.instagram.com/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
        ),
        # Some app/account combinations return a method error for GET;
        # retrying as POST keeps the flow resilient.
        (
            "POST",
            "https://graph.instagram.com/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
        ),
        (
            "GET",
            f"{GRAPH_API_BASE_IG}/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
        ),
        (
            "GET",
            "https://graph.instagram.com/v24.0/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
        ),
        (
            "POST",
            "https://graph.instagram.com/v24.0/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
        ),
    ]

    last_status = None
    last_body = ""
    async with httpx.AsyncClient(timeout=15) as client:
        for method, url, payload in exchange_candidates:
            if method == "POST":
                resp = await client.post(url, data=payload)
            else:
                resp = await client.get(url, params=payload)

            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if token:
                    return token, int(data.get("expires_in", 5184000))
                last_status = resp.status_code
                last_body = resp.text
                logger.warning(
                    "exchange_for_long_lived_token success response without access_token (%s %s): %s",
                    method,
                    url,
                    resp.text,
                )
                continue

            last_status = resp.status_code
            last_body = resp.text
            logger.warning(
                "exchange_for_long_lived_token attempt failed (%s %s): %s %s",
                method,
                url,
                resp.status_code,
                resp.text,
            )

    logger.error(
        "exchange_for_long_lived_token failed after fallbacks: %s %s",
        last_status,
        last_body,
    )
    raise RuntimeError(f"Long-lived token exchange failed: {last_body}")


async def refresh_long_lived_token(access_token: str) -> Tuple[str, int]:
    """Refresh a long-lived token for another 60 days.

    The token must be at least 24 hours old and not yet expired.
    Returns (new_token, expires_in_seconds).
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://graph.instagram.com/refresh_access_token",
            params={
                "grant_type": "ig_refresh_token",
                "access_token": access_token,
            },
        )
    if resp.status_code != 200:
        logger.error("refresh_long_lived_token failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"Token refresh failed: {resp.text}")
    data = resp.json()
    return data["access_token"], int(data.get("expires_in", 5184000))


async def subscribe_webhooks(ig_user_id: str, access_token: str) -> bool:
    """Programmatically subscribe the app to webhook events for this IG account.

    Calls POST /{ig_user_id}/subscribed_apps with the user's token.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_IG}/{ig_user_id}/subscribed_apps",
            params={
                "subscribed_fields": "messages,messaging_postbacks",
                "access_token": access_token,
            },
        )
    if resp.status_code != 200:
        logger.error("subscribe_webhooks failed: %s %s", resp.status_code, resp.text)
        return False
    result = resp.json()
    ok = result.get("success", False)
    if ok:
        logger.info("Webhook subscription successful for ig_user_id=%s", ig_user_id)
    else:
        logger.warning("Webhook subscription returned success=false for ig_user_id=%s: %s", ig_user_id, result)
    return ok


async def get_ig_account_info(access_token: str) -> dict:
    """Get the Instagram professional account info (username, name, etc.)."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GRAPH_API_BASE_IG}/me",
            params={
                "fields": "user_id,username,name,profile_picture_url,account_type",
                "access_token": access_token,
            },
        )
    if resp.status_code != 200:
        logger.error("get_ig_account_info failed: %s %s", resp.status_code, resp.text)
        return {}
    return resp.json()


async def get_ig_webhook_igsid(access_token: str) -> Optional[str]:
    """Resolve the IGSID (webhook recipient ID) for an Instagram account.

    The OAuth token exchange returns an app-scoped user ID, but webhooks
    deliver messages using a different IGSID (the older Instagram-scoped ID).

    We try two strategies:
    1. Call graph.instagram.com/me with ``id`` field — sometimes returns the IGSID
    2. Call graph.facebook.com/me — IGAA tokens may work here too

    Returns the IGSID string or None if it cannot be resolved.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        # Strategy 1: graph.instagram.com/me — check if 'id' differs from 'user_id'
        resp = await client.get(
            f"{GRAPH_API_BASE_IG}/me",
            params={"fields": "id,user_id,username", "access_token": access_token},
        )
        if resp.status_code == 200:
            data = resp.json()
            ig_id = str(data.get("id", ""))
            ig_user_id = str(data.get("user_id", ""))
            if ig_id and ig_id != ig_user_id:
                logger.info("get_ig_webhook_igsid: id=%s differs from user_id=%s (IG graph)", ig_id, ig_user_id)
                return ig_id

        # Strategy 2: graph.facebook.com/me
        resp2 = await client.get(
            f"{GRAPH_API_BASE_FB}/me",
            params={"fields": "id,name", "access_token": access_token},
        )
        if resp2.status_code == 200:
            fb_id = str(resp2.json().get("id", ""))
            if fb_id:
                logger.info("get_ig_webhook_igsid: resolved IGSID=%s via FB graph", fb_id)
                return fb_id

    logger.warning("get_ig_webhook_igsid: could not resolve IGSID")
    return None
