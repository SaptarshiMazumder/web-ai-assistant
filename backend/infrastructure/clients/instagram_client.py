"""Thin wrapper around Meta's Graph API for Instagram Messaging.

Uses raw httpx calls -- no facebook-sdk dependency needed.
"""

import hashlib
import hmac
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE_FB = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
GRAPH_API_BASE_IG = f"https://graph.instagram.com/{GRAPH_API_VERSION}"


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
