"""Thin wrapper around the LINE Messaging API.

Uses raw httpx calls -- no line-bot-sdk dependency needed.
"""

import hashlib
import hmac
import base64
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_API_DATA = "https://api-data.line.me/v2/bot"

# ── Signature verification ────────────────────────────────────────────


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify LINE webhook signature (HMAC-SHA256)."""
    digest = hmac.new(
        channel_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# ── Sending messages ──────────────────────────────────────────────────


def _text_message(text: str) -> dict:
    return {"type": "text", "text": text}


def _flex_image_card(name: str, image_url: str, link_url: Optional[str] = None) -> dict:
    """Build a LINE Flex Message bubble with a hero image and title."""
    body_contents: List[dict] = [
        {"type": "text", "text": name, "weight": "bold", "size": "md", "wrap": True},
    ]
    hero: dict = {
        "type": "image",
        "url": image_url,
        "size": "full",
        "aspectRatio": "20:13",
        "aspectMode": "cover",
    }
    if link_url:
        hero["action"] = {"type": "uri", "label": name, "uri": link_url}
    bubble: dict = {
        "type": "bubble",
        "hero": hero,
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": body_contents,
        },
    }
    if link_url:
        bubble["footer"] = {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "button",
                    "action": {"type": "uri", "label": "View", "uri": link_url},
                    "style": "primary",
                    "height": "sm",
                }
            ],
        }
    return {
        "type": "flex",
        "altText": name,
        "contents": bubble,
    }


async def reply_message(
    reply_token: str,
    texts: List[str],
    access_token: str,
    *,
    asset_cards: Optional[List[dict]] = None,
) -> bool:
    """Reply to a webhook event using the reply token (free, no quota cost)."""
    messages: List[dict] = [_text_message(t) for t in texts[:5]]
    # Append asset flex cards (up to remaining slots; LINE allows max 5 per reply)
    if asset_cards:
        remaining = 5 - len(messages)
        logger.info(f"LINE reply_message: adding {len(asset_cards[:remaining])} asset cards")
        for card in asset_cards[:remaining]:
            flex_card = _flex_image_card(
                card.get("name", ""),
                card.get("image_url", ""),
                card.get("link_url") or None,
            )
            logger.info(f"LINE flex card: {flex_card}")
            messages.append(flex_card)
    
    logger.info(f"LINE reply_message: sending {len(messages)} total messages")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/message/reply",
            json={"replyToken": reply_token, "messages": messages},
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.error("LINE reply_message failed: %s %s", resp.status_code, resp.text)
        return False
    return True



async def push_message(
    user_id: str,
    texts: List[str],
    access_token: str,
    *,
    asset_cards: Optional[List[dict]] = None,
) -> bool:
    """Push a message to a user proactively (costs message quota)."""
    messages: List[dict] = [_text_message(t) for t in texts[:5]]
    if asset_cards:
        remaining = 5 - len(messages)
        for card in asset_cards[:remaining]:
            messages.append(
                _flex_image_card(
                    card.get("name", ""),
                    card.get("image_url", ""),
                    card.get("link_url") or None,
                )
            )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/message/push",
            json={"to": user_id, "messages": messages},
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.error("LINE push_message failed: %s %s", resp.status_code, resp.text)
        return False
    return True


async def get_profile(
    user_id: str,
    access_token: str,
) -> Optional[dict]:
    """Get LINE user profile (displayName, pictureUrl, etc.)."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{LINE_API_BASE}/profile/{user_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.warning("LINE get_profile failed: %s %s", resp.status_code, resp.text)
        return None
    return resp.json()
