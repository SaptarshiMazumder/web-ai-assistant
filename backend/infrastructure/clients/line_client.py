"""Thin wrapper around the LINE Messaging API.

Uses raw httpx calls -- no line-bot-sdk dependency needed.
"""

import hashlib
import hmac
import base64
import logging
from typing import List, Optional

import httpx

from application.utils.text_formatter import format_for_messaging

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


def _text_message(text: str, *, quick_reply: Optional[dict] = None) -> dict:
    formatted_text = format_for_messaging(text)
    msg: dict = {"type": "text", "text": formatted_text}
    if quick_reply:
        msg["quickReply"] = quick_reply
    return msg


def build_quick_reply(suggested_messages: list) -> Optional[dict]:
    """Convert widget suggestedMessages config into a LINE quickReply object.

    Each suggested message becomes a quick reply button that sends
    the label text back as a user message.
    """
    if not suggested_messages:
        return None
    items = []
    for sm in suggested_messages[:13]:  # LINE allows max 13 quick reply items
        label = (sm.get("label") or "").strip()
        if not label:
            continue
        # LINE quick reply labels max 20 chars
        display_label = label[:20]
        items.append({
            "type": "action",
            "action": {
                "type": "message",
                "label": display_label,
                "text": label,  # full label sent as user message
            },
        })
    if not items:
        return None
    return {"items": items}


def _create_image_bubble(name: str, image_url: str, link_url: Optional[str] = None) -> dict:
    """Build a LINE Flex Message bubble with a hero image."""
    # User requested:
    # 1. "nano is too small, revert to older sze" -> size="micro"
    # 2. "img should fill the carousel card completely" -> use 'hero' block (full bleed)
    # 3. "truncate the text size so that its not more than 2 lines max" -> maxLines=2, text size small
    
    hero: dict = {
        "type": "image",
        "url": image_url,
        "size": "full",
        "aspectRatio": "4:3",
        "aspectMode": "cover",
    }
    
    # Action on the bubble container so the whole card is clickable
    bubble_action = None
    if link_url:
        label = (name or "View")[:40]
        bubble_action = {"type": "uri", "label": label, "uri": link_url}

    body_contents: List[dict] = []
    if name:
        body_contents.append(
            {
                "type": "text", 
                "text": name, 
                "weight": "bold", 
                "size": "xs", # Keep text small
                "wrap": True,
                "maxLines": 2, # Truncate to 2 lines max
                "color": "#ffffff", # White text
            }
        )
    
    bubble: dict = {
        "type": "bubble",
        "size": "micro", # Reverted to micro (larger than nano)
        "hero": hero,
    }
    
    if bubble_action:
        bubble["action"] = bubble_action

    if body_contents:
        bubble["body"] = {
            "type": "box",
            "layout": "vertical",
            "contents": body_contents,
            "paddingAll": "sm", # Standard padding for text area
            "justifyContent": "center",
            "backgroundColor": "#333333", # Dark background
        }

    return bubble


async def reply_message(
    reply_token: str,
    texts: List[str],
    access_token: str,
    *,
    asset_cards: Optional[List[dict]] = None,
    quick_reply: Optional[dict] = None,
) -> bool:
    """Reply to a webhook event using the reply token (free, no quota cost)."""

    # 1. Text messages first (quick reply goes on the LAST text message)
    messages: List[dict] = [_text_message(t) for t in texts[:4]] # Leave room for 1 carousel if needed
    
    # 2. Asset Carousel
    if asset_cards:
        bubbles = []
        # Max bubbles in a carousel is 12
        for card in asset_cards[:12]: 
            bubbles.append(
                _create_image_bubble(
                    card.get("name", ""),
                    card.get("image_url", ""),
                    card.get("link_url") or None,
                )
            )
        
        if bubbles:
            carousel_message = {
                "type": "flex",
                "altText": "Images sent",
                "contents": {
                    "type": "carousel",
                    "contents": bubbles
                }
            }
            # Ensure we don't exceed 5 messages total
            if len(messages) >= 5:
                messages = messages[:4]
            messages.append(carousel_message)
    
    # Attach quick reply buttons to the last message
    if quick_reply and messages:
        messages[-1]["quickReply"] = quick_reply

    logger.info(f"LINE reply_message: sending {len(messages)} messages (text + carousel)")
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
    quick_reply: Optional[dict] = None,
) -> bool:
    """Push a message to a user proactively (costs message quota)."""

    messages: List[dict] = [_text_message(t) for t in texts[:4]]
    
    if asset_cards:
        bubbles = []
        for card in asset_cards[:12]:
            bubbles.append(
                _create_image_bubble(
                    card.get("name", ""),
                    card.get("image_url", ""),
                    card.get("link_url") or None,
                )
            )
            
        if bubbles:
            carousel_message = {
                "type": "flex",
                "altText": "Images sent",
                "contents": {
                    "type": "carousel",
                    "contents": bubbles
                }
            }
            if len(messages) >= 5:
                messages = messages[:4]
            messages.append(carousel_message)

    # Attach quick reply buttons to the last message
    if quick_reply and messages:
        messages[-1]["quickReply"] = quick_reply

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
