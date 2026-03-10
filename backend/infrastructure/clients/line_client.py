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
MAX_ASSET_CAROUSEL_CARDS = 6

# Menu flow (matches Instagram) — from config
from domain.platform_profiles import get_line_menu_quick_payload, get_line_menu_page_payload_prefix

LINE_MENU_QUICK_PAYLOAD = get_line_menu_quick_payload()
LINE_MENU_PAGE_PAYLOAD_PREFIX = get_line_menu_page_payload_prefix()

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


# ── Typing indicator ──────────────────────────────────────────────────


async def show_typing(user_id: str, access_token: str) -> None:
    """Show a loading animation (typing dots) in the LINE chat.

    Uses the /chat/loading/start endpoint. The animation automatically
    disappears when the bot sends a reply or after loadingSeconds elapses.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{LINE_API_BASE}/chat/loading/start",
                json={"chatId": user_id, "loadingSeconds": 60},
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
            )
    except Exception:
        pass  # Non-critical, don't block message handling


# ── Sending messages ──────────────────────────────────────────────────


def _text_message(text: str, quick_reply_items: Optional[List[dict]] = None) -> dict:
    formatted_text = format_for_messaging(text)
    msg: dict = {"type": "text", "text": formatted_text}
    if quick_reply_items:
        msg["quickReply"] = {
            "items": [
                {
                    "type": "action",
                    "action": {
                        "type": "message",
                        "label": (item.get("label") or item.get("title") or "")[:20],
                        "text": str(item.get("text") or item.get("payload") or "")[:1000],
                    },
                }
                for item in quick_reply_items[:13]
                if item.get("text") or item.get("payload")
            ]
        }
    return msg


def build_text_with_quick_replies(text: str, quick_reply_items: List[dict]) -> dict:
    """Build a text message with quick reply buttons (for menu 'View more' etc)."""
    return _text_message(text, quick_reply_items)


def build_buttons_template(alt_text: str, text: str, buttons: List[dict]) -> dict:
    """Build a LINE buttons template (e.g. View full menu)."""
    actions = []
    for btn in buttons[:3]:
        if btn.get("type") == "uri":
            actions.append({
                "type": "uri",
                "label": (btn.get("label") or "View")[:20],
                "uri": btn.get("uri", ""),
            })
    return {
        "type": "template",
        "altText": alt_text[:400],
        "template": {
            "type": "buttons",
            "text": format_for_messaging(text)[:160],
            "actions": actions,
        },
    }


def build_suggested_flex(suggested_messages: list) -> Optional[dict]:
    """Build a LINE Flex Message with vertically stacked tappable rows.

    Uses box components with text inside (no char limit on display) and
    a postback action on the box so taps can be resolved by suggestion id.
    """
    if not suggested_messages:
        return None
    rows = []
    for sm in suggested_messages[:10]:
        label = (sm.get("label") or "").strip()
        suggested_id = str(sm.get("id") or "").strip()
        if not label:
            continue
        rows.append({
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": label,
                    "size": "sm",
                    "color": "#555555",
                    "align": "center",
                    "wrap": True,
                },
            ],
            "action": {
                "type": "postback",
                "label": label[:20],
                "displayText": label,
                "data": f"lineux:suggest:{suggested_id or label[:20]}",
            },
            "paddingAll": "md",
            "cornerRadius": "md",
            "backgroundColor": "#F0F0F0",
            "margin": "sm",
        })
    if not rows:
        return None
    return {
        "type": "flex",
        "altText": "Suggested messages",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": rows,
                "paddingAll": "lg",
            },
        },
    }


def _is_valid_image_url(url: Optional[str]) -> bool:
    """LINE requires non-empty, https (or http) image URLs with a valid host."""
    if not url or not str(url).strip():
        return False
    s = str(url).strip().lower()
    if s.startswith("https://"):
        return len(s) > 8 and s[8] != "/"  # need host after "https://"
    if s.startswith("http://"):
        return len(s) > 7 and s[7] != "/"  # need host after "http://"
    return False


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
    suggested_flex: Optional[dict] = None,
) -> bool:
    """Reply to a webhook event using the reply token (free, no quota cost)."""

    # LINE allows max 5 message objects per reply.
    has_assets = bool(asset_cards)
    has_suggestions = suggested_flex is not None
    reserved_slots = (1 if has_assets else 0) + (1 if has_suggestions else 0)
    max_text_slots = max(1, 5 - reserved_slots)
    messages: List[dict] = [_text_message(t) for t in texts[:max_text_slots]]

    # 2. Asset Carousel (skip cards with invalid/empty image URLs — LINE rejects them)
    if asset_cards:
        bubbles = []
        valid_cards = [c for c in asset_cards[:MAX_ASSET_CAROUSEL_CARDS] if _is_valid_image_url(c.get("image_url"))]
        for card in valid_cards:
            bubbles.append(
                _create_image_bubble(
                    card.get("name", ""),
                    (card.get("image_url") or "").strip(),
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
            # Ensure we don't exceed 5 messages total.
            if len(messages) >= 5:
                messages = messages[:4]
            messages.append(carousel_message)

    # 3. Suggested messages as a separate flex message with vertical buttons.
    if suggested_flex and len(messages) < 5:
        messages.append(suggested_flex)

    logger.info(f"LINE reply_message: sending {len(messages)} messages (text + carousel)")
    return await _send_reply(reply_token, messages, access_token)


async def reply_with_messages(
    reply_token: str,
    messages: List[dict],
    access_token: str,
) -> bool:
    """Reply with a custom list of messages (e.g. for menu flow). Max 5 messages."""
    msgs = messages[:5]
    return await _send_reply(reply_token, msgs, access_token)


async def _send_reply(reply_token: str, messages: List[dict], access_token: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/message/reply",
            json={"replyToken": reply_token, "messages": messages},
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.error("LINE reply failed: %s %s", resp.status_code, resp.text)
        return False
    return True



async def push_message(
    user_id: str,
    texts: List[str],
    access_token: str,
    *,
    asset_cards: Optional[List[dict]] = None,
    suggested_flex: Optional[dict] = None,
) -> bool:
    """Push a message to a user proactively (costs message quota)."""

    has_assets = bool(asset_cards)
    has_suggestions = suggested_flex is not None
    reserved_slots = (1 if has_assets else 0) + (1 if has_suggestions else 0)
    max_text_slots = max(1, 5 - reserved_slots)
    messages: List[dict] = [_text_message(t) for t in texts[:max_text_slots]]

    if asset_cards:
        bubbles = []
        valid_cards = [c for c in asset_cards[:MAX_ASSET_CAROUSEL_CARDS] if _is_valid_image_url(c.get("image_url"))]
        for card in valid_cards:
            bubbles.append(
                _create_image_bubble(
                    card.get("name", ""),
                    (card.get("image_url") or "").strip(),
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

    # Suggested messages as a separate flex with vertical buttons
    if suggested_flex and len(messages) < 5:
        messages.append(suggested_flex)

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


async def create_rich_menu(rich_menu: dict, access_token: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/richmenu",
            json=rich_menu,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
    if resp.status_code != 200:
        logger.error("LINE create_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE rich menu create failed ({resp.status_code})")
    payload = resp.json()
    rich_menu_id = str(payload.get("richMenuId") or "").strip()
    if not rich_menu_id:
        raise RuntimeError("LINE rich menu create returned no richMenuId")
    return rich_menu_id


async def upload_rich_menu_image(rich_menu_id: str, image_bytes: bytes, access_token: str) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{LINE_API_DATA}/richmenu/{rich_menu_id}/content",
            content=image_bytes,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "image/png",
            },
        )
    if resp.status_code != 200:
        logger.error("LINE upload_rich_menu_image failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE rich menu image upload failed ({resp.status_code})")


async def set_default_rich_menu(rich_menu_id: str, access_token: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/user/all/richmenu/{rich_menu_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.error("LINE set_default_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE default rich menu link failed ({resp.status_code})")


async def clear_default_rich_menu(access_token: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(
            f"{LINE_API_BASE}/user/all/richmenu",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code not in (200, 204):
        logger.error("LINE clear_default_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE default rich menu unlink failed ({resp.status_code})")


async def link_user_rich_menu(user_id: str, rich_menu_id: str, access_token: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{LINE_API_BASE}/user/{user_id}/richmenu/{rich_menu_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        logger.error("LINE link_user_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE user rich menu link failed ({resp.status_code})")


async def unlink_user_rich_menu(user_id: str, access_token: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(
            f"{LINE_API_BASE}/user/{user_id}/richmenu",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code not in (200, 204):
        logger.error("LINE unlink_user_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE user rich menu unlink failed ({resp.status_code})")


async def delete_rich_menu(rich_menu_id: str, access_token: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(
            f"{LINE_API_BASE}/richmenu/{rich_menu_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code not in (200, 204):
        logger.error("LINE delete_rich_menu failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"LINE rich menu delete failed ({resp.status_code})")
