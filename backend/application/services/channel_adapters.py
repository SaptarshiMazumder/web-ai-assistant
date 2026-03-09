from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from infrastructure.clients.line_client import reply_message


@dataclass(frozen=True)
class ParsedLineEvent:
    event_type: str
    text: str
    reply_token: str
    user_id: str
    postback_data: Optional[str] = None


class LineChannelAdapter:
    channel_id = "line"

    def parse_event(self, raw_event: Dict[str, Any]) -> Optional[ParsedLineEvent]:
        event_type = str(raw_event.get("type") or "").strip().lower()
        source = raw_event.get("source") or {}
        user_id = str(source.get("userId") or "").strip()
        reply_token = str(raw_event.get("replyToken") or "").strip()
        if not user_id or not reply_token:
            return None
        if event_type == "message":
            message = raw_event.get("message") or {}
            if str(message.get("type") or "").strip() != "text":
                return None
            text = str(message.get("text") or "").strip()
            if not text:
                return None
            return ParsedLineEvent(event_type="message", text=text, reply_token=reply_token, user_id=user_id)
        if event_type == "postback":
            postback = raw_event.get("postback") or {}
            return ParsedLineEvent(
                event_type="postback",
                text=str(postback.get("displayText") or "").strip(),
                reply_token=reply_token,
                user_id=user_id,
                postback_data=str(postback.get("data") or "").strip() or None,
            )
        if event_type == "follow":
            return ParsedLineEvent(
                event_type="follow",
                text="",
                reply_token=reply_token,
                user_id=user_id,
            )
        return None

    def format_reply(self, texts: List[str]) -> List[str]:
        return [str(t or "") for t in texts if str(t or "").strip()]

    async def send_reply(
        self,
        *,
        reply_token: str,
        access_token: str,
        texts: List[str],
        asset_cards: Optional[List[Dict[str, str]]] = None,
        suggested_flex: Optional[dict] = None,
    ) -> bool:
        return await reply_message(
            reply_token,
            self.format_reply(texts),
            access_token,
            asset_cards=asset_cards,
            suggested_flex=suggested_flex,
        )
