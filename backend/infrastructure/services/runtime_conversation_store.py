from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from redis import Redis

from common.config import config
from domain.entities import (
    ConversationChannelContact,
    ConversationMessage,
    ConversationSession,
    ConversationSessionHandoff,
    EscalationRecord,
    InstagramUserSession,
    LineUserSession,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_conversation_id() -> str:
    return "conv_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_message_id() -> str:
    return "msg_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_contact_id() -> str:
    return "cc_" + secrets.token_urlsafe(18).replace("-", "_").replace(".", "_")


def _new_handoff_id() -> str:
    return "hof_" + secrets.token_urlsafe(18).replace("-", "_").replace(".", "_")


def _new_escalation_id() -> str:
    return "esc_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_event_id() -> str:
    return "evt_" + secrets.token_urlsafe(18).replace("-", "_").replace(".", "_")


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_loads(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


class RuntimeConversationStore:
    def __init__(self) -> None:
        self._logger = logging.getLogger("runtime_conversation_store")
        self._trace_storage = bool(getattr(config, "CHAT_STORAGE_TRACE_LOGS", False))
        self._redis_url = (
            (getattr(config, "REDIS_URL", "") or "").strip()
            or (getattr(config, "CELERY_BROKER_URL", "") or "").strip()
        )
        self._redis: Optional[Redis] = None
        self._last_ping_ok = False
        self._last_ping_ts = 0.0
        self._last_ping_error = ""
        self._ping_ttl_seconds = 2.0
        self._ns = (getattr(config, "CHAT_RUNTIME_REDIS_NAMESPACE", "") or "webai:chatruntime:v1").strip()
        ttl_minutes = int(getattr(config, "CONVERSATION_SESSION_TTL_MINUTES", 30))
        self._session_ttl_seconds = max(900, min(ttl_minutes * 60, 7 * 24 * 60 * 60))
        self._history_max = int(getattr(config, "CHAT_RUNTIME_HISTORY_MAX_MESSAGES", 100))
        self._history_max = max(20, min(self._history_max, 300))
        self._lock_ttl = int(getattr(config, "CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS", 30))
        self._lock_ttl = max(5, min(self._lock_ttl, 300))
        self._persist_stream_key = (
            getattr(config, "CHAT_RUNTIME_PERSIST_STREAM_KEY", "") or "webai:chatruntime:persist:v1"
        ).strip()
        self._persist_stream_maxlen = int(getattr(config, "CHAT_RUNTIME_PERSIST_STREAM_MAXLEN", 200000))
        self._persist_stream_maxlen = max(1000, min(self._persist_stream_maxlen, 2_000_000))

    def _trace_log(self, message: str, *args: object) -> None:
        if self._trace_storage:
            self._logger.warning(message, *args)

    def _key(self, *parts: str) -> str:
        clean = [str(p or "").strip() for p in parts if str(p or "").strip()]
        return f"{self._ns}:{':'.join(clean)}"

    def _redis_client(self) -> Optional[Redis]:
        if not self._redis_url:
            return None
        if self._redis is not None:
            return self._redis
        try:
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)
            return self._redis
        except Exception as exc:
            self._logger.warning("Failed to create Redis client for chat runtime: %s", exc)
            self._redis = None
            return None

    def is_available(self, *, force: bool = False) -> bool:
        redis = self._redis_client()
        if redis is None:
            return False
        now = time.time()
        if not force and (now - self._last_ping_ts) < self._ping_ttl_seconds:
            return self._last_ping_ok
        try:
            redis.ping()
            self._last_ping_ok = True
            self._last_ping_error = ""
        except Exception as exc:
            self._last_ping_ok = False
            self._last_ping_error = str(exc)
            self._logger.warning("Redis runtime store unavailable: %s", exc)
            self._redis = None
        finally:
            self._last_ping_ts = now
        return self._last_ping_ok

    def last_ping_error(self) -> str:
        return self._last_ping_error

    def _session_key(self, session_id: str) -> str:
        return self._key("session", session_id)

    def _history_key(self, session_id: str) -> str:
        return self._key("history", session_id)

    def _pending_key(self, session_id: str) -> str:
        return self._key("pending", session_id)

    def _seq_key(self, session_id: str) -> str:
        return self._key("seq", session_id)

    def _lock_key(self, session_id: str) -> str:
        return self._key("turn_lock", session_id)

    def _contact_key(self, *, channel: str, bot_id: str, external_user_id: str) -> str:
        return self._key("contact", channel, bot_id, external_user_id)

    def _line_user_key(self, *, bot_id: str, line_user_id: str) -> str:
        return self._key("line_user", bot_id, line_user_id)

    def _line_sid_key(self, *, session_id: str) -> str:
        return self._key("line_sid", session_id)

    def _ig_user_key(self, *, bot_id: str, ig_user_id: str) -> str:
        return self._key("ig_user", bot_id, ig_user_id)

    def _ig_sid_key(self, *, session_id: str) -> str:
        return self._key("ig_sid", session_id)

    def _handoff_key(self, session_id: str) -> str:
        return self._key("handoff", session_id)

    def _escalation_key(self, escalation_id: str) -> str:
        return self._key("escalation", escalation_id)

    def _session_latest_escalation_key(self, session_id: str) -> str:
        return self._key("session_latest_escalation", session_id)

    def _save_json(self, key: str, payload: Dict[str, Any], *, ttl_seconds: Optional[int] = None) -> None:
        redis = self._redis_client()
        if redis is None:
            return
        ttl = int(ttl_seconds or self._session_ttl_seconds)
        redis.setex(key, max(1, ttl), _json_dumps(payload))

    def _load_json_dict(self, key: str) -> Optional[Dict[str, Any]]:
        redis = self._redis_client()
        if redis is None:
            return None
        raw = redis.get(key)
        data = _json_loads(raw)
        if isinstance(data, dict):
            return data
        return None

    def _session_from_payload(self, payload: Dict[str, Any]) -> Optional[ConversationSession]:
        if not payload:
            return None
        try:
            return ConversationSession(
                session_id=str(payload.get("session_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                org_id=str(payload.get("org_id") or ""),
                channel=str(payload.get("channel") or ""),
                status=str(payload.get("status") or "active"),
                title=(str(payload.get("title")) if payload.get("title") is not None else None),
                site_url=(str(payload.get("site_url")) if payload.get("site_url") is not None else None),
                site_title=(str(payload.get("site_title")) if payload.get("site_title") is not None else None),
                message_count=int(payload.get("message_count") or 0),
                started_at=str(payload.get("started_at") or ""),
                last_active_at=str(payload.get("last_active_at") or ""),
                ended_at=(str(payload.get("ended_at")) if payload.get("ended_at") is not None else None),
                user_agent=(str(payload.get("user_agent")) if payload.get("user_agent") is not None else None),
                ip=(str(payload.get("ip")) if payload.get("ip") is not None else None),
                assistant_state=str(payload.get("assistant_state") or "bot"),
                handoff_active=bool(payload.get("handoff_active")),
                support_request_id=(
                    str(payload.get("support_request_id")) if payload.get("support_request_id") is not None else None
                ),
                support_request_status=(
                    str(payload.get("support_request_status"))
                    if payload.get("support_request_status") is not None
                    else None
                ),
            )
        except Exception:
            return None

    def _contact_from_payload(self, payload: Dict[str, Any]) -> Optional[ConversationChannelContact]:
        if not payload:
            return None
        try:
            return ConversationChannelContact(
                contact_id=str(payload.get("contact_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                channel=str(payload.get("channel") or ""),
                external_user_id=str(payload.get("external_user_id") or ""),
                display_name=(str(payload.get("display_name")) if payload.get("display_name") is not None else None),
                current_session_id=(
                    str(payload.get("current_session_id")) if payload.get("current_session_id") is not None else None
                ),
                created_at=str(payload.get("created_at") or ""),
                updated_at=str(payload.get("updated_at") or ""),
                metadata=_as_dict(payload.get("metadata")),
            )
        except Exception:
            return None

    def _message_from_payload(self, payload: Dict[str, Any]) -> Optional[ConversationMessage]:
        if not payload:
            return None
        try:
            return ConversationMessage(
                message_id=str(payload.get("message_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                role=str(payload.get("role") or ""),
                content=str(payload.get("content") or ""),
                sender_name=(str(payload.get("sender_name")) if payload.get("sender_name") is not None else None),
                citations=[x for x in _as_list(payload.get("citations")) if isinstance(x, dict)],
                created_at=str(payload.get("created_at") or ""),
            )
        except Exception:
            return None

    def _line_mapping_from_payload(self, payload: Dict[str, Any]) -> Optional[LineUserSession]:
        if not payload:
            return None
        try:
            return LineUserSession(
                line_user_id=str(payload.get("line_user_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                is_escalated=bool(payload.get("is_escalated")),
                created_at=str(payload.get("created_at") or ""),
                updated_at=str(payload.get("updated_at") or ""),
                awaiting_escalation_msg=bool(payload.get("awaiting_escalation_msg")),
                awaiting_staff_takeover=bool(payload.get("awaiting_staff_takeover")),
                display_name=(str(payload.get("display_name")) if payload.get("display_name") is not None else None),
            )
        except Exception:
            return None

    def _ig_mapping_from_payload(self, payload: Dict[str, Any]) -> Optional[InstagramUserSession]:
        if not payload:
            return None
        try:
            return InstagramUserSession(
                ig_user_id=str(payload.get("ig_user_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                is_escalated=bool(payload.get("is_escalated")),
                created_at=str(payload.get("created_at") or ""),
                updated_at=str(payload.get("updated_at") or ""),
                awaiting_escalation_msg=bool(payload.get("awaiting_escalation_msg")),
                awaiting_staff_takeover=bool(payload.get("awaiting_staff_takeover")),
            )
        except Exception:
            return None

    def _handoff_from_payload(self, payload: Dict[str, Any]) -> Optional[ConversationSessionHandoff]:
        if not payload:
            return None
        try:
            return ConversationSessionHandoff(
                handoff_id=str(payload.get("handoff_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                contact_id=(str(payload.get("contact_id")) if payload.get("contact_id") is not None else None),
                source_channel=(str(payload.get("source_channel")) if payload.get("source_channel") is not None else None),
                assistant_state=str(payload.get("assistant_state") or "bot"),
                support_request_id=(
                    str(payload.get("support_request_id")) if payload.get("support_request_id") is not None else None
                ),
                started_by=(str(payload.get("started_by")) if payload.get("started_by") is not None else None),
                ended_reason=(str(payload.get("ended_reason")) if payload.get("ended_reason") is not None else None),
                created_at=str(payload.get("created_at") or ""),
                updated_at=str(payload.get("updated_at") or ""),
                ended_at=(str(payload.get("ended_at")) if payload.get("ended_at") is not None else None),
            )
        except Exception:
            return None

    def _escalation_from_payload(self, payload: Dict[str, Any]) -> Optional[EscalationRecord]:
        if not payload:
            return None
        try:
            return EscalationRecord(
                escalation_id=str(payload.get("escalation_id") or ""),
                bot_id=str(payload.get("bot_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                visitor_email=str(payload.get("visitor_email") or ""),
                details=(str(payload.get("details")) if payload.get("details") is not None else None),
                created_at=str(payload.get("created_at") or ""),
                status=str(payload.get("status") or "open"),
                session_title=(str(payload.get("session_title")) if payload.get("session_title") is not None else None),
                site_url=(str(payload.get("site_url")) if payload.get("site_url") is not None else None),
                site_title=(str(payload.get("site_title")) if payload.get("site_title") is not None else None),
                last_active_at=(str(payload.get("last_active_at")) if payload.get("last_active_at") is not None else None),
                session_status=(str(payload.get("session_status")) if payload.get("session_status") is not None else None),
                visitor_name=(str(payload.get("visitor_name")) if payload.get("visitor_name") is not None else None),
                linked_session_id=(
                    str(payload.get("linked_session_id")) if payload.get("linked_session_id") is not None else None
                ),
                notification_read_at=(
                    str(payload.get("notification_read_at"))
                    if payload.get("notification_read_at") is not None
                    else None
                ),
                notification_is_unread=bool(payload.get("notification_is_unread")),
            )
        except Exception:
            return None

    def _session_expired(self, session: ConversationSession) -> bool:
        try:
            last = datetime.fromisoformat(session.last_active_at)
        except Exception:
            return False
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        ttl_minutes = int(getattr(config, "CONVERSATION_SESSION_TTL_MINUTES", 30))
        return datetime.now(timezone.utc) - last > timedelta(minutes=ttl_minutes)

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        payload = self._load_json_dict(self._session_key(sid))
        return self._session_from_payload(payload or {})

    def _save_session(self, session: ConversationSession) -> None:
        self._save_json(self._session_key(session.session_id), asdict(session))
        self._trace_log(
            "chat_storage redis_session_upsert session_id=%s bot_id=%s channel=%s message_count=%s",
            session.session_id,
            session.bot_id,
            session.channel,
            session.message_count,
        )

    def _next_seq(self, session_id: str) -> int:
        sid = (session_id or "").strip()
        if not sid:
            return 0
        redis = self._redis_client()
        if redis is None:
            return 0
        key = self._seq_key(sid)
        seq = int(redis.incr(key))
        redis.expire(key, self._session_ttl_seconds)
        return seq

    def _queue_event(
        self,
        *,
        event_type: str,
        session_id: str,
        channel: Optional[str],
        bot_id: Optional[str],
        org_id: Optional[str],
        role: Optional[str] = None,
        content: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        redis = self._redis_client()
        if redis is None:
            return
        event_id = _new_event_id()
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "session_id": sid,
            "channel": (channel or "").strip(),
            "bot_id": (bot_id or "").strip(),
            "org_id": (org_id or "").strip(),
            "seq": self._next_seq(sid),
            "role": (role or "").strip(),
            "content": content or "",
            "citations": citations or [],
            "transport_sent_at": "",
            "created_at": _utc_now(),
            "idempotency_key": (idempotency_key or event_id).strip(),
            "payload": payload or {},
            "retry_count": 0,
        }
        pending_key = self._pending_key(sid)
        redis.rpush(pending_key, _json_dumps(event))
        redis.expire(pending_key, self._session_ttl_seconds)
        self._trace_log(
            "chat_storage redis_pending_event event_id=%s event_type=%s session_id=%s seq=%s pending_key=%s",
            event_id,
            event_type,
            sid,
            event.get("seq", ""),
            pending_key,
        )

    def get_or_create_session(
        self,
        *,
        bot_id: str,
        org_id: str,
        channel: str,
        session_id: Optional[str],
        site_url: Optional[str],
        site_title: Optional[str],
        user_agent: Optional[str],
        ip: Optional[str],
    ) -> ConversationSession:
        bid = (bot_id or "").strip()
        oid = (org_id or "").strip()
        ch = (channel or "").strip().lower()
        sid = (session_id or "").strip()
        if not bid or not oid or not ch:
            raise ValueError("bot_id, org_id and channel are required")

        existing = self.get_session(sid) if sid else None
        if existing and existing.bot_id == bid and existing.status == "active" and not self._session_expired(existing):
            now = _utc_now()
            existing.last_active_at = now
            if site_url:
                existing.site_url = site_url
            if site_title:
                existing.site_title = site_title
            if user_agent:
                existing.user_agent = user_agent
            if ip:
                existing.ip = ip
            self._save_session(existing)
            self._queue_event(
                event_type="session_upsert",
                session_id=existing.session_id,
                channel=existing.channel,
                bot_id=existing.bot_id,
                org_id=existing.org_id,
                payload={"session": asdict(existing)},
            )
            return existing

        now = _utc_now()
        next_sid = _new_conversation_id()
        session = ConversationSession(
            session_id=next_sid,
            bot_id=bid,
            org_id=oid,
            channel=ch,
            status="active",
            title=None,
            site_url=site_url,
            site_title=site_title,
            message_count=0,
            started_at=now,
            last_active_at=now,
            ended_at=None,
            user_agent=user_agent,
            ip=ip,
            assistant_state="bot",
            handoff_active=False,
            support_request_id=None,
            support_request_status=None,
        )
        self._save_session(session)
        self._queue_event(
            event_type="session_upsert",
            session_id=session.session_id,
            channel=ch,
            bot_id=bid,
            org_id=oid,
            payload={"session": asdict(session)},
        )
        return session

    def list_recent_messages(self, session_id: str, *, limit: int) -> List[ConversationMessage]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        lim = max(1, min(int(limit or 20), 200))
        redis = self._redis_client()
        if redis is None:
            self._trace_log(
                "chat_storage redis_history_read session_id=%s limit=%s returned=0 redis_available=0",
                sid,
                lim,
            )
            return []
        history_key = self._history_key(sid)
        items = redis.lrange(history_key, -lim, -1)
        out: List[ConversationMessage] = []
        for raw in items or []:
            payload = _json_loads(raw)
            if not isinstance(payload, dict):
                continue
            msg = self._message_from_payload(payload)
            if msg is not None:
                out.append(msg)
        self._trace_log(
            "chat_storage redis_history_read session_id=%s limit=%s returned=%s redis_available=1 history_key=%s",
            sid,
            lim,
            len(out),
            history_key,
        )
        return out

    def add_message(
        self,
        *,
        session_id: str,
        bot_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]],
        sender_name: Optional[str],
    ) -> ConversationMessage:
        sid = (session_id or "").strip()
        bid = (bot_id or "").strip()
        if not sid or not bid:
            raise ValueError("session_id and bot_id are required")
        session = self.get_session(sid)
        if session is None:
            raise ValueError("unknown session_id for runtime message append")
        now = _utc_now()
        message = ConversationMessage(
            message_id=_new_message_id(),
            session_id=sid,
            bot_id=bid,
            role=(role or "").strip(),
            sender_name=(sender_name or "").strip() or None,
            content=content or "",
            citations=[c for c in (citations or []) if isinstance(c, dict)],
            created_at=now,
        )
        redis = self._redis_client()
        if redis is None:
            return message
        history_key = self._history_key(sid)
        with redis.pipeline(transaction=False) as pipe:
            pipe.rpush(history_key, _json_dumps(asdict(message)))
            pipe.ltrim(history_key, -self._history_max, -1)
            pipe.expire(history_key, self._session_ttl_seconds)
            pipe.execute()
        self._trace_log(
            "chat_storage redis_history_append message_id=%s session_id=%s bot_id=%s role=%s history_key=%s",
            message.message_id,
            sid,
            bid,
            message.role,
            history_key,
        )

        session.message_count = int(session.message_count or 0) + 1
        session.last_active_at = now
        if (session.title or "").strip() == "" and message.role == "user" and (message.content or "").strip():
            session.title = message.content
        self._save_session(session)
        self._queue_event(
            event_type="message_upsert",
            session_id=sid,
            channel=session.channel,
            bot_id=session.bot_id,
            org_id=session.org_id,
            role=message.role,
            content=message.content,
            citations=message.citations,
            payload={
                "message": asdict(message),
                "session": asdict(session),
            },
            idempotency_key=message.message_id,
        )
        return message

    def get_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
    ) -> Optional[ConversationChannelContact]:
        bid = (bot_id or "").strip()
        ch = (channel or "").strip().lower()
        uid = (external_user_id or "").strip()
        if not bid or not ch or not uid:
            return None
        payload = self._load_json_dict(self._contact_key(channel=ch, bot_id=bid, external_user_id=uid))
        return self._contact_from_payload(payload or {})

    def upsert_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
        current_session_id: Optional[str],
        display_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> ConversationChannelContact:
        bid = (bot_id or "").strip()
        ch = (channel or "").strip().lower()
        uid = (external_user_id or "").strip()
        if not bid or not ch or not uid:
            raise ValueError("bot_id, channel and external_user_id are required")

        existing = self.get_channel_contact(bot_id=bid, channel=ch, external_user_id=uid)
        now = _utc_now()
        contact = ConversationChannelContact(
            contact_id=(existing.contact_id if existing else _new_contact_id()),
            bot_id=bid,
            channel=ch,
            external_user_id=uid,
            display_name=((display_name or "").strip() or (existing.display_name if existing else None)),
            current_session_id=((current_session_id or "").strip() or (existing.current_session_id if existing else None)),
            created_at=(existing.created_at if existing else now),
            updated_at=now,
            metadata={**(existing.metadata if existing else {}), **_as_dict(metadata)},
        )
        self._save_json(self._contact_key(channel=ch, bot_id=bid, external_user_id=uid), asdict(contact))
        session = self.get_session(contact.current_session_id or "")
        self._queue_event(
            event_type="contact_upsert",
            session_id=contact.current_session_id or "",
            channel=ch,
            bot_id=bid,
            org_id=(session.org_id if session else ""),
            payload={"contact": asdict(contact)},
            idempotency_key=f"contact:{bid}:{ch}:{uid}:{contact.updated_at}",
        )
        return contact

    def resolve_or_create_channel_session(
        self,
        *,
        bot_id: str,
        org_id: str,
        channel: str,
        external_user_id: str,
        current_session_id: Optional[str],
        display_name: Optional[str],
        metadata: Optional[Dict[str, str]],
        site_url: Optional[str],
        site_title: Optional[str],
        user_agent: Optional[str],
        ip: Optional[str],
    ) -> Tuple[ConversationSession, ConversationChannelContact, bool]:
        existing_contact = self.get_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
        )
        seed_sid = (current_session_id or "").strip() or (
            (existing_contact.current_session_id or "").strip() if existing_contact else ""
        )
        session = self.get_or_create_session(
            bot_id=bot_id,
            org_id=org_id,
            channel=channel,
            session_id=seed_sid or None,
            site_url=site_url,
            site_title=site_title,
            user_agent=user_agent,
            ip=ip,
        )
        contact = self.upsert_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
            current_session_id=session.session_id,
            display_name=display_name,
            metadata=metadata,
        )
        changed = bool(seed_sid and seed_sid != session.session_id)
        return session, contact, changed

    def set_session_title(self, session_id: str, title: Optional[str]) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        session = self.get_session(sid)
        if session is None:
            return False
        next_title = (title or "").strip()
        if not next_title:
            return False
        session.title = next_title
        session.last_active_at = _utc_now()
        self._save_session(session)
        self._queue_event(
            event_type="session_upsert",
            session_id=session.session_id,
            channel=session.channel,
            bot_id=session.bot_id,
            org_id=session.org_id,
            payload={"session": asdict(session)},
        )
        return True

    def _save_handoff(self, handoff: ConversationSessionHandoff) -> None:
        self._save_json(self._handoff_key(handoff.session_id), asdict(handoff))

    def get_session_handoff(self, session_id: str) -> Optional[ConversationSessionHandoff]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        payload = self._load_json_dict(self._handoff_key(sid))
        return self._handoff_from_payload(payload or {})

    def upsert_handoff(
        self,
        *,
        bot_id: str,
        session_id: str,
        assistant_state: str,
        contact_id: Optional[str],
        source_channel: Optional[str],
        support_request_id: Optional[str],
        started_by: Optional[str],
        ended_reason: Optional[str],
        ended_at: Optional[str],
    ) -> ConversationSessionHandoff:
        sid = (session_id or "").strip()
        bid = (bot_id or "").strip()
        if not sid or not bid:
            raise ValueError("bot_id and session_id are required")
        now = _utc_now()
        existing = self.get_session_handoff(sid)
        handoff = ConversationSessionHandoff(
            handoff_id=(existing.handoff_id if existing else _new_handoff_id()),
            bot_id=bid,
            session_id=sid,
            contact_id=contact_id if contact_id is not None else (existing.contact_id if existing else None),
            source_channel=(
                source_channel if source_channel is not None else (existing.source_channel if existing else None)
            ),
            assistant_state=(assistant_state or "bot").strip() or "bot",
            support_request_id=(
                support_request_id if support_request_id is not None else (existing.support_request_id if existing else None)
            ),
            started_by=started_by if started_by is not None else (existing.started_by if existing else None),
            ended_reason=ended_reason,
            created_at=(existing.created_at if existing else now),
            updated_at=now,
            ended_at=ended_at,
        )
        self._save_handoff(handoff)

        session = self.get_session(sid)
        if session is not None:
            session.assistant_state = handoff.assistant_state
            session.handoff_active = handoff.assistant_state in {"awaiting_support_details", "human_handoff"}
            session.support_request_id = handoff.support_request_id
            if handoff.assistant_state == "bot":
                if handoff.ended_reason:
                    session.support_request_status = handoff.ended_reason
            else:
                session.support_request_status = "open" if handoff.support_request_id else session.support_request_status
            session.last_active_at = now
            self._save_session(session)
            org_id = session.org_id
            channel = session.channel
        else:
            org_id = ""
            channel = source_channel or ""

        self._queue_event(
            event_type="handoff_upsert",
            session_id=sid,
            channel=channel,
            bot_id=bid,
            org_id=org_id,
            payload={"handoff": asdict(handoff)},
            idempotency_key=f"handoff:{sid}:{now}",
        )
        if session is not None:
            self._queue_event(
                event_type="session_upsert",
                session_id=sid,
                channel=session.channel,
                bot_id=session.bot_id,
                org_id=session.org_id,
                payload={"session": asdict(session)},
                idempotency_key=f"session:{sid}:{now}",
            )
        return handoff

    def create_escalation(
        self,
        *,
        bot_id: str,
        session_id: str,
        visitor_email: str,
        details: Optional[str],
        status: str = "open",
    ) -> EscalationRecord:
        sid = (session_id or "").strip()
        bid = (bot_id or "").strip()
        if not sid or not bid:
            raise ValueError("bot_id and session_id are required")
        now = _utc_now()
        record = EscalationRecord(
            escalation_id=_new_escalation_id(),
            bot_id=bid,
            session_id=sid,
            visitor_email=(visitor_email or "").strip(),
            details=(details or "").strip() or None,
            created_at=now,
            status=(status or "open").strip() or "open",
        )
        self._save_json(self._escalation_key(record.escalation_id), asdict(record))
        self._save_json(
            self._session_latest_escalation_key(sid),
            {"escalation_id": record.escalation_id, "updated_at": now},
        )
        session = self.get_session(sid)
        self._queue_event(
            event_type="escalation_upsert",
            session_id=sid,
            channel=(session.channel if session else ""),
            bot_id=bid,
            org_id=(session.org_id if session else ""),
            payload={"escalation": asdict(record)},
            idempotency_key=record.escalation_id,
        )
        return record

    def get_escalation(self, escalation_id: str) -> Optional[EscalationRecord]:
        eid = (escalation_id or "").strip()
        if not eid:
            return None
        payload = self._load_json_dict(self._escalation_key(eid))
        return self._escalation_from_payload(payload or {})

    def get_latest_escalation_for_session(self, session_id: str) -> Optional[EscalationRecord]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        ref = self._load_json_dict(self._session_latest_escalation_key(sid)) or {}
        escalation_id = (str(ref.get("escalation_id") or "")).strip()
        if not escalation_id:
            return None
        return self.get_escalation(escalation_id)

    def update_escalation_status(self, escalation_id: str, status: str) -> Optional[EscalationRecord]:
        record = self.get_escalation(escalation_id)
        if record is None:
            return None
        record.status = (status or "").strip() or record.status
        self._save_json(self._escalation_key(record.escalation_id), asdict(record))
        session = self.get_session(record.session_id)
        self._queue_event(
            event_type="escalation_upsert",
            session_id=record.session_id,
            channel=(session.channel if session else ""),
            bot_id=record.bot_id,
            org_id=(session.org_id if session else ""),
            payload={"escalation": asdict(record)},
            idempotency_key=f"{record.escalation_id}:{record.status}",
        )
        return record

    def get_line_user_session(self, *, line_user_id: str, bot_id: str) -> Optional[LineUserSession]:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return None
        payload = self._load_json_dict(self._line_user_key(bot_id=bid, line_user_id=uid))
        return self._line_mapping_from_payload(payload or {})

    def get_line_user_session_by_session_id(self, session_id: str) -> Optional[LineUserSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        payload = self._load_json_dict(self._line_sid_key(session_id=sid))
        return self._line_mapping_from_payload(payload or {})

    def upsert_line_user_session(
        self,
        *,
        line_user_id: str,
        bot_id: str,
        session_id: str,
        is_escalated: Optional[bool] = None,
        awaiting_escalation_msg: Optional[bool] = None,
        awaiting_staff_takeover: Optional[bool] = None,
        display_name: Optional[str] = None,
    ) -> Optional[LineUserSession]:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        if not uid or not bid or not sid:
            return None
        existing = self.get_line_user_session(line_user_id=uid, bot_id=bid)
        now = _utc_now()
        mapping = LineUserSession(
            line_user_id=uid,
            bot_id=bid,
            session_id=sid,
            is_escalated=bool(is_escalated) if is_escalated is not None else bool(existing.is_escalated if existing else False),
            created_at=(existing.created_at if existing else now),
            updated_at=now,
            awaiting_escalation_msg=(
                bool(awaiting_escalation_msg)
                if awaiting_escalation_msg is not None
                else bool(existing.awaiting_escalation_msg if existing else False)
            ),
            awaiting_staff_takeover=(
                bool(awaiting_staff_takeover)
                if awaiting_staff_takeover is not None
                else bool(existing.awaiting_staff_takeover if existing else False)
            ),
            display_name=(
                (display_name or "").strip() or (existing.display_name if existing else None)
            ),
        )
        self._save_json(self._line_user_key(bot_id=bid, line_user_id=uid), asdict(mapping))
        self._save_json(self._line_sid_key(session_id=sid), asdict(mapping))
        if existing and existing.session_id and existing.session_id != sid:
            redis = self._redis_client()
            if redis is not None:
                redis.delete(self._line_sid_key(session_id=existing.session_id))
        session = self.get_session(sid)
        self._queue_event(
            event_type="line_user_session_upsert",
            session_id=sid,
            channel="line",
            bot_id=bid,
            org_id=(session.org_id if session else ""),
            payload={"line_user_session": asdict(mapping)},
            idempotency_key=f"line:{bid}:{uid}:{now}",
        )
        return mapping

    def get_instagram_user_session(self, *, ig_user_id: str, bot_id: str) -> Optional[InstagramUserSession]:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return None
        payload = self._load_json_dict(self._ig_user_key(bot_id=bid, ig_user_id=uid))
        return self._ig_mapping_from_payload(payload or {})

    def get_instagram_user_session_by_session_id(self, session_id: str) -> Optional[InstagramUserSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        payload = self._load_json_dict(self._ig_sid_key(session_id=sid))
        return self._ig_mapping_from_payload(payload or {})

    def upsert_instagram_user_session(
        self,
        *,
        ig_user_id: str,
        bot_id: str,
        session_id: str,
        is_escalated: Optional[bool] = None,
        awaiting_escalation_msg: Optional[bool] = None,
        awaiting_staff_takeover: Optional[bool] = None,
    ) -> Optional[InstagramUserSession]:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        if not uid or not bid or not sid:
            return None
        existing = self.get_instagram_user_session(ig_user_id=uid, bot_id=bid)
        now = _utc_now()
        mapping = InstagramUserSession(
            ig_user_id=uid,
            bot_id=bid,
            session_id=sid,
            is_escalated=bool(is_escalated) if is_escalated is not None else bool(existing.is_escalated if existing else False),
            created_at=(existing.created_at if existing else now),
            updated_at=now,
            awaiting_escalation_msg=(
                bool(awaiting_escalation_msg)
                if awaiting_escalation_msg is not None
                else bool(existing.awaiting_escalation_msg if existing else False)
            ),
            awaiting_staff_takeover=(
                bool(awaiting_staff_takeover)
                if awaiting_staff_takeover is not None
                else bool(existing.awaiting_staff_takeover if existing else False)
            ),
        )
        self._save_json(self._ig_user_key(bot_id=bid, ig_user_id=uid), asdict(mapping))
        self._save_json(self._ig_sid_key(session_id=sid), asdict(mapping))
        if existing and existing.session_id and existing.session_id != sid:
            redis = self._redis_client()
            if redis is not None:
                redis.delete(self._ig_sid_key(session_id=existing.session_id))
        session = self.get_session(sid)
        self._queue_event(
            event_type="instagram_user_session_upsert",
            session_id=sid,
            channel="instagram",
            bot_id=bid,
            org_id=(session.org_id if session else ""),
            payload={"instagram_user_session": asdict(mapping)},
            idempotency_key=f"instagram:{bid}:{uid}:{now}",
        )
        return mapping

    def acquire_turn_lock(
        self,
        *,
        session_id: str,
        owner: str,
        wait_timeout_ms: int = 8000,
    ) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        redis = self._redis_client()
        if redis is None:
            return False
        key = self._lock_key(sid)
        deadline = time.monotonic() + (max(0, int(wait_timeout_ms)) / 1000.0)
        while time.monotonic() <= deadline:
            try:
                if redis.set(key, owner, nx=True, ex=self._lock_ttl):
                    return True
            except Exception:
                return False
            time.sleep(0.05)
        return False

    def release_turn_lock(self, *, session_id: str, owner: str) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        redis = self._redis_client()
        if redis is None:
            return False
        key = self._lock_key(sid)
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        try:
            result = redis.eval(script, 1, key, owner)
            return bool(int(result or 0))
        except Exception:
            return False

    def flush_pending_events(
        self,
        *,
        session_id: str,
        channel: str,
        transport_sent: bool,
        force: bool = False,
    ) -> int:
        sid = (session_id or "").strip()
        if not sid:
            return 0
        if not transport_sent and not force:
            return 0
        redis = self._redis_client()
        if redis is None:
            return 0
        if not self._persist_stream_key:
            return 0
        now = _utc_now()
        pushed = 0
        pending_key = self._pending_key(sid)
        while True:
            raw = redis.lpop(pending_key)
            if raw is None:
                break
            event = _json_loads(raw)
            if not isinstance(event, dict):
                continue
            if transport_sent:
                event["transport_sent_at"] = now
                event["transport_delivered"] = "1"
            else:
                event["transport_sent_at"] = event.get("transport_sent_at") or now
                event["transport_delivered"] = "0"
            event["channel"] = event.get("channel") or channel
            fields: Dict[str, str] = {}
            for key, value in event.items():
                if isinstance(value, (dict, list)):
                    fields[key] = _json_dumps(value)
                elif value is None:
                    fields[key] = ""
                else:
                    fields[key] = str(value)
            try:
                redis.xadd(
                    self._persist_stream_key,
                    fields,
                    maxlen=self._persist_stream_maxlen,
                    approximate=True,
                )
                pushed += 1
                self._trace_log(
                    "chat_storage redis_stream_enqueue stream=%s event_id=%s event_type=%s session_id=%s seq=%s transport_delivered=%s",
                    self._persist_stream_key,
                    str(event.get("event_id") or ""),
                    str(event.get("event_type") or ""),
                    sid,
                    str(event.get("seq") or ""),
                    str(event.get("transport_delivered") or ""),
                )
            except Exception as exc:
                self._logger.warning("Failed to enqueue runtime persistence event: %s", exc)
                redis.lpush(pending_key, _json_dumps(event))
                redis.expire(pending_key, self._session_ttl_seconds)
                break
        self._trace_log(
            "chat_storage redis_stream_flush session_id=%s channel=%s transport_sent=%s force=%s pushed=%s",
            sid,
            channel,
            int(bool(transport_sent)),
            int(bool(force)),
            pushed,
        )
        return pushed
