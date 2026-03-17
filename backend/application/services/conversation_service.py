import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from application.services.chat_runtime_context import get_runtime_mode
from domain.entities import (
    ConversationChannelContact,
    ConversationMessage,
    ConversationSession,
    ConversationSessionHandoff,
    InstagramUserSession,
    LineUserSession,
)
from infrastructure.db.repositories import (
    PostgresConversationRepository,
    PostgresInstagramUserSessionRepository,
    PostgresLineUserSessionRepository,
)
from infrastructure.services.runtime_conversation_store import RuntimeConversationStore
from redis import Redis
from common.config import config

def _safe_int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or str(default)).strip())
    except ValueError:
        return default


CONVERSATION_HISTORY_MESSAGES = max(5, min(_safe_int_env("CONVERSATION_HISTORY_MESSAGES", 20), 100))


class ConversationService:
    def __init__(self, repo: Optional[PostgresConversationRepository] = None):
        self._logger = logging.getLogger(__name__)
        self._trace_storage = bool(getattr(config, "CHAT_STORAGE_TRACE_LOGS", False))
        self._repo = repo or PostgresConversationRepository()
        self._line_user_repo = PostgresLineUserSessionRepository()
        self._ig_user_repo = PostgresInstagramUserSessionRepository()
        self._runtime_store = RuntimeConversationStore()
        self._ttl = timedelta(minutes=config.CONVERSATION_SESSION_TTL_MINUTES)
        self._redis = Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)

    def _trace_storage_log(self, message: str, *args: object) -> None:
        if self._trace_storage:
            self._logger.warning(message, *args)

    def resolve_runtime_request_mode(
        self,
        *,
        request_path: str,
        channel: str,
        bot_id: Optional[str] = None,
    ) -> bool:
        if not bool(getattr(config, "CHAT_RUNTIME_REDIS_ENABLED", False)):
            return False
        available = self._runtime_store.is_available(force=True)
        if available:
            return True
        reason = self._runtime_store.last_ping_error() or "unknown"
        request_label = f"path={request_path} channel={channel} bot_id={bot_id or ''}".strip()
        if bool(getattr(config, "CHAT_RUNTIME_REDIS_OUTAGE_DB_FALLBACK", True)):
            self._logger.warning(
                "chat_runtime redis unavailable; falling back to DB flow (%s reason=%s)",
                request_label,
                reason,
            )
            return False
        message = f"chat_runtime redis unavailable ({request_label} reason={reason})"
        if bool(getattr(config, "CHAT_RUNTIME_STRICT_NO_DB_PRE_RESPONSE", False)):
            raise RuntimeError(message)
        self._logger.warning("%s; fallback disabled but strict mode off, using DB flow", message)
        return False

    def _runtime_enabled(self) -> bool:
        if not bool(get_runtime_mode()):
            return False
        if not bool(getattr(config, "CHAT_RUNTIME_REDIS_ENABLED", False)):
            return False
        if self._runtime_store.is_available(force=False):
            return True
        reason = self._runtime_store.last_ping_error() or "unknown"
        if bool(getattr(config, "CHAT_RUNTIME_REDIS_OUTAGE_DB_FALLBACK", True)):
            self._logger.warning("chat_runtime fallback to DB mid-request (reason=%s)", reason)
            return False
        if bool(getattr(config, "CHAT_RUNTIME_STRICT_NO_DB_PRE_RESPONSE", False)):
            raise RuntimeError(f"chat_runtime redis unavailable mid-request (reason={reason})")
        return False

    def acquire_turn_lock(
        self,
        *,
        session_id: str,
        owner: str,
        wait_timeout_ms: Optional[int] = None,
    ) -> bool:
        if not self._runtime_enabled():
            return True
        timeout_ms = (
            int(wait_timeout_ms)
            if wait_timeout_ms is not None
            else int(getattr(config, "CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS", 8000))
        )
        return self._runtime_store.acquire_turn_lock(
            session_id=session_id,
            owner=owner,
            wait_timeout_ms=timeout_ms,
        )

    def release_turn_lock(self, *, session_id: str, owner: str) -> bool:
        if not self._runtime_enabled():
            return True
        return self._runtime_store.release_turn_lock(session_id=session_id, owner=owner)

    def finalize_runtime_turn(
        self,
        *,
        session_id: str,
        channel: str,
        transport_sent: bool,
        force: bool = False,
    ) -> int:
        if not bool(getattr(config, "CHAT_RUNTIME_REDIS_ENABLED", False)):
            return 0
        if not self._runtime_store.is_available(force=False):
            return 0
        try:
            pushed = self._runtime_store.flush_pending_events(
                session_id=session_id,
                channel=channel,
                transport_sent=transport_sent,
                force=force,
            )
            self._trace_storage_log(
                "chat_storage redis_flush session_id=%s channel=%s transport_sent=%s force=%s pushed=%s",
                session_id,
                channel,
                int(bool(transport_sent)),
                int(bool(force)),
                pushed,
            )
            return pushed
        except Exception:
            self._logger.exception(
                "chat_runtime finalize failed session_id=%s channel=%s transport_sent=%s",
                session_id,
                channel,
                transport_sent,
            )
            return 0

    def get_or_create_session(
        self,
        *,
        bot_id: str,
        org_id: str,
        channel: str,
        session_id: Optional[str] = None,
        site_url: Optional[str] = None,
        site_title: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> ConversationSession:
        if self._runtime_enabled():
            return self._runtime_store.get_or_create_session(
                bot_id=bot_id,
                org_id=org_id,
                channel=channel,
                session_id=session_id,
                site_url=site_url,
                site_title=site_title,
                user_agent=user_agent,
                ip=ip,
            )

        existing = None
        if session_id:
            existing = self._repo.get_session(session_id)
        if existing and existing.bot_id == bot_id and existing.status == "active":
            if not self._is_expired(existing.last_active_at):
                self._repo.touch_session(existing.session_id)
                return existing
            self.expire_session_runtime_state(bot_id=bot_id, session_id=existing.session_id)
            self._repo.end_session(existing.session_id, status="expired")
        return self._repo.create_session(
            bot_id=bot_id,
            org_id=org_id,
            channel=channel,
            site_url=site_url,
            site_title=site_title,
            user_agent=user_agent,
            ip=ip,
        )

    def resolve_or_create_channel_session(
        self,
        *,
        bot_id: str,
        org_id: str,
        channel: str,
        external_user_id: str,
        current_session_id: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        site_url: Optional[str] = None,
        site_title: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> tuple[ConversationSession, ConversationChannelContact, bool]:
        if self._runtime_enabled():
            return self._runtime_store.resolve_or_create_channel_session(
                bot_id=bot_id,
                org_id=org_id,
                channel=channel,
                external_user_id=external_user_id,
                current_session_id=current_session_id,
                display_name=display_name,
                metadata=metadata,
                site_url=site_url,
                site_title=site_title,
                user_agent=user_agent,
                ip=ip,
            )

        existing_contact = self._repo.get_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
        )
        seed_session_id = (current_session_id or "").strip() or getattr(existing_contact, "current_session_id", None)
        session = self.get_or_create_session(
            bot_id=bot_id,
            org_id=org_id,
            channel=channel,
            session_id=seed_session_id,
            site_url=site_url,
            site_title=site_title,
            user_agent=user_agent,
            ip=ip,
        )
        session_changed = bool(seed_session_id and seed_session_id != session.session_id)
        if session_changed:
            self.expire_session_runtime_state(bot_id=bot_id, session_id=seed_session_id)
        contact = self._repo.upsert_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
            current_session_id=session.session_id,
            display_name=display_name,
            metadata=metadata,
        )
        refreshed = self._repo.get_session(session.session_id)
        return refreshed or session, contact, session_changed

    def add_message(
        self,
        *,
        session_id: str,
        bot_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, str]]] = None,
        sender_name: Optional[str] = None,
    ) -> ConversationMessage:
        if self._runtime_enabled():
            msg = self._runtime_store.add_message(
                session_id=session_id,
                bot_id=bot_id,
                role=role,
                content=content,
                citations=citations or [],
                sender_name=sender_name,
            )
            self._trace_storage_log(
                "chat_storage redis_write message_id=%s session_id=%s bot_id=%s role=%s content_len=%s",
                getattr(msg, "message_id", ""),
                session_id,
                bot_id,
                role,
                len(content or ""),
            )
            self._redis.sadd("analytics:dirty_bots", bot_id)
            return msg

        msg = self._repo.add_message(
            session_id=session_id,
            bot_id=bot_id,
            role=role,
            content=content,
            citations=citations or [],
            sender_name=sender_name,
        )
        self._trace_storage_log(
            "chat_storage db_write_direct message_id=%s session_id=%s bot_id=%s role=%s content_len=%s",
            getattr(msg, "message_id", ""),
            session_id,
            bot_id,
            role,
            len(content or ""),
        )
        self._redis.sadd("analytics:dirty_bots", bot_id)
        return msg

    def list_sessions(self, bot_id: str, *, limit: int = 50, before: Optional[str] = None) -> List[ConversationSession]:
        return self._repo.list_sessions_for_bot(bot_id, limit=limit, before=before)

    def count_sessions(self, bot_id: str) -> int:
        return self._repo.count_sessions_for_bot(bot_id)

    def list_messages(self, session_id: str, *, limit: int = 200) -> List[ConversationMessage]:
        return self._repo.list_messages(session_id, limit=limit)

    def list_recent_messages(self, session_id: str, *, limit: int = CONVERSATION_HISTORY_MESSAGES) -> List[ConversationMessage]:
        if self._runtime_enabled():
            messages = self._runtime_store.list_recent_messages(session_id, limit=limit)
            trace_bot_id = ""
            if self._trace_storage:
                try:
                    trace_session = self._runtime_store.get_session(session_id)
                    trace_bot_id = (getattr(trace_session, "bot_id", "") or "").strip()
                except Exception:
                    trace_bot_id = ""
            self._trace_storage_log(
                "chat_storage history_read source=redis_runtime bot_id=%s session_id=%s limit=%s returned=%s",
                trace_bot_id,
                session_id,
                limit,
                len(messages),
            )
            return messages
        messages = self._repo.list_messages_recent(session_id, limit=limit)
        trace_bot_id = ""
        if self._trace_storage:
            try:
                trace_session = self._repo.get_session(session_id)
                trace_bot_id = (getattr(trace_session, "bot_id", "") or "").strip()
            except Exception:
                trace_bot_id = ""
        self._trace_storage_log(
            "chat_storage history_read source=db_repo bot_id=%s session_id=%s limit=%s returned=%s",
            trace_bot_id,
            session_id,
            limit,
            len(messages),
        )
        return messages

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        if self._runtime_enabled():
            return self._runtime_store.get_session(session_id)
        return self._repo.get_session(session_id)

    def set_session_title(self, session_id: str, title: Optional[str]) -> bool:
        if self._runtime_enabled():
            return self._runtime_store.set_session_title(session_id, title)
        return self._repo.update_session_title(session_id, title)

    def end_session(self, session_id: str, status: str = "ended") -> None:
        session = self._repo.get_session(session_id)
        if session:
            self._redis.sadd("analytics:dirty_bots", session.bot_id)
            if status == "expired":
                self.expire_session_runtime_state(bot_id=session.bot_id, session_id=session_id)
            elif session.handoff_active:
                self.release_handoff(bot_id=session.bot_id, session_id=session_id, ended_reason="staff_released")
        self._repo.end_session(session_id, status=status)

    def create_escalation(self, *, bot_id: str, session_id: str, visitor_email: str, details: Optional[str] = None):
        if self._runtime_enabled():
            record = self._runtime_store.create_escalation(
                bot_id=bot_id,
                session_id=session_id,
                visitor_email=visitor_email,
                details=details,
                status="open",
            )
            self._redis.sadd("analytics:dirty_bots", bot_id)
            return record
        record = self._repo.create_escalation(
            bot_id=bot_id,
            session_id=session_id,
            visitor_email=visitor_email,
            details=details,
        )
        self._redis.sadd("analytics:dirty_bots", bot_id)
        return record

    def get_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
    ) -> Optional[ConversationChannelContact]:
        if self._runtime_enabled():
            return self._runtime_store.get_channel_contact(
                bot_id=bot_id,
                channel=channel,
                external_user_id=external_user_id,
            )
        return self._repo.get_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
        )

    def upsert_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
        current_session_id: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ConversationChannelContact:
        if self._runtime_enabled():
            return self._runtime_store.upsert_channel_contact(
                bot_id=bot_id,
                channel=channel,
                external_user_id=external_user_id,
                current_session_id=current_session_id,
                display_name=display_name,
                metadata=metadata,
            )
        return self._repo.upsert_channel_contact(
            bot_id=bot_id,
            channel=channel,
            external_user_id=external_user_id,
            current_session_id=current_session_id,
            display_name=display_name,
            metadata=metadata,
        )

    def get_session_handoff(self, session_id: str) -> Optional[ConversationSessionHandoff]:
        if self._runtime_enabled():
            return self._runtime_store.get_session_handoff(session_id)
        return self._repo.get_session_handoff(session_id)

    def set_awaiting_support_details(
        self,
        *,
        bot_id: str,
        session_id: str,
        contact_id: Optional[str] = None,
        source_channel: Optional[str] = None,
        started_by: str = "user",
    ) -> ConversationSessionHandoff:
        if self._runtime_enabled():
            return self._runtime_store.upsert_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="awaiting_support_details",
                contact_id=contact_id,
                source_channel=source_channel,
                support_request_id=None,
                started_by=started_by,
                ended_reason=None,
                ended_at=None,
            )
        return self._repo.upsert_session_handoff(
            bot_id=bot_id,
            session_id=session_id,
            assistant_state="awaiting_support_details",
            contact_id=contact_id,
            source_channel=source_channel,
            support_request_id=None,
            started_by=started_by,
            ended_reason=None,
            ended_at=None,
        )

    def begin_handoff(
        self,
        *,
        bot_id: str,
        session_id: str,
        contact_id: Optional[str] = None,
        source_channel: Optional[str] = None,
        support_request_id: Optional[str] = None,
        started_by: str = "staff",
    ) -> ConversationSessionHandoff:
        if self._runtime_enabled():
            active_request_id = support_request_id
            if active_request_id is None:
                latest = self._runtime_store.get_latest_escalation_for_session(session_id)
                if latest and latest.status == "open":
                    active_request_id = latest.escalation_id
            return self._runtime_store.upsert_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="human_handoff",
                contact_id=contact_id,
                source_channel=source_channel,
                support_request_id=active_request_id,
                started_by=started_by,
                ended_reason=None,
                ended_at=None,
            )

        active_request_id = support_request_id
        if active_request_id is None:
            open_request = self._repo.get_open_escalation_for_session(bot_id, session_id)
            active_request_id = open_request.escalation_id if open_request else None
        return self._repo.upsert_session_handoff(
            bot_id=bot_id,
            session_id=session_id,
            assistant_state="human_handoff",
            contact_id=contact_id,
            source_channel=source_channel,
            support_request_id=active_request_id,
            started_by=started_by,
            ended_reason=None,
            ended_at=None,
        )

    def request_support(
        self,
        *,
        bot_id: str,
        session_id: str,
        visitor_email: str,
        details: Optional[str] = None,
        contact_id: Optional[str] = None,
        source_channel: Optional[str] = None,
    ):
        if self._runtime_enabled():
            record = self._runtime_store.create_escalation(
                bot_id=bot_id,
                session_id=session_id,
                visitor_email=visitor_email,
                details=details,
                status="open",
            )
            self._runtime_store.upsert_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="human_handoff",
                contact_id=contact_id,
                source_channel=source_channel,
                support_request_id=record.escalation_id,
                started_by="user",
                ended_reason=None,
                ended_at=None,
            )
            self._redis.sadd("analytics:dirty_bots", bot_id)
            return record

        record = self._repo.create_escalation(
            bot_id=bot_id,
            session_id=session_id,
            visitor_email=visitor_email,
            details=details,
        )
        self._repo.upsert_session_handoff(
            bot_id=bot_id,
            session_id=session_id,
            assistant_state="human_handoff",
            contact_id=contact_id,
            source_channel=source_channel,
            support_request_id=record.escalation_id,
            started_by="user",
            ended_reason=None,
            ended_at=None,
        )
        self._redis.sadd("analytics:dirty_bots", bot_id)
        return record

    def cancel_handoff(self, *, bot_id: str, session_id: str) -> Optional[ConversationSessionHandoff]:
        if self._runtime_enabled():
            handoff = self._runtime_store.get_session_handoff(session_id)
            latest = self._runtime_store.get_latest_escalation_for_session(session_id)
            active_request_id = handoff.support_request_id if handoff and handoff.support_request_id else None
            if latest and latest.status == "open":
                active_request_id = latest.escalation_id
                self._runtime_store.update_escalation_status(latest.escalation_id, "canceled")
            return self._runtime_store.upsert_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="bot",
                contact_id=handoff.contact_id if handoff else None,
                source_channel=handoff.source_channel if handoff else None,
                support_request_id=active_request_id,
                started_by=handoff.started_by if handoff else None,
                ended_reason="user_canceled",
                ended_at=datetime.now(timezone.utc).isoformat(),
            )

        handoff = self._repo.get_session_handoff(session_id)
        open_request = self._repo.get_open_escalation_for_session(bot_id, session_id)
        active_request_id = handoff.support_request_id if handoff and handoff.support_request_id else None
        if open_request:
            active_request_id = open_request.escalation_id
            self._repo.update_escalation_status(bot_id, open_request.escalation_id, "canceled")
        return self._repo.upsert_session_handoff(
            bot_id=bot_id,
            session_id=session_id,
            assistant_state="bot",
            contact_id=handoff.contact_id if handoff else None,
            source_channel=handoff.source_channel if handoff else None,
            support_request_id=active_request_id,
            started_by=handoff.started_by if handoff else None,
            ended_reason="user_canceled",
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    def resolve_support_request(self, *, bot_id: str, escalation_id: str) -> Optional[str]:
        if self._runtime_enabled():
            escalation = self._runtime_store.get_escalation(escalation_id)
            if not escalation:
                return None
            self._runtime_store.update_escalation_status(escalation_id, "resolved")
            handoff = self._runtime_store.get_session_handoff(escalation.session_id)
            if handoff and handoff.assistant_state in {"awaiting_support_details", "human_handoff"}:
                if not handoff.support_request_id or handoff.support_request_id == escalation_id:
                    self._runtime_store.upsert_handoff(
                        bot_id=bot_id,
                        session_id=escalation.session_id,
                        assistant_state="bot",
                        contact_id=handoff.contact_id,
                        source_channel=handoff.source_channel,
                        support_request_id=escalation_id,
                        started_by=handoff.started_by,
                        ended_reason="staff_resolved",
                        ended_at=datetime.now(timezone.utc).isoformat(),
                    )
            return escalation.session_id

        escalation = self._repo.get_escalation_by_id(bot_id, escalation_id)
        if not escalation:
            return None
        self._repo.update_escalation_status(bot_id, escalation_id, "resolved")
        handoff = self._repo.get_session_handoff(escalation.session_id)
        if handoff and handoff.assistant_state in {"awaiting_support_details", "human_handoff"}:
            if not handoff.support_request_id or handoff.support_request_id == escalation_id:
                self._repo.upsert_session_handoff(
                    bot_id=bot_id,
                    session_id=escalation.session_id,
                    assistant_state="bot",
                    contact_id=handoff.contact_id,
                    source_channel=handoff.source_channel,
                    support_request_id=escalation_id,
                    started_by=handoff.started_by,
                    ended_reason="staff_resolved",
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
        return escalation.session_id

    def release_handoff(
        self,
        *,
        bot_id: str,
        session_id: str,
        ended_reason: str,
    ) -> Optional[ConversationSessionHandoff]:
        if self._runtime_enabled():
            handoff = self._runtime_store.get_session_handoff(session_id)
            return self._runtime_store.upsert_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="bot",
                contact_id=handoff.contact_id if handoff else None,
                source_channel=handoff.source_channel if handoff else None,
                support_request_id=handoff.support_request_id if handoff else None,
                started_by=handoff.started_by if handoff else None,
                ended_reason=ended_reason,
                ended_at=datetime.now(timezone.utc).isoformat(),
            )

        handoff = self._repo.get_session_handoff(session_id)
        return self._repo.upsert_session_handoff(
            bot_id=bot_id,
            session_id=session_id,
            assistant_state="bot",
            contact_id=handoff.contact_id if handoff else None,
            source_channel=handoff.source_channel if handoff else None,
            support_request_id=handoff.support_request_id if handoff else None,
            started_by=handoff.started_by if handoff else None,
            ended_reason=ended_reason,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    def expire_session_runtime_state(self, *, bot_id: str, session_id: str) -> Optional[ConversationSessionHandoff]:
        if self._runtime_enabled():
            handoff = self._runtime_store.get_session_handoff(session_id)
            if not handoff:
                return None
            if handoff.assistant_state in {"awaiting_support_details", "human_handoff"}:
                latest = self._runtime_store.get_latest_escalation_for_session(session_id)
                if latest and latest.status == "open":
                    self._runtime_store.update_escalation_status(latest.escalation_id, "expired")
                    support_request_id = latest.escalation_id
                else:
                    support_request_id = handoff.support_request_id
                return self._runtime_store.upsert_handoff(
                    bot_id=bot_id,
                    session_id=session_id,
                    assistant_state="bot",
                    contact_id=handoff.contact_id,
                    source_channel=handoff.source_channel,
                    support_request_id=support_request_id,
                    started_by=handoff.started_by,
                    ended_reason="session_expired",
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
            return handoff

        handoff = self._repo.get_session_handoff(session_id)
        if not handoff:
            return None
        if handoff.assistant_state in {"awaiting_support_details", "human_handoff"}:
            open_request = self._repo.get_open_escalation_for_session(bot_id, session_id)
            if open_request:
                self._repo.update_escalation_status(bot_id, open_request.escalation_id, "expired")
                support_request_id = open_request.escalation_id
            else:
                support_request_id = handoff.support_request_id
            return self._repo.upsert_session_handoff(
                bot_id=bot_id,
                session_id=session_id,
                assistant_state="bot",
                contact_id=handoff.contact_id,
                source_channel=handoff.source_channel,
                support_request_id=support_request_id,
                started_by=handoff.started_by,
                ended_reason="session_expired",
                ended_at=datetime.now(timezone.utc).isoformat(),
            )
        return handoff

    def list_escalations(self, bot_id: str, *, limit: int = 10, before: Optional[str] = None):
        return self._repo.list_escalations_for_bot(bot_id, limit=limit, before=before)

    def list_escalations_for_user(
        self,
        bot_id: str,
        *,
        user_id: str,
        limit: int = 10,
        before: Optional[str] = None,
    ):
        return self._repo.list_escalations_for_bot(bot_id, limit=limit, before=before, user_id=user_id)

    def count_escalations(self, bot_id: str) -> int:
        return self._repo.count_escalations_for_bot(bot_id)

    def count_open_escalations(self, bot_id: str) -> int:
        return self._repo.count_open_escalations_for_bot(bot_id)

    def count_unread_escalations(self, bot_id: str, user_id: str) -> int:
        return self._repo.count_unread_escalations_for_bot(bot_id, user_id)

    def get_escalation_for_session(self, bot_id: str, session_id: str, *, user_id: Optional[str] = None):
        return self._repo.get_escalation_for_session(bot_id, session_id, user_id=user_id)

    def get_latest_escalation_for_visitor(self, bot_id: str, visitor_email: str, *, user_id: Optional[str] = None):
        return self._repo.get_latest_escalation_for_visitor(bot_id, visitor_email, user_id=user_id)

    def get_escalation_by_id(self, bot_id: str, escalation_id: str, *, user_id: Optional[str] = None):
        return self._repo.get_escalation_by_id(bot_id, escalation_id, user_id=user_id)

    def update_escalation_status(self, bot_id: str, escalation_id: str, status: str) -> bool:
        if self._runtime_enabled():
            updated = self._runtime_store.update_escalation_status(escalation_id, status)
            return bool(updated)
        return self._repo.update_escalation_status(bot_id, escalation_id, status)

    def mark_escalation_read(self, bot_id: str, escalation_id: str, user_id: str) -> bool:
        return self._repo.mark_escalation_read(bot_id, escalation_id, user_id)

    def get_line_user_session(self, *, line_user_id: str, bot_id: str) -> Optional[LineUserSession]:
        if self._runtime_enabled():
            return self._runtime_store.get_line_user_session(line_user_id=line_user_id, bot_id=bot_id)
        return self._line_user_repo.get(line_user_id=line_user_id, bot_id=bot_id)

    def get_line_user_session_by_session_id(self, session_id: str) -> Optional[LineUserSession]:
        if self._runtime_enabled():
            return self._runtime_store.get_line_user_session_by_session_id(session_id)
        return self._line_user_repo.get_by_session_id(session_id)

    def get_or_create_line_user_session(
        self,
        *,
        line_user_id: str,
        bot_id: str,
        session_id: str,
        display_name: Optional[str] = None,
    ) -> Optional[LineUserSession]:
        if self._runtime_enabled():
            existing = self._runtime_store.get_line_user_session(line_user_id=line_user_id, bot_id=bot_id)
            if existing:
                if existing.session_id == session_id and (
                    (display_name or "").strip() == "" or (existing.display_name or "").strip() == (display_name or "").strip()
                ):
                    return existing
            return self._runtime_store.upsert_line_user_session(
                line_user_id=line_user_id,
                bot_id=bot_id,
                session_id=session_id,
                display_name=display_name,
            )
        return self._line_user_repo.get_or_create(
            line_user_id=line_user_id,
            bot_id=bot_id,
            session_id=session_id,
            display_name=display_name,
        )

    def update_line_user_session_id(self, *, line_user_id: str, bot_id: str, session_id: str) -> bool:
        if self._runtime_enabled():
            current = self._runtime_store.get_line_user_session(line_user_id=line_user_id, bot_id=bot_id)
            if not current:
                current = self._runtime_store.upsert_line_user_session(
                    line_user_id=line_user_id,
                    bot_id=bot_id,
                    session_id=session_id,
                )
                return current is not None
            updated = self._runtime_store.upsert_line_user_session(
                line_user_id=line_user_id,
                bot_id=bot_id,
                session_id=session_id,
                is_escalated=current.is_escalated,
                awaiting_escalation_msg=current.awaiting_escalation_msg,
                awaiting_staff_takeover=current.awaiting_staff_takeover,
                display_name=current.display_name,
            )
            return updated is not None
        return self._line_user_repo.update_session_id(
            line_user_id=line_user_id,
            bot_id=bot_id,
            session_id=session_id,
        )

    def set_line_user_display_name(self, *, line_user_id: str, bot_id: str, display_name: Optional[str]) -> bool:
        if self._runtime_enabled():
            current = self._runtime_store.get_line_user_session(line_user_id=line_user_id, bot_id=bot_id)
            if not current:
                return False
            updated = self._runtime_store.upsert_line_user_session(
                line_user_id=line_user_id,
                bot_id=bot_id,
                session_id=current.session_id,
                is_escalated=current.is_escalated,
                awaiting_escalation_msg=current.awaiting_escalation_msg,
                awaiting_staff_takeover=current.awaiting_staff_takeover,
                display_name=display_name,
            )
            return updated is not None
        return self._line_user_repo.set_display_name(
            line_user_id=line_user_id,
            bot_id=bot_id,
            display_name=display_name,
        )

    def get_instagram_user_session(self, *, ig_user_id: str, bot_id: str) -> Optional[InstagramUserSession]:
        if self._runtime_enabled():
            return self._runtime_store.get_instagram_user_session(ig_user_id=ig_user_id, bot_id=bot_id)
        return self._ig_user_repo.get(ig_user_id=ig_user_id, bot_id=bot_id)

    def get_instagram_user_session_by_session_id(self, session_id: str) -> Optional[InstagramUserSession]:
        if self._runtime_enabled():
            return self._runtime_store.get_instagram_user_session_by_session_id(session_id)
        return self._ig_user_repo.get_by_session_id(session_id)

    def get_or_create_instagram_user_session(
        self,
        *,
        ig_user_id: str,
        bot_id: str,
        session_id: str,
    ) -> Optional[InstagramUserSession]:
        if self._runtime_enabled():
            existing = self._runtime_store.get_instagram_user_session(ig_user_id=ig_user_id, bot_id=bot_id)
            if existing and existing.session_id == session_id:
                return existing
            return self._runtime_store.upsert_instagram_user_session(
                ig_user_id=ig_user_id,
                bot_id=bot_id,
                session_id=session_id,
            )
        return self._ig_user_repo.get_or_create(
            ig_user_id=ig_user_id,
            bot_id=bot_id,
            session_id=session_id,
        )

    def update_instagram_user_session_id(self, *, ig_user_id: str, bot_id: str, session_id: str) -> bool:
        if self._runtime_enabled():
            current = self._runtime_store.get_instagram_user_session(ig_user_id=ig_user_id, bot_id=bot_id)
            if not current:
                current = self._runtime_store.upsert_instagram_user_session(
                    ig_user_id=ig_user_id,
                    bot_id=bot_id,
                    session_id=session_id,
                )
                return current is not None
            updated = self._runtime_store.upsert_instagram_user_session(
                ig_user_id=ig_user_id,
                bot_id=bot_id,
                session_id=session_id,
                is_escalated=current.is_escalated,
                awaiting_escalation_msg=current.awaiting_escalation_msg,
                awaiting_staff_takeover=current.awaiting_staff_takeover,
            )
            return updated is not None
        return self._ig_user_repo.update_session_id(
            ig_user_id=ig_user_id,
            bot_id=bot_id,
            session_id=session_id,
        )

    def _is_expired(self, last_active_at: str) -> bool:
        try:
            last_dt = datetime.fromisoformat(last_active_at)
        except Exception:
            return False
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last_dt > self._ttl
