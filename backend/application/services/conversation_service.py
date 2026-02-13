from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from domain.entities import ConversationMessage, ConversationSession
from infrastructure.db.repositories import PostgresConversationRepository
from redis import Redis
from common.config import config

CONVERSATION_HISTORY_MESSAGES = 20


class ConversationService:
    def __init__(self, repo: Optional[PostgresConversationRepository] = None):
        self._repo = repo or PostgresConversationRepository()
        self._ttl = timedelta(minutes=30)
        self._redis = Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)

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
        existing = None
        if session_id:
            existing = self._repo.get_session(session_id)
        if existing and existing.bot_id == bot_id and existing.status == "active":
            if not self._is_expired(existing.last_active_at):
                self._repo.touch_session(existing.session_id)
                return existing
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
        msg = self._repo.add_message(
            session_id=session_id,
            bot_id=bot_id,
            role=role,
            content=content,
            citations=citations or [],
            sender_name=sender_name,
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
        return self._repo.list_messages_recent(session_id, limit=limit)

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._repo.get_session(session_id)

    def end_session(self, session_id: str, status: str = "ended") -> None:
        session = self._repo.get_session(session_id)
        if session:
            self._redis.sadd("analytics:dirty_bots", session.bot_id)
        self._repo.end_session(session_id, status=status)

    def create_escalation(self, *, bot_id: str, session_id: str, visitor_email: str, details: Optional[str] = None):
        return self._repo.create_escalation(
            bot_id=bot_id,
            session_id=session_id,
            visitor_email=visitor_email,
            details=details,
        )

    def list_escalations(self, bot_id: str, *, limit: int = 10, before: Optional[str] = None):
        return self._repo.list_escalations_for_bot(bot_id, limit=limit, before=before)

    def count_escalations(self, bot_id: str) -> int:
        return self._repo.count_escalations_for_bot(bot_id)

    def count_open_escalations(self, bot_id: str) -> int:
        return self._repo.count_open_escalations_for_bot(bot_id)

    def get_escalation_for_session(self, bot_id: str, session_id: str):
        return self._repo.get_escalation_for_session(bot_id, session_id)

    def update_escalation_status(self, bot_id: str, escalation_id: str, status: str) -> bool:
        return self._repo.update_escalation_status(bot_id, escalation_id, status)

    def _is_expired(self, last_active_at: str) -> bool:
        try:
            last_dt = datetime.fromisoformat(last_active_at)
        except Exception:
            return False
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last_dt > self._ttl
