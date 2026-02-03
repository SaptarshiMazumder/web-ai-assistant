from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from domain.entities import ConversationMessage, ConversationSession
from infrastructure.db.repositories import PostgresConversationRepository


class ConversationService:
    def __init__(self, repo: Optional[PostgresConversationRepository] = None):
        self._repo = repo or PostgresConversationRepository()
        self._ttl = timedelta(minutes=30)

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
        return self._repo.add_message(
            session_id=session_id,
            bot_id=bot_id,
            role=role,
            content=content,
            citations=citations or [],
            sender_name=sender_name,
        )

    def list_sessions(self, bot_id: str, *, limit: int = 50, before: Optional[str] = None) -> List[ConversationSession]:
        return self._repo.list_sessions_for_bot(bot_id, limit=limit, before=before)

    def count_sessions(self, bot_id: str) -> int:
        return self._repo.count_sessions_for_bot(bot_id)

    def list_messages(self, session_id: str, *, limit: int = 200) -> List[ConversationMessage]:
        return self._repo.list_messages(session_id, limit=limit)

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._repo.get_session(session_id)

    def end_session(self, session_id: str, status: str = "ended") -> None:
        self._repo.end_session(session_id, status=status)

    def _is_expired(self, last_active_at: str) -> bool:
        try:
            last_dt = datetime.fromisoformat(last_active_at)
        except Exception:
            return False
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last_dt > self._ttl
