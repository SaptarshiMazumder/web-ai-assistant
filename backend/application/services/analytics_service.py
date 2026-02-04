from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from infrastructure.db.repositories import PostgresAnalyticsRepository


def _parse_range(range_str: str) -> int:
    s = (range_str or "").strip().lower()
    if s in ("7d", "7", "week"):
        return 7
    if s in ("30d", "30", "month"):
        return 30
    if s in ("90d", "90", "quarter"):
        return 90
    # default
    return 30


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _parse_day(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    v = str(s).strip()
    if not v:
        return None
    try:
        yy, mm, dd = v.split("-", 2)
        return date(int(yy), int(mm), int(dd))
    except Exception:
        return None


def _resolve_window(*, range_str: str, from_day: Optional[str], to_day: Optional[str]) -> tuple[date, date]:
    fd = _parse_day(from_day)
    td = _parse_day(to_day)
    if fd and td:
        return (fd, td) if fd <= td else (td, fd)
    if fd and not td:
        return (fd, _utc_today())
    if td and not fd:
        # default 30d ending at to_day
        start = td - timedelta(days=29)
        return (start, td)
    days = _parse_range(range_str)
    end = _utc_today()
    start = end - timedelta(days=days - 1)
    return start, end

@dataclass
class AnalyticsSummary:
    start_day: str
    end_day: str
    conversations: int
    messages_user: int
    messages_bot: int
    escalations: int
    unique_visitors_est: int
    messages_per_conversation: float
    escalation_rate: float
    positive_feedback: int
    negative_feedback: int


class AnalyticsService:
    def __init__(self, repo: Optional[PostgresAnalyticsRepository] = None):
        self._repo = repo or PostgresAnalyticsRepository()

    def recompute(self, *, org_id: str, bot_id: str, range_str: str, from_day: Optional[str] = None, to_day: Optional[str] = None) -> Dict[str, Any]:
        start_day, end_day = _resolve_window(range_str=range_str, from_day=from_day, to_day=to_day)
        self._repo.recompute_bot_rollups(org_id=org_id, bot_id=bot_id, start_day=start_day, end_day=end_day)
        return {"ok": True, "start_day": start_day.isoformat(), "end_day": end_day.isoformat()}

    def summary(self, *, org_id: str, bot_id: str, range_str: str, from_day: Optional[str] = None, to_day: Optional[str] = None) -> AnalyticsSummary:
        start_day, end_day = _resolve_window(range_str=range_str, from_day=from_day, to_day=to_day)
        series = self._repo.get_usage_timeseries(org_id=org_id, bot_id=bot_id, start_day=start_day, end_day=end_day)
        conv = sum(int(p.get("conversations") or 0) for p in series)
        mu = sum(int(p.get("messages_user") or 0) for p in series)
        mb = sum(int(p.get("messages_bot") or 0) for p in series)
        esc = sum(int(p.get("escalations") or 0) for p in series)
        uniq = sum(int(p.get("unique_visitors_est") or 0) for p in series)

        start_iso = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc).isoformat()
        end_excl = end_day + timedelta(days=1)
        end_iso = datetime(end_excl.year, end_excl.month, end_excl.day, tzinfo=timezone.utc).isoformat()
        fb = self._repo.get_feedback_counts(org_id=org_id, bot_id=bot_id, start_iso=start_iso, end_iso=end_iso)

        mpc = float(mu + mb) / float(conv) if conv else 0.0
        er = float(esc) / float(conv) if conv else 0.0
        return AnalyticsSummary(
            start_day=start_day.isoformat(),
            end_day=end_day.isoformat(),
            conversations=conv,
            messages_user=mu,
            messages_bot=mb,
            escalations=esc,
            unique_visitors_est=uniq,
            messages_per_conversation=mpc,
            escalation_rate=er,
            positive_feedback=int(fb.get("positive") or 0),
            negative_feedback=int(fb.get("negative") or 0),
        )

    def timeseries(self, *, org_id: str, bot_id: str, range_str: str, from_day: Optional[str] = None, to_day: Optional[str] = None) -> Dict[str, Any]:
        start_day, end_day = _resolve_window(range_str=range_str, from_day=from_day, to_day=to_day)
        points = self._repo.get_usage_timeseries(org_id=org_id, bot_id=bot_id, start_day=start_day, end_day=end_day)
        return {"start_day": start_day.isoformat(), "end_day": end_day.isoformat(), "points": points}

    def top_sources(
        self, *, org_id: str, bot_id: str, range_str: str, limit: int = 10, from_day: Optional[str] = None, to_day: Optional[str] = None
    ) -> Dict[str, Any]:
        start_day, end_day = _resolve_window(range_str=range_str, from_day=from_day, to_day=to_day)
        items = self._repo.get_top_sources(org_id=org_id, bot_id=bot_id, start_day=start_day, end_day=end_day, limit=limit)
        return {"start_day": start_day.isoformat(), "end_day": end_day.isoformat(), "items": items}

    def topics(
        self, *, org_id: str, bot_id: str, range_str: str, limit: int = 20, from_day: Optional[str] = None, to_day: Optional[str] = None
    ) -> Dict[str, Any]:
        start_day, end_day = _resolve_window(range_str=range_str, from_day=from_day, to_day=to_day)
        items = self._repo.get_top_topics(org_id=org_id, bot_id=bot_id, start_day=start_day, end_day=end_day, limit=limit)
        return {"start_day": start_day.isoformat(), "end_day": end_day.isoformat(), "items": items}

    def record_feedback(
        self,
        *,
        org_id: str,
        bot_id: str,
        session_id: str,
        rating: int,
        comment: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        fid = uuid.uuid4().hex
        self._repo.insert_feedback(
            feedback_id=fid,
            org_id=org_id,
            bot_id=bot_id,
            session_id=session_id,
            message_id=message_id,
            rating=rating,
            comment=comment,
            created_at=now,
        )
        return {"feedback_id": fid, "created_at": now}

