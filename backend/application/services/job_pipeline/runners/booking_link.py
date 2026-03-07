from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict

from domain.entities import BookingLinkJob
from domain.interfaces import JobResult
from infrastructure.db.repositories import PostgresBookingLinkJobRepository
from infrastructure.tasks.booking_link_tasks import booking_link_job_task


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_booking_link_job_id() -> str:
    return "blj_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


class BookingLinkRunner:
    def run(self, context: Dict[str, Any]) -> JobResult:
        bot_id = str(context.get("bot_id") or "").strip()
        index_job_id = str(context.get("index_job_id") or "").strip() or None
        root_url = str(context.get("root_url") or "").strip()
        if not bot_id:
            return JobResult(status="error", error="Missing bot_id for booking_link runner")

        delay_sec = int((os.environ.get("BOOKING_RAG_START_DELAY_SEC") or "90").strip())
        repo = PostgresBookingLinkJobRepository()
        now = _utc_now()
        job = BookingLinkJob(
            job_id=_new_booking_link_job_id(),
            bot_id=bot_id,
            index_job_id=index_job_id,
            root_url=root_url,
            status="queued",
            links=[],
            error=None,
            celery_task_id=None,
            created_at=now,
            updated_at=now,
        )
        repo.create(job)
        async_result = booking_link_job_task.apply_async((job.job_id, bot_id), countdown=max(0, delay_sec))
        job.celery_task_id = async_result.id
        repo.update(job)
        return JobResult(
            status="done",
            linked_job_type="booking_link",
            linked_job_id=job.job_id,
            output={"status": "queued"},
        )
