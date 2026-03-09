from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from domain.interfaces import JobResult
from infrastructure.db.repositories import (
    AssetExtractionJob,
    PostgresAssetExtractionJobRepository,
    PostgresBotRepository,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssetExtractionRunner:
    """Pipeline runner for generic asset extraction (non-profiled sites)."""

    def run(self, context: Dict[str, Any]) -> JobResult:
        bot_id = str(context.get("bot_id") or "").strip()
        gcs_prefix = str(context.get("gcs_prefix") or "").strip()
        if not bot_id:
            return JobResult(status="error", error="Missing bot_id for asset_extraction runner")
        if not gcs_prefix:
            return JobResult(status="done", output={"status": "skipped", "reason": "missing_gcs_prefix"})

        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            return JobResult(status="error", error=f"Bot not found: {bot_id}")

        from infrastructure.tasks.crawl_tasks import asset_extraction_task

        job_id = "assetext_" + uuid.uuid4().hex
        now = _utc_now()
        job = AssetExtractionJob(
            job_id=job_id,
            bot_id=bot_id,
            org_id=bot.org_id,
            status="queued",
            created_at=now,
            updated_at=now,
            gcs_prefix=gcs_prefix,
            page_urls=None,
        )
        repo = PostgresAssetExtractionJobRepository()
        repo.create_job(job)
        async_result = asset_extraction_task.delay(
            bot_id=bot_id,
            org_id=bot.org_id,
            gcs_prefix=gcs_prefix,
            job_id=job_id,
        )
        job.celery_task_id = async_result.id
        repo.update_job(job)
        return JobResult(
            status="done",
            linked_job_type="asset_extraction",
            linked_job_id=job_id,
            output={"status": "queued"},
        )
