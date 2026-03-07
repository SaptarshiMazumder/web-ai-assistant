from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict

from celery import Task

from application.services.job_pipeline.engine import JobPipelineEngine
from application.services.job_pipeline.registry import JobRunnerRegistry
from domain.entities import JobPipelineStepEvent
from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresBotRepository, PostgresJobPipelineRepository

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    return "jpe_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


@celery_app.task(
    name="infrastructure.tasks.job_pipeline_tasks.run_job_pipeline_task",
    bind=True,
    max_retries=0,
)
def run_job_pipeline_task(self: Task, run_id: str, is_resume: bool = False) -> Dict[str, Any]:
    pipeline_repo = PostgresJobPipelineRepository()
    engine = JobPipelineEngine(
        pipeline_repo=pipeline_repo,
        bot_repo=PostgresBotRepository(),
        runner_registry=JobRunnerRegistry(),
    )

    try:
        snapshot = engine.execute(
            run_id,
            is_resume=bool(is_resume),
            celery_task_id=(self.request.id if self.request else None),
        )
    except ValueError as exc:
        logger.warning("Pipeline run task received unknown run_id=%s: %s", run_id, exc)
        return {"status": "error", "run_id": run_id, "error": str(exc)}
    except Exception as exc:
        logger.exception("Pipeline run task crashed (run_id=%s): %s", run_id, exc)
        run = pipeline_repo.get_run(run_id)
        if run and (run.status or "").lower() not in ("done", "error"):
            error_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            run.status = "error"
            run.current_stage_key = "error"
            run.current_message = error_msg
            run.last_error = error_msg
            run.updated_at = _utc_now()
            pipeline_repo.update_run(run)
            pipeline_repo.append_event(
                JobPipelineStepEvent(
                    event_id=_new_event_id(),
                    run_id=run_id,
                    step_index=None,
                    event_type="error",
                    stage_key="error",
                    message=error_msg,
                    progress_pct=run.progress_pct,
                    details={"task": "run_job_pipeline_task"},
                    created_at=_utc_now(),
                )
            )
        return {"status": "error", "run_id": run_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    run = snapshot.get("run") if isinstance(snapshot, dict) else None
    return {
        "status": str((run or {}).get("status") or "unknown"),
        "run_id": run_id,
        "is_resume": bool(is_resume),
    }

