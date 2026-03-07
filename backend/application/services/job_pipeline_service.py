from __future__ import annotations

from typing import Any, Dict, List, Optional

from domain.platform_profiles import get_job_pipeline_config
from domain.repositories import BotRepository, JobPipelineRepository

from .job_pipeline.engine import JobPipelineEngine
from .job_pipeline.registry import JobRunnerRegistry


class JobPipelineService:
    def __init__(self, *, pipeline_repo: JobPipelineRepository, bot_repo: BotRepository) -> None:
        self._engine = JobPipelineEngine(
            pipeline_repo=pipeline_repo,
            bot_repo=bot_repo,
            runner_registry=JobRunnerRegistry(),
        )

    def _task_queue(self) -> Optional[str]:
        cfg = get_job_pipeline_config()
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
        queue = str(defaults.get("task_queue") or "").strip() if isinstance(defaults, dict) else ""
        return queue or None

    def _enqueue(self, *, run_id: str, is_resume: bool) -> Optional[str]:
        from infrastructure.tasks.job_pipeline_tasks import run_job_pipeline_task

        kwargs = {"run_id": run_id, "is_resume": bool(is_resume)}
        queue = self._task_queue()
        if queue:
            async_result = run_job_pipeline_task.apply_async(kwargs=kwargs, queue=queue)
        else:
            async_result = run_job_pipeline_task.apply_async(kwargs=kwargs)
        return getattr(async_result, "id", None)

    def start_post_crawl(
        self,
        *,
        bot_id: str,
        index_job_id: str,
        gcs_prefix: str,
        root_url: str,
        crawled_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        snapshot = self._engine.start(
            bot_id=bot_id,
            workflow_id="default",
            trigger="post_crawl",
            context={
                "index_job_id": index_job_id,
                "gcs_prefix": gcs_prefix,
                "root_url": root_url,
                "crawled_urls": list(crawled_urls or []),
            },
            execute_now=False,
        )
        run = snapshot.get("run") if isinstance(snapshot, dict) else None
        run_id = str((run or {}).get("run_id") or "").strip() if isinstance(run, dict) else ""
        if run_id:
            try:
                self._enqueue(run_id=run_id, is_resume=False)
            except Exception:
                return self._engine.execute(run_id, is_resume=False)
            latest = self._engine.get_status(run_id)
            if latest:
                return latest
        return snapshot

    def resume(self, run_id: str) -> Dict[str, Any]:
        snapshot = self._engine.resume(run_id, execute_now=False)
        run = snapshot.get("run") if isinstance(snapshot, dict) else None
        status = str((run or {}).get("status") or "").strip().lower() if isinstance(run, dict) else ""
        stage_key = str((run or {}).get("current_stage_key") or "").strip().lower() if isinstance(run, dict) else ""
        if status == "queued" and stage_key == "resume_requested":
            try:
                self._enqueue(run_id=run_id, is_resume=True)
            except Exception:
                return self._engine.execute(run_id, is_resume=True)
            latest = self._engine.get_status(run_id)
            if latest:
                return latest
        return snapshot

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._engine.get_status(run_id)

    def get_latest_for_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        return self._engine.get_latest_for_bot(bot_id)
