from __future__ import annotations

from typing import Any, Dict, List, Optional

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

    def start_post_crawl(
        self,
        *,
        bot_id: str,
        index_job_id: str,
        gcs_prefix: str,
        root_url: str,
        crawled_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._engine.start(
            bot_id=bot_id,
            workflow_id="default",
            trigger="post_crawl",
            context={
                "index_job_id": index_job_id,
                "gcs_prefix": gcs_prefix,
                "root_url": root_url,
                "crawled_urls": list(crawled_urls or []),
            },
        )

    def resume(self, run_id: str) -> Dict[str, Any]:
        return self._engine.resume(run_id)

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._engine.get_status(run_id)

    def get_latest_for_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        return self._engine.get_latest_for_bot(bot_id)
