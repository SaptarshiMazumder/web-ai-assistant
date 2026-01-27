from typing import Dict, List, Optional
from datetime import datetime, timezone

from domain.entities import IndexJob
from domain.repositories import IndexJobRepository


class InMemoryIndexJobRepository(IndexJobRepository):
    def __init__(self) -> None:
        self._jobs: Dict[str, IndexJob] = {}

    def _job_key(self, bot_id: str, hostname: str) -> str:
        return f"{bot_id}::{hostname}"

    def _batch_job_key(self, bot_id: str, job_id: str) -> str:
        return f"{bot_id}::batch::{job_id}"

    def create_job(self, job: IndexJob) -> None:
        if job.hostname == "batch":
            key = self._batch_job_key(job.bot_id, job.job_id)
        else:
            key = self._job_key(job.bot_id, job.hostname)
        self._jobs[key] = job

    def get_job(self, bot_id: str, job_key: str) -> Optional[IndexJob]:
        # Try batch key first
        batch_key = self._batch_job_key(bot_id, job_key)
        if batch_key in self._jobs:
            return self._jobs[batch_key]
        # Try regular key
        if job_key in self._jobs:
            return self._jobs[job_key]
        # Search by job_id
        for job in self._jobs.values():
            if job.bot_id == bot_id and job.job_id == job_key:
                return job
        return None

    def get_job_by_hostname(self, bot_id: str, hostname: str) -> Optional[IndexJob]:
        key = self._job_key(bot_id, hostname)
        return self._jobs.get(key)

    def update_job(self, job: IndexJob) -> None:
        job.updated_at = datetime.now(timezone.utc).isoformat()
        if job.hostname == "batch":
            key = self._batch_job_key(job.bot_id, job.job_id)
        else:
            key = self._job_key(job.bot_id, job.hostname)
        if key in self._jobs:
            self._jobs[key] = job

    def list_jobs_for_bot(self, bot_id: str) -> List[IndexJob]:
        jobs = [job for job in self._jobs.values() if job.bot_id == bot_id]
        return sorted(jobs, key=lambda j: j.updated_at, reverse=True)
