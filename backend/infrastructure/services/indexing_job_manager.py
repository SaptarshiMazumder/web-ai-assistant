import asyncio
import json
import subprocess
from typing import Dict, Optional

from domain.entities import IndexJob
from domain.repositories import IndexJobRepository


class IndexingJobManager:
    """Manages worker processes and updates job state from worker output."""

    def __init__(self, job_repo: IndexJobRepository) -> None:
        self._job_repo = job_repo
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_tasks: Dict[str, asyncio.Task] = {}

    def start_job_process(
        self,
        job: IndexJob,
        worker_args: list[str],
        worker_env: dict[str, str],
    ) -> None:
        """Start a worker process for the job."""
        proc = subprocess.Popen(
            worker_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=worker_env,
        )

        self._processes[job.job_id] = proc
        task = asyncio.create_task(self._consume_logs(job, proc))
        self._log_tasks[job.job_id] = task

    def cancel_job_process(self, job_id: str) -> bool:
        """Cancel a running job process."""
        proc = self._processes.get(job_id)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                return True
            except Exception:
                pass
        return False

    async def _consume_logs(self, job: IndexJob, proc: subprocess.Popen) -> None:
        """Consume worker output and update job state."""
        job.stage = "crawling"
        self._job_repo.update_job(job)

        prefix = "WEB_AI_EVENT "
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)  # type: ignore[union-attr]
            if not line:
                break
            s = (line or "").strip()
            if not s:
                continue

            if prefix not in s:
                continue

            payload = s.split(prefix, 1)[1].strip()
            try:
                msg = json.loads(payload)
            except Exception:
                continue

            t = msg.get("type")
            if t == "stage":
                job.stage = str(msg.get("stage") or job.stage)
                self._job_repo.update_job(job)
            elif t == "progress":
                job.pages_crawled = int(msg.get("pages_crawled") or job.pages_crawled)
                job.last_crawled_url = str(msg.get("url") or job.last_crawled_url)
                job.last_depth = int(msg.get("depth") if msg.get("depth") is not None else job.last_depth)
                self._job_repo.update_job(job)
            elif t == "result":
                job.docs_count = int(msg.get("docs_count") or job.docs_count)
                self._job_repo.update_job(job)
            elif t == "gcs_prefix":
                job.gcs_prefix = str(msg.get("gcs_prefix") or job.gcs_prefix)
                self._job_repo.update_job(job)
            elif t == "error":
                job.stage = "error"
                job.last_error = str(msg.get("error") or "")
                self._job_repo.update_job(job)

        rc = proc.poll()
        if rc is None:
            return
        if rc == 0:
            if job.stage not in ("import_submitted", "prompt_queued", "prompt_generating", "done"):
                job.stage = "done"
            job.last_error = ""
        else:
            if job.stage != "error":
                job.stage = "error"
                job.last_error = job.last_error or f"Worker exited with code {rc}"

        self._job_repo.update_job(job)

        # Cleanup
        if job.job_id in self._processes:
            del self._processes[job.job_id]
        if job.job_id in self._log_tasks:
            del self._log_tasks[job.job_id]
