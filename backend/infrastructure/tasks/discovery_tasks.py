"""
Background URL discovery task: full discovery (no time limit) for a bot.
Enqueued when the user starts training from create-bot; results shown on Knowledge tab.
Updates discovered_urls periodically so the dashboard can show "X URLs so far" while running.
"""
import asyncio
import logging

from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresDiscoveryJobRepository
from infrastructure.rag.url_discovery_adapter import HttpUrlDiscoveryAdapter

logger = logging.getLogger(__name__)

_PROGRESS_UPDATE_EVERY = 50  # update DB every N URLs so dashboard poll sees progress


@celery_app.task(bind=True, name="discovery.discovery_job_task")
def discovery_job_task(self, job_id: str, bot_id: str, root_url: str, method: str) -> None:
    """
    Run full URL discovery (no max_duration_sec). Update discovery_jobs row with status and
    discovered_urls; write progress periodically so the UI can show "X URLs so far".
    """
    repo = PostgresDiscoveryJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        logger.warning("Discovery job %s not found for bot %s", job_id, bot_id)
        return
    try:
        job.status = "running"
        job.celery_task_id = self.request.id
        repo.update(job)

        async def run_stream() -> None:
            adapter = HttpUrlDiscoveryAdapter()
            accumulated: list[str] = []
            stream = adapter.discover_stream(root_url, method or "auto")
            async for evt in stream:
                if evt.get("type") == "discovered" and evt.get("url"):
                    url = evt["url"]
                    if url not in accumulated:
                        accumulated.append(url)
                    n = len(accumulated)
                    # Update DB so dashboard poll shows "X URLs so far" (often in first 10, then every 50)
                    if n <= 10 or n % _PROGRESS_UPDATE_EVERY == 0:
                        job.discovered_urls = list(accumulated)
                        job.status = "running"
                        repo.update(job)
                if evt.get("type") == "done":
                    if evt.get("urls"):
                        accumulated[:] = list(evt["urls"])
                    break
                if evt.get("type") == "error":
                    raise RuntimeError(evt.get("message", "Discovery error"))
            job.discovered_urls = accumulated
            job.status = "done"
            job.error = None
            repo.update(job)
            logger.info("Discovery job %s completed: %s URLs", job_id, len(accumulated))

        asyncio.run(run_stream())
    except Exception as e:
        logger.exception("Discovery job %s failed", job_id)
        job.status = "failed"
        job.error = f"{type(e).__name__}: {str(e)}"
        repo.update(job)
