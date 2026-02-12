"""
Background URL discovery task: full discovery (no time limit) for a bot.
Enqueued when the user starts training from create-bot; results shown on Knowledge tab.
Updates discovered_urls periodically so the dashboard can show "X URLs so far" while running.
"""
import asyncio
import logging
import time

from common.di.container import url_discovery
from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresDiscoveryJobRepository

logger = logging.getLogger(__name__)

_PROGRESS_UPDATE_EVERY = 50  # update DB every N URLs so dashboard poll sees progress
_MAX_DISCOVERY_DURATION_SEC = 180  # HARD 3-MINUTE LIMIT for background discovery


@celery_app.task(bind=True, name="discovery.discovery_job_task", time_limit=_MAX_DISCOVERY_DURATION_SEC + 10, soft_time_limit=_MAX_DISCOVERY_DURATION_SEC)
def discovery_job_task(self, job_id: str, bot_id: str, root_url: str, method: str) -> None:
    """
    Run full URL discovery with HARD 3-MINUTE LIMIT. Update discovery_jobs row with status and
    discovered_urls; write progress periodically so the UI can show "X URLs so far".
    
    CRITICAL: This task has a hard timeout of 3 minutes. Under NO circumstances should it run longer.
    """
    repo = PostgresDiscoveryJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        logger.warning("Discovery job %s not found for bot %s", job_id, bot_id)
        return
    
    start_time = time.monotonic()
    
    try:
        job.status = "running"
        job.celery_task_id = self.request.id
        repo.update(job)

        async def run_stream() -> None:
            adapter = url_discovery()
            accumulated: list[str] = []
            # ENFORCE HARD 3-MINUTE TIMEOUT ON BACKEND
            stream = adapter.discover_stream(root_url, method or "auto", max_duration_sec=_MAX_DISCOVERY_DURATION_SEC)
            async for evt in stream:
                # Double-check timeout in case adapter doesn't enforce it
                elapsed = time.monotonic() - start_time
                if elapsed >= _MAX_DISCOVERY_DURATION_SEC:
                    logger.warning(
                        "Background discovery TIMEOUT for job %s (bot %s) after %.1fs. Stopping with %d URLs.",
                        job_id, bot_id, elapsed, len(accumulated)
                    )
                    break
                    
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
            
            elapsed = time.monotonic() - start_time
            job.discovered_urls = accumulated
            job.status = "done"
            job.error = None
            repo.update(job)
            logger.info(
                "Background discovery job %s COMPLETED: discovered=%d URLs in %.1fs (bot=%s, url=%s)",
                job_id, len(accumulated), elapsed, bot_id, root_url
            )

        asyncio.run(run_stream())
    except asyncio.CancelledError:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Background discovery job %s CANCELLED after %.1fs (bot=%s). Task was terminated.",
            job_id, elapsed, bot_id
        )
        job.status = "failed"
        job.error = f"Discovery cancelled/timeout after {int(elapsed)}s"
        repo.update(job)
        raise
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.exception(
            "Background discovery job %s FAILED after %.1fs (bot=%s): %s",
            job_id, elapsed, bot_id, e
        )
        job.status = "failed"
        job.error = f"{type(e).__name__}: {str(e)}"
        repo.update(job)
