"""Periodic task to check and run scheduled source syncs."""
import asyncio
import logging
from datetime import datetime, timezone

from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresBotSourceRepository

logger = logging.getLogger(__name__)


@celery_app.task(
    name="infrastructure.tasks.sync_tasks.check_and_run_source_syncs",
    bind=True,
    max_retries=0,
    time_limit=300,
    soft_time_limit=280,
)
def check_and_run_source_syncs(self):
    """Check for URL sources due for auto-sync and dispatch re-crawl for each."""
    source_repo = PostgresBotSourceRepository()
    try:
        due_sources = source_repo.list_sources_due_for_sync()
    except Exception as e:
        logger.error("Failed to query sources due for sync: %s", e)
        return {"status": "error", "error": str(e)}

    if not due_sources:
        logger.info("No sources due for sync")
        return {"status": "ok", "synced": 0}

    synced = 0
    errors = []
    for source in due_sources:
        bot_id = source.bot_id
        source_id = source.source_id
        url = (source.config or {}).get("url", "") if isinstance(source.config, dict) else ""
        logger.info("Auto-syncing source %s (bot %s, url %s)", source_id, bot_id, url)
        try:
            # Reuse the indexing service to start a crawl for this source
            from common.di.container import indexing_service
            svc = indexing_service()
            result = asyncio.run(
                svc.start_indexing_for_source(bot_id, source_id)
            )
            # Update last_synced_at
            source_repo.update_last_synced(bot_id, source_id)
            synced += 1
            logger.info("Auto-sync started for source %s: %s", source_id, result.get("job_id", "?"))
        except Exception as e:
            logger.error("Failed to auto-sync source %s (bot %s): %s", source_id, bot_id, e)
            errors.append({"source_id": source_id, "bot_id": bot_id, "error": str(e)})

    return {"status": "ok", "synced": synced, "errors": errors}
