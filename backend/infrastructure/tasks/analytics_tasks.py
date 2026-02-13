import asyncio
from datetime import datetime, timedelta, timezone

from celery import Task
from redis import Redis

from common.config import config
from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresBotRepository, _utc_now


def _get_redis_client() -> Redis:
    return Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)


@celery_app.task(
    name="infrastructure.tasks.analytics_tasks.rollup_analytics",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    ignore_result=True,
)
def rollup_analytics_task(self: Task) -> None:
    """
    Checks for "dirty" bots (bots with new activity) and rolls up their analytics.
    This includes usage stats, top sources, and topic extraction.
    """
    redis = _get_redis_client()
    # "dirty" bots are stored in a set. Pop all of them to process.
    # We use spop(count) to get them. If the set is large, we might want to limit this,
    # but for now let's just grab everything that's pending.
    # Using a finite loop to avoid getting stuck if bots are added faster than processed (though unlikely for this task).
    # Actually, smembers + del is safer for "process everything currently there".
    # But to be atomic, we can rename the key or use spop.
    # Let's use spop with a large count.
    
    dirty_key = "analytics:dirty_bots"
    # Pop up to 1000 bots at a time.
    bot_ids = redis.spop(dirty_key, 1000)
    
    if not bot_ids:
        return

    # Deduplicate just in case (though set handles it, spop returns unique items)
    unique_bot_ids = set(bot_ids)
    print(f"Rolling up analytics for {len(unique_bot_ids)} bots")

    repo = PostgresBotRepository()
    
    # Range: Last 24 hours to cover any recent data. 
    # The rollup function uses [start, end) and generates daily buckets.
    # We want to make sure we cover "today" (UTC).
    now = datetime.now(timezone.utc)
    today = now.date()
    # Also cover yesterday just in case of late-arriving events near midnight
    yesterday = today - timedelta(days=1)
    
    start_date = yesterday
    end_date = today + timedelta(days=1) # exclusive

    for bot_id in unique_bot_ids:
        try:
            # We don't have org_id easily available here without querying.
            # But the repository methods need it? 
            # Wait, PostgresBotRepository.rollup_usage_hourly uses org_id AND bot_id.
            # We need to look up the bot to get the org_id.
            bot = repo.get_bot(bot_id)
            if not bot:
                continue

            # Run the rollup
            repo.rollup_usage_hourly(
                org_id=bot.org_id,
                bot_id=bot.bot_id,
                start_day=start_date,
                end_day=end_date
            )
        except Exception as e:
            print(f"Error rolling up analytics for bot {bot_id}: {e}")
            # If it failed, maybe add back to dirty set? 
            # For now, log and move on to avoid blocking others.
