"""Celery tasks for Instagram integration maintenance.

- Token refresh: refreshes OAuth tokens that expire within 7 days.
"""

import logging

from infrastructure.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="infrastructure.tasks.instagram_tasks.refresh_instagram_tokens")
def refresh_instagram_tokens():
    """Refresh Instagram OAuth tokens expiring within 7 days.

    Runs daily via Celery Beat. Tokens are 60-day long-lived tokens that
    can be refreshed once they're at least 24 hours old.
    """
    import asyncio
    asyncio.run(_refresh_instagram_tokens_async())


async def _refresh_instagram_tokens_async():
    from datetime import datetime, timedelta, timezone
    from infrastructure.db.repositories import PostgresInstagramChannelRepository
    from infrastructure.clients.instagram_client import refresh_long_lived_token

    repo = PostgresInstagramChannelRepository()
    channels = repo.get_channels_expiring_soon(days=7)

    if not channels:
        logger.info("Instagram token refresh: no tokens expiring soon")
        return

    logger.info("Instagram token refresh: %d tokens to refresh", len(channels))
    refreshed = 0
    failed = 0

    for channel in channels:
        try:
            new_token, expires_in = await refresh_long_lived_token(channel.page_access_token)
            new_expiry = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
            repo.update_token(
                channel.channel_id,
                access_token=new_token,
                token_expires_at=new_expiry,
            )
            refreshed += 1
            logger.info(
                "Instagram token refreshed for bot_id=%s (ig_user_id=%s), new expiry=%s",
                channel.bot_id, channel.ig_user_id, new_expiry,
            )
        except Exception:
            failed += 1
            logger.exception(
                "Instagram token refresh FAILED for bot_id=%s (ig_user_id=%s)",
                channel.bot_id, channel.ig_user_id,
            )

    logger.info("Instagram token refresh complete: %d refreshed, %d failed", refreshed, failed)
