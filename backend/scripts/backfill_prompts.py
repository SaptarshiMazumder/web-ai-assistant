"""
Script to backfill auto-generated system prompts for existing bots.

Iterates through all bots, finds their last successful crawl job, and triggers
the prompt generation task if the bot has no custom instructions.

Usage:
    python scripts/backfill_prompts.py
"""

import logging
import os
import sys

# Add backend directory to path so we can import app modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from infrastructure.db.repositories import PostgresBotRepository, _connect
from infrastructure.tasks.crawl_tasks import prompt_generation_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_prompts():
    logger.info("Starting prompt backfill...")

    bot_repo = PostgresBotRepository()
    bots = bot_repo.list_bots()
    logger.info(f"Found {len(bots)} bots.")

    triggered_count = 0
    skipped_count = 0

    con = _connect()
    try:
        for bot in bots:
            # 1. Check if bot needs a prompt
            existing_instructions = ""
            if bot.agent_config:
                import json
                try:
                    config = json.loads(bot.agent_config)
                    existing_instructions = (config.get("instructions") or "").strip()
                except Exception:
                    pass
            
            if existing_instructions:
                logger.info(f"Skipping bot {bot.bot_id}: already has instructions")
                skipped_count += 1
                continue

            # 2. Find last successful crawl job with GCS prefix
            row = con.execute(
                """
                SELECT gcs_prefix, url, crawled_urls
                FROM index_jobs
                WHERE bot_id = %s AND gcs_prefix IS NOT NULL AND gcs_prefix != ''
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (bot.bot_id,),
            ).fetchone()

            if not row:
                logger.info(f"Skipping bot {bot.bot_id}: no successful crawl job found")
                skipped_count += 1
                continue

            gcs_prefix = row[0]
            root_url = row[1]
            crawled_urls = row[2]

            # If root_url is empty, try to get it from crawled_urls
            if not root_url and crawled_urls:
                try:
                    model_crawled = json.loads(crawled_urls) if isinstance(crawled_urls, str) else crawled_urls
                    if model_crawled and isinstance(model_crawled, list):
                        root_url = model_crawled[0]
                except Exception:
                    pass

            logger.info(f"Triggering prompt generation for bot {bot.bot_id} (URL: {root_url or 'unknown'})")
            
            # Use .delay() to trigger Celery task asynchronously
            prompt_generation_task.delay(
                bot_id=bot.bot_id,
                org_id=bot.org_id,
                gcs_prefix=gcs_prefix,
                root_url=root_url or "",
            )
            triggered_count += 1

    finally:
        con.close()

    logger.info(f"Backfill complete. Triggered: {triggered_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    backfill_prompts()
