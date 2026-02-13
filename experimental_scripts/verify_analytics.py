import sys
import os
import time
from datetime import datetime, timezone
from redis import Redis

# Add backend to path so we can import modules
sys.path.append(os.path.join(os.getcwd(), "backend"))

from infrastructure.db.connection import get_connection
from domain.repositories import PostgresBotRepository, PostgresOrgRepository
from application.services.conversation_service import ConversationService
from common.config import config
from infrastructure.tasks.analytics_tasks import rollup_analytics_task

def verify():
    print("=== Starting Analytics Verification ===")
    
    # 1. Setup Data
    repo_bot = PostgresBotRepository()
    repo_org = PostgresOrgRepository()
    service_conv = ConversationService()
    redis = Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)

    # Ensure clean state
    # Create test org/bot
    org = repo_org.create_org("Test Org Analytics")
    bot = repo_bot.create_bot("Test Bot Analytics", org)
    print(f"Created Bot: {bot.bot_id} in Org: {org}")

    # 2. Simulate Conversation
    print("Simulating conversation...")
    session = service_conv.get_or_create_session(
        bot_id=bot.bot_id,
        org_id=org,
        channel="web",
        site_url="https://example.com"
    )
    
    # Add a user message that should trigger topic extraction
    # The topic extraction regex in repositories.py splits by space and takes first 6 words.
    msg_content = "I want to book a demo for your product pricing"
    service_conv.add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=msg_content
    )
    print(f"Added message: '{msg_content}'")

    # 3. Verify Redis Dirty Flag
    print("Checking Redis for dirty flag...")
    dirty_bots = redis.smembers("analytics:dirty_bots")
    if bot.bot_id in dirty_bots:
        print("✅ SUCCESS: Bot ID found in analytics:dirty_bots")
    else:
        print(f"❌ FAILURE: Bot ID {bot.bot_id} NOT found in analytics:dirty_bots. Found: {dirty_bots}")
        return

    # 4. Run Rollup Task Manually
    print("Running rollup task...")
    try:
        rollup_analytics_task() # Call directly (synchronously)
        print("Rollup task completed.")
    except Exception as e:
        print(f"❌ FAILURE: Rollup task raised exception: {e}")
        return

    # 5. Verify Database
    print("Verifying database for extracted topics...")
    # Clean up Redis just in case (task should have done it)
    remaining_dirty = redis.smembers("analytics:dirty_bots")
    if bot.bot_id in remaining_dirty:
        print("⚠️ WARNING: Bot ID still in dirty set (Task might use spop so this race is possible if local run didn't use real redis properly? No wait task uses spop).")
        # Ensure we check if it was popped. 
        # Actually task uses spop, so it should be gone.
        # Let's check if it's gone.
        pass
    else:
        print("✅ SUCCESS: Bot ID removed from Redis dirty set")

    # Check bot_topics_daily
    con = get_connection()
    try:
        rows = con.execute(
            "SELECT topic, count FROM bot_topics_daily WHERE bot_id = %s",
            (bot.bot_id,)
        ).fetchall()
        
        print(f"Found {len(rows)} topic rows.")
        for r in rows:
            print(f" - Topic: '{r[0]}', Count: {r[1]}")
            
        # We expect "i want to book a demo" or similar (first 6 words)
        expected_topic = "i want to book a demo"
        found = any(expected_topic in r[0] for r in rows)
        
        if found:
            print("✅ SUCCESS: Expected topic found in database!")
        else:
            print("❌ FAILURE: Expected topic NOT found.")
            
    finally:
        con.close()
        # Cleanup
        # repo_bot.delete_bot(bot.bot_id) # Optional: keep for inspection

if __name__ == "__main__":
    verify()
