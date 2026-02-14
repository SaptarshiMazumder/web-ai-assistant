from celery import Task
from celery.schedules import crontab

# Task execution settings
task_acks_late = True  # Acknowledge task after completion, not before
worker_prefetch_multiplier = 1  # Fair distribution (one task per worker at a time)
task_reject_on_worker_lost = True  # Re-queue if worker dies

# Time limits
task_time_limit = 3600  # Hard limit: 1 hour (kills task)
task_soft_time_limit = 3300  # Soft limit: 55 minutes (raises exception)

# Retry settings
task_default_retry_delay = 60  # Wait 60 seconds before retry
task_max_retries = 3  # Max retries for failed tasks

# Serialization
task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "UTC"
enable_utc = True

# Result backend settings
result_expires = 3600  # Results expire after 1 hour

# Worker settings (can be overridden via command line)
worker_concurrency = 10  # Default concurrency
worker_max_tasks_per_child = 50  # Restart worker after 50 tasks (prevent memory leaks)
worker_disable_rate_limits = False

# Task routing (disabled for now - use default queue)
# task_routes = {
#     "infrastructure.tasks.crawl_tasks.crawl_job": {"queue": "crawling"},
# }

# Beat schedule (for periodic tasks - not needed now, but ready for future)
beat_schedule = {
    "rollup-analytics-hourly": {
        "task": "infrastructure.tasks.analytics_tasks.rollup_analytics",
        "schedule": crontab(minute="0"),  # Run once an hour
    },
    "refresh-instagram-tokens-daily": {
        "task": "infrastructure.tasks.instagram_tasks.refresh_instagram_tokens",
        "schedule": crontab(hour="3", minute="0"),  # Run daily at 3 AM UTC
    },
}
