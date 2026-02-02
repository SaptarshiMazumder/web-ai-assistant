from celery import Celery
from common.config import config

# Get Redis URL from config
redis_url = config.CELERY_BROKER_URL
result_backend = config.CELERY_RESULT_BACKEND

# Create Celery app
celery_app = Celery(
    "web_ai",
    broker=redis_url,
    backend=result_backend,
)

# Load configuration
celery_app.config_from_object("infrastructure.celery_config")

# Import tasks to register them (must be after config and app creation)
# This import registers the task decorators
try:
    from infrastructure.tasks import crawl_tasks  # noqa: E402, F401
except ImportError as e:
    import sys
    print(f"WARNING: Failed to import crawl_tasks: {e}", file=sys.stderr)
try:
    from infrastructure.tasks import discovery_tasks  # noqa: E402, F401
except ImportError as e:
    import sys
    print(f"WARNING: Failed to import discovery_tasks: {e}", file=sys.stderr)

if __name__ == "__main__":
    print(f"Registered tasks: {list(celery_app.tasks.keys())}")
    celery_app.start()
