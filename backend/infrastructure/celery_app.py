import importlib
import sys

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

# Import task modules to register decorated tasks.
_TASK_MODULES = (
    "infrastructure.tasks.analytics_tasks",
    "infrastructure.tasks.availability_tasks",
    "infrastructure.tasks.booking_link_tasks",
    "infrastructure.tasks.crawl_tasks",
    "infrastructure.tasks.discovery_tasks",
    "infrastructure.tasks.docs_source_tasks",
    "infrastructure.tasks.instagram_tasks",
    "infrastructure.tasks.job_pipeline_tasks",
    "infrastructure.tasks.pdf_source_tasks",
    "infrastructure.tasks.single_page_crawl_tasks",
    "infrastructure.tasks.sync_tasks",
    "infrastructure.tasks.text_source_tasks",
)

for module_path in _TASK_MODULES:
    try:
        importlib.import_module(module_path)
    except Exception as exc:
        print(f"WARNING: Failed to import {module_path}: {exc}", file=sys.stderr)

if __name__ == "__main__":
    print(f"Registered tasks: {list(celery_app.tasks.keys())}")
    celery_app.start()
