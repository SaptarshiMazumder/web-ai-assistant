import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from celery import Task
from celery.exceptions import Retry

from infrastructure.celery_app import celery_app
from infrastructure.rag.crawl_service import CRAWL_MAX_DEPTH, CRAWL_MAX_CONCURRENCY
from infrastructure.repositories import Crawl4AICrawlerRepository, GCSDocumentStorageRepository, VertexRAGRepository
from infrastructure.db.repositories import PostgresIndexJobRepository
from google.cloud import storage
import google.auth
import vertexai


def _emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit event for progress tracking (for backward compatibility)."""
    # In Celery, we update DB directly, but can also log for monitoring
    print(f"WEB_AI_EVENT {json.dumps({'type': event_type, **data}, ensure_ascii=False)}", flush=True)


async def _execute_crawl(
    job_id: str,
    bot_id: str,
    url: Optional[str],
    urls: Optional[List[str]],
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
) -> Dict[str, Any]:
    """Execute the crawl job (async function)."""
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    try:
        job.stage = "crawling"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "starting_browser"})
        _emit_event("progress", {"pages_crawled": 0, "url": (url or ""), "depth": 0})
        _emit_event("stage", {"stage": "crawling"})

        crawler_repo = Crawl4AICrawlerRepository()

        def _on_progress(evt: Dict[str, Any]):
            if evt.get("type") == "page_crawled":
                job.pages_crawled = int(evt.get("count") or 0)
                job.last_crawled_url = str(evt.get("url") or "")
                job.last_depth = int(evt.get("depth") if evt.get("depth") is not None else -1)
                job_repo.update_job(job)
                _emit_event("progress", {
                    "pages_crawled": job.pages_crawled,
                    "url": job.last_crawled_url,
                    "depth": job.last_depth,
                })
            elif evt.get("type") == "fetch":
                _emit_event("fetch", {
                    "url": str(evt.get("url") or ""),
                    "success": bool(evt.get("success")),
                    "status_code": evt.get("status_code"),
                    "error": str(evt.get("error") or ""),
                    "content_source": str(evt.get("content_source") or ""),
                    "markdown_len": int(evt.get("markdown_len") or 0),
                    "text_len": int(evt.get("text_len") or 0),
                    "extracted_text_len": int(evt.get("extracted_text_len") or 0),
                    "cleaned_html_len": int(evt.get("cleaned_html_len") or 0),
                    "html_len": int(evt.get("html_len") or 0),
                    "raw_html_len": int(evt.get("raw_html_len") or 0),
                })

        if urls:
            docs = await crawler_repo.crawl_urls_list(
                urls,
                max_concurrent=CRAWL_MAX_CONCURRENCY,
                progress_cb=_on_progress,
            )
        else:
            docs = await crawler_repo.crawl_urls_bfs(
                url or "",
                max_depth=CRAWL_MAX_DEPTH,
                max_concurrent=CRAWL_MAX_CONCURRENCY,
                stop_event=None,
                progress_cb=_on_progress,
            )

        job.docs_count = len(docs)
        job_repo.update_job(job)
        _emit_event("result", {"docs_count": len(docs)})

        if not docs:
            job.stage = "done"
            job_repo.update_job(job)
            _emit_event("stage", {"stage": "done"})
            return {"status": "done", "docs_count": 0}

        job.stage = "uploading"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "uploading"})

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds_path:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

        creds, proj = google.auth.load_credentials_from_file(creds_path)
        _emit_event("auth", {
            "creds_path": creds_path,
            "creds_type": "service_account" if getattr(creds, "service_account_email", None) else "non_service_account",
            "project": proj,
        })

        bot_id_from_prefix = ""
        if "/bots/" in base_prefix:
            parts = base_prefix.split("/bots/")
            if len(parts) > 1:
                bot_id_from_prefix = parts[1].split("/")[0]

        storage_client = storage.Client(credentials=creds, project=proj)
        storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
        gcs_prefix = storage_repo.save_documents(bot_id_from_prefix, docs)
        job.gcs_prefix = gcs_prefix
        job_repo.update_job(job)
        _emit_event("gcs_prefix", {"gcs_prefix": gcs_prefix})

        job.stage = "importing"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "importing"})

        try:
            vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
        except Exception:
            pass

        rag_repo = VertexRAGRepository()
        rag_repo.import_documents(corpus_resource, gcs_prefix)

        job.stage = "import_submitted"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "import_submitted"})

        return {"status": "done", "docs_count": len(docs), "gcs_prefix": gcs_prefix}

    except Exception as e:
        job.stage = "error"
        job.last_error = str(e)
        job_repo.update_job(job)
        _emit_event("error", {"error": str(e)})
        raise


@celery_app.task(
    name="infrastructure.tasks.crawl_tasks.crawl_job",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def crawl_job_task(
    self: Task,
    job_id: str,
    bot_id: str,
    url: Optional[str],
    urls: Optional[List[str]],
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
) -> Dict[str, Any]:
    """
    Celery task to execute crawling job.
    
    Args:
        self: Celery task instance (for retries)
        job_id: Job identifier
        bot_id: Bot identifier
        url: Single URL for BFS crawl (if urls is None)
        urls: List of URLs for direct crawl (if url is None)
        bucket_name: GCS bucket name
        base_prefix: GCS base prefix
        corpus_resource: Vertex AI RAG corpus resource name
    
    Returns:
        Dict with status and results
    """
    # Store celery_task_id in job
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if job:
        job.celery_task_id = self.request.id
        job_repo.update_job(job)
    
    try:
        # Run async function - create new event loop for Celery worker
        result = asyncio.run(
            _execute_crawl(job_id, bot_id, url, urls, bucket_name, base_prefix, corpus_resource)
        )
        return result
    except (ConnectionError, TimeoutError, OSError) as exc:
        # Retry transient errors
        raise self.retry(exc=exc)
    except Exception as exc:
        # Don't retry other errors (permanent failures)
        raise
