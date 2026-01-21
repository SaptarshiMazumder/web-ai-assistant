from fastapi import APIRouter, Query
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from models import WebsiteRagRequest
from .vertex_rag_eg import run_vertex_rag
from config import config
from .crawl_to_gcs import (
    list_existing_site_prefixes,
    crawl_site_bfs,
    upload_markdown_docs_to_gcs,
    import_gcs_prefix_into_corpus,
    RAG_CORPUS,
    CRAWL_MAX_DEPTH,
    CRAWL_MAX_CONCURRENCY,
)
import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


router = APIRouter()

@dataclass
class _IndexJob:
    job_id: str
    url: str
    stop_event: asyncio.Event
    task: asyncio.Task
    created_at: str
    stage: str = "queued"  # queued|crawling|uploading|importing|import_submitted|cancelled|error|done
    pages_crawled: int = 0
    last_crawled_url: str = ""
    last_depth: int = -1
    gcs_prefix: str = ""
    last_error: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

_index_jobs_by_url: Dict[str, _IndexJob] = {}

def _job_key(url: str) -> str:
    return (url or "").strip()

def _attach_task_exception_sink(task: asyncio.Task) -> None:
    # Prevent "Task exception was never retrieved" by consuming exceptions.
    def _done_cb(t: asyncio.Task):
        try:
            _ = t.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    task.add_done_callback(_done_cb)

def _touch(job: _IndexJob) -> None:
    job.updated_at = datetime.now(timezone.utc).isoformat()


@router.post("/ask-website-rag")
async def ask_website_rag(request: WebsiteRagRequest):
    domain = request.domain
    try:
        if not domain and request.page_url:
            domain = urlparse(request.page_url).hostname
    except Exception:
        pass
    print(f"[Website RAG] domain: {domain}")
    result = run_vertex_rag(request.question)
    return result


@router.get("/is-indexed")
async def is_indexed(url: str):
    print("Checking indexed or not...")
    from urllib.parse import urlparse as _urlparse
    host = ""
    try:
        parsed = _urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = ""
    if not host and url:
        host = url.strip()

    def host_variants(h: str):
        if not h:
            return []
        base = h.split(":")[0].lower()
        nowww = base[4:] if base.startswith("www.") else base
        parts = nowww.split(".")
        apex = nowww if len(parts) < 2 else ".".join(parts[-2:])
        return [v for v in {base, nowww, apex, f"www.{apex}"} if v]

    variants = host_variants(host)
    try:
        bucket_and_prefix = (config.GCS_BUCKET or '').strip('/').split('/', 1)
        if len(bucket_and_prefix) == 2:
            bucket_name, base_prefix = bucket_and_prefix[0], bucket_and_prefix[1]
        else:
            bucket_name, base_prefix = bucket_and_prefix[0], ''
        for hv in variants:
            try:
                prefixes = list_existing_site_prefixes(
                    bucket_name=bucket_name,
                    base_prefix=base_prefix,
                    site_url=f"https://{hv}",
                )
                if prefixes:
                    return {"indexed": True, "host": host, "source": "gcs"}
            except Exception:
                pass
    except Exception:
        pass

    # Only use GCS status; do not fall back to local raw_pages
    return {"indexed": False, "host": host, "source": "gcs"}


@router.post("/index-site")
async def index_site(payload: Dict[str, Any]):
    url: Optional[str] = (payload or {}).get("url") if isinstance(payload, dict) else None
    if not url:
        return {"status": "error", "message": "Missing 'url' in request body."}
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        bucket_and_prefix = (config.GCS_BUCKET or '').strip('/').split('/', 1)
        if len(bucket_and_prefix) == 2:
            bucket_name, base_prefix = bucket_and_prefix[0], bucket_and_prefix[1]
        else:
            bucket_name, base_prefix = bucket_and_prefix[0], ''
    except Exception:
        return {"status": "error", "message": "Invalid GCS_BUCKET configuration."}

    key = _job_key(url)
    # If there's an existing job for this URL, stop it and replace it.
    existing = _index_jobs_by_url.get(key)
    if existing and not existing.stop_event.is_set():
        existing.stop_event.set()

    job_id = uuid.uuid4().hex
    stop_event = asyncio.Event()

    async def crawl_and_upload():
        try:
            cur = _index_jobs_by_url.get(key)
            if cur and cur.job_id == job_id:
                cur.stage = "crawling"
                _touch(cur)

            def _on_progress(evt: Dict[str, Any]):
                cur2 = _index_jobs_by_url.get(key)
                if not cur2 or cur2.job_id != job_id:
                    return
                if evt.get("type") == "page_crawled":
                    cur2.pages_crawled = int(evt.get("count") or cur2.pages_crawled)
                    cur2.last_crawled_url = str(evt.get("url") or "")
                    cur2.last_depth = int(evt.get("depth") if evt.get("depth") is not None else cur2.last_depth)
                    _touch(cur2)

            docs = await crawl_site_bfs(
                url,
                max_depth=CRAWL_MAX_DEPTH,
                max_concurrent=CRAWL_MAX_CONCURRENCY,
                stop_event=stop_event,
                progress_cb=_on_progress,
            )
            if docs:
                cur = _index_jobs_by_url.get(key)
                if cur and cur.job_id == job_id:
                    cur.stage = "uploading"
                    _touch(cur)
                gcs_prefix = upload_markdown_docs_to_gcs(
                    bucket_name=bucket_name,
                    base_prefix=base_prefix,
                    docs=docs,
                )
                cur = _index_jobs_by_url.get(key)
                if cur and cur.job_id == job_id:
                    cur.gcs_prefix = gcs_prefix
                    cur.stage = "importing"
                    _touch(cur)
                # Make uploaded pages searchable by Website RAG (Vertex RAG corpus).
                # Note: import is async on the Vertex side; it can take a few minutes to become queryable.
                try:
                    import_gcs_prefix_into_corpus(
                        corpus_resource=RAG_CORPUS,
                        bucket_name=bucket_name,
                        prefix=gcs_prefix,
                    )
                    cur = _index_jobs_by_url.get(key)
                    if cur and cur.job_id == job_id:
                        cur.stage = "import_submitted"
                        _touch(cur)
                except Exception as e:
                    # Don't fail the whole job if import fails; GCS upload succeeded.
                    print(f"[index-site] Warning: RAG import failed for {gcs_prefix}: {e}")
                    cur = _index_jobs_by_url.get(key)
                    if cur and cur.job_id == job_id:
                        cur.stage = "error"
                        cur.last_error = f"RAG import failed: {e}"
                        _touch(cur)
            else:
                cur = _index_jobs_by_url.get(key)
                if cur and cur.job_id == job_id:
                    cur.stage = "done"
                    _touch(cur)
        except asyncio.CancelledError:
            cur = _index_jobs_by_url.get(key)
            if cur and cur.job_id == job_id:
                cur.stage = "cancelled"
                _touch(cur)
            raise
        except Exception as e:
            try:
                print(f"[index-site] Error while crawling/uploading {url}: {e}")
                print(traceback.format_exc())
            except Exception:
                pass
            cur = _index_jobs_by_url.get(key)
            if cur and cur.job_id == job_id:
                cur.stage = "error"
                cur.last_error = str(e)
                _touch(cur)
        finally:
            # Cleanup job registry
            try:
                cur = _index_jobs_by_url.get(key)
                if cur and cur.job_id == job_id:
                    _index_jobs_by_url.pop(key, None)
            except Exception:
                pass

    task = asyncio.create_task(crawl_and_upload())
    _attach_task_exception_sink(task)
    _index_jobs_by_url[key] = _IndexJob(
        job_id=job_id,
        url=url,
        stop_event=stop_event,
        task=task,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return {"status": "started", "job_id": job_id}


@router.get("/index-job-status")
async def index_job_status(url: str):
    """
    Returns best-effort progress for an active /index-site job.

    Note: Vertex RAG import is asynchronous on Google side; we can only report that
    an import request was submitted, not % completion.
    """
    if not url:
        return {"status": "error", "message": "Missing url"}
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    job = _index_jobs_by_url.get(_job_key(url))
    if not job:
        return {"status": "not_found"}
    return {
        "status": "ok",
        "job_id": job.job_id,
        "url": job.url,
        "stage": job.stage,
        "pages_crawled": job.pages_crawled,
        "last_crawled_url": job.last_crawled_url,
        "last_depth": job.last_depth,
        "gcs_prefix": job.gcs_prefix,
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.post("/cancel-index-site")
async def cancel_index_site(payload: Dict[str, Any]):
    """
    Gracefully stop an in-flight /index-site crawl and upload what has been crawled so far.

    Body: { "url": "https://example.com" }
    """
    url: Optional[str] = (payload or {}).get("url") if isinstance(payload, dict) else None
    if not url:
        return {"status": "error", "message": "Missing 'url' in request body."}
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    job = _index_jobs_by_url.get(_job_key(url))
    if not job:
        return {"status": "not_found"}

    job.stop_event.set()
    # Also cancel the running task so Playwright/crawl4ai unwinds promptly.
    try:
        job.task.cancel()
    except Exception:
        pass

    return {"status": "stopping", "job_id": job.job_id}


