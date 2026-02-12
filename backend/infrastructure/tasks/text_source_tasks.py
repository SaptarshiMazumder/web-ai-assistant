import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import google.auth
from google.cloud import storage
import vertexai

from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresIndexJobRepository, PostgresBotSourceRepository
from infrastructure.repositories.gcs_document_storage_repository import GCSDocumentStorageRepository
from infrastructure.repositories.vertex_rag_repository import VertexRAGRepository
from infrastructure.repositories.gcs_source_file_repository import GcsSourceFileRepository
from infrastructure.rag.text_chunker import chunk_text


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@celery_app.task(
    name="infrastructure.tasks.text_source_tasks.text_source_ingest_job",
    bind=True,
    max_retries=6,
    default_retry_delay=20,
)
def text_source_ingest_job(
    self,
    *,
    job_id: str,
    bot_id: str,
    source_id: str,
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
) -> Dict[str, Any]:
    """
    Ingest a text/custom source:
    - read content from source config (or download from GCS for large entries)
    - chunk into overlapping segments
    - upload extracted markdown docs to GCS
    - import into Vertex RAG corpus
    """
    job_repo = PostgresIndexJobRepository()
    source_repo = PostgresBotSourceRepository()

    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise RuntimeError("Index job not found")

    source = source_repo.get_source(bot_id, source_id)
    if not source:
        job.stage = "error"
        job.last_error = "Source not found"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    cfg = source.config or {}
    title = (cfg.get("title") or "").strip() if isinstance(cfg, dict) else ""
    content = (cfg.get("content") or "").strip() if isinstance(cfg, dict) else ""
    gcs_content_blob = (cfg.get("gcs_content_blob") or "").strip() if isinstance(cfg, dict) else ""

    job.stage = "crawling"
    job.last_error = ""
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

    creds, proj = google.auth.load_credentials_from_file(creds_path)
    storage_client = storage.Client(credentials=creds, project=proj)

    # If content was too large to store inline, download from GCS
    if not content and gcs_content_blob:
        file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
        try:
            content = file_repo.download(blob_name=gcs_content_blob).decode("utf-8", errors="replace")
        except Exception as exc:
            job.stage = "error"
            job.last_error = f"Failed to download text content from GCS: {str(exc)[:200]}"
            job.updated_at = _utc_now()
            job_repo.update_job(job)
            return {"status": "error", "error": job.last_error}

    if not content:
        job.stage = "done"
        job.docs_count = 0
        job.last_error = "No text content found"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "gcs_prefix": ""}

    source_url = f"https://text.local/{bot_id}/{source_id}"
    docs = chunk_text(content, title=title or None, source_url=source_url)

    if not docs:
        job.stage = "done"
        job.docs_count = 0
        job.last_error = "No chunks produced from text"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "gcs_prefix": ""}

    job.pages_crawled = len(docs)
    job.docs_count = len(docs)
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    logger.info("[TEXT %s] chunked title=%r chunks=%d", source_id[:8], title, len(docs))

    # Upload chunks as markdown docs to GCS
    job.stage = "uploading"
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
    gcs_prefix = storage_repo.save_documents(bot_id, docs)
    job.gcs_prefix = gcs_prefix
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    # Import into Vertex RAG
    job.stage = "importing"
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    try:
        vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
    except Exception:
        pass

    rag_repo = VertexRAGRepository()
    rag_repo.import_documents(corpus_resource, gcs_prefix)
    job.stage = "import_submitted"
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    return {"status": "done", "docs_count": job.docs_count, "gcs_prefix": gcs_prefix}
