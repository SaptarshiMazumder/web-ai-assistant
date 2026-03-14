import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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


def _format_stage_error(
    stage: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    error_kind: Optional[str] = None,
    max_len: int = 500,
) -> str:
    stage_name = (stage or "runtime").strip().lower() or "runtime"
    detail = (message or "").strip() or "(no details)"
    kind = (error_kind or "").strip() or (type(exc).__name__ if exc is not None else "")
    if kind:
        return f"{stage_name}: {kind}: {detail}"[:max_len]
    return f"{stage_name}: {detail}"[:max_len]


def _set_job_stage(job_repo: PostgresIndexJobRepository, job: Any, bot_id: str, source_id: str, stage: str, *, detail: str = "") -> None:
    previous = str(getattr(job, "stage", "") or "").strip()
    job.stage = stage
    job.updated_at = _utc_now()
    job_repo.update_job(job)
    logger.info(
        "[TEXT %s] job=%s bot=%s stage %s -> %s docs=%s pages=%s detail=%s",
        source_id[:8],
        getattr(job, "job_id", ""),
        bot_id,
        previous or "-",
        stage,
        getattr(job, "docs_count", 0),
        getattr(job, "pages_crawled", 0),
        detail[:220] if detail else "",
    )


def _mark_job_error(
    job_repo: PostgresIndexJobRepository,
    job: Any,
    bot_id: str,
    source_id: str,
    stage: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    error_kind: Optional[str] = None,
) -> str:
    error_msg = _format_stage_error(stage, message, exc=exc, error_kind=error_kind)
    job.last_error = error_msg
    _set_job_stage(job_repo, job, bot_id, source_id, "error", detail=error_msg)
    logger.error("[TEXT %s] job=%s bot=%s terminal_error=%s", source_id[:8], getattr(job, "job_id", ""), bot_id, error_msg)
    return error_msg


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
    logger.info("[TEXT %s] task_started job=%s bot=%s", source_id[:8], job_id, bot_id)

    source = source_repo.get_source(bot_id, source_id)
    if not source:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "Source not found",
            error_kind="SourceMissing",
        )
        return {"status": "error", "error": error_msg}

    cfg = source.config or {}
    title = (cfg.get("title") or "").strip() if isinstance(cfg, dict) else ""
    content = (cfg.get("content") or "").strip() if isinstance(cfg, dict) else ""
    gcs_content_blob = (cfg.get("gcs_content_blob") or "").strip() if isinstance(cfg, dict) else ""

    _set_job_stage(job_repo, job, bot_id, source_id, "crawling")
    job.last_error = ""
    job_repo.update_job(job)

    try:
        from common.gcp_auth import load_gcp_credentials
        creds, proj = load_gcp_credentials()
        storage_client = storage.Client(credentials=creds, project=proj)
    except Exception as exc:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "Failed to initialize GCP clients",
            exc=exc,
            error_kind="AuthInitFailed",
        )
        return {"status": "error", "error": error_msg}

    # If content was too large to store inline, download from GCS
    if not content and gcs_content_blob:
        file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
        try:
            content = file_repo.download(blob_name=gcs_content_blob).decode("utf-8", errors="replace")
        except Exception as exc:
            error_msg = _mark_job_error(
                job_repo,
                job,
                bot_id,
                source_id,
                "crawling",
                "Failed to download text content from GCS",
                exc=exc,
                error_kind="TextDownloadFailed",
            )
            return {"status": "error", "error": error_msg}

    if not content:
        job.docs_count = 0
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "No text content found",
            error_kind="NoTextContent",
        )
        return {"status": "error", "docs_count": 0, "gcs_prefix": "", "error": error_msg}

    source_url = f"https://text.local/{bot_id}/{source_id}"
    docs = chunk_text(content, title=title or None, source_url=source_url)

    if not docs:
        job.docs_count = 0
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "No chunks produced from text",
            error_kind="NoChunks",
        )
        return {"status": "error", "docs_count": 0, "gcs_prefix": "", "error": error_msg}

    job.pages_crawled = len(docs)
    job.docs_count = len(docs)
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    logger.info("[TEXT %s] chunked title=%r chunks=%d", source_id[:8], title, len(docs))

    # Upload chunks as markdown docs to GCS
    _set_job_stage(job_repo, job, bot_id, source_id, "uploading")

    try:
        storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
        gcs_prefix = storage_repo.save_documents(bot_id, docs)
    except Exception as exc:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "uploading",
            "Failed to upload text chunks",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": "", "error": error_msg}
    job.gcs_prefix = gcs_prefix
    job.updated_at = _utc_now()
    job_repo.update_job(job)
    logger.info("[TEXT %s] upload_complete job=%s bot=%s gcs_prefix=%s docs=%d", source_id[:8], job_id, bot_id, gcs_prefix, len(docs))

    # Import into Vertex RAG
    _set_job_stage(job_repo, job, bot_id, source_id, "importing")

    try:
        vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
    except Exception:
        pass

    try:
        rag_repo = VertexRAGRepository()
        rag_repo.import_documents(corpus_resource, gcs_prefix)
    except Exception as exc:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "importing",
            "Failed to import text chunks into RAG",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": gcs_prefix, "error": error_msg}
    _set_job_stage(job_repo, job, bot_id, source_id, "import_submitted")
    logger.info("[TEXT %s] completed job=%s bot=%s docs=%d stage=%s", source_id[:8], job_id, bot_id, job.docs_count, job.stage)

    return {"status": "done", "docs_count": job.docs_count, "gcs_prefix": gcs_prefix}
