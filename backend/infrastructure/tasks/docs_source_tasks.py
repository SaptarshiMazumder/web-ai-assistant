import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import google.auth
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions
import vertexai
from celery.exceptions import SoftTimeLimitExceeded

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
        "[DOCS %s] job=%s bot=%s stage %s -> %s docs=%s pages=%s detail=%s",
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
    logger.error("[DOCS %s] job=%s bot=%s terminal_error=%s", source_id[:8], getattr(job, "job_id", ""), bot_id, error_msg)
    return error_msg


def _extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from .txt, .md, .docx, or .doc files.
    Returns empty string if extraction fails.
    """
    ext = os.path.splitext((filename or "").lower())[1]

    if ext in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="replace")

    if ext == ".docx":
        try:
            import io
            import docx  # python-docx
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            return "\n\n".join(paragraphs)
        except Exception as exc:
            logger.warning("Failed to extract .docx text: %s", exc)
            return ""

    if ext == ".doc":
        try:
            import io
            import mammoth  # type: ignore
            result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
            return result.value or ""
        except Exception as exc:
            logger.warning("Failed to extract .doc text via mammoth: %s", exc)
            return ""

    logger.warning("Unsupported file extension for text extraction: %s", ext)
    return ""


_DOCS_INGEST_SOFT_LIMIT_SEC = 300
_DOCS_INGEST_HARD_LIMIT_SEC = 320


@celery_app.task(
    name="infrastructure.tasks.docs_source_tasks.docs_source_ingest_job",
    bind=True,
    max_retries=6,
    default_retry_delay=20,
    soft_time_limit=_DOCS_INGEST_SOFT_LIMIT_SEC,
    time_limit=_DOCS_INGEST_HARD_LIMIT_SEC,
)
def docs_source_ingest_job(
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
    Ingest a document source (.txt, .md, .docx, .doc):
    - download raw file from GCS
    - extract text based on file type
    - chunk into overlapping segments
    - upload extracted markdown docs to GCS
    - import into Vertex RAG corpus
    """
    job_repo = PostgresIndexJobRepository()
    source_repo = PostgresBotSourceRepository()

    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise RuntimeError("Index job not found")
    logger.info("[DOCS %s] task_started job=%s bot=%s", source_id[:8], job_id, bot_id)

    try:
        return _execute_docs_ingest(job_repo, source_repo, job, job_id=job_id, bot_id=bot_id, source_id=source_id, bucket_name=bucket_name, base_prefix=base_prefix, corpus_resource=corpus_resource)
    except SoftTimeLimitExceeded:
        logger.error("[DOCS %s] job=%s bot=%s TIMEOUT after %ds", source_id[:8], job_id, bot_id, _DOCS_INGEST_SOFT_LIMIT_SEC)
        error_msg = _mark_job_error(job_repo, job, bot_id, source_id, job.stage or "importing", "Task timed out — the RAG import may still be processing. Please retry.", error_kind="SoftTimeLimitExceeded")
        return {"status": "error", "error": error_msg}


def _execute_docs_ingest(job_repo, source_repo, job, *, job_id, bot_id, source_id, bucket_name, base_prefix, corpus_resource):
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
    gcs_blob = (cfg.get("gcs_blob") or "").strip() if isinstance(cfg, dict) else ""
    filename = (cfg.get("filename") or "").strip() if isinstance(cfg, dict) else ""

    if not gcs_blob:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "File not uploaded (missing gcs_blob)",
            error_kind="MissingFileBlob",
        )
        return {"status": "error", "error": error_msg}

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

    file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
    try:
        file_bytes = file_repo.download(blob_name=gcs_blob)
    except gcs_exceptions.NotFound:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            f"{filename or 'File'} is missing from storage (GCS 404). Re-upload the file.",
            error_kind="BlobNotFound",
        )
        return {"status": "error", "error": error_msg}
    except Exception as exc:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "Failed to download file",
            exc=exc,
            error_kind="FileDownloadFailed",
        )
        return {"status": "error", "error": error_msg}

    logger.info("[DOCS %s] ingest start filename=%s bytes=%d", source_id[:8], filename, len(file_bytes))

    text = _extract_text_from_file(file_bytes, filename)
    if not text.strip():
        job.docs_count = 0
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "No text extracted from file",
            error_kind="NoTextExtracted",
        )
        return {"status": "error", "docs_count": 0, "gcs_prefix": "", "error": error_msg}

    # Update source config with char count
    try:
        source.config = dict(source.config or {})
        source.config["char_count"] = len(text)
        source.updated_at = _utc_now()
        source_repo.update_source(source)
    except Exception:
        pass

    source_url = f"https://docs.local/{bot_id}/{source_id}/{filename}"
    title = filename
    docs = chunk_text(text, title=title, source_url=source_url)

    if not docs:
        job.docs_count = 0
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "No chunks produced from file",
            error_kind="NoChunks",
        )
        return {"status": "error", "docs_count": 0, "gcs_prefix": "", "error": error_msg}

    job.pages_crawled = len(docs)
    job.docs_count = len(docs)
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    logger.info("[DOCS %s] chunked filename=%s chunks=%d", source_id[:8], filename, len(docs))

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
            "Failed to upload document chunks",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": "", "error": error_msg}
    job.gcs_prefix = gcs_prefix
    job.updated_at = _utc_now()
    job_repo.update_job(job)
    logger.info("[DOCS %s] upload_complete job=%s bot=%s gcs_prefix=%s docs=%d", source_id[:8], job_id, bot_id, gcs_prefix, len(docs))

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
            f"Failed to import document chunks into RAG: {exc}",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": gcs_prefix, "error": error_msg}
    _set_job_stage(job_repo, job, bot_id, source_id, "import_submitted")
    _set_job_stage(job_repo, job, bot_id, source_id, "done")
    logger.info("[DOCS %s] completed job=%s bot=%s docs=%d stage=%s", source_id[:8], job_id, bot_id, job.docs_count, job.stage)

    return {"status": "done", "docs_count": job.docs_count, "gcs_prefix": gcs_prefix}
