import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import google.auth
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions
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


@celery_app.task(
    name="infrastructure.tasks.docs_source_tasks.docs_source_ingest_job",
    bind=True,
    max_retries=6,
    default_retry_delay=20,
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

    source = source_repo.get_source(bot_id, source_id)
    if not source:
        job.stage = "error"
        job.last_error = "Source not found"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    cfg = source.config or {}
    gcs_blob = (cfg.get("gcs_blob") or "").strip() if isinstance(cfg, dict) else ""
    filename = (cfg.get("filename") or "").strip() if isinstance(cfg, dict) else ""

    if not gcs_blob:
        job.stage = "error"
        job.last_error = "File not uploaded (missing gcs_blob)"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    job.stage = "crawling"
    job.last_error = ""
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

    creds, proj = google.auth.load_credentials_from_file(creds_path)
    storage_client = storage.Client(credentials=creds, project=proj)

    file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
    try:
        file_bytes = file_repo.download(blob_name=gcs_blob)
    except gcs_exceptions.NotFound:
        job.stage = "error"
        job.last_error = f"{filename or 'File'} is missing from storage (GCS 404). Re-upload the file."
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}
    except Exception as exc:
        job.stage = "error"
        job.last_error = f"Failed to download file: {str(exc)[:200]}"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    logger.info("[DOCS %s] ingest start filename=%s bytes=%d", source_id[:8], filename, len(file_bytes))

    text = _extract_text_from_file(file_bytes, filename)
    if not text.strip():
        job.stage = "done"
        job.docs_count = 0
        job.last_error = "No text extracted from file"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "gcs_prefix": ""}

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
        job.stage = "done"
        job.docs_count = 0
        job.last_error = "No chunks produced from file"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "gcs_prefix": ""}

    job.pages_crawled = len(docs)
    job.docs_count = len(docs)
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    logger.info("[DOCS %s] chunked filename=%s chunks=%d", source_id[:8], filename, len(docs))

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
