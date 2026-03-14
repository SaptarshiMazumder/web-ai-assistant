import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import google.auth
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions
import vertexai

from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresIndexJobRepository, PostgresBotSourceRepository
from infrastructure.repositories.gcs_document_storage_repository import GCSDocumentStorageRepository
from infrastructure.repositories.vertex_rag_repository import VertexRAGRepository
from infrastructure.repositories.gcs_source_file_repository import GcsSourceFileRepository
from infrastructure.rag.pdf_extractor import extract_pdf_pages_text

from domain.entities import Document


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
        "[PDF %s] job=%s bot=%s stage %s -> %s docs=%s pages=%s detail=%s",
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
    logger.error("[PDF %s] job=%s bot=%s terminal_error=%s", source_id[:8], getattr(job, "job_id", ""), bot_id, error_msg)
    return error_msg


def _make_doc_url(bot_id: str, source_id: str, filename: str, page: int) -> str:
    safe = (filename or "document.pdf").strip().replace(" ", "%20")
    return f"https://pdf.local/{bot_id}/{source_id}/{safe}?page={page}"


@celery_app.task(
    name="infrastructure.tasks.pdf_source_tasks.pdf_source_ingest_job",
    bind=True,
    # Vertex RAG corpus imports can be temporarily busy; allow retries.
    max_retries=6,
    default_retry_delay=20,
)
def pdf_source_ingest_job(
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
    Ingest a PDF source:
    - download raw PDF from GCS (stored on BotSource.config)
    - extract text per page (fast: PyMuPDF; fallback: pdfplumber; optional OCR: Gemini/Tesseract)
    - upload extracted markdown docs to GCS
    - import into Vertex RAG corpus
    """
    job_repo = PostgresIndexJobRepository()
    source_repo = PostgresBotSourceRepository()
    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise RuntimeError("Index job not found")
    logger.info("[PDF %s] task_started job=%s bot=%s", source_id[:8], job_id, bot_id)
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
    gcs_blob = (cfg.get("gcs_pdf_blob") or "").strip() if isinstance(cfg, dict) else ""
    filename = (cfg.get("filename") or "").strip() if isinstance(cfg, dict) else ""
    if not gcs_blob:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "PDF not uploaded (missing gcs_pdf_blob)",
            error_kind="MissingPdfBlob",
        )
        return {"status": "error", "error": error_msg}

    _set_job_stage(job_repo, job, bot_id, source_id, "crawling")
    job.last_error = ""
    job_repo.update_job(job)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "GOOGLE_APPLICATION_CREDENTIALS is not set for worker",
            error_kind="MissingCredentials",
        )
        return {"status": "error", "error": error_msg}
    try:
        creds, proj = google.auth.load_credentials_from_file(creds_path)
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

    # Download raw PDF
    file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
    try:
        pdf_bytes = file_repo.download(blob_name=gcs_blob)
    except gcs_exceptions.NotFound:
        display_name = filename or "PDF"
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            f"{display_name} is missing from storage (GCS 404). Re-upload the PDF or recreate the bucket.",
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
            "Failed to download PDF",
            exc=exc,
            error_kind="PdfDownloadFailed",
        )
        return {"status": "error", "error": error_msg}
    if not pdf_bytes:
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "Failed to download PDF from storage",
            error_kind="PdfDownloadEmpty",
        )
        return {"status": "error", "error": error_msg}

    logger.info("[PDF %s] ingest start filename=%s bytes=%d gcs_blob=%s", source_id[:8], filename, len(pdf_bytes), gcs_blob)

    # Extract text per page (logs previews)
    pages = extract_pdf_pages_text(pdf_bytes, filename=filename, log_prefix=f"[PDF {source_id[:8]}] ")

    docs: List[Document] = []
    methods_count: Dict[str, int] = {}
    for p in pages:
        methods_count[p.method] = methods_count.get(p.method, 0) + 1
        text = (p.text or "").strip()
        if not text:
            continue
        url = _make_doc_url(bot_id, source_id, filename, p.page)
        content = f"Source: {filename}\nPage: {p.page}\nMethod: {p.method}\n\n{text}"
        docs.append(Document(url=url, content=content, metadata={"source_type": "pdf", "filename": filename, "page": p.page, "method": p.method}))

        job.pages_crawled = p.page
        job.last_crawled_url = url
        job.docs_count = len(docs)
        job.updated_at = _utc_now()
        job_repo.update_job(job)

    # Update source config with extraction stats
    try:
        source.config = dict(source.config or {})
        source.config.update(
            {
                "page_count": len(pages),
                "docs_count": len(docs),
                "extract_methods": methods_count,
            }
        )
        source.updated_at = _utc_now()
        source_repo.update_source(source)
    except Exception:
        pass

    if not docs:
        job.docs_count = 0
        error_msg = _mark_job_error(
            job_repo,
            job,
            bot_id,
            source_id,
            "crawling",
            "No text extracted from PDF",
            error_kind="NoTextExtracted",
        )
        return {"status": "error", "docs_count": 0, "gcs_prefix": "", "error": error_msg}

    # Upload extracted docs as markdown
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
            "Failed to upload extracted PDF chunks",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": "", "error": error_msg}
    job.gcs_prefix = gcs_prefix
    job.updated_at = _utc_now()
    job_repo.update_job(job)
    logger.info("[PDF %s] upload_complete job=%s bot=%s gcs_prefix=%s docs=%d", source_id[:8], job_id, bot_id, gcs_prefix, len(docs))

    # Import into RAG
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
            "Failed to import PDF chunks into RAG",
            exc=exc,
        )
        return {"status": "error", "docs_count": len(docs), "gcs_prefix": gcs_prefix, "error": error_msg}
    _set_job_stage(job_repo, job, bot_id, source_id, "import_submitted")
    logger.info("[PDF %s] completed job=%s bot=%s docs=%d stage=%s", source_id[:8], job_id, bot_id, job.docs_count, job.stage)

    return {"status": "done", "docs_count": job.docs_count, "gcs_prefix": gcs_prefix}

