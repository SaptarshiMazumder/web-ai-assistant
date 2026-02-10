import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import google.auth
from google.cloud import storage
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
    source = source_repo.get_source(bot_id, source_id)
    if not source:
        job.stage = "error"
        job.last_error = "Source not found"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    cfg = source.config or {}
    gcs_blob = (cfg.get("gcs_pdf_blob") or "").strip() if isinstance(cfg, dict) else ""
    filename = (cfg.get("filename") or "").strip() if isinstance(cfg, dict) else ""
    if not gcs_blob:
        job.stage = "error"
        job.last_error = "PDF not uploaded (missing gcs_pdf_blob)"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

    job.stage = "crawling"  # reuse existing stage label
    job.last_error = ""
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

    creds, proj = google.auth.load_credentials_from_file(creds_path)
    storage_client = storage.Client(credentials=creds, project=proj)

    # Download raw PDF
    file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix, storage_client=storage_client)
    pdf_bytes = file_repo.download(blob_name=gcs_blob)
    if not pdf_bytes:
        job.stage = "error"
        job.last_error = "Failed to download PDF from storage"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "error", "error": job.last_error}

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
        job.stage = "done"
        job.docs_count = 0
        job.last_error = "No text extracted from PDF"
        job.updated_at = _utc_now()
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "gcs_prefix": ""}

    # Upload extracted docs as markdown
    job.stage = "uploading"
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
    gcs_prefix = storage_repo.save_documents(bot_id, docs)
    job.gcs_prefix = gcs_prefix
    job.updated_at = _utc_now()
    job_repo.update_job(job)

    # Import into RAG
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

