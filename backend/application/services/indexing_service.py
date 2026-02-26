import asyncio
import os
import subprocess
import sys
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from google.cloud import storage

from common.config import config
from domain.entities import BotSource, IndexJob
from domain.platform_profiles import normalize_url_for_crawl, should_allow_url, resolve_platform_profile
from domain.repositories import (
    BotCorpusRepository,
    BotDomainRepository,
    BotRepository,
    BotSourceRepository,
    CrawlerRepository,
    DocumentStorageRepository,
    IndexJobRepository,
    RAGRepository,
)

from infrastructure.celery_app import celery_app
from infrastructure.tasks.crawl_tasks import crawl_job_task
from infrastructure.tasks.single_page_crawl_tasks import single_page_crawl_job
from infrastructure.tasks.pdf_source_tasks import pdf_source_ingest_job
from infrastructure.tasks.text_source_tasks import text_source_ingest_job
from infrastructure.tasks.docs_source_tasks import docs_source_ingest_job
from infrastructure.services.indexing_service import (
    _bot_base_prefix,
    _display_name_from_url,
    _parse_and_validate_url,
    _parse_bucket_and_prefix,
    _validate_urls_for_bot,
)
from infrastructure.repositories.gcs_source_file_repository import GcsSourceFileRepository


class IndexingService:
    def __init__(
        self,
        bot_repo: BotRepository,
        domain_repo: BotDomainRepository,
        corpus_repo: BotCorpusRepository,
        source_repo: BotSourceRepository,
        job_repo: IndexJobRepository,
        crawler_repo: CrawlerRepository,
        storage_repo: DocumentStorageRepository,
        rag_repo: RAGRepository,
    ) -> None:
        self._bot_repo = bot_repo
        self._domain_repo = domain_repo
        self._corpus_repo = corpus_repo
        self._source_repo = source_repo
        self._job_repo = job_repo
        self._crawler_repo = crawler_repo
        self._storage_repo = storage_repo
        self._rag_repo = rag_repo

    def list_sources_for_bot(self, bot_id: str) -> List[BotSource]:
        """List all sources for a bot."""
        return self._source_repo.list_sources_for_bot(bot_id)

    def create_source(self, bot_id: str, type: str, config: Dict[str, Any], display_name: Optional[str] = None) -> BotSource:
        """Create a source for a bot. Returns the created source."""
        now = datetime.now(timezone.utc).isoformat()
        source_id = uuid.uuid4().hex
        source = BotSource(
            source_id=source_id,
            bot_id=bot_id,
            type=type,
            config=config,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        self._source_repo.create_source(source)
        return source

    def get_source(self, bot_id: str, source_id: str) -> Optional[BotSource]:
        """Get a source by id."""
        return self._source_repo.get_source(bot_id, source_id)

    def delete_source(self, bot_id: str, source_id: str) -> None:
        """Delete a source and remove its content from GCS. Jobs that reference it keep source_id (stale)."""
        jobs = self._job_repo.list_jobs_for_bot(bot_id)
        try:
            bucket_name, _ = _parse_bucket_and_prefix()
        except Exception:
            bucket_name = ""
        for job in jobs:
            if getattr(job, "source_id", None) != source_id:
                continue
            gcs_prefix = (getattr(job, "gcs_prefix", None) or "").strip()
            if not gcs_prefix or not bucket_name:
                continue
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                for blob in bucket.list_blobs(prefix=gcs_prefix):
                    blob.delete()
            except Exception:
                pass
        self._source_repo.delete_source(bot_id, source_id)

    async def start_indexing_for_bot(
        self,
        bot_id: str,
        raw_url: str,
        source_id: Optional[str] = None,
        *,
        headless: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Start BFS crawling from a single URL. Creates a source (unless source_id given) and links the job to it."""
        return await self._start_indexing_for_bot_inner(bot_id, raw_url, source_id=source_id, headless=headless)

    async def _start_indexing_for_bot_inner(
        self,
        bot_id: str,
        raw_url: str,
        source_id: Optional[str] = None,
        *,
        headless: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Inner: start crawl for URL; if source_id is None, create a new source first."""
        url, host = _parse_and_validate_url(raw_url)

        if config.REQUIRE_DOMAIN_VERIFICATION:
            verified_hosts = set(self._domain_repo.list_verified_hosts(bot_id))
            if host not in verified_hosts:
                raise PermissionError(f"Domain '{host}' is not verified for this bot")

        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

        corpus = self._rag_repo.ensure_corpus(bot_id)
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        if source_id is None:
            source = self.create_source(bot_id, "url", {"url": url})
            source_id = source.source_id

        job_id = uuid.uuid4().hex
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=url,
            hostname=host,
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source_id=source_id,
        )

        existing = self._job_repo.get_job_by_hostname(bot_id, host)
        if existing:
            pass  # Job cancellation handled separately

        self._job_repo.create_job(job)

        try:
            task = crawl_job_task.delay(job_id, bot_id, url, None, bucket_name, base_prefix, corpus, headless)
            task_id = task.id if task else None
        except Exception as e:
            job.stage = "error"
            job.last_error = f"Failed to queue task: {str(e)}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue crawl task: {str(e)}")

        return {"status": "started", "job_id": job_id, "hostname": host, "task_id": task_id}

    async def start_indexing_for_source(
        self,
        bot_id: str,
        source_id: str,
        *,
        headless: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Start crawling for an existing source (type=url). Creates a job linked to this source and adds to RAG."""
        source = self._source_repo.get_source(bot_id, source_id)
        if not source:
            raise ValueError("Source not found")
        if (source.type or "").lower() != "url":
            raise ValueError("Only URL sources can be crawled; use the source's URL")
        raw_url = (source.config or {}).get("url") if isinstance(source.config, dict) else None
        if not raw_url or not isinstance(raw_url, str):
            raise ValueError("Source has no URL in config")
        return await self._start_indexing_for_bot_inner(bot_id, raw_url, source_id=source_id, headless=headless)

    async def start_single_page_crawl_for_source(self, bot_id: str, source_id: str) -> Dict[str, Any]:
        """Start a dedicated single-page crawl for an existing source (type=url)."""
        source = self._source_repo.get_source(bot_id, source_id)
        if not source:
            raise ValueError("Source not found")
        if (source.type or "").lower() != "url":
            raise ValueError("Only URL sources can be crawled; use the source's URL")
        raw_url = (source.config or {}).get("url") if isinstance(source.config, dict) else None
        if not raw_url or not isinstance(raw_url, str):
            raise ValueError("Source has no URL in config")

        url, host = _parse_and_validate_url(raw_url)

        if config.REQUIRE_DOMAIN_VERIFICATION:
            verified_hosts = set(self._domain_repo.list_verified_hosts(bot_id))
            if host not in verified_hosts:
                raise PermissionError(f"Domain '{host}' is not verified for this bot")

        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

        corpus = self._rag_repo.ensure_corpus(bot_id)
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        job_id = uuid.uuid4().hex
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=url,
            hostname=host,
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source_id=source_id,
        )
        self._job_repo.create_job(job)

        try:
            task = single_page_crawl_job.delay(job_id, bot_id, url, bucket_name, base_prefix, corpus)
            task_id = task.id if task else None
        except Exception as e:
            job.stage = "error"
            job.last_error = f"Failed to queue task: {str(e)}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue crawl task: {str(e)}")

        return {"status": "started", "job_id": job_id, "hostname": host, "task_id": task_id}

    async def create_pdf_source_and_start_ingest(
        self,
        *,
        bot_id: str,
        filename: str,
        pdf_bytes: bytes,
        display_name: Optional[str] = None,
    ) -> Tuple[BotSource, str]:
        """
        Create a PDF source and immediately enqueue a background ingestion job.
        Returns (source, job_id).
        """
        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")
        if not bot_id:
            raise ValueError("Missing bot_id")
        if not pdf_bytes:
            raise ValueError("Empty PDF")

        corpus = self._rag_repo.ensure_corpus(bot_id)
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        # Create source record first.
        source = self.create_source(
            bot_id,
            "pdf",
            {
                "filename": (filename or "").strip() or "document.pdf",
            },
            display_name=display_name or (filename or "").strip() or None,
        )

        # Upload raw PDF to GCS and persist location on the source config.
        file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix)
        uploaded = file_repo.upload_pdf(bot_id=bot_id, source_id=source.source_id, filename=filename, data=pdf_bytes)
        source.config = dict(source.config or {})
        source.config.update(
            {
                "filename": uploaded.filename,
                "gcs_pdf_uri": uploaded.gcs_uri,
                "gcs_pdf_blob": uploaded.blob_name,
                "bytes": uploaded.bytes,
            }
        )
        source.updated_at = datetime.now(timezone.utc).isoformat()
        self._source_repo.update_source(source)

        # Create an index job linked to the source.
        job_id = uuid.uuid4().hex
        job_url = f"https://pdf.local/{bot_id}/{source.source_id}/{uploaded.filename}"
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=job_url,
            hostname="pdf.local",
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source_id=source.source_id,
        )
        self._job_repo.create_job(job)

        try:
            task = pdf_source_ingest_job.delay(job_id=job_id, bot_id=bot_id, source_id=source.source_id, bucket_name=bucket_name, base_prefix=base_prefix, corpus_resource=corpus)
            task_id = task.id if task else None
            if task_id:
                job.celery_task_id = task_id
                self._job_repo.update_job(job)
        except Exception as e:
            job.stage = "error"
            job.last_error = f"Failed to queue PDF task: {str(e)[:200]}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue PDF ingestion task: {str(e)}")

        return source, job_id

    async def create_text_source_and_start_ingest(
        self,
        *,
        bot_id: str,
        content: str,
        title: Optional[str] = None,
    ) -> Tuple[BotSource, str]:
        """
        Create a text/custom source and enqueue a background ingestion job.
        For content >50 000 chars the raw text is stored in GCS instead of inline.
        Returns (source, job_id).
        """
        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")
        if not bot_id:
            raise ValueError("Missing bot_id")
        content = (content or "").strip()
        if not content:
            raise ValueError("Empty text content")

        corpus = self._rag_repo.ensure_corpus(bot_id)
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        title_str = (title or "").strip() or None
        display = title_str or f"Text ({len(content)} chars)"

        source_cfg: dict = {
            "title": title_str or "",
            "char_count": len(content),
        }

        # Store large content in GCS to avoid DB bloat
        if len(content) > 50_000:
            source_cfg["content"] = ""  # placeholder; will be overwritten after GCS upload
        else:
            source_cfg["content"] = content

        source = self.create_source(bot_id, "text", source_cfg, display_name=display)

        if len(content) > 50_000:
            file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix)
            uploaded = file_repo.upload_text(bot_id=bot_id, source_id=source.source_id, content=content)
            source.config = dict(source.config or {})
            source.config.update({"gcs_content_blob": uploaded.blob_name, "gcs_content_uri": uploaded.gcs_uri})
            source.updated_at = datetime.now(timezone.utc).isoformat()
            self._source_repo.update_source(source)

        job_id = uuid.uuid4().hex
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=f"https://text.local/{bot_id}/{source.source_id}",
            hostname="text.local",
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source_id=source.source_id,
        )
        self._job_repo.create_job(job)

        try:
            task = text_source_ingest_job.delay(
                job_id=job_id,
                bot_id=bot_id,
                source_id=source.source_id,
                bucket_name=bucket_name,
                base_prefix=base_prefix,
                corpus_resource=corpus,
            )
            task_id = task.id if task else None
            if task_id:
                job.celery_task_id = task_id
                self._job_repo.update_job(job)
        except Exception as e:
            job.stage = "error"
            job.last_error = f"Failed to queue text task: {str(e)[:200]}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue text ingestion task: {str(e)}")

        return source, job_id

    async def create_docs_source_and_start_ingest(
        self,
        *,
        bot_id: str,
        file_bytes: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> Tuple[BotSource, str]:
        """
        Create a docs source (.txt/.md/.docx/.doc) and enqueue a background ingestion job.
        Returns (source, job_id).
        """
        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")
        if not bot_id:
            raise ValueError("Missing bot_id")
        if not file_bytes:
            raise ValueError("Empty file")

        corpus = self._rag_repo.ensure_corpus(bot_id)
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        source = self.create_source(
            bot_id,
            "docs",
            {"filename": (filename or "").strip() or "document.txt", "content_type": content_type},
            display_name=(filename or "").strip() or None,
        )

        file_repo = GcsSourceFileRepository(bucket_name=bucket_name, base_prefix=base_prefix)
        uploaded = file_repo.upload_file(
            bot_id=bot_id,
            source_id=source.source_id,
            filename=filename,
            data=file_bytes,
            content_type=content_type,
        )
        source.config = dict(source.config or {})
        source.config.update({
            "filename": uploaded.filename,
            "gcs_uri": uploaded.gcs_uri,
            "gcs_blob": uploaded.blob_name,
            "bytes": uploaded.bytes,
        })
        source.updated_at = datetime.now(timezone.utc).isoformat()
        self._source_repo.update_source(source)

        job_id = uuid.uuid4().hex
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=f"https://docs.local/{bot_id}/{source.source_id}/{uploaded.filename}",
            hostname="docs.local",
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source_id=source.source_id,
        )
        self._job_repo.create_job(job)

        try:
            task = docs_source_ingest_job.delay(
                job_id=job_id,
                bot_id=bot_id,
                source_id=source.source_id,
                bucket_name=bucket_name,
                base_prefix=base_prefix,
                corpus_resource=corpus,
            )
            task_id = task.id if task else None
            if task_id:
                job.celery_task_id = task_id
                self._job_repo.update_job(job)
        except Exception as e:
            job.stage = "error"
            job.last_error = f"Failed to queue docs task: {str(e)[:200]}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue docs ingestion task: {str(e)}")

        return source, job_id

    async def start_indexing_batch_for_bot(
        self,
        bot_id: str,
        urls: List[str],
        *,
        headless: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Start crawling for a list of URLs (no BFS expansion). Creates one source per URL so the Sources table shows each URL."""
        cleaned = _validate_urls_for_bot(bot_id, urls)
        # Apply platform-specific crawl profiles (e.g., restaurant platform include/exclude rules)
        cleaned = apply_platform_profiles_to_urls(cleaned)

        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

        now = datetime.now(timezone.utc).isoformat()
        for u in cleaned:
            display_name = _display_name_from_url(u)
            source = BotSource(
                source_id=uuid.uuid4().hex,
                bot_id=bot_id,
                type="url",
                config={"url": u},
                display_name=display_name,
                created_at=now,
                updated_at=now,
            )
            self._source_repo.create_source(source)

        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

        job_id = uuid.uuid4().hex
        job = IndexJob(
            job_id=job_id,
            bot_id=bot_id,
            url=cleaned[0] if cleaned else "",
            hostname="batch",
            stage="queued",
            pages_crawled=0,
            docs_count=0,
            last_crawled_url="",
            last_depth=-1,
            gcs_prefix="",
            last_error="",
            created_at=now,
            updated_at=now,
        )

        self._job_repo.create_job(job)

        # Queue Celery task immediately; worker will call ensure_corpus so API returns fast
        try:
            task = crawl_job_task.delay(job_id, bot_id, None, cleaned, bucket_name, base_prefix, None, headless)
            task_id = task.id if task else None
        except Exception as e:
            # If Celery task fails to queue, mark job as error
            job.stage = "error"
            job.last_error = f"Failed to queue task: {str(e)}"
            self._job_repo.update_job(job)
            raise RuntimeError(f"Failed to queue crawl task: {str(e)}")

        return {"status": "started", "job_id": job_id, "hostname": job.hostname, "task_id": task_id}

    def get_job_status(self, bot_id: str, job_key: str) -> Dict[str, Any]:
        """Get job status by job_id or hostname."""
        job = self._job_repo.get_job(bot_id, job_key)
        if not job:
            return {"status": "not_found"}

        try:
            bucket_name, base_prefix_root = _parse_bucket_and_prefix()
        except Exception:
            bucket_name, base_prefix_root = "", ""
        corpus_resource = self._corpus_repo.get_bot_corpus(bot_id) or ""

        return {
            "status": "ok",
            "job_id": job.job_id,
            "url": job.url,
            "hostname": job.hostname,
            "stage": job.stage,
            "pages_crawled": job.pages_crawled,
            "last_crawled_url": job.last_crawled_url,
            "last_depth": job.last_depth,
            "gcs_prefix": job.gcs_prefix,
            "docs_count": job.docs_count,
            "bucket": bucket_name,
            "base_prefix": _bot_base_prefix(base_prefix_root, bot_id) if bot_id else base_prefix_root,
            "corpus_resource": corpus_resource,
            "last_error": job.last_error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def get_job_status_by_hostname(self, bot_id: str, raw_url: str) -> Dict[str, Any]:
        """Get job status by URL (finds job by hostname)."""
        url, host = _parse_and_validate_url(raw_url)
        job = self._job_repo.get_job_by_hostname(bot_id, host)
        if not job:
            return {"status": "not_found"}

        return self.get_job_status(bot_id, job.job_id)

    def list_jobs_for_bot(self, bot_id: str) -> List[IndexJob]:
        """List all jobs for a bot."""
        return self._job_repo.list_jobs_for_bot(bot_id)

    def cancel_job(self, bot_id: str, raw_url: str) -> Dict[str, Any]:
        """Cancel a job by URL."""
        url, host = _parse_and_validate_url(raw_url)
        job = self._job_repo.get_job_by_hostname(bot_id, host)
        if not job:
            return {"status": "not_found"}

        # Revoke Celery task if it has a task_id
        if job.celery_task_id:
            try:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception:
                pass  # Task may already be done

        job.stage = "cancelled"
        self._job_repo.update_job(job)

        return {"status": "stopping", "job_id": job.job_id, "hostname": job.hostname}


# ═══════════════════════════════════════════════════════════════════════════
# Platform Profile URL Filter
# Logic lives in domain.platform_profiles; this is just the pipeline entry point.
# ═══════════════════════════════════════════════════════════════════════════

def apply_platform_profiles_to_urls(urls: List[str]) -> List[str]:
    """
    Filter and deduplicate a URL list using platform profile rules.

    Delegates filtering to domain.platform_profiles.should_allow_url (junk detection +
    profile include/exclude rules). Then deduplicates platform URLs whose profile sets
    strip_query_params=True, collapsing ?RDT=YYYYMMDD variations into one entry.
    Unknown-domain URLs are passed through untouched.
    """
    filtered = [url for url in urls if should_allow_url(url)]

    # Deduplication: only for platforms with strip_query_params=True
    deduplicated: List[str] = []
    seen_normalized: set = set()

    for url in filtered:
        profile, _ = resolve_platform_profile(url)
        if profile is not None and profile.strip_query_params:
            normalized = normalize_url_for_crawl(url)
            if normalized not in seen_normalized:
                deduplicated.append(url)
                seen_normalized.add(normalized)
        else:
            deduplicated.append(url)

    return deduplicated
