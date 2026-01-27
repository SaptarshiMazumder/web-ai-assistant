import asyncio
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from common.config import config
from domain.entities import IndexJob
from domain.repositories import (
    BotCorpusRepository,
    BotDomainRepository,
    BotRepository,
    CrawlerRepository,
    DocumentStorageRepository,
    IndexJobRepository,
    RAGRepository,
)

from infrastructure.services.indexing_job_manager import IndexingJobManager
from infrastructure.services.indexing_service import (
    _bot_base_prefix,
    _parse_and_validate_url,
    _parse_bucket_and_prefix,
)


class IndexingService:
    def __init__(
        self,
        bot_repo: BotRepository,
        domain_repo: BotDomainRepository,
        corpus_repo: BotCorpusRepository,
        job_repo: IndexJobRepository,
        crawler_repo: CrawlerRepository,
        storage_repo: DocumentStorageRepository,
        rag_repo: RAGRepository,
        job_manager: IndexingJobManager,
    ) -> None:
        self._bot_repo = bot_repo
        self._domain_repo = domain_repo
        self._corpus_repo = corpus_repo
        self._job_repo = job_repo
        self._crawler_repo = crawler_repo
        self._storage_repo = storage_repo
        self._rag_repo = rag_repo
        self._job_manager = job_manager

    async def start_indexing_for_bot(self, bot_id: str, raw_url: str) -> Dict[str, Any]:
        """Start BFS crawling from a single URL."""
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
        )

        # Check for existing job and cancel if needed
        existing = self._job_repo.get_job_by_hostname(bot_id, host)
        if existing:
            # Cancel existing job if running
            pass  # Job cancellation handled separately

        self._job_repo.create_job(job)

        # Start worker process
        worker_args, worker_env = self._build_worker_args(job_id, url, None, bucket_name, base_prefix, corpus)
        self._job_manager.start_job_process(job, worker_args, worker_env)

        return {"status": "started", "job_id": job_id, "hostname": host}

    async def start_indexing_batch_for_bot(self, bot_id: str, urls: List[str]) -> Dict[str, Any]:
        """Start crawling for a list of URLs (no BFS expansion)."""
        cleaned = self._validate_urls_for_bot(bot_id, urls)

        if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
            raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

        corpus = self._rag_repo.ensure_corpus(bot_id)
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
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        self._job_repo.create_job(job)

        # Start worker process with URL list
        worker_args, worker_env = self._build_worker_args(job_id, None, cleaned, bucket_name, base_prefix, corpus)
        self._job_manager.start_job_process(job, worker_args, worker_env)

        return {"status": "started", "job_id": job_id, "hostname": job.hostname}

    def _build_worker_args(
        self,
        job_id: str,
        url: Optional[str],
        urls: Optional[List[str]],
        bucket_name: str,
        base_prefix: str,
        corpus: str,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Build worker process arguments and environment."""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "infrastructure",
            "workers",
            "worker_index_job.py",
        )
        worker_path = os.path.abspath(worker_path)
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        args = [
            (config.WORKER_PYTHON or sys.executable),
            worker_path,
            "--bucket",
            bucket_name,
            "--base-prefix",
            base_prefix,
            "--corpus",
            corpus,
            "--creds",
            (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
        ]

        if urls:
            import json

            args.extend(["--urls-json", json.dumps(urls)])
        else:
            args.extend(["--url", url or ""])

        env = {
            **os.environ,
            "GOOGLE_APPLICATION_CREDENTIALS": (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
            "LOCATION": (config.LOCATION or "us-central1"),
            "PYTHONPATH": backend_root,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        return args, env

    def _validate_urls_for_bot(self, bot_id: str, urls: List[str]) -> List[str]:
        """Validate and clean URLs for a bot."""
        from urllib.parse import urlparse

        cleaned: List[str] = []
        seen = set()
        for raw_url in urls:
            url, host = _parse_and_validate_url(raw_url)
            if url in seen:
                continue
            seen.add(url)
            cleaned.append(url)

        if not cleaned:
            raise ValueError("Missing url list")

        if config.REQUIRE_DOMAIN_VERIFICATION:
            verified_hosts = set(self._domain_repo.list_verified_hosts(bot_id))
            for url in cleaned:
                host = (urlparse(url).hostname or "").lower().split(":")[0]
                if host not in verified_hosts:
                    raise PermissionError(f"Domain '{host}' is not verified for this bot")

        return cleaned

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

        self._job_manager.cancel_job_process(job.job_id)
        job.stage = "cancelled"
        self._job_repo.update_job(job)

        return {"status": "stopping", "job_id": job.job_id, "hostname": job.hostname}
