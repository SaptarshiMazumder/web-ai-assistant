import asyncio
import json
import os
import sys
import subprocess
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlparse

import vertexai
from vertexai import rag as vx_rag

from common.config import config
from infrastructure.db.repositories import (
    PostgresBotCorpusRepository,
    PostgresBotDomainRepository,
    PostgresBotRepository,
)

_bot_repo = PostgresBotRepository()
_bot_domain_repo = PostgresBotDomainRepository()
_bot_corpus_repo = PostgresBotCorpusRepository()


@dataclass
class IndexJob:
    job_id: str
    bot_id: str
    url: str
    hostname: str
    created_at: str
    process: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None
    last_log_line: str = ""
    log_tail: list[str] = field(default_factory=list)
    worker_runtime: Dict[str, Any] = field(default_factory=dict)
    docs_count: int = 0
    stage: str = "queued"  # queued|crawling|uploading|importing|import_submitted|prompt_queued|prompt_generating|cancelled|error|done
    pages_crawled: int = 0
    last_crawled_url: str = ""
    last_depth: int = -1
    gcs_prefix: str = ""
    last_error: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_index_jobs_by_key: Dict[str, IndexJob] = {}


def _job_key(bot_id: str, hostname: str) -> str:
    return f"{bot_id}::{hostname}"


def _batch_job_key(bot_id: str, job_id: str) -> str:
    return f"{bot_id}::batch::{job_id}"


def _touch(job: IndexJob) -> None:
    job.updated_at = datetime.now(timezone.utc).isoformat()


def _parse_and_validate_url(raw_url: str) -> Tuple[str, str]:
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("Missing url")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower().split(":")[0]
    if not host:
        raise ValueError("Could not parse hostname from url")
    return url, host


def _validate_urls_for_bot(bot_id: str, urls: List[str]) -> List[str]:
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
        verified_hosts = set(_bot_domain_repo.list_verified_hosts(bot_id))
        for url in cleaned:
            host = (urlparse(url).hostname or "").lower().split(":")[0]
            if host not in verified_hosts:
                raise PermissionError(f"Domain '{host}' is not verified for this bot")

    return cleaned


def _display_name_from_url(url: str) -> Optional[str]:
    """Derive a short display name from URL: pathname or hostname if path is / or empty."""
    try:
        parsed = urlparse((url or "").strip())
        path = (parsed.path or "").strip("/")
        if path:
            return "/" + path[:80] if len(path) > 80 else "/" + path
        host = (parsed.hostname or "").lower().split(":")[0]
        return host or None
    except Exception:
        return None


def _parse_bucket_and_prefix() -> Tuple[str, str]:
    bucket_and_prefix = (config.GCS_BUCKET or "").strip("/").split("/", 1)
    if len(bucket_and_prefix) == 2:
        return bucket_and_prefix[0], bucket_and_prefix[1]
    if len(bucket_and_prefix) == 1:
        return bucket_and_prefix[0], ""
    raise ValueError("Invalid GCS_BUCKET configuration")


def _org_slug(bot_id: str) -> str:
    bot = _bot_repo.get_bot(bot_id)
    if not bot or not bot.org_id:
        raise ValueError("Missing org_id for bot")
    return _slugify_name(bot.org_id)


def _slugify_name(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "bot"


def _env_slug() -> str:
    raw = (os.environ.get("ENVIRONMENT") or os.environ.get("ENV") or "dev").strip()
    return _slugify_name(raw)


def _rag_display_name(bot_id: str) -> str:
    bot = _bot_repo.get_bot(bot_id)
    bot_slug = _slugify_name(bot.display_name if bot else "")
    env = _env_slug()
    tenant = _org_slug(bot_id)
    prefix = f"web-rag-bot-{env}-{tenant}-"
    suffix = f"-{bot_id}"
    max_total = 120
    max_slug_len = max(12, max_total - len(prefix) - len(suffix))
    bot_slug = bot_slug[:max_slug_len]
    return f"{prefix}{bot_slug}{suffix}"


def _bot_base_prefix(base_prefix_root: str, bot_id: str) -> str:
    base_prefix_root = (base_prefix_root or "").strip("/")
    tenant = _org_slug(bot_id)
    if base_prefix_root:
        return f"{base_prefix_root}/{tenant}/bots/{bot_id}"
    return f"{tenant}/bots/{bot_id}"


def ensure_bot_corpus(bot_id: str, *, force_new: bool = False) -> str:
    existing = _bot_corpus_repo.get_bot_corpus(bot_id)
    if existing and not force_new:
        # Verify the corpus still exists (it might have been deleted manually).
        try:
            vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
            try:
                vx_rag.get_corpus(existing)
                return existing
            except Exception:
                # Fall through to recreate
                pass
        except Exception:
            # If validation fails, fall back to creating a new corpus.
            pass
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    display_name = _rag_display_name(bot_id)
    from infrastructure.rag.crawl_service import EMBEDDING_PUBLISHER_MODEL
    emb_cfg = vx_rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=vx_rag.VertexPredictionEndpoint(
            publisher_model=EMBEDDING_PUBLISHER_MODEL
        )
    )
    corpus = vx_rag.create_corpus(
        display_name=display_name,
        backend_config=vx_rag.RagVectorDbConfig(rag_embedding_model_config=emb_cfg),
    )
    _bot_corpus_repo.upsert_bot_corpus(bot_id, corpus.name)
    return corpus.name


async def start_index_for_bot(
    bot_id: str,
    raw_url: str,
    *,
    headless: Optional[bool] = None,
) -> Dict[str, Any]:
    url, host = _parse_and_validate_url(raw_url)

    if config.REQUIRE_DOMAIN_VERIFICATION:
        verified_hosts = set(_bot_domain_repo.list_verified_hosts(bot_id))
        if host not in verified_hosts:
            raise PermissionError(f"Domain '{host}' is not verified for this bot")

    # Hard fail early if creds are missing; otherwise the worker may start with
    # surprising ADC/user credentials depending on environment/reload behavior.
    if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
        raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

    corpus = ensure_bot_corpus(bot_id)
    bucket_name, base_prefix_root = _parse_bucket_and_prefix()

    job_key = _job_key(bot_id, host)
    existing = _index_jobs_by_key.get(job_key)
    if existing and existing.process and existing.process.poll() is None:
        try:
            existing.process.terminate()
        except Exception:
            pass

    job_id = uuid.uuid4().hex
    base_prefix = _bot_base_prefix(base_prefix_root, bot_id)

    # Start a separate worker process for crawling/indexing to avoid Windows+uvicorn event loop issues.
    worker_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "workers",
        "worker_index_job.py",
    )
    worker_path = os.path.abspath(worker_path)

    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    args = [
        (config.WORKER_PYTHON or sys.executable),
        worker_path,
        "--url",
        url,
        "--bucket",
        bucket_name,
        "--base-prefix",
        base_prefix,
        "--corpus",
        corpus,
        "--creds",
        (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
    ]
    if headless is not None:
        args.extend(["--headless", "true" if headless else "false"])

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            # Force the worker to use the same credential file as the API process.
            "GOOGLE_APPLICATION_CREDENTIALS": (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
            "LOCATION": (config.LOCATION or "us-central1"),
            # Ensure the worker can import backend packages (api, infrastructure, etc).
            "PYTHONPATH": backend_root,
            # Ensure unicode output doesn't crash on Windows codepages.
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )

    job = IndexJob(
        job_id=job_id,
        bot_id=bot_id,
        url=url,
        hostname=host,
        process=proc,
        created_at=datetime.now(timezone.utc).isoformat(),
        stage="queued",
    )
    _index_jobs_by_key[job_key] = job

    async def _consume_logs():
        job.stage = "crawling"
        _touch(job)

        prefix = "WEB_AI_EVENT "
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)  # type: ignore[union-attr]
            if not line:
                break
            s = (line or "").strip()
            if not s:
                continue
            job.last_log_line = s
            job.log_tail.append(s)
            if len(job.log_tail) > 200:
                job.log_tail = job.log_tail[-200:]
            _touch(job)
            if prefix not in s:
                continue
            payload = s.split(prefix, 1)[1].strip()
            try:
                msg = json.loads(payload)
            except Exception:
                continue
            t = msg.get("type")
            if t == "stage":
                job.stage = str(msg.get("stage") or job.stage)
                _touch(job)
            elif t == "progress":
                job.pages_crawled = int(msg.get("pages_crawled") or job.pages_crawled)
                job.last_crawled_url = str(msg.get("url") or job.last_crawled_url)
                job.last_depth = int(msg.get("depth") if msg.get("depth") is not None else job.last_depth)
                _touch(job)
            elif t == "result":
                job.docs_count = int(msg.get("docs_count") or job.docs_count)
                _touch(job)
            elif t == "gcs_prefix":
                job.gcs_prefix = str(msg.get("gcs_prefix") or job.gcs_prefix)
                _touch(job)
            elif t == "runtime":
                job.worker_runtime = dict(msg)
                _touch(job)
            elif t == "error":
                job.stage = "error"
                job.last_error = str(msg.get("error") or "")
                _touch(job)

        rc = proc.poll()
        if rc is None:
            return
        if rc == 0:
            if job.stage not in ("import_submitted", "prompt_queued", "prompt_generating", "done"):
                job.stage = "done"
            job.last_error = ""
            _touch(job)
        else:
            if job.stage != "error":
                job.stage = "error"
                job.last_error = job.last_error or f"Worker exited with code {rc}"
            _touch(job)

    job.log_task = asyncio.create_task(_consume_logs())

    return {"status": "started", "job_id": job_id, "hostname": host}


async def start_index_for_bot_batch(
    bot_id: str,
    urls: List[str],
    *,
    headless: Optional[bool] = None,
) -> Dict[str, Any]:
    cleaned = _validate_urls_for_bot(bot_id, urls)

    # Hard fail early if creds are missing; otherwise the worker may start with
    # surprising ADC/user credentials depending on environment/reload behavior.
    if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
        raise RuntimeError("Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker")

    corpus = ensure_bot_corpus(bot_id)
    bucket_name, base_prefix_root = _parse_bucket_and_prefix()

    job_id = uuid.uuid4().hex
    base_prefix = _bot_base_prefix(base_prefix_root, bot_id)
    job_key = _batch_job_key(bot_id, job_id)

    worker_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "workers",
        "worker_index_job.py",
    )
    worker_path = os.path.abspath(worker_path)
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    args = [
        (config.WORKER_PYTHON or sys.executable),
        worker_path,
        "--urls-json",
        json.dumps(cleaned),
        "--bucket",
        bucket_name,
        "--base-prefix",
        base_prefix,
        "--corpus",
        corpus,
        "--creds",
        (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
    ]
    if headless is not None:
        args.extend(["--headless", "true" if headless else "false"])

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "GOOGLE_APPLICATION_CREDENTIALS": (config.GOOGLE_APPLICATION_CREDENTIALS or ""),
            "LOCATION": (config.LOCATION or "us-central1"),
            "PYTHONPATH": backend_root,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )

    job = IndexJob(
        job_id=job_id,
        bot_id=bot_id,
        url=cleaned[0],
        hostname="batch",
        process=proc,
        created_at=datetime.now(timezone.utc).isoformat(),
        stage="queued",
    )
    _index_jobs_by_key[job_key] = job

    async def _consume_logs():
        job.stage = "crawling"
        _touch(job)

        prefix = "WEB_AI_EVENT "
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)  # type: ignore[union-attr]
            if not line:
                break
            s = (line or "").strip()
            if not s:
                continue
            job.last_log_line = s
            job.log_tail.append(s)
            if len(job.log_tail) > 200:
                job.log_tail = job.log_tail[-200:]
            _touch(job)
            if prefix not in s:
                continue
            payload = s.split(prefix, 1)[1].strip()
            try:
                msg = json.loads(payload)
            except Exception:
                continue
            t = msg.get("type")
            if t == "stage":
                job.stage = str(msg.get("stage") or job.stage)
                _touch(job)
            elif t == "progress":
                job.pages_crawled = int(msg.get("pages_crawled") or job.pages_crawled)
                job.last_crawled_url = str(msg.get("url") or job.last_crawled_url)
                job.last_depth = int(msg.get("depth") if msg.get("depth") is not None else job.last_depth)
                _touch(job)
            elif t == "result":
                job.docs_count = int(msg.get("docs_count") or job.docs_count)
                _touch(job)
            elif t == "gcs_prefix":
                job.gcs_prefix = str(msg.get("gcs_prefix") or job.gcs_prefix)
                _touch(job)
            elif t == "runtime":
                job.worker_runtime = dict(msg)
                _touch(job)
            elif t == "error":
                job.stage = "error"
                job.last_error = str(msg.get("error") or "")
                _touch(job)

        rc = proc.poll()
        if rc is None:
            return
        if rc == 0:
            if job.stage not in ("import_submitted", "prompt_queued", "prompt_generating", "done"):
                job.stage = "done"
            job.last_error = ""
            _touch(job)
        else:
            if job.stage != "error":
                job.stage = "error"
                job.last_error = job.last_error or f"Worker exited with code {rc}"
            _touch(job)

    job.log_task = asyncio.create_task(_consume_logs())

    return {"status": "started", "job_id": job_id, "hostname": job.hostname}


def get_index_status_for_bot(bot_id: str, raw_url: str) -> Dict[str, Any]:
    url, host = _parse_and_validate_url(raw_url)
    job = _index_jobs_by_key.get(_job_key(bot_id, host))
    if not job:
        return {"status": "not_found"}


def get_index_status_by_job_id(bot_id: str, job_id: str) -> Dict[str, Any]:
    # Try batch job first
    batch_key = _batch_job_key(bot_id, job_id)
    job = _index_jobs_by_key.get(batch_key)
    if not job:
        # Fallback: search all jobs for this bot
        for key, candidate in _index_jobs_by_key.items():
            if candidate.bot_id == bot_id and candidate.job_id == job_id:
                job = candidate
                break
    if not job:
        return {"status": "not_found"}
    return _format_job_status(bot_id, job)


def _format_job_status(bot_id: str, job: IndexJob) -> Dict[str, Any]:
    # If the worker exited but we didn't observe a terminal stage yet, surface it.
    try:
        if job.process is not None and job.process.poll() is not None and job.stage not in ("done", "error", "cancelled", "import_submitted", "prompt_queued", "prompt_generating"):
            rc = job.process.returncode
            if rc == 0:
                job.stage = "done"
                job.last_error = ""
            else:
                job.stage = "error"
                job.last_error = job.last_error or f"Worker exited with code {rc}"
            _touch(job)
    except Exception:
        pass
    # Useful debug fields to understand where indexing is happening.
    try:
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
    except Exception:
        bucket_name, base_prefix_root = "", ""
    corpus_resource = _bot_corpus_repo.get_bot_corpus(bot_id) or ""
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
        "worker_pid": (job.process.pid if job.process is not None else None),
        "worker_running": (job.process.poll() is None) if job.process is not None else None,
        "worker_exit_code": (job.process.returncode if job.process is not None else None),
        "worker_runtime": job.worker_runtime,
        "last_log_line": job.last_log_line,
        "log_tail": job.log_tail[-50:],
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def cancel_index_for_bot(bot_id: str, raw_url: str) -> Dict[str, Any]:
    url, host = _parse_and_validate_url(raw_url)
    job = _index_jobs_by_key.get(_job_key(bot_id, host))
    if not job:
        return {"status": "not_found"}
    try:
        if job.process and job.process.poll() is None:
            job.process.terminate()
    except Exception:
        pass
    job.stage = "cancelled"
    _touch(job)
    return {"status": "stopping", "job_id": job.job_id, "hostname": job.hostname}


def list_index_jobs_for_bot(bot_id: str) -> List[IndexJob]:
    bid = (bot_id or "").strip()
    if not bid:
        return []
    jobs = [job for job in _index_jobs_by_key.values() if job.bot_id == bid]
    return sorted(jobs, key=lambda j: j.updated_at, reverse=True)
