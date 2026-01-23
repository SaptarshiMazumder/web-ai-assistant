import time
import urllib.request
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, Request

import asyncio
import uuid
import os
import sys
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bot_registry import (
    create_bot,
    add_domain,
    get_bot_by_publishable_key,
    get_bot_by_secret_key,
    get_bot_corpus,
    list_verified_hosts,
    mark_domain_verified,
    upsert_bot_corpus,
)
from config import config
from models import (
    BotCreateRequest,
    BotCreateResponse,
    BotDomainAddRequest,
    BotDomainAddResponse,
    BotDomainVerifyResponse,
    WidgetChatRequest,
    WidgetChatResponse,
    Citation,
    BotIndexRequest,
)
from features.website_rag.vertex_rag_eg import run_vertex_rag

import vertexai
from vertexai import rag as vx_rag


router = APIRouter()


def _chat_debug_emit(event: Dict[str, Any]) -> None:
    """
    Extremely verbose debug logging for widget chat.
    Prints JSON to terminal AND appends to backend/chat_debug.log.
    """
    # Terminal (pretty-print chunks for readability)
    try:
        t = str(event.get("type") or "")
        trace_id = str(event.get("trace_id") or "")
        if t in ("retrieved_chunk", "filtered_chunk"):
            idx = event.get("idx")
            url = str(event.get("url") or "")
            snippet = str(event.get("snippet") or "")
            label = "RETRIEVED" if t == "retrieved_chunk" else "FILTERED"
            print(f"\nWEB_AI_CHAT_DEBUG_CHUNK {label} trace_id={trace_id} #{idx}\nURL: {url}\n---\n{snippet}\n---\n", flush=True)
        else:
            try:
                line = "WEB_AI_CHAT_DEBUG " + json.dumps(event, ensure_ascii=False)
            except Exception:
                line = "WEB_AI_CHAT_DEBUG " + str(event)
            print(line, flush=True)
    except Exception:
        pass

    # File (always JSONL, even for chunk events)
    try:
        line = "WEB_AI_CHAT_DEBUG " + json.dumps(event, ensure_ascii=False)
    except Exception:
        line = "WEB_AI_CHAT_DEBUG " + str(event)
    # File (best-effort)
    try:
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chat_debug.log")
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _require_admin_key(x_admin_key: Optional[str]) -> None:
    required = (getattr(config, "ADMIN_API_KEY", None) or "").strip()
    if not required:
        # If ADMIN_API_KEY is not set, allow bootstrapping in dev.
        return
    if (x_admin_key or "").strip() != required:
        raise HTTPException(status_code=401, detail="Missing/invalid admin key")


def _require_bot_secret(authorization: Optional[str]) -> str:
    auth = (authorization or "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <bot_secret_key>")
    token = auth.split(" ", 1)[1].strip()
    bot = get_bot_by_secret_key(token)
    if not bot:
        raise HTTPException(status_code=401, detail="Invalid bot secret key")
    return bot.bot_id


def _origin_host(origin: Optional[str]) -> str:
    if not origin:
        return ""
    try:
        u = urlparse(origin)
        host = (u.hostname or "").lower()
        return host.split(":")[0]
    except Exception:
        return ""


# Very small in-memory rate limiter (per bot, per minute).
_rl_window_s = 60
_rl_max_per_window = 60
_rl_state: Dict[str, Tuple[float, int]] = {}


def _rate_limit(bot_id: str) -> None:
    now = time.time()
    start, count = _rl_state.get(bot_id, (now, 0))
    if now - start > _rl_window_s:
        start, count = now, 0
    count += 1
    _rl_state[bot_id] = (start, count)
    if count > _rl_max_per_window:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def _verification_url(hostname: str, token: str) -> str:
    # Well-known file approach (simple and automatable).
    return f"https://{hostname}/.well-known/web-ai-bot-verification.txt"


def _check_domain_verification(hostname: str, token: str, *, timeout_s: float = 5.0) -> Tuple[bool, str]:
    url = _verification_url(hostname, token)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "web-ai-bot-verifier/1.0",
            "Accept": "text/plain,*/*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(64 * 1024).decode("utf-8", errors="ignore")
            if token in body:
                return True, "Verification token found"
            return False, "Well-known file fetched but token was not found in body"
    except Exception as e:
        return False, f"Failed to fetch verification URL: {e}"


def _parse_bucket_and_prefix() -> Tuple[str, str]:
    bucket_and_prefix = (config.GCS_BUCKET or "").strip("/").split("/", 1)
    if len(bucket_and_prefix) == 2:
        return bucket_and_prefix[0], bucket_and_prefix[1]
    if len(bucket_and_prefix) == 1:
        return bucket_and_prefix[0], ""
    raise ValueError("Invalid GCS_BUCKET configuration")


def _bot_base_prefix(base_prefix_root: str, bot_id: str) -> str:
    base_prefix_root = (base_prefix_root or "").strip("/")
    if base_prefix_root:
        return f"{base_prefix_root}/bot={bot_id}"
    return f"bot={bot_id}"


def _ensure_bot_corpus(bot_id: str, *, force_new: bool = False) -> str:
    existing = get_bot_corpus(bot_id)
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
    display_name = f"web-rag-bot-{bot_id[:40]}"
    emb_cfg = vx_rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=vx_rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
    corpus = vx_rag.create_corpus(
        display_name=display_name,
        backend_config=vx_rag.RagVectorDbConfig(rag_embedding_model_config=emb_cfg),
    )
    upsert_bot_corpus(bot_id, corpus.name)
    return corpus.name


@dataclass
class _IndexJob:
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
    stage: str = "queued"  # queued|crawling|uploading|importing|import_submitted|cancelled|error|done
    pages_crawled: int = 0
    last_crawled_url: str = ""
    last_depth: int = -1
    gcs_prefix: str = ""
    last_error: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_index_jobs_by_key: Dict[str, _IndexJob] = {}


def _job_key(bot_id: str, hostname: str) -> str:
    return f"{bot_id}::{hostname}"


def _touch(job: _IndexJob) -> None:
    job.updated_at = datetime.now(timezone.utc).isoformat()

def _parse_and_validate_url(raw_url: str) -> Tuple[str, str]:
    url = (raw_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower().split(":")[0]
    if not host:
        raise HTTPException(status_code=400, detail="Could not parse hostname from url")
    return url, host


async def _start_index_for_bot(bot_id: str, raw_url: str) -> Dict[str, Any]:
    url, host = _parse_and_validate_url(raw_url)

    if config.REQUIRE_DOMAIN_VERIFICATION:
        verified_hosts = set(list_verified_hosts(bot_id))
        if host not in verified_hosts:
            raise HTTPException(status_code=403, detail=f"Domain '{host}' is not verified for this bot")

    # Hard fail early if creds are missing; otherwise the worker may start with
    # surprising ADC/user credentials depending on environment/reload behavior.
    if not (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip():
        raise HTTPException(
            status_code=500,
            detail="Server is missing GOOGLE_APPLICATION_CREDENTIALS; cannot start indexing worker",
        )

    corpus = _ensure_bot_corpus(bot_id)
    try:
        bucket_name, base_prefix_root = _parse_bucket_and_prefix()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    worker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "worker_index_job.py")
    # features/saas/router.py -> backend/worker_index_job.py
    worker_path = os.path.abspath(worker_path)

    proc = subprocess.Popen(
        [
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
        ],
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
            # Ensure unicode output doesn't crash on Windows codepages.
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )

    job = _IndexJob(
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
                # Helpful to debug interpreter/env mismatches vs manual runs.
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
            if job.stage not in ("import_submitted", "done"):
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


def _get_index_status_for_bot(bot_id: str, raw_url: str) -> Dict[str, Any]:
    url, host = _parse_and_validate_url(raw_url)
    job = _index_jobs_by_key.get(_job_key(bot_id, host))
    if not job:
        return {"status": "not_found"}
    # If the worker exited but we didn't observe a terminal stage yet, surface it.
    try:
        if job.process is not None and job.process.poll() is not None and job.stage not in ("done", "error", "cancelled", "import_submitted"):
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
    corpus_resource = get_bot_corpus(bot_id) or ""
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


def _cancel_index_for_bot(bot_id: str, raw_url: str) -> Dict[str, Any]:
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


@router.post("/v1/bots", response_model=BotCreateResponse)
async def v1_create_bot(payload: BotCreateRequest, x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    b = create_bot(payload.display_name)
    return BotCreateResponse(
        bot_id=b.bot_id,
        display_name=b.display_name,
        publishable_key=b.publishable_key,
        secret_key=b.secret_key,
    )


@router.post("/v1/bots/{bot_id}/domains", response_model=BotDomainAddResponse)
async def v1_add_domain(
    bot_id: str,
    payload: BotDomainAddRequest,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    status, token = add_domain(bot_id, payload.hostname)
    hostname = payload.hostname.strip().lower().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    return BotDomainAddResponse(
        bot_id=bot_id,
        hostname=hostname,
        status=status,
        verification_token=token,
        verification_url=_verification_url(hostname, token),
    )


@router.post("/v1/bots/{bot_id}/domains/{hostname}/verify", response_model=BotDomainVerifyResponse)
async def v1_verify_domain(
    bot_id: str,
    hostname: str,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")

    # Find existing token by re-adding (idempotent)
    status, token = add_domain(bot_id, hostname)
    ok, msg = _check_domain_verification(hostname, token)
    if ok:
        mark_domain_verified(bot_id, hostname)
        return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status="verified", verified=True, message=msg)
    return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status=status, verified=False, message=msg)


@router.post("/v1/bots/{bot_id}/index")
async def v1_start_index(
    bot_id: str,
    payload: BotIndexRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Starts server-side crawl+upload+RAG import for the hostname in the provided URL."""
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")

    return await _start_index_for_bot(bot_id, payload.url)


@router.get("/v1/bots/{bot_id}/index/status")
async def v1_index_status(
    bot_id: str,
    url: str,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    return _get_index_status_for_bot(bot_id, url)


@router.post("/v1/bots/{bot_id}/index/cancel")
async def v1_cancel_index(
    bot_id: str,
    payload: BotIndexRequest,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    return _cancel_index_for_bot(bot_id, payload.url)


@router.post("/v1/pk/{publishable_key}/index")
async def v1_pk_start_index(
    publishable_key: str,
    payload: BotIndexRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    _rate_limit(bot.bot_id)
    return await _start_index_for_bot(bot.bot_id, payload.url)


@router.get("/v1/pk/{publishable_key}/index/status")
async def v1_pk_index_status(
    publishable_key: str,
    url: str,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    return _get_index_status_for_bot(bot.bot_id, url)


@router.post("/v1/pk/{publishable_key}/index/cancel")
async def v1_pk_cancel_index(
    publishable_key: str,
    payload: BotIndexRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    return _cancel_index_for_bot(bot.bot_id, payload.url)


@router.post("/v1/pk/{publishable_key}/chat", response_model=WidgetChatResponse)
async def v1_widget_chat(
    publishable_key: str,
    payload: WidgetChatRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    trace_id = uuid.uuid4().hex
    _chat_debug_emit(
        {
            "type": "chat_request_received",
            "trace_id": trace_id,
            "publishable_key": publishable_key,
            "payload": payload.model_dump() if hasattr(payload, "model_dump") else getattr(payload, "__dict__", {}),
        }
    )
    bot = get_bot_by_publishable_key(publishable_key)
    if not bot:
        _chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Unknown bot publishable key"})
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")

    # Enforce origin allowlist only when verification is enabled.
    if config.REQUIRE_DOMAIN_VERIFICATION:
        origin_h = _origin_host(origin)
        if origin_h:
            verified_hosts = set(list_verified_hosts(bot.bot_id))
            if origin_h not in verified_hosts:
                raise HTTPException(status_code=403, detail="Origin not allowed for this bot")

    _rate_limit(bot.bot_id)

    corpus = _ensure_bot_corpus(bot.bot_id)

    msg = (payload.message or "").strip()
    if not msg:
        _chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Missing message"})
        raise HTTPException(status_code=400, detail="Missing message")

    # The iframe runs on the API origin, so "Origin" won't be the host website.
    # Use the embedding page context passed from widget.js to reduce "this service" ambiguity.
    site_url = (getattr(payload, "site_url", None) or "").strip()
    site_title = (getattr(payload, "site_title", None) or "").strip()
    ctx_lines = []
    if site_title:
        ctx_lines.append(f"Site title: {site_title}")
    if site_url:
        ctx_lines.append(f"Site URL: {site_url}")
    query = msg
    if ctx_lines:
        query = f"{msg}\n\nContext:\n" + "\n".join(ctx_lines)

    allowed_host = ""
    if site_url:
        try:
            allowed_host = (urlparse(site_url).hostname or "").lower().split(":")[0]
        except Exception:
            allowed_host = ""

    _chat_debug_emit(
        {
            "type": "chat_context",
            "trace_id": trace_id,
            "bot_id": bot.bot_id,
            "corpus_resource": corpus,
            "allowed_host": allowed_host,
            "site_url": site_url,
            "site_title": site_title,
            "raw_message": msg,
            "final_query": query,
        }
    )

    def _rag_dbg(evt: Dict[str, Any]) -> None:
        evt2 = dict(evt)
        evt2["trace_id"] = trace_id
        _chat_debug_emit(evt2)

    result = run_vertex_rag(query, rag_corpus=corpus, allowed_host=allowed_host or None, debug_cb=_rag_dbg)
    _chat_debug_emit({"type": "chat_rag_result", "trace_id": trace_id, "result": result})
    sources = result.get("sources") or []
    citations = []
    for s in sources:
        citations.append(Citation(url=str(s.get("url") or ""), snippet=str(s.get("excerpt") or "")))

    # Strict demarcation: if we can't retrieve any sources for THIS host, refuse to answer.
    if not citations:
        host_label = allowed_host or "this site"
        _chat_debug_emit(
            {
                "type": "chat_refusal",
                "trace_id": trace_id,
                "reason": "no_citations_after_host_filter",
                "host_label": host_label,
            }
        )
        return WidgetChatResponse(
            answer=f"I can’t find that in the indexed content for {host_label}. Try asking about something on the site, or re-run Crawl.",
            citations=[],
        )

    _chat_debug_emit(
        {
            "type": "chat_response",
            "trace_id": trace_id,
            "answer": str(result.get("answer") or ""),
            "citations": [c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
        }
    )
    return WidgetChatResponse(answer=str(result.get("answer") or ""), citations=citations)
