import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag

from celery import Task

from infrastructure.celery_app import celery_app
from infrastructure.clients.rag_client import run_vertex_rag
from infrastructure.db.repositories import (
    PostgresBookingLinkJobRepository,
    PostgresBotCorpusRepository,
    PostgresIndexJobRepository,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_url(url: str) -> str:
    url = urldefrag((url or "").strip())[0]
    return url


def _url_key(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return ""
    if not p.scheme or not p.netloc:
        return ""
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"


def _is_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


def _extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)\]}>\"']+", text)
    return [u.strip() for u in urls if u.strip()]


def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _collect_evidence_urls(sources: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        src_url = _normalize_url(src.get("url") or "")
        snippet = src.get("excerpt") or ""
        if _is_http_url(src_url):
            evidence[_url_key(src_url)] = {
                "url": src_url,
                "source_urls": [src_url],
                "snippets": [snippet] if snippet else [],
            }
        for u in _extract_urls_from_text(snippet):
            norm = _normalize_url(u)
            if not _is_http_url(norm):
                continue
            key = _url_key(norm)
            entry = evidence.get(key)
            if not entry:
                evidence[key] = {
                    "url": norm,
                    "source_urls": [src_url] if src_url else [],
                    "snippets": [snippet] if snippet else [],
                }
            else:
                if src_url and src_url not in entry.get("source_urls", []):
                    entry.setdefault("source_urls", []).append(src_url)
                if snippet and snippet not in entry.get("snippets", []):
                    entry.setdefault("snippets", []).append(snippet)
    return evidence


def _merge_candidate(store: Dict[str, Dict[str, Any]], url: str, confidence: float, reason: str, evidence: Dict[str, Any]) -> None:
    key = _url_key(url)
    if not key:
        return
    entry = store.get(key)
    if not entry:
        store[key] = {
            "url": url,
            "confidence": confidence,
            "reasons": [reason] if reason else [],
            "sources": list(evidence.get("source_urls", [])),
            "snippets": list(evidence.get("snippets", []))[:3],
        }
        return
    if confidence > float(entry.get("confidence") or 0):
        entry["confidence"] = confidence
    if reason and reason not in entry.get("reasons", []):
        entry.setdefault("reasons", []).append(reason)
    for src in evidence.get("source_urls", []):
        if src not in entry.get("sources", []):
            entry.setdefault("sources", []).append(src)
    for snip in evidence.get("snippets", []):
        if snip not in entry.get("snippets", []):
            entry.setdefault("snippets", []).append(snip)
    entry["snippets"] = entry.get("snippets", [])[:3]


def _build_system_instruction() -> str:
    return (
        "You are extracting booking/availability URLs for hotels and restaurants. "
        "Return ONLY JSON with schema: "
        "{\"booking_urls\":[{\"url\":\"...\",\"confidence\":0-1,\"reason\":\"...\"}]}. "
        "Only include URLs that appear verbatim in the evidence snippets or in the source URLs. "
        "If none, return {\"booking_urls\":[]}."
    )


def _build_queries(root_url: str) -> List[str]:
    base = root_url or "this site"
    return [
        (
            "Find booking, reservation, or availability URLs for hotel room booking or restaurant table booking "
            f"on {base}. Return the booking page URLs only."
        ),
        (
            "List URLs for booking rooms, checking availability, or making restaurant reservations on this site. "
            "Return only URLs."
        ),
        (
            "Extract URLs used for booking or reservation flows (including international pages)."
        ),
    ]


@celery_app.task(
    name="booking.booking_link_job_task",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def booking_link_job_task(self: Task, job_id: str, bot_id: str) -> Dict[str, Any]:
    repo = PostgresBookingLinkJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        return {"status": "error", "error": "job not found"}

    job.status = "running"
    job.celery_task_id = self.request.id
    repo.update(job)

    corpus_repo = PostgresBotCorpusRepository()
    corpus = corpus_repo.get_bot_corpus(bot_id)
    if not corpus:
        job.status = "failed"
        job.error = "No RAG corpus found for bot"
        repo.update(job)
        return {"status": "error", "error": job.error}

    index_repo = PostgresIndexJobRepository()
    index_job = None
    if job.index_job_id:
        index_job = index_repo.get_job(bot_id, job.index_job_id)
    if not index_job:
        jobs = index_repo.list_jobs_for_bot(bot_id)
        index_job = jobs[0] if jobs else None

    root_url = job.root_url or (index_job.url if index_job else "")
    allowed_host = ""
    if index_job and index_job.hostname:
        allowed_host = index_job.hostname
    elif root_url:
        try:
            allowed_host = urlparse(root_url).netloc
        except Exception:
            allowed_host = ""

    system_instruction = _build_system_instruction()
    model_name = (os.environ.get("BOOKING_RAG_MODEL") or "").strip() or None
    temperature = float(os.environ.get("BOOKING_RAG_TEMPERATURE", "0.2"))
    max_candidates = int(os.environ.get("BOOKING_RAG_MAX_CANDIDATES", "30"))
    min_conf = float(os.environ.get("BOOKING_RAG_MIN_CONF", "0.35"))

    candidates: Dict[str, Dict[str, Any]] = {}

    for query in _build_queries(root_url):
        try:
            result = run_vertex_rag(
                query,
                rag_corpus=corpus,
                allowed_host=allowed_host or None,
                system_instruction=system_instruction,
                model_name=model_name,
                temperature=temperature,
            )
        except Exception:
            continue

        sources = result.get("sources") or []
        evidence_map = _collect_evidence_urls(sources)
        answer = (result.get("answer") or "").strip()
        parsed = _parse_json_from_text(answer) or {}
        booking_urls = parsed.get("booking_urls") or []
        if not isinstance(booking_urls, list):
            booking_urls = []

        for item in booking_urls:
            if not isinstance(item, dict):
                continue
            raw_url = (item.get("url") or "").strip()
            if not raw_url:
                continue
            if not _is_http_url(raw_url):
                if root_url:
                    raw_url = _normalize_url(urljoin(root_url, raw_url))
            if not _is_http_url(raw_url):
                continue
            key = _url_key(raw_url)
            evidence = evidence_map.get(key)
            if not evidence:
                continue
            conf = float(item.get("confidence") or 0)
            if conf < min_conf:
                continue
            reason = (item.get("reason") or "").strip()
            _merge_candidate(candidates, evidence.get("url") or raw_url, conf, reason, evidence)

    ordered = sorted(
        candidates.values(),
        key=lambda c: float(c.get("confidence") or 0),
        reverse=True,
    )
    if max_candidates > 0:
        ordered = ordered[:max_candidates]

    job.status = "done"
    job.links = ordered
    job.error = None
    repo.update(job)

    return {"status": "done", "count": len(ordered)}
