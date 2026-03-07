import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery import Task
from celery.exceptions import Retry, SoftTimeLimitExceeded

from infrastructure.celery_app import celery_app
from infrastructure.rag.crawl_service import CRAWL_MAX_CONCURRENCY
import httpx
from markdownify import markdownify
import re
import codecs

from infrastructure.repositories import Crawl4AICrawlerRepository, GCSDocumentStorageRepository, VertexRAGRepository
from infrastructure.db.repositories import (
    PostgresBookingLinkJobRepository,
    PostgresBotAssetRepository,
    PostgresBotRepository,
    PostgresIndexJobRepository,
    PostgresJobPipelineRepository,
    PostgresTopicJobRepository,
)
from infrastructure.rag.error_handling import safe_execute
from google.cloud import storage
import google.auth
import vertexai
from domain.entities import BookingLinkJob, Document, TopicJob
from application.services.default_prompt_service import (
    build_default_system_instruction,
    extract_business_type_from_widget_config,
)
from domain.platform_profiles import (
    get_default_post_crawl_jobs,
    get_default_source_language,
    get_post_crawl_jobs_for_widget,
)
from infrastructure.tasks.booking_link_tasks import booking_link_job_task
from application.services.job_pipeline_service import JobPipelineService

logger = logging.getLogger(__name__)

_MAX_CRAWL_DURATION_SEC = 600  # HARD 10-MINUTE LIMIT for training/crawl to GCS/RAG
try:
    _MENU_EXTRACTION_SOFT_LIMIT_SEC = max(120, int((os.environ.get("MENU_EXTRACTION_SOFT_LIMIT_SEC") or "600").strip()))
except ValueError:
    _MENU_EXTRACTION_SOFT_LIMIT_SEC = 600
_MENU_EXTRACTION_HARD_LIMIT_SEC = _MENU_EXTRACTION_SOFT_LIMIT_SEC + 20


def _asset_limit() -> int:
    raw = (os.environ.get("ASSET_MAX_PER_BOT") or "15").strip()
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 15


def _emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit event for progress tracking (for backward compatibility)."""
    # In Celery, we update DB directly, but can also log for monitoring
    print(f"WEB_AI_EVENT {json.dumps({'type': event_type, **data}, ensure_ascii=False)}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_topic_job_id() -> str:
    return "tjob_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_booking_link_job_id() -> str:
    return "blj_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _get_source_language(bot_id: str, source_id: Optional[str]) -> str:
    try:
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if bot and getattr(bot, "widget_config", None):
            raw = (bot.widget_config or "").strip()
            if raw:
                cfg = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(cfg, dict):
                    lang = str(cfg.get("language") or cfg.get("botLanguage") or "").strip().lower()
                    if lang:
                        return "ja" if lang in ("ja", "jp") else "en"
    except Exception:
        pass
    return get_default_source_language()


def _get_bot_language(bot: Any) -> str:
    """Extract bot language from widget_config; defaults to English."""
    raw = getattr(bot, "widget_config", None)
    if not raw or not str(raw).strip():
        return "en"
    try:
        cfg = json.loads(raw)
        lang = str(cfg.get("language") or "").strip().lower()
        if lang in ("ja", "jp"):
            return "ja"
    except (TypeError, ValueError, AttributeError):
        pass
    return "en"


def _detect_declared_charset(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        head = raw[:8192].decode("latin-1", errors="ignore")
    except Exception:
        return ""
    match = re.search(r"charset\s*=\s*['\"]?([a-zA-Z0-9_\-]+)", head, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return ""


def _score_decoded_text(text: str) -> float:
    if not text:
        return -1e9
    total = max(1, len(text))
    replacement = text.count("\ufffd")
    controls = sum(1 for ch in text if (ord(ch) < 32 and ch not in "\n\r\t"))
    jp = 0
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            jp += 1
    latin = sum(1 for ch in text if "A" <= ch <= "Z" or "a" <= ch <= "z")
    mojibake = sum(text.count(tok) for tok in ("Ã", "Â", "â", "ã", "¤", "¢", "»", "œ", "ƒ"))
    jp_ratio = jp / total
    return (jp * 5.0) + (latin * 0.1) + (jp_ratio * 200.0) - (replacement * 15.0) - (controls * 5.0) - (mojibake * 8.0)


def _decode_bytes_with_charset(
    data: bytes, content_type: str | None = None
) -> tuple[str, str, float, str, str]:
    if not data:
        return "", "", -1e9, "", ""
    header_charset = ""
    meta_charset = ""
    if content_type:
        match = re.search(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", content_type, re.IGNORECASE)
        if match:
            header_charset = match.group(1).strip().lower()
    meta_charset = _detect_declared_charset(data)

    candidates: list[tuple[str, float]] = []

    def add_candidate(enc: str, bonus: float = 0.0) -> None:
        enc = (enc or "").strip().lower()
        if not enc:
            return
        for existing, _ in candidates:
            if existing == enc:
                return
        try:
            codecs.lookup(enc)
        except Exception:
            return
        candidates.append((enc, bonus))

    if data.startswith(codecs.BOM_UTF8):
        add_candidate("utf-8-sig", bonus=3.0)
    if header_charset:
        add_candidate(header_charset, bonus=2.5)
    if meta_charset:
        add_candidate(meta_charset, bonus=2.0)
    try:
        from charset_normalizer import from_bytes  # type: ignore
        best = from_bytes(data).best()
        if best is not None and getattr(best, "encoding", None):
            add_candidate(str(best.encoding), bonus=1.5)
    except Exception:
        pass

    for enc in ("utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp", "latin-1"):
        add_candidate(enc)

    best_text = ""
    best_enc = ""
    best_score = -1e9
    for enc, bonus in candidates:
        try:
            decoded = data.decode(enc, errors="replace")
        except Exception:
            continue
        score = _score_decoded_text(decoded) + bonus
        if score > best_score:
            best_score = score
            best_text = decoded
            best_enc = enc

    if not best_text:
        try:
            fallback = data.decode("utf-8", errors="replace")
        except Exception:
            fallback = ""
        return fallback, "utf-8", _score_decoded_text(fallback), header_charset, meta_charset

    return best_text, best_enc, best_score, header_charset, meta_charset


def _default_user_agent() -> str:
    return (os.environ.get("CRAWL_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


def _is_japanese_lang(lang: str) -> bool:
    return lang in ("ja", "jpn", "jp", "japanese")


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    total = max(1, len(text))
    printable = sum(1 for ch in text if ch.isprintable())
    replacement = text.count("\ufffd")
    return (printable / total) - (replacement * 0.02)


def _preview_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _repair_docs_for_language(docs: List[Document], *, lang: str) -> List[Document]:
    if not docs or not _is_japanese_lang(lang):
        return docs
    timeout = float(os.environ.get("CRAWL_REPAIR_TIMEOUT", "15"))
    max_chars = int(os.environ.get("CRAWL_REPAIR_MAX_CHARS", "120000"))
    headers = {"User-Agent": _default_user_agent(), "Accept-Language": "ja,en;q=0.8"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        for doc in docs:
            url = (doc.url or "").strip()
            if not url:
                continue
            try:
                resp = client.get(url)
                if resp.status_code >= 400:
                    continue
                data = resp.content or b""
                if not data:
                    continue
                html, enc, score, header_cs, meta_cs = _decode_bytes_with_charset(
                    data, resp.headers.get("content-type")
                )
                if not html:
                    continue
                logger.info(
                    "JP decode url=%s header_charset=%s meta_charset=%s chosen=%s score=%.2f",
                    url,
                    header_cs or "",
                    meta_cs or "",
                    enc or "",
                    score,
                )
                md = markdownify(html, heading_style="ATX")
                if not md:
                    continue
                if len(md) > max_chars:
                    md = md[:max_chars]
                doc.content = f"Source URL: {url}\n\n{md}"
            except Exception:
                continue
    return docs


def _start_topic_extraction_job(bot_id: str, gcs_prefix: str) -> None:
    if not bot_id or not gcs_prefix:
        return
    try:
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            logger.warning(f"Cannot start topic job: bot {bot_id} not found")
            return
        repo = PostgresTopicJobRepository()
        job_id = _new_topic_job_id()
        now = _utc_now()
        job = TopicJob(
            job_id=job_id,
            org_id=bot.org_id,
            bot_id=bot_id,
            status="queued",
            stage="queued",
            gcs_prefix=gcs_prefix,
            docs_count=0,
            topics_count=0,
            last_error=None,
            celery_task_id=None,
            created_at=now,
            updated_at=now,
        )
        repo.create(job)
        async_result = topic_extraction_job.delay(job_id=job_id, bot_id=bot_id, org_id=bot.org_id, gcs_prefix=gcs_prefix)
        job.celery_task_id = async_result.id
        repo.update(job)
    except Exception as e:
        logger.warning(f"Failed to start topic extraction job for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")


def _get_post_crawl_jobs(bot_id: str) -> List[str]:
    """Get post-crawl job names from platform config for this bot."""
    default_jobs = get_default_post_crawl_jobs()
    try:
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot or not getattr(bot, "widget_config", None):
            return default_jobs
        raw = (bot.widget_config or "").strip()
        if not raw:
            return default_jobs
        cfg = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(cfg, dict):
            return default_jobs
        return get_post_crawl_jobs_for_widget(cfg)
    except Exception:
        return default_jobs


def _start_menu_extraction_job(bot_id: str, gcs_prefix: str) -> None:
    if not bot_id or not gcs_prefix:
        return
    try:
        import uuid
        from infrastructure.db.repositories import PostgresAssetExtractionJobRepository, AssetExtractionJob

        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            logger.warning(f"Cannot start menu extraction: bot {bot_id} not found")
            return
        job_id = "menuext_" + uuid.uuid4().hex
        now = _utc_now()
        job = AssetExtractionJob(
            job_id=job_id,
            bot_id=bot_id,
            org_id=bot.org_id,
            status="queued",
            created_at=now,
            updated_at=now,
            gcs_prefix=gcs_prefix,
            page_urls=None,
        )
        extract_repo = PostgresAssetExtractionJobRepository()
        extract_repo.create_job(job)
        async_result = menu_extraction_task.delay(
            bot_id=bot_id,
            org_id=bot.org_id,
            gcs_prefix=gcs_prefix,
            job_id=job_id,
        )
        job.celery_task_id = async_result.id
        extract_repo.update_job(job)
    except Exception as e:
        logger.warning(f"Failed to start menu extraction job for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")


def _start_booking_link_job(*, bot_id: str, index_job_id: str, root_url: str) -> None:
    if not bot_id or not index_job_id:
        return
    delay_sec = int(os.environ.get("BOOKING_RAG_START_DELAY_SEC", "90"))
    repo = PostgresBookingLinkJobRepository()
    now = _utc_now()
    job = BookingLinkJob(
        job_id=_new_booking_link_job_id(),
        bot_id=bot_id,
        index_job_id=index_job_id,
        root_url=root_url or "",
        status="queued",
        links=[],
        error=None,
        celery_task_id=None,
        created_at=now,
        updated_at=now,
    )
    repo.create(job)
    async_result = booking_link_job_task.apply_async((job.job_id, bot_id), countdown=max(0, delay_sec))
    job.celery_task_id = async_result.id
    repo.update(job)


def _run_post_crawl_pipeline(
    *,
    bot_id: str,
    index_job_id: str,
    gcs_prefix: str,
    root_url: str,
    crawled_urls: Optional[List[str]] = None,
) -> None:
    service = JobPipelineService(
        pipeline_repo=PostgresJobPipelineRepository(),
        bot_repo=PostgresBotRepository(),
    )
    service.start_post_crawl(
        bot_id=bot_id,
        index_job_id=index_job_id,
        gcs_prefix=gcs_prefix,
        root_url=root_url,
        crawled_urls=crawled_urls or [],
    )


def _extract_topics_from_docs(bot_id: str, docs: List[Any]) -> None:
    """Extract topics from crawled documents and save them for the bot."""
    if not docs:
        return
    
    try:
        from application.services.topic_extraction_service import topic_extraction_service
        
        # Get org_id for the bot
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            logger.warning(f"Cannot extract topics: bot {bot_id} not found")
            return
        
        org_id = bot.org_id
        
        # Prepare documents for extraction
        documents = []
        for doc in docs:
            content = getattr(doc, "content", None) or (doc.get("content") if isinstance(doc, dict) else "")
            url = getattr(doc, "url", None) or (doc.get("url") if isinstance(doc, dict) else "")
            if content:
                documents.append({"content": content, "url": url})
        
        if not documents:
            return
        
        # Extract and save topics
        service = topic_extraction_service()
        extracted = service.extract_topics_from_documents(
            org_id=org_id,
            bot_id=bot_id,
            documents=documents,
            clear_existing=False,  # Merge with existing topics
        )
        
        logger.info(f"Extracted {len(extracted)} topics for bot {bot_id}")
        
    except Exception as e:
        logger.warning(f"Topic extraction failed for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")


def _load_docs_from_gcs_prefix(gcs_prefix: str) -> List[Dict[str, Any]]:
    if not gcs_prefix:
        return []
    try:
        from common.config import config
    except Exception:
        return []
    bucket_raw = (config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")).strip()
    if not bucket_raw:
        return []
    bucket_name = bucket_raw.strip("/").split("/", 1)[0]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Use no max_results limit to find all .md files under any sub-prefix
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    print(f"[TopicExtraction] _load_docs_from_gcs_prefix: prefix={gcs_prefix}, total_blobs={len(blobs)}")
    documents: List[Dict[str, Any]] = []
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        try:
            content = blob.download_as_text(encoding="utf-8")
        except Exception:
            continue
        url = ""
        if content.startswith("Source URL:"):
            first_line = content.split("\n")[0]
            url = first_line.replace("Source URL:", "").strip()
        if content:
            documents.append({"content": content, "url": url})
    print(f"[TopicExtraction] _load_docs_from_gcs_prefix: loaded {len(documents)} .md documents")
    return documents


@celery_app.task(name="infrastructure.tasks.crawl_tasks.topic_extraction_job", bind=True)
def topic_extraction_job(
    self: Task,
    job_id: str,
    bot_id: str,
    org_id: str,
    gcs_prefix: str,
) -> Dict[str, Any]:
    repo = PostgresTopicJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        return {"status": "error", "error": "job not found"}
    try:
        job.status = "running"
        job.stage = "loading_docs"
        repo.update(job)

        documents = _load_docs_from_gcs_prefix(gcs_prefix)
        job.docs_count = len(documents)
        repo.update(job)

        if not documents:
            job.status = "done"
            job.stage = "done"
            repo.update(job)
            return {"status": "done", "docs_count": 0, "topics_count": 0}

        job.stage = "extracting"
        repo.update(job)

        from application.services.topic_extraction_service import topic_extraction_service
        service = topic_extraction_service()
        extracted = service.extract_topics_from_documents(
            org_id=org_id,
            bot_id=bot_id,
            documents=documents,
            clear_existing=False,
        )

        job.topics_count = len(extracted)
        job.stage = "done"
        job.status = "done"
        repo.update(job)
        return {"status": "done", "docs_count": job.docs_count, "topics_count": job.topics_count}
    except Exception as e:
        job.status = "error"
        job.stage = "error"
        job.last_error = f"{type(e).__name__}: {str(e)[:200]}"
        repo.update(job)
        return {"status": "error", "error": job.last_error}


@celery_app.task(name="infrastructure.tasks.crawl_tasks.asset_extraction_task", bind=True)
def asset_extraction_task(
    self: Task,
    bot_id: str,
    org_id: str,
    gcs_prefix: str,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract business assets from crawled documents using LLM."""
    # update job status to running
    job_repo = None
    current_job = None
    if job_id:
        try:
            from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
            job_repo = PostgresAssetExtractionJobRepository()
            current_job = job_repo.get_job(job_id)
            if current_job and current_job.status != "cancelled":
                current_job.status = "running"
                job_repo.update_job(current_job)
        except Exception as e:
            logger.warning(f"Failed to update job status to running: {e}")

    try:
        limit = _asset_limit()
        repo = PostgresBotAssetRepository()
        existing_count = len(repo.list_assets_for_bot(bot_id, active_only=False))
        remaining = max(0, limit - existing_count)
        if remaining <= 0:
            if job_repo and current_job:
                current_job.status = "done"
                job_repo.update_job(current_job)
            return {
                "status": "done",
                "assets_count": 0,
                "assets_total": existing_count,
                "assets_limit": limit,
            }

        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {
                    "status": "cancelled",
                    "assets_count": 0,
                    "assets_total": existing_count,
                    "assets_limit": limit,
                }

        from application.services.asset_extraction_service import asset_extraction_service
        service = asset_extraction_service()
        count = service.extract_from_gcs_prefix(
            org_id=org_id,
            bot_id=bot_id,
            gcs_prefix=gcs_prefix,
            max_assets=remaining,
            page_urls=(current_job.page_urls if current_job else None),
            job_id=job_id,
        )
        total_count = len(repo.list_assets_for_bot(bot_id, active_only=False))

        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {
                    "status": "cancelled",
                    "assets_count": count,
                    "assets_total": total_count,
                    "assets_limit": limit,
                }
            current_job.status = "done"
            job_repo.update_job(current_job)

        logger.info(f"Asset extraction completed for bot {bot_id}: {count} assets created")
        return {
            "status": "done",
            "assets_count": count,
            "assets_total": total_count,
            "assets_limit": limit,
        }
    except Exception as e:
        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {"status": "cancelled"}
        if job_repo and current_job:
            current_job.status = "error"
            current_job.error = str(e)[:200]
            try:
                job_repo.update_job(current_job)
            except Exception:
                pass

        logger.warning(f"Asset extraction failed for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")
        return {"status": "error", "error": str(e)[:200]}


@celery_app.task(
    name="infrastructure.tasks.crawl_tasks.menu_extraction_task",
    bind=True,
    time_limit=_MENU_EXTRACTION_HARD_LIMIT_SEC,
    soft_time_limit=_MENU_EXTRACTION_SOFT_LIMIT_SEC,
)
def menu_extraction_task(
    self: Task,
    bot_id: str,
    org_id: str,
    gcs_prefix: str,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract menu items from crawled restaurant pages using LLM."""
    job_repo = None
    current_job = None
    if job_id:
        try:
            from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
            job_repo = PostgresAssetExtractionJobRepository()
            current_job = job_repo.get_job(job_id)
            if current_job and current_job.status != "cancelled":
                current_job.status = "running"
                job_repo.update_job(current_job)
        except Exception as e:
            logger.warning(f"Failed to update menu extraction job status: {e}")

    try:
        limit = int((os.environ.get("MENU_MAX_PER_BOT") or os.environ.get("ASSET_MAX_PER_BOT") or "500").strip() or 500)
        repo = PostgresBotAssetRepository()
        existing_count = len(repo.list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item"))
        remaining = max(0, limit - existing_count)
        if remaining <= 0:
            if job_repo and current_job:
                current_job.status = "done"
                job_repo.update_job(current_job)
            return {"status": "done", "items_count": 0, "items_total": existing_count, "items_limit": limit}

        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {"status": "cancelled", "items_count": 0, "items_total": existing_count, "items_limit": limit}

        from application.services.menu_extraction_service import menu_extraction_service
        service = menu_extraction_service()
        count = service.extract_from_gcs_prefix(
            org_id=org_id,
            bot_id=bot_id,
            gcs_prefix=gcs_prefix,
            max_items=remaining,
            page_urls=(current_job.page_urls if current_job else None),
            job_id=job_id,
        )
        total_count = len(repo.list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item"))

        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {"status": "cancelled", "items_count": count, "items_total": total_count, "items_limit": limit}
            current_job.status = "done"
            job_repo.update_job(current_job)

        logger.info(f"Menu extraction completed for bot {bot_id}: {count} items created")
        return {"status": "done", "items_count": count, "items_total": total_count, "items_limit": limit}
    except Exception as e:
        if job_repo and current_job:
            latest = job_repo.get_job(current_job.job_id)
            if latest and latest.status == "cancelled":
                return {"status": "cancelled"}
        if job_repo and current_job:
            current_job.status = "error"
            current_job.error = str(e)[:200]
            try:
                job_repo.update_job(current_job)
            except Exception:
                pass
        logger.warning(f"Menu extraction failed for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")
        return {"status": "error", "error": str(e)[:200]}


@celery_app.task(
    name="infrastructure.tasks.crawl_tasks.prompt_generation_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def prompt_generation_task(
    self: Task,
    bot_id: str,
    org_id: str,
    gcs_prefix: str,
    root_url: str = "",
    index_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate deterministic system instructions (no LLM)."""
    job_repo: Optional[PostgresIndexJobRepository] = None
    task_outcome = "error"
    try:
        if index_job_id:
            try:
                job_repo = PostgresIndexJobRepository()
                job = job_repo.get_job(bot_id, index_job_id)
                if job and (job.stage or "").lower() == "cancelled":
                    task_outcome = "cancelled"
                    return {"status": "cancelled", "reason": "index job cancelled"}
                if job and (job.stage or "").lower() != "error":
                    job.stage = "prompt_generating"
                    if self.request and self.request.id:
                        job.celery_task_id = self.request.id
                    job_repo.update_job(job)
            except Exception as stage_error:
                logger.warning(
                    "[PromptGen] Failed to mark prompt_generating for bot %s job %s: %s: %s",
                    bot_id,
                    index_job_id,
                    type(stage_error).__name__,
                    str(stage_error)[:200],
                )

        # 1. Check if bot already has custom instructions
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            task_outcome = "skipped"
            return {"status": "skipped", "reason": "bot not found"}

        existing_config = {}
        if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
            try:
                existing_config = json.loads(bot.agent_config)
            except (TypeError, ValueError):
                pass

        existing_instructions = (existing_config.get("instructions") or "").strip()
        if existing_instructions:
            logger.info(f"[PromptGen] Bot {bot_id} already has custom instructions, skipping auto-generation")
            task_outcome = "skipped"
            return {"status": "skipped", "reason": "custom instructions exist"}

        widget_config: Dict[str, Any] = {}
        if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
            try:
                widget_config = json.loads(bot.widget_config)
            except (TypeError, ValueError):
                widget_config = {}

        # 2. Build deterministic instruction from bot metadata.
        bot_lang = _get_bot_language(bot)
        generated_prompt = build_default_system_instruction(
            bot_name=(bot.display_name or "").strip(),
            business_type=extract_business_type_from_widget_config(widget_config),
            lang=bot_lang,
        )

        # 3. Save to agent_config.instructions
        existing_config["instructions"] = generated_prompt
        config_json = json.dumps(existing_config, ensure_ascii=False)
        bot_repo.update_agent_config(bot_id, config_json)

        logger.info(f"[PromptGen] Deterministic prompt set for bot {bot_id} ({len(generated_prompt)} chars)")
        task_outcome = "done"
        return {"status": "done", "prompt_length": len(generated_prompt)}

    except Exception as e:
        task_outcome = "error"
        logger.warning(f"[PromptGen] Failed for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")
        return {"status": "error", "error": str(e)[:200]}
    finally:
        if index_job_id:
            try:
                repo = job_repo or PostgresIndexJobRepository()
                latest = repo.get_job(bot_id, index_job_id)
                if latest:
                    latest_stage = (latest.stage or "").lower()
                    # Respect explicit terminal updates from other flows (cancel/error).
                    if latest_stage not in ("cancelled", "error") and task_outcome != "cancelled":
                        # Prompt generation is optional; once it exits, indexing
                        # should be considered ready for normal post-import flows.
                        latest.stage = "import_submitted"
                        repo.update_job(latest)
            except Exception as finalize_error:
                logger.warning(
                    "[PromptGen] Failed to finalize index job stage for bot %s job %s: %s: %s",
                    bot_id,
                    index_job_id,
                    type(finalize_error).__name__,
                    str(finalize_error)[:200],
                )


async def _execute_crawl(
    job_id: str,
    bot_id: str,
    url: Optional[str],
    urls: Optional[List[str]],
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
    headless: Optional[bool] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute the crawl job (async function) with timeout enforcement.
    
    CRITICAL: This function must complete within 10 minutes (enforced by Celery soft_time_limit).
    """
    if start_time is None:
        start_time = time.monotonic()
        
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    try:
        source_lang = _get_source_language(bot_id, job.source_id)
        job.stage = "crawling"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "starting_browser"})
        _emit_event("progress", {"pages_crawled": 0, "url": (url or ""), "depth": 0})
        _emit_event("stage", {"stage": "crawling"})

        crawler_repo = Crawl4AICrawlerRepository()

        def _on_progress(evt: Dict[str, Any]):
            if evt.get("type") == "page_crawled":
                job.pages_crawled = int(evt.get("count") or 0)
                job.last_crawled_url = str(evt.get("url") or "")
                job.last_depth = int(evt.get("depth") if evt.get("depth") is not None else -1)
                job_repo.update_job(job)
                _emit_event("progress", {
                    "pages_crawled": job.pages_crawled,
                    "url": job.last_crawled_url,
                    "depth": job.last_depth,
                })
            elif evt.get("type") == "fetch":
                _emit_event("fetch", {
                    "url": str(evt.get("url") or ""),
                    "success": bool(evt.get("success")),
                    "status_code": evt.get("status_code"),
                    "error": str(evt.get("error") or ""),
                    "content_source": str(evt.get("content_source") or ""),
                    "markdown_len": int(evt.get("markdown_len") or 0),
                    "text_len": int(evt.get("text_len") or 0),
                    "extracted_text_len": int(evt.get("extracted_text_len") or 0),
                    "cleaned_html_len": int(evt.get("cleaned_html_len") or 0),
                    "html_len": int(evt.get("html_len") or 0),
                    "raw_html_len": int(evt.get("raw_html_len") or 0),
                })

        # Crawl with comprehensive error handling - always returns partial results
        # Single URL or list of URLs: crawl only those pages (no link-following / BFS)
        docs: List[Any] = []
        try:
            if urls:
                docs = await crawler_repo.crawl_urls_list(
                    urls,
                    max_concurrent=CRAWL_MAX_CONCURRENCY,
                    progress_cb=_on_progress,
                    headless=headless,
                )
            else:
                # Single URL: crawl only that page (no nested pages)
                single_url = (url or "").strip()
                if single_url:
                    docs = await crawler_repo.crawl_urls_list(
                        [single_url],
                        max_concurrent=CRAWL_MAX_CONCURRENCY,
                        progress_cb=_on_progress,
                        headless=headless,
                    )
        except Exception as crawl_error:
            # Log error but continue - we might have partial results
            error_msg = str(crawl_error)[:200]
            logger.warning(f"Crawl error (continuing with partial results): {type(crawl_error).__name__}: {error_msg}")
            job.last_error = f"Crawl error: {error_msg}"
            # Don't raise - continue to process whatever we got

        if docs:
            preview_limit = int(os.environ.get("CRAWL_LOG_MAX_CHARS", "4000"))
            for idx, doc in enumerate(docs):
                url = getattr(doc, "url", None) or (doc.get("url") if isinstance(doc, dict) else "") or ""
                content = getattr(doc, "content", None) or (doc.get("content") if isinstance(doc, dict) else "") or ""
                if content:
                    logger.info("Crawl raw preview [%s] %s:\n%s", idx + 1, url, _preview_text(content, preview_limit))

            docs = _repair_docs_for_language(docs, lang=source_lang)

            for idx, doc in enumerate(docs):
                url = getattr(doc, "url", None) or (doc.get("url") if isinstance(doc, dict) else "") or ""
                content = getattr(doc, "content", None) or (doc.get("content") if isinstance(doc, dict) else "") or ""
                if content:
                    logger.info(
                        "GCS upload preview [%s] %s:\n%s",
                        idx + 1,
                        url,
                        _preview_text(content, preview_limit),
                    )

        job.docs_count = len(docs) if docs else 0
        # Store every URL we discovered and indexed (for display in dashboard)
        crawled_urls = []
        for d in docs or []:
            u = getattr(d, "url", None) or (d.get("url") if isinstance(d, dict) else "")
            if u:
                crawled_urls.append(str(u))
        job.crawled_urls = crawled_urls
        job_repo.update_job(job)
        _emit_event("result", {"docs_count": job.docs_count})

        # Even if no docs, continue to completion (might be a valid empty site)
        if not docs:
            job.stage = "done"
            job_repo.update_job(job)
            _emit_event("stage", {"stage": "done"})
            return {"status": "done", "docs_count": 0}

        job.stage = "uploading"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "uploading"})

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds_path:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

        creds, proj = google.auth.load_credentials_from_file(creds_path)
        _emit_event("auth", {
            "creds_path": creds_path,
            "creds_type": "service_account" if getattr(creds, "service_account_email", None) else "non_service_account",
            "project": proj,
        })

        bot_id_from_prefix = ""
        if "/bots/" in base_prefix:
            parts = base_prefix.split("/bots/")
            if len(parts) > 1:
                bot_id_from_prefix = parts[1].split("/")[0]

        # Upload to GCS with error handling
        gcs_prefix = ""
        try:
            storage_client = storage.Client(credentials=creds, project=proj)
            storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
            gcs_prefix = storage_repo.save_documents(bot_id_from_prefix, docs)
            job.gcs_prefix = gcs_prefix
            job_repo.update_job(job)
            _emit_event("gcs_prefix", {"gcs_prefix": gcs_prefix})
        except Exception as upload_error:
            error_msg = str(upload_error)[:200]
            logger.error(f"GCS upload error: {type(upload_error).__name__}: {error_msg}")
            job.last_error = f"Upload error: {error_msg}"
            job_repo.update_job(job)
            # Continue even if upload fails - at least we tried

        # Import to RAG with error handling
        job.stage = "importing"
        job_repo.update_job(job)
        _emit_event("stage", {"stage": "importing"})

        try:
            vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
        except Exception as init_error:
            logger.debug(f"Vertex AI init error (may already be initialized): {type(init_error).__name__}")

        try:
            rag_repo = VertexRAGRepository()
            if gcs_prefix:
                rag_repo.import_documents(corpus_resource, gcs_prefix)
            job.stage = "import_submitted"
            job_repo.update_job(job)
            _emit_event("stage", {"stage": "import_submitted"})
            try:
                _run_post_crawl_pipeline(
                    bot_id=bot_id,
                    index_job_id=job_id,
                    gcs_prefix=gcs_prefix,
                    root_url=job.url if (job.url or "").startswith(("http://", "https://")) else "",
                    crawled_urls=getattr(job, "crawled_urls", None) or [],
                )
            except Exception as pipeline_error:
                logger.warning(
                    f"Post-crawl pipeline start failed: {type(pipeline_error).__name__}: {str(pipeline_error)[:200]}"
                )
        except Exception as import_error:
            error_msg = str(import_error)[:200]
            logger.error(f"RAG import error: {type(import_error).__name__}: {error_msg}")
            job.last_error = f"Import error: {error_msg}"
            job.stage = "error"
            job_repo.update_job(job)
            # Don't raise - return what we have

        return {"status": "done", "docs_count": len(docs), "gcs_prefix": gcs_prefix}

    except Exception as e:
        # Last resort error handling - update job and return error status
        error_msg = str(e)[:500]
        logger.error(f"Critical crawl error: {type(e).__name__}: {error_msg}")
        try:
            job.stage = "error"
            job.last_error = error_msg
            job_repo.update_job(job)
            _emit_event("error", {"error": error_msg})
        except Exception:
            # Even job update failed - log and continue
            pass
        # Re-raise so Celery can handle retry
        raise


@celery_app.task(
    name="infrastructure.tasks.crawl_tasks.crawl_job",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    time_limit=_MAX_CRAWL_DURATION_SEC + 20,  # Hard kill after 10min + 20s
    soft_time_limit=_MAX_CRAWL_DURATION_SEC,  # Soft timeout at 10min
)
def crawl_job_task(
    self: Task,
    job_id: str,
    bot_id: str,
    url: Optional[str],
    urls: Optional[List[str]],
    bucket_name: str,
    base_prefix: str,
    corpus_resource: Optional[str],
    headless: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Celery task to execute crawling job with HARD 10-MINUTE LIMIT.
    
    CRITICAL: This task has a hard timeout of 10 minutes for the entire flow:
    crawling -> GCS upload -> Vertex RAG import. Under NO circumstances should it run longer.
    
    Args:
        self: Celery task instance (for retries)
        job_id: Job identifier
        bot_id: Bot identifier
        url: Single URL for BFS crawl (if urls is None)
        urls: List of URLs for direct crawl (if url is None)
        bucket_name: GCS bucket name
        base_prefix: GCS base prefix
        corpus_resource: Vertex AI RAG corpus resource name (None = resolve in worker for fast API return)
    
    Returns:
        Dict with status and results
    """
    start_time = time.monotonic()
    
    # Store celery_task_id in job
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if job:
        job.celery_task_id = self.request.id
        job_repo.update_job(job)

    # Resolve corpus in worker so the API can return job_id immediately (ensure_corpus can take 30+ s)
    if not corpus_resource:
        rag_repo = VertexRAGRepository()
        corpus_resource = rag_repo.ensure_corpus(bot_id)
    
    try:
        # Run async function - create new event loop for Celery worker
        result = asyncio.run(
            _execute_crawl(job_id, bot_id, url, urls, bucket_name, base_prefix, corpus_resource, headless, start_time)
        )
        
        elapsed = time.monotonic() - start_time
        logger.info(
            "Crawl job %s COMPLETED successfully in %.1fs (bot=%s, docs=%d)",
            job_id, elapsed, bot_id, result.get("docs_count", 0)
        )
        return result
    except SoftTimeLimitExceeded:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Crawl job %s TIMEOUT after %.1fs (bot=%s). 10-minute limit exceeded!",
            job_id, elapsed, bot_id
        )
        if job:
            job.stage = "error"
            job.last_error = f"Training timeout: exceeded 10-minute limit (ran {int(elapsed)}s)"
            job_repo.update_job(job)
        raise
    except (ConnectionError, TimeoutError, OSError) as exc:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "Crawl job %s transient error after %.1fs (bot=%s): %s. Retrying...",
            job_id, elapsed, bot_id, type(exc).__name__
        )
        # Retry transient errors
        raise self.retry(exc=exc)
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.exception(
            "Crawl job %s FAILED after %.1fs (bot=%s): %s",
            job_id, elapsed, bot_id, exc
        )
        # Don't retry other errors (permanent failures)
        raise
