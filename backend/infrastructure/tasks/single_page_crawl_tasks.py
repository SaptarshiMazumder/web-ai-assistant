import asyncio
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import google.auth
import vertexai
import httpx
import re
import codecs
from celery import Task
from google.cloud import storage
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from markdownify import markdownify

from domain.entities import BookingLinkJob, Document, TopicJob
from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import (
    PostgresBookingLinkJobRepository,
    PostgresIndexJobRepository,
    PostgresBotRepository,
    PostgresTopicJobRepository,
)
from infrastructure.rag.crawl_service import CRAWL_WAIT_FOR_CONTENT, _normalize_text_encoding
from infrastructure.rag.robots_policy import robots_policy
from infrastructure.repositories import GCSDocumentStorageRepository, VertexRAGRepository
from infrastructure.tasks.booking_link_tasks import booking_link_job_task

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_topic_job_id() -> str:
    return "tjob_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_booking_link_job_id() -> str:
    return "blj_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _get_source_language(bot_id: str, source_id: Optional[str]) -> str:
    # Hardcode JP for now (do not read env or source config).
    return "ja"


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


def _preview_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _repair_doc_for_language(doc: Document, *, lang: str) -> None:
    if not doc or not doc.url or not _is_japanese_lang(lang):
        return
    timeout = float(os.environ.get("CRAWL_REPAIR_TIMEOUT", "15"))
    max_chars = int(os.environ.get("CRAWL_REPAIR_MAX_CHARS", "120000"))
    headers = {"User-Agent": _default_user_agent(), "Accept-Language": "ja,en;q=0.8"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(doc.url)
            if resp.status_code >= 400:
                return
            data = resp.content or b""
            if not data:
                return
            html, enc, score, header_cs, meta_cs = _decode_bytes_with_charset(
                data, resp.headers.get("content-type")
            )
            if not html:
                return
            logger.info(
                "JP decode url=%s header_charset=%s meta_charset=%s chosen=%s score=%.2f",
                doc.url,
                header_cs or "",
                meta_cs or "",
                enc or "",
                score,
            )
            md = markdownify(html, heading_style="ATX")
            if not md:
                return
            if len(md) > max_chars:
                md = md[:max_chars]
            doc.content = f"Source URL: {doc.url}\n\n{md}"
    except Exception:
        return


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
        from infrastructure.tasks.crawl_tasks import topic_extraction_job
        async_result = topic_extraction_job.delay(job_id=job_id, bot_id=bot_id, org_id=bot.org_id, gcs_prefix=gcs_prefix)
        job.celery_task_id = async_result.id
        repo.update(job)
    except Exception as e:
        logger.warning(f"Failed to start topic extraction job for bot {bot_id}: {type(e).__name__}: {str(e)[:200]}")


def _start_booking_link_job(*, bot_id: str, index_job_id: str, root_url: str) -> None:
    if not bot_id or not index_job_id:
        return
    auto_start = (os.environ.get("BOOKING_RAG_AUTO_START") or "true").strip().lower()
    if auto_start not in ("1", "true", "yes", "on"):
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


async def _execute_single_page_crawl(
    job_id: str,
    bot_id: str,
    url: str,
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
) -> Dict[str, Any]:
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    source_lang = _get_source_language(bot_id, job.source_id)

    job.stage = "crawling"
    job_repo.update_job(job)

    # Robots.txt compliance: never fetch disallowed URLs.
    try:
        allowed = await robots_policy().is_allowed(url)
    except Exception:
        allowed = True
    if not allowed:
        job.docs_count = 0
        job.last_error = "Blocked by robots.txt"
        job.stage = "done"
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0, "blocked_by_robots": True}

    wait_for = (os.environ.get("SINGLE_PAGE_WAIT_FOR") or "").strip() or ""
    delay_before_return_html = float(os.environ.get("SINGLE_PAGE_DELAY_BEFORE_RETURN_HTML", "3.0"))
    viewport_width = int(os.environ.get("SINGLE_PAGE_VIEWPORT_WIDTH", "1920"))
    viewport_height = int(os.environ.get("SINGLE_PAGE_VIEWPORT_HEIGHT", "1080"))
    user_agent = (os.environ.get("SINGLE_PAGE_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    min_text_len = int(os.environ.get("SINGLE_PAGE_MIN_TEXT_LEN", "1800"))
    scroll_steps = int(os.environ.get("SINGLE_PAGE_SCROLL_STEPS", "4"))
    scroll_delay = float(os.environ.get("SINGLE_PAGE_SCROLL_DELAY", "0.6"))
    nav_retry_delay = float(os.environ.get("SINGLE_PAGE_NAV_RETRY_DELAY", "2.5"))

    # Domain-agnostic wait: require a meaningful amount of visible text.
    text_ready_wait = f"js:() => document.body && document.body.innerText && document.body.innerText.length > {min_text_len}"

    adapter = None
    browser_type = None
    try:
        # crawl4ai >= 0.8.0 (if provided)
        from crawl4ai import UndetectedAdapter as _UndetectedAdapter  # type: ignore
        adapter = _UndetectedAdapter()
    except Exception:
        try:
            # fallback for older crawl4ai
            from crawl4ai.async_configs import UndetectedAdapter as _UndetectedAdapter  # type: ignore
            adapter = _UndetectedAdapter()
        except Exception:
            # last-resort: use undetected browser type if available in this version
            browser_type = "undetected"

    def build_browser_config(*, headless: bool, use_undetected: bool) -> BrowserConfig:
        browser_kwargs = dict(
            headless=headless,
            verbose=True,
            enable_stealth=True,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        try:
            import inspect
            params = inspect.signature(BrowserConfig).parameters
            if "adapter" in params:
                browser_kwargs["adapter"] = adapter if use_undetected else None
            if "browser_type" in params and use_undetected and browser_type:
                browser_kwargs["browser_type"] = browser_type
        except Exception:
            pass
        return BrowserConfig(**browser_kwargs)

    def build_run_config(*, wait_for_expr: str, delay: float) -> CrawlerRunConfig:
        run_kwargs = dict(
            cache_mode=CacheMode.BYPASS,
            wait_for=wait_for_expr or None,
            delay_before_return_html=delay,
        )
        try:
            import inspect
            params = inspect.signature(CrawlerRunConfig).parameters
            if "headers" in params:
                run_kwargs["headers"] = {"User-Agent": user_agent}
        except Exception:
            pass
        return CrawlerRunConfig(**run_kwargs)

    def score_content(text: str) -> bool:
        if not text:
            return False
        if len(text) < min_text_len:
            return False
        # simple heuristic: enough unique tokens
        tokens = [t for t in text.split() if t.isalpha()]
        if len(tokens) < 200:
            return False
        if len(set(tokens)) < 120:
            return False
        return True

    async def auto_scroll(crawler: AsyncWebCrawler) -> None:
        try:
            for _ in range(max(1, scroll_steps)):
                await crawler.arun(
                    url=url,
                    config=build_run_config(wait_for_expr="", delay=0.1),
                )
        except Exception:
            pass

    # Pass B: JS-heavy, stealth, headful
    pass_configs = [
        ("js_heavy", build_browser_config(headless=False, use_undetected=False), build_run_config(wait_for_expr=text_ready_wait, delay=delay_before_return_html)),
        # Pass C: anti-bot hardening, use undetected if available
        ("anti_bot", build_browser_config(headless=False, use_undetected=True), build_run_config(wait_for_expr=text_ready_wait, delay=delay_before_return_html + 1.5)),
    ]

    docs: list[Document] = []
    last_error: str | None = None
    for pass_name, browser_cfg, run_cfg in pass_configs:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = None
            for attempt in range(3):
                try:
                    result = await crawler.arun(url=url, config=run_cfg)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "page.content" in msg and "navigating" in msg:
                        await asyncio.sleep(nav_retry_delay)
                        # Increase delay on retry to let navigation settle
                        run_cfg = build_run_config(
                            wait_for_expr=text_ready_wait,
                            delay=delay_before_return_html + (attempt + 1) * 1.5,
                        )
                        continue
                    raise
            if not result or not getattr(result, "success", False):
                last_error = str(getattr(result, "error_message", "") or getattr(result, "error", "") or "crawl failed")
                continue
            content = getattr(result, "markdown", None) or getattr(result, "cleaned_html", None) or ""
            content = _normalize_text_encoding(content)
            if not score_content(content):
                # Try a light scroll to trigger lazy content, then retry once
                try:
                    for _ in range(max(1, scroll_steps)):
                        await crawler.arun(url=url, config=build_run_config(wait_for_expr="", delay=scroll_delay))
                except Exception:
                    pass
                result = await crawler.arun(url=url, config=run_cfg)
                content = getattr(result, "markdown", None) or getattr(result, "cleaned_html", None) or ""
                content = _normalize_text_encoding(content)
            if content:
                preview_limit = int(os.environ.get("SINGLE_PAGE_LOG_MAX_CHARS", "8000"))
                preview = content if len(content) <= preview_limit else content[:preview_limit] + "\n...[truncated]"
                logger.info("Single-page crawl [%s] content preview for %s:\n%s", pass_name, url, preview)
            if content:
                docs.append(
                    Document(
                        url=url,
                        content=f"Source URL: {url}\n\n{content}",
                        metadata={"source": f"single_page_{pass_name}"},
                    )
                )
                break
    if not docs and last_error:
        job.last_error = last_error

    job.docs_count = len(docs) if docs else 0
    if docs:
        job.crawled_urls = [getattr(d, "url", "") or "" for d in docs]
    job_repo.update_job(job)

    if not docs:
        job.stage = "done"
        job_repo.update_job(job)
        return {"status": "done", "docs_count": 0}

    if docs:
        preview_limit = int(os.environ.get("CRAWL_LOG_MAX_CHARS", "4000"))
        for idx, doc in enumerate(docs):
            content = getattr(doc, "content", None) or ""
            if content:
                logger.info(
                    "Crawl raw preview [%s] %s:\n%s",
                    idx + 1,
                    getattr(doc, "url", "") or "",
                    _preview_text(content, preview_limit),
                )

    if source_lang:
        for doc in docs:
            _repair_doc_for_language(doc, lang=source_lang)
        preview_limit = int(os.environ.get("CRAWL_LOG_MAX_CHARS", "4000"))
        for idx, doc in enumerate(docs):
            content = getattr(doc, "content", None) or ""
            if content:
                logger.info(
                    "GCS upload preview [%s] %s:\n%s",
                    idx + 1,
                    getattr(doc, "url", "") or "",
                    _preview_text(content, preview_limit),
                )

    job.stage = "uploading"
    job_repo.update_job(job)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

    creds, proj = google.auth.load_credentials_from_file(creds_path)
    storage_client = storage.Client(credentials=creds, project=proj)
    storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
    gcs_prefix = storage_repo.save_documents(bot_id, docs)
    job.gcs_prefix = gcs_prefix
    job_repo.update_job(job)

    if gcs_prefix:
        _start_topic_extraction_job(bot_id, gcs_prefix)

    job.stage = "importing"
    job_repo.update_job(job)

    try:
        vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
    except Exception:
        pass

    rag_repo = VertexRAGRepository()
    rag_repo.import_documents(corpus_resource, gcs_prefix)
    job.stage = "import_submitted"
    job_repo.update_job(job)
    try:
        root_url = url if (url or "").startswith(("http://", "https://")) else ""
        _start_booking_link_job(bot_id=bot_id, index_job_id=job_id, root_url=root_url)
    except Exception as booking_error:
        logger.warning(
            f"Booking link extraction queue failed: {type(booking_error).__name__}: {str(booking_error)[:200]}"
        )
    return {"status": "done", "docs_count": job.docs_count, "gcs_prefix": gcs_prefix}


@celery_app.task(
    name="infrastructure.tasks.single_page_crawl_tasks.single_page_crawl_job",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def single_page_crawl_job(
    self: Task,
    job_id: str,
    bot_id: str,
    url: str,
    bucket_name: str,
    base_prefix: str,
    corpus_resource: Optional[str],
) -> Dict[str, Any]:
    job_repo = PostgresIndexJobRepository()
    job = job_repo.get_job(bot_id, job_id)
    if job:
        job.celery_task_id = self.request.id
        job_repo.update_job(job)

    if not corpus_resource:
        rag_repo = VertexRAGRepository()
        corpus_resource = rag_repo.ensure_corpus(bot_id)

    try:
        result = asyncio.run(
            _execute_single_page_crawl(job_id, bot_id, url, bucket_name, base_prefix, corpus_resource)
        )
        return result
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise self.retry(exc=exc)
    except Exception as exc:
        if job:
            job.stage = "error"
            job.last_error = str(exc)[:200]
            job_repo.update_job(job)
        raise
