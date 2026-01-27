import asyncio
import os
import re
import hashlib
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from urllib.parse import urlparse, urldefrag

from google.cloud import storage
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# --- Vertex AI RAG
from vertexai import rag as vx_rag
import vertexai

from common.config import config

# =========================
# ---- CONFIG -------------
# =========================
PROJECT_ID = (config.PROJECT_ID or "").strip()
VERTEX_LOCATION = (config.LOCATION or "us-central1").strip()

# If you already have a corpus, put its full resource name here.
# Else leave empty ("") and the script will create one and print its name.
RAG_CORPUS = os.environ.get("DEFAULT_RAG_CORPUS", "").strip()

# Embedding model used by the RAG index
EMBEDDING_PUBLISHER_MODEL = "publishers/google/models/text-embedding-005"

def _parse_bucket_and_prefix() -> Tuple[str, str]:
    bucket_and_prefix = (config.GCS_BUCKET or "").strip("/").split("/", 1)
    if len(bucket_and_prefix) == 2:
        return bucket_and_prefix[0], bucket_and_prefix[1]
    if len(bucket_and_prefix) == 1 and bucket_and_prefix[0]:
        return bucket_and_prefix[0], ""
    return "", ""

BUCKET_NAME, GCS_SUBPATH = _parse_bucket_and_prefix()
CRAWL_MAX_DEPTH = 8
CRAWL_MAX_CONCURRENCY = 25
HEADLESS = True

# RAG import (chunking) config
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64

# =========================
# ---- UTILITIES ----------
# =========================
def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def _get_str(result: Any, attr: str) -> str:
    try:
        v = getattr(result, attr, None)
    except Exception:
        v = None
    return v if isinstance(v, str) else ""


def _len_attr(result: Any, attr: str) -> int:
    try:
        v = getattr(result, attr, None)
    except Exception:
        v = None
    if isinstance(v, str):
        return len(v)
    return 0


def _meta_attr(result: Any, attr: str) -> Any:
    try:
        return getattr(result, attr, None)
    except Exception:
        return None


def _best_text(result: Any) -> Tuple[str, str]:
    """
    crawl4ai sometimes returns empty markdown even when the fetch succeeded.
    Use fallbacks so we don't end up with docs_count=0 for successful pages.
    """
    md = _get_str(result, "markdown").strip()
    extracted = _get_str(result, "extracted_text").strip()
    text = _get_str(result, "text").strip()

    # Prefer markdown if it's reasonably complete; otherwise use extracted/text.
    primary = md
    primary_src = "markdown" if md else ""
    if (len(md) < 400 and len(extracted) > len(md)) or (len(extracted) > len(md) * 1.5):
        primary, primary_src = extracted, "extracted_text"
    elif (len(md) < 400 and len(text) > len(md)) or (len(text) > len(md) * 1.5):
        primary, primary_src = text, "text"

    # If we have both markdown and extracted_text and they differ, concatenate to
    # catch content that markdown conversion might drop (e.g. accordions/FAQ).
    parts: List[str] = []
    used_src = primary_src
    if md:
        parts.append(md)
    if extracted and extracted not in md and extracted not in primary:
        parts.append("\n\n---\n\n" + extracted)
        used_src = used_src or "markdown+extracted_text"
    elif extracted and extracted not in md and primary_src == "extracted_text":
        # primary is extracted; still append markdown if it has unique bits
        if md and md not in extracted:
            parts.append("\n\n---\n\n" + md)
            used_src = "extracted_text+markdown"

    if not parts:
        # Last resort: HTML variants (can be large).
        for attr in ("cleaned_html", "html", "raw_html", "content"):
            v = _get_str(result, attr).strip()
            if v:
                return v[:120_000], attr
        return "", ""

    combined = "\n".join(parts).strip()
    if len(combined) > 120_000:
        combined = combined[:120_000]
    return combined, used_src or primary_src or ""

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "index"

def _ensure_url(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if not s.startswith(("http://", "https://")):
        return f"https://{s}"
    return s

def _site_slug_from_url(url: str) -> str:
    # NOTE: kept for backwards compatibility, but indexing now prefers exact-host prefixes.
    return _slugify(urlparse(url).netloc or "site")

def host_prefix_from_url(url: str) -> str:
    """
    Exact-hostname isolation prefix.
    Example: saas/<tenant>/bots/<bot_id>/hosts/www.example.com/<timestamp>/...
    """
    host = (urlparse(url).hostname or "").strip().lower()
    host = host.split(":")[0]
    return f"hosts/{host or 'unknown-host'}"

# =========================
# ---- RAG HELPERS --------
# =========================
def get_or_create_corpus(project_id: str, location: str, corpus_hint: str = RAG_CORPUS) -> str:
    """Return a corpus resource name. Create one if not provided."""
    vertexai.init(project=project_id, location=location)

    if corpus_hint:
        return corpus_hint

    emb_cfg = vx_rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=vx_rag.VertexPredictionEndpoint(
            publisher_model=EMBEDDING_PUBLISHER_MODEL
        )
    )
    corpus = vx_rag.create_corpus(
        display_name="web_corpus",
        backend_config=vx_rag.RagVectorDbConfig(
            rag_embedding_model_config=emb_cfg
        ),
    )
    print(f"[RAG] Created corpus: {corpus.name}")
    return corpus.name

def import_gcs_prefix_into_corpus(corpus_resource: str, bucket_name: str, prefix: str) -> None:
    """Import all files under gs://bucket/prefix/ into the given RAG corpus."""
    gcs_uri = f"gs://{bucket_name}/{prefix}/"
    try:
        vx_rag.import_files(
            corpus_resource,
            [gcs_uri],
            transformation_config=vx_rag.TransformationConfig(
                chunking_config=vx_rag.ChunkingConfig(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                # (Later) you can switch to semantic/html chunking configs
            ),
            max_embedding_requests_per_min=1000,
        )
        print(f"[RAG] Imported: {gcs_uri}")
    except Exception as e:
        # Provide actionable diagnostics
        print("[RAG] Import failed.")
        print(f"  Corpus:   {corpus_resource}")
        print(f"  GCS URI:  {gcs_uri}")
        print(f"  Project:  {PROJECT_ID}")
        print(f"  Location: {VERTEX_LOCATION}")
        print(f"  Error:    {e}")
        print("\nCommon fixes:\n"
              "  1) Grant the Vertex AI service agent Storage Object Viewer on your bucket:\n"
              "     gsutil iam ch serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com:roles/storage.objectViewer gs://" + bucket_name + "\n"
              "  2) Ensure the corpus exists in the same Vertex location (" + VERTEX_LOCATION + ") and you initialized vertexai with that location.\n"
              "  3) Verify the prefix exists and contains files: gsutil ls " + gcs_uri + "\n"
              "  4) Check that your account has Vertex AI permissions in project " + PROJECT_ID + ".")
        raise

# =========================
# ---- GCS HELPERS --------
# =========================
def upload_markdown_docs_to_gcs(
    bucket_name: str,
    base_prefix: str,
    docs: List[Dict[str, Any]],
    *,
    storage_client: "storage.Client | None" = None,
) -> str:
    """Uploads docs as markdown files to GCS and returns the prefix used (without trailing slash)."""
    if not docs:
        raise ValueError("No docs to upload to GCS.")
    client = storage_client or storage.Client()
    bucket = client.bucket(bucket_name)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    first_url = docs[0]["url"]
    host_prefix = host_prefix_from_url(first_url)
    prefix = f"{base_prefix}/{host_prefix}/{timestamp}"

    for doc in docs:
        url = doc["url"]
        md = doc["markdown"]
        parsed = urlparse(url)
        path_slug = _slugify(parsed.path or "index")
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"{path_slug or 'index'}-{url_hash}.md"
        blob_name = f"{prefix}/{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(md, content_type="text/markdown")

    return prefix  # e.g., saas/<tenant>/bots/<bot_id>/hosts/example.com/20250814-010203

def list_existing_site_prefixes(bucket_name: str, base_prefix: str, site_url: str) -> List[str]:
    """
    Returns sorted list of GCS prefixes for previous crawls of this site.
    Format: {base_prefix}/hosts/<host>/<timestamp>
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    base_prefix = (base_prefix or "").strip("/")
    # Exact-host isolation: list only this hostname’s prefixes.
    site_root = f"{base_prefix}/{host_prefix_from_url(site_url)}/"

    prefixes = set()
    for blob in bucket.list_blobs(prefix=site_root):
        name = blob.name or ""
        if not name.startswith(site_root):
            continue
        # Expected: {base_prefix}/hosts/<hostname>/<timestamp>/file.md
        rel = name[len(site_root):]
        ts = rel.split("/", 1)[0]
        if ts:
            prefixes.add(f"{site_root.rstrip('/')}/{ts}")

    def _ts_key(pref: str) -> Tuple[datetime, str]:
        ts = pref.rstrip("/").split("/")[-1]
        try:
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
            return (dt, ts)
        except Exception:
            return (datetime.min, ts)

    return sorted(prefixes, key=_ts_key)

def choose_prefix_interactively(prefixes: List[str]) -> str | None:
    if not prefixes:
        return None
    print("\nFound previous crawls:\n")
    for i, p in enumerate(prefixes, 1):
        print(f"  {i}. gs://{BUCKET_NAME}/{p}/")
    print("  0. Cancel")
    while True:
        choice = input("Pick a number (Enter=latest): ").strip()
        if choice == "":
            return prefixes[-1]
        if choice == "0":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(prefixes):
                return prefixes[idx - 1]
        print("Invalid choice. Try again.")

# =========================
# ---- CRAWLER ------------
# =========================
async def crawl_site_bfs(
    root_url: str,
    max_depth: int,
    max_concurrent: int,
    *,
    stop_event: Optional[asyncio.Event] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    parsed_root = urlparse(root_url)
    root_netloc = parsed_root.netloc
    visited = set()
    current_urls = set([_normalize_url(root_url)])
    all_results: List[Dict[str, Any]] = []

    def is_internal(url: str) -> bool:
        return urlparse(url).netloc == root_netloc

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(max_depth):
                if stop_event and stop_event.is_set():
                    print(f"[crawl] Stop requested; returning {len(all_results)} crawled page(s) so far.")
                    break
                urls_to_crawl = [u for u in current_urls if u not in visited]
                if not urls_to_crawl:
                    break

                print(f"[Depth {depth}] Crawling {len(urls_to_crawl)} page(s)...")
                try:
                    results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                except asyncio.CancelledError:
                    # Graceful cancel: keep what we already crawled
                    print(f"[crawl] Cancelled; returning {len(all_results)} crawled page(s) so far.")
                    return all_results

                next_level_urls = set()

                for result in results:
                    norm = _normalize_url(result.url)
                    visited.add(norm)
                    if result.success:
                        content, src = _best_text(result)
                        if content:
                            # Prefix the content with its source URL so retrieval can match vague queries
                            # without the user having to type the site/service name.
                            all_results.append({"url": result.url, "markdown": f"Source URL: {result.url}\n\n{content}"})
                            if src != "markdown":
                                # Helpful when debugging "0 pages" issues.
                                print(f"[crawl] Used fallback content '{src}' for {result.url}")
                        # Emit per-fetch diagnostics (even if content is empty) so callers
                        # can see why docs_count is 0 (blocked, empty render, etc.).
                        if progress_cb:
                            try:
                                progress_cb(
                                    {
                                        "type": "fetch",
                                        "url": result.url,
                                        "success": True,
                                        "status_code": _meta_attr(result, "status_code")
                                        or _meta_attr(result, "http_status")
                                        or _meta_attr(result, "status"),
                                        "error": _meta_attr(result, "error")
                                        or _meta_attr(result, "error_message")
                                        or _meta_attr(result, "message"),
                                        "content_source": src or "",
                                        "markdown_len": _len_attr(result, "markdown"),
                                        "text_len": _len_attr(result, "text"),
                                        "extracted_text_len": _len_attr(result, "extracted_text"),
                                        "cleaned_html_len": _len_attr(result, "cleaned_html"),
                                        "html_len": _len_attr(result, "html"),
                                        "raw_html_len": _len_attr(result, "raw_html"),
                                    }
                                )
                            except Exception:
                                pass
                        if progress_cb:
                            try:
                                progress_cb({
                                    "type": "page_crawled",
                                    "count": len(all_results),
                                    "url": result.url,
                                    "depth": depth,
                                })
                            except Exception:
                                pass
                        for link in result.links.get("internal", []):
                            href = _normalize_url(link.get("href", ""))
                            if href and href not in visited and is_internal(href):
                                next_level_urls.add(href)
                    else:
                        if progress_cb:
                            try:
                                progress_cb(
                                    {
                                        "type": "fetch",
                                        "url": getattr(result, "url", ""),
                                        "success": False,
                                        "status_code": _meta_attr(result, "status_code")
                                        or _meta_attr(result, "http_status")
                                        or _meta_attr(result, "status"),
                                        "error": _meta_attr(result, "error")
                                        or _meta_attr(result, "error_message")
                                        or _meta_attr(result, "message"),
                                        "content_source": "",
                                        "markdown_len": _len_attr(result, "markdown"),
                                        "text_len": _len_attr(result, "text"),
                                        "extracted_text_len": _len_attr(result, "extracted_text"),
                                        "cleaned_html_len": _len_attr(result, "cleaned_html"),
                                        "html_len": _len_attr(result, "html"),
                                        "raw_html_len": _len_attr(result, "raw_html"),
                                    }
                                )
                            except Exception:
                                pass
                current_urls = next_level_urls
    except asyncio.CancelledError:
        # If cancellation happens while entering/exiting crawler context, still return partials.
        print(f"[crawl] Cancelled during setup/teardown; returning {len(all_results)} crawled page(s) so far.")
        return all_results

    return all_results


async def discover_internal_urls(
    root_url: str,
    max_depth: int,
    max_concurrent: int,
    *,
    max_urls: int = 2000,
) -> List[str]:
    browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    parsed_root = urlparse(root_url)
    root_netloc = parsed_root.netloc
    visited = set()
    current_urls = set([_normalize_url(root_url)])
    discovered: List[str] = []

    def is_internal(url: str) -> bool:
        return urlparse(url).netloc == root_netloc

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for _depth in range(max_depth):
                urls_to_crawl = [u for u in current_urls if u not in visited]
                if not urls_to_crawl:
                    break
                if len(discovered) >= max_urls:
                    break
                results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                next_level_urls = set()
                for result in results:
                    norm = _normalize_url(result.url)
                    visited.add(norm)
                    if norm and norm not in discovered and is_internal(norm):
                        discovered.append(norm)
                        if len(discovered) >= max_urls:
                            break
                    for link in result.links.get("internal", []):
                        href = _normalize_url(link.get("href", ""))
                        if href and href not in visited and is_internal(href):
                            next_level_urls.add(href)
                current_urls = next_level_urls
    except Exception:
        return discovered

    return discovered


async def crawl_urls(
    urls: List[str],
    *,
    max_concurrent: int,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    if not urls:
        return []

    browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
    except Exception:
        return []

    docs: List[Dict[str, Any]] = []
    for result in results:
        norm = _normalize_url(result.url)
        if result.success:
            content, src = _best_text(result)
            if content:
                docs.append({"url": result.url, "markdown": f"Source URL: {result.url}\n\n{content}"})
            if progress_cb:
                try:
                    progress_cb(
                        {
                            "type": "fetch",
                            "url": result.url,
                            "success": True,
                            "status_code": _meta_attr(result, "status_code")
                            or _meta_attr(result, "http_status")
                            or _meta_attr(result, "status"),
                            "error": _meta_attr(result, "error")
                            or _meta_attr(result, "error_message")
                            or _meta_attr(result, "message"),
                            "content_source": src or "",
                            "markdown_len": _len_attr(result, "markdown"),
                            "text_len": _len_attr(result, "text"),
                            "extracted_text_len": _len_attr(result, "extracted_text"),
                            "cleaned_html_len": _len_attr(result, "cleaned_html"),
                            "html_len": _len_attr(result, "html"),
                            "raw_html_len": _len_attr(result, "raw_html"),
                        }
                    )
                except Exception:
                    pass
            if progress_cb:
                try:
                    progress_cb({"type": "page_crawled", "count": len(docs), "url": result.url, "depth": 0})
                except Exception:
                    pass
        else:
            if progress_cb:
                try:
                    progress_cb(
                        {
                            "type": "fetch",
                            "url": norm,
                            "success": False,
                            "status_code": _meta_attr(result, "status_code")
                            or _meta_attr(result, "http_status")
                            or _meta_attr(result, "status"),
                            "error": _meta_attr(result, "error")
                            or _meta_attr(result, "error_message")
                            or _meta_attr(result, "message"),
                            "content_source": "",
                            "markdown_len": _len_attr(result, "markdown"),
                            "text_len": _len_attr(result, "text"),
                            "extracted_text_len": _len_attr(result, "extracted_text"),
                            "cleaned_html_len": _len_attr(result, "cleaned_html"),
                            "html_len": _len_attr(result, "html"),
                            "raw_html_len": _len_attr(result, "raw_html"),
                        }
                    )
                except Exception:
                    pass

    return docs
