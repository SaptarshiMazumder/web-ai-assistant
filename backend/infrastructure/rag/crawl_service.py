import asyncio
import json
import logging
import os
import re
import hashlib
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from urllib.parse import urlparse, urldefrag, urljoin, urlunparse

logger = logging.getLogger(__name__)

from google.cloud import storage
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# --- Vertex AI RAG
from vertexai import rag as vx_rag
import vertexai

from common.config import config
from infrastructure.rag.url_map import URL_MAP_FILENAME

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

# Wait for body to have some visible text (fast on server-rendered pages, ~1–3s on JS-heavy).
# No fixed per-page delay; capture as soon as content appears or page_timeout.
CRAWL_WAIT_FOR_CONTENT = "js:() => document.body && document.body.innerText.trim().length > 80"

# =========================
# ---- UTILITIES ----------
# =========================
def _normalize_url(url: str) -> str:
    """Normalize URL: strip fragment, and collapse path '' and '/' so example.com and example.com/ are the same."""
    url = urldefrag(url)[0]
    try:
        p = urlparse(url)
        path = (p.path or "/").strip() or "/"
        if path == "/":
            # Canonical form for root: scheme://netloc/ so example.com and example.com/ dedupe
            return urlunparse((p.scheme, p.netloc, "/", "", p.query, ""))
    except Exception:
        pass
    return url


# Extensions that are almost certainly not HTML content pages.
_NON_PAGE_EXTENSIONS = {
    # images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".bmp",
    ".tiff",
    ".avif",
    # styles/scripts
    ".css",
    ".js",
    ".mjs",
    ".map",
    # fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    # media
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".webm",
    ".mov",
    ".m4a",
    # docs/archives
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".rar",
    ".7z",
    ".tar",
    ".dmg",
    ".exe",
    # data
    ".json",
    ".xml",
    ".rss",
}


def _is_url_under_root_path(url: str, root_url: str) -> bool:
    """
    True if url has same domain as root_url AND path is under root_url's path.
    E.g. root https://example.com/hotel/tokyoshiodome/ → allows /hotel/tokyoshiodome/* only.
    If root path is / (domain root), allows all paths on that domain.
    """
    try:
        pu = urlparse(url)
        pr = urlparse(root_url)
    except Exception:
        return False
    if pu.netloc != pr.netloc:
        return False
    root_path = (pr.path or "/").rstrip("/") or "/"
    url_path = (pu.path or "/").rstrip("/") or "/"
    if root_path == "/":
        return True
    return url_path == root_path or url_path.startswith(root_path + "/")


def _is_probably_page_url(url: str, *, root_netloc: str) -> bool:
    """
    Heuristic filter to keep discovery focused on content pages.
    We intentionally skip obvious static assets (png/js/css/etc.).
    """
    try:
        p = urlparse(url)
    except Exception:
        return False

    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc or p.netloc != root_netloc:
        return False

    path = (p.path or "/").strip()
    if not path:
        path = "/"

    # Strip trailing slash for extension checks.
    path_no_slash = path[:-1] if path.endswith("/") and path != "/" else path
    lower = path_no_slash.lower()

    # If the last path segment ends with a known non-page extension, skip it.
    for ext in _NON_PAGE_EXTENSIONS:
        if lower.endswith(ext):
            return False

    return True


# Regexes for fallback link extraction from HTML/markdown (so we never miss URLs).
_RE_HREF = re.compile(
    r'\bhref\s*=\s*["\']([^"\']+)["\']|\bhref\s*=\s*([^\s>"\']+)',
    re.IGNORECASE,
)
_RE_MD_LINK = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')


def _extract_urls_from_content(
    content: str,
    base_url: str,
    root_netloc: str,
) -> List[str]:
    """
    Extract same-domain page URLs from HTML or markdown when crawl4ai's links are empty.
    Ensures discovery never misses links that are present in the page content.
    """
    if not content or not isinstance(content, str):
        return []
    seen: set = set()
    out: List[str] = []
    # href="..." or href='...' or href=...
    for m in _RE_HREF.finditer(content):
        raw = (m.group(1) or m.group(2) or "").strip()
        if not raw or raw.startswith("#") or raw.startswith("javascript:"):
            continue
        try:
            full = urljoin(base_url, raw)
            norm = _normalize_url(full)
            if not norm or norm in seen:
                continue
            if urlparse(norm).netloc != root_netloc:
                continue
            if not _is_probably_page_url(norm, root_netloc=root_netloc):
                continue
            seen.add(norm)
            out.append(norm)
        except Exception:
            continue
    # Markdown [text](url)
    for m in _RE_MD_LINK.finditer(content):
        raw = (m.group(2) or "").strip()
        if not raw or raw.startswith("#") or raw.startswith("javascript:"):
            continue
        try:
            full = urljoin(base_url, raw)
            norm = _normalize_url(full)
            if not norm or norm in seen:
                continue
            if urlparse(norm).netloc != root_netloc:
                continue
            if not _is_probably_page_url(norm, root_netloc=root_netloc):
                continue
            seen.add(norm)
            out.append(norm)
        except Exception:
            continue
    return out


# ---- Discovery reliability (without freezing): bounded retries/backoff ----
# Discovery runs in the request path (and can stream). Long backoffs (e.g. 60s waits)
# make the UX unusable. We keep retries, but cap the total time per URL.
_DISCOVERY_MAX_RETRIES = 4
_DISCOVERY_RETRY_BACKOFF_BASE = 0.35  # ~0.35, 0.7, 1.4, 2.8 seconds
_DISCOVERY_RETRY_BACKOFF_CAP_S = 2.0
_DISCOVERY_PER_URL_BUDGET_S = 6.0


def _is_successful_result(result: Any) -> bool:
    """True if we got a real response (not a failed/placeholder result)."""
    if result is None:
        return False
    return getattr(result, "success", True) is not False


def _result_has_links(result: Any) -> bool:
    """True if result has at least one internal or external link (for discovery)."""
    if result is None:
        return False
    links = getattr(result, "links", None) or {}
    if not isinstance(links, dict):
        return False
    internal = links.get("internal") or []
    external = links.get("external") or []
    return len(internal) > 0 or len(external) > 0


class _FailedDiscoveryResult:
    """Placeholder for a URL we could not fetch after retries. Has .url and .links = {}."""

    def __init__(self, url: str):
        self.url = url
        self.success = False
        self.links = {}


async def _fetch_single_url_with_retries(
    crawler: Any,
    url: str,
    run_config: Any,
    *,
    max_retries: int = _DISCOVERY_MAX_RETRIES,
) -> Any:
    """Fetch one URL with exponential backoff. Returns result or _FailedDiscoveryResult."""
    start = time.monotonic()
    last_error = None
    for attempt in range(max_retries):
        try:
            result = await crawler.arun(url=url, config=run_config)
            if result:
                return result
        except Exception as e:
            last_error = e
            msg = str(e)
            # Fail fast on connection/navigation errors - each attempt can pause for many seconds,
            # so retrying would freeze the flow. Prefer moving on over 2–3 long waits per URL.
            if "ERR_CONNECTION_CLOSED" in msg or "Failed on navigating" in msg or "ACS-GOTO" in msg:
                break
            if time.monotonic() - start >= _DISCOVERY_PER_URL_BUDGET_S:
                break
            if attempt < max_retries - 1:
                wait = min(_DISCOVERY_RETRY_BACKOFF_CAP_S, _DISCOVERY_RETRY_BACKOFF_BASE * (2 ** attempt))
                # Don't sleep past remaining budget.
                remaining = _DISCOVERY_PER_URL_BUDGET_S - (time.monotonic() - start)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(wait, max(0.0, remaining)))
    return _FailedDiscoveryResult(url)


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
    return _normalize_text_encoding(combined), used_src or primary_src or ""


def _quality_score(text: str) -> float:
    if not text:
        return 0.0
    total = max(1, len(text))
    printable = sum(1 for ch in text if ch.isprintable())
    replacement = text.count("\ufffd")
    # Favor printable text and penalize replacement chars.
    return (printable / total) - (replacement * 0.01)


def _normalize_text_encoding(text: str) -> str:
    """Best-effort cleanup using charset detection (no language-specific rules)."""
    if not text:
        return text
    fixed = text
    try:
        import ftfy  # type: ignore
        fixed = ftfy.fix_text(fixed, normalization="NFC")
    except Exception:
        fixed = text
    # Skip re-encoding for text that contains non-Latin-1 characters (e.g., Japanese)
    if any(ord(c) > 0xFF for c in fixed):
        return fixed
    try:
        from charset_normalizer import from_bytes  # type: ignore
        # Treat the current string as a byte sequence that may have been
        # decoded with the wrong codec. This is language-agnostic.
        raw_bytes = fixed.encode("latin-1", errors="ignore")
        if raw_bytes:
            best = from_bytes(raw_bytes).best()
            if best and best.encoding:
                repaired = str(best)
                if _quality_score(repaired) > _quality_score(fixed):
                    fixed = repaired
    except Exception:
        pass
    return fixed

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

    url_map: Dict[str, str] = {}  # filename -> page URL for citation resolution at retrieval
    for doc in docs:
        url = doc["url"]
        md = doc["markdown"]
        parsed = urlparse(url)
        path_slug = _slugify(parsed.path or "index")
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"{path_slug or 'index'}-{url_hash}.md"
        url_map[filename] = url
        blob_name = f"{prefix}/{filename}"
        blob = bucket.blob(blob_name)
        md_bytes = md.encode("utf-8") if isinstance(md, str) else md
        blob.upload_from_string(md_bytes, content_type="text/markdown; charset=utf-8")

    map_blob = bucket.blob(f"{prefix}/{URL_MAP_FILENAME}")
    map_blob.upload_from_string(
        json.dumps(url_map, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
    )
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
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_for=CRAWL_WAIT_FOR_CONTENT,
    )
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
        return _is_url_under_root_path(url, root_url)

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
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_for=CRAWL_WAIT_FOR_CONTENT,
    )
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
    failed_urls: set = set()
    max_depth_reached = -1

    def is_internal(url: str) -> bool:
        return _is_url_under_root_path(url, root_url)

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for _depth in range(max_depth):
                urls_to_crawl = sorted(
                    u
                    for u in current_urls
                    if u not in visited and _is_probably_page_url(u, root_netloc=root_netloc)
                )
                if not urls_to_crawl:
                    # No more URLs at this depth, but continue to next depth if we have discovered URLs
                    # This handles cases where some pages don't have links but others do
                    if len(discovered) > 0:
                        continue
                    break
                if len(discovered) >= max_urls:
                    break
                
                # Try batch crawl with retry and fallback (5 retries for reliability)
                results = None
                for retry_attempt in range(5):
                    try:
                        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
                        break
                    except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError) as e:
                        if retry_attempt < 4:
                            await asyncio.sleep(1.5 * (2 ** retry_attempt))
                        else:
                            from infrastructure.repositories.crawl4ai_crawler_repository import _crawl_urls_individually
                            results = await _crawl_urls_individually(crawler, urls_to_crawl, run_config)
                            break
                    except Exception as e:
                        from infrastructure.repositories.crawl4ai_crawler_repository import _crawl_urls_individually
                        results = await _crawl_urls_individually(crawler, urls_to_crawl, run_config)
                        break
                
                if not results:
                    # If all crawling failed, continue to next depth level instead of breaking
                    current_urls = set()
                    continue
                # Ensure every requested URL has a result: retry missing and failed until success or max retries
                result_by_norm = {_normalize_url(getattr(r, "url", None) or ""): r for r in results}
                missing = [u for u in urls_to_crawl if _normalize_url(u) not in result_by_norm]
                for u in sorted(missing):
                    r = await _fetch_single_url_with_retries(crawler, u, run_config)
                    result_by_norm[_normalize_url(u)] = r
                failed = [u for u in urls_to_crawl if not _is_successful_result(result_by_norm.get(_normalize_url(u)))]
                for u in sorted(failed):
                    r = await _fetch_single_url_with_retries(crawler, u, run_config)
                    if _is_successful_result(r):
                        result_by_norm[_normalize_url(u)] = r
                results = [result_by_norm[_normalize_url(u)] for u in urls_to_crawl]
                results = sorted(results, key=lambda r: (getattr(r, "url", None) or ""))
                max_depth_reached = _depth
                for r in results:
                    if not _is_successful_result(r):
                        u = getattr(r, "url", None) or ""
                        if u:
                            failed_urls.add(u)
                level_norms: set = set()
                next_level_urls = set()
                for result in results:
                    try:
                        result_url = getattr(result, "url", None) or ""
                        norm = _normalize_url(result_url) if result_url else ""

                        if norm:
                            visited.add(norm)
                            if is_internal(norm) and _is_probably_page_url(norm, root_netloc=root_netloc):
                                level_norms.add(norm)

                        links = getattr(result, "links", None) or {}
                        for link in links.get("internal", []):
                            try:
                                href = _normalize_url(link.get("href", ""))
                                if (
                                    href
                                    and href not in visited
                                    and is_internal(href)
                                    and _is_probably_page_url(href, root_netloc=root_netloc)
                                ):
                                    next_level_urls.add(href)
                            except Exception:
                                continue
                        for link in links.get("external", []):
                            try:
                                href = _normalize_url(link.get("href", ""))
                                if (
                                    href
                                    and is_internal(href)
                                    and href not in visited
                                    and _is_probably_page_url(href, root_netloc=root_netloc)
                                ):
                                    next_level_urls.add(href)
                            except Exception:
                                continue
                        page_url = getattr(result, "url", None) or norm or root_url
                        for content_attr in ("markdown", "html", "raw_html", "cleaned_html", "content"):
                            raw = getattr(result, content_attr, None)
                            if not raw or not isinstance(raw, str):
                                continue
                            for href in _extract_urls_from_content(raw, page_url, root_netloc):
                                if href not in visited and is_internal(href):
                                    next_level_urls.add(href)
                    except Exception:
                        continue

                candidates = set(discovered) | level_norms | next_level_urls
                discovered = sorted(candidates)[:max_urls]
                current_urls = set(discovered) - visited
    except Exception as e:
        logger.warning(
            "Discovery error in crawl_service: %s: %s, returning %s URLs",
            type(e).__name__,
            str(e)[:100],
            len(discovered),
        )
        pass

    # Always return at least the root URL if we have nothing else
    if not discovered:
        root_norm = _normalize_url(root_url)
        if root_norm:
            discovered.append(root_norm)
    # Log summary for ops (depths + failed URLs); not exposed to UI
    logger.info(
        "URL discovery completed: depths=%s, discovered=%s, failed=%s. Failed URLs: %s",
        max_depth_reached + 1,
        len(discovered),
        len(failed_urls),
        sorted(failed_urls),
    )
    # Return in deterministic order so counts are stable across runs
    return sorted(discovered)


async def discover_internal_urls_stream(
    root_url: str,
    max_depth: int,
    max_concurrent: int,
    *,
    max_urls: int = 2000,
    max_duration_sec: Optional[int] = None,
):
    """
    Async generator version of discover_internal_urls().

    Yields NDJSON-friendly dict events:
      - {"type":"start","root_url":...}
      - {"type":"batch","depth":...,"queued":...}
      - {"type":"discovered","url":...,"count":...,"depth":...}
      - {"type":"error","message":...}
      - {"type":"done","urls":[...],"timed_out":bool}
    """
    browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_for=CRAWL_WAIT_FOR_CONTENT,
    )
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
    failed_urls: set = set()
    max_depth_reached = -1
    yielded_global: set = set()

    def is_internal(url: str) -> bool:
        return _is_url_under_root_path(url, root_url)

    yield {"type": "start", "root_url": root_url, "max_depth": max_depth, "max_urls": max_urls}

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(max_depth):
                urls_to_crawl = sorted(
                    u
                    for u in current_urls
                    if u not in visited and _is_probably_page_url(u, root_netloc=root_netloc)
                )
                if not urls_to_crawl:
                    break
                if len(discovered) >= max_urls:
                    break

                # Chunk batches so the client receives frequent updates (closer to crawl logs).
                batch_size = 8
                yield {
                    "type": "batch",
                    "depth": depth,
                    "queued": len(urls_to_crawl),
                    "discovered": len(discovered),
                    "batch_size": batch_size,
                }

                level_norms_stream: set = set()
                next_level_urls = set()
                for start in range(0, len(urls_to_crawl), batch_size):
                    chunk = urls_to_crawl[start : start + batch_size]
                    if not chunk:
                        continue

                    yield {"type": "chunk", "depth": depth, "from": start, "size": len(chunk)}

                    results = None
                    for retry_attempt in range(5):
                        try:
                            results = await crawler.arun_many(urls=chunk, config=run_config, dispatcher=dispatcher)
                            break
                        except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError):
                            if retry_attempt < 4:
                                await asyncio.sleep(1.5 * (2 ** retry_attempt))
                            else:
                                from infrastructure.repositories.crawl4ai_crawler_repository import _crawl_urls_individually
                                results = await _crawl_urls_individually(crawler, chunk, run_config)
                                break
                        except Exception:
                            from infrastructure.repositories.crawl4ai_crawler_repository import _crawl_urls_individually
                            results = await _crawl_urls_individually(crawler, chunk, run_config)
                            break

                    if not results:
                        yield {"type": "error", "message": f"Failed to crawl chunk at depth {depth}. Continuing."}
                        continue
                    result_by_norm = {_normalize_url(getattr(r, "url", None) or ""): r for r in results}
                    missing = [u for u in chunk if _normalize_url(u) not in result_by_norm]
                    for u in sorted(missing):
                        r = await _fetch_single_url_with_retries(crawler, u, run_config)
                        result_by_norm[_normalize_url(u)] = r
                    failed = [u for u in chunk if not _is_successful_result(result_by_norm.get(_normalize_url(u)))]
                    for u in sorted(failed):
                        r = await _fetch_single_url_with_retries(crawler, u, run_config)
                        if _is_successful_result(r):
                            result_by_norm[_normalize_url(u)] = r
                    results = [result_by_norm[_normalize_url(u)] for u in chunk]
                    results = sorted(results, key=lambda r: (getattr(r, "url", None) or ""))
                    max_depth_reached = depth
                    for r in results:
                        if not _is_successful_result(r):
                            u = getattr(r, "url", None) or ""
                            if u:
                                failed_urls.add(u)
                    for result in results:
                        try:
                            result_url = getattr(result, "url", None) or ""
                            norm = _normalize_url(result_url) if result_url else ""

                            if norm:
                                visited.add(norm)
                                if is_internal(norm) and _is_probably_page_url(norm, root_netloc=root_netloc):
                                    level_norms_stream.add(norm)

                            links = getattr(result, "links", None) or {}
                            for link in links.get("internal", []):
                                try:
                                    href = _normalize_url(link.get("href", ""))
                                    if (
                                        href
                                        and href not in visited
                                        and is_internal(href)
                                        and _is_probably_page_url(href, root_netloc=root_netloc)
                                    ):
                                        next_level_urls.add(href)
                                except Exception:
                                    continue
                            for link in links.get("external", []):
                                try:
                                    href = _normalize_url(link.get("href", ""))
                                    if (
                                        href
                                        and is_internal(href)
                                        and href not in visited
                                        and _is_probably_page_url(href, root_netloc=root_netloc)
                                    ):
                                        next_level_urls.add(href)
                                except Exception:
                                    continue
                            page_url = getattr(result, "url", None) or norm or root_url
                            for content_attr in ("markdown", "html", "raw_html", "cleaned_html", "content"):
                                raw = getattr(result, content_attr, None)
                                if not raw or not isinstance(raw, str):
                                    continue
                                for href in _extract_urls_from_content(raw, page_url, root_netloc):
                                    if href not in visited and is_internal(href):
                                        next_level_urls.add(href)
                        except Exception:
                            continue

                candidates = set(discovered) | level_norms_stream | next_level_urls
                discovered = sorted(candidates)[:max_urls]
                for u in sorted(set(discovered) - yielded_global):
                    yielded_global.add(u)
                    yield {"type": "discovered", "url": u, "count": len(discovered), "depth": depth}
                current_urls = set(discovered) - visited
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {str(e)}"}

    if not discovered:
        root_norm = _normalize_url(root_url)
        if root_norm:
            discovered.append(root_norm)
            yield {"type": "discovered", "url": root_norm, "count": len(discovered), "depth": 0}
    # Log summary for ops (depths + failed URLs); not exposed to UI
    logger.info(
        "URL discovery (stream) completed: depths=%s, discovered=%s, failed=%s. Failed URLs: %s",
        max_depth_reached + 1,
        len(discovered),
        len(failed_urls),
        sorted(failed_urls),
    )
    # Return in deterministic order so counts are stable across runs
    discovered = sorted(discovered)
    yield {"type": "done", "urls": discovered, "timed_out": False}


async def crawl_urls(
    urls: List[str],
    *,
    max_concurrent: int,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    if not urls:
        return []

    browser_config = BrowserConfig(headless=HEADLESS, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_for=CRAWL_WAIT_FOR_CONTENT,
    )
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
