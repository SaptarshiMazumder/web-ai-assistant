# save as: crawl_site_to_gcs.py
# Usage: python crawl_site_to_gcs.py
# Type a website (e.g., "tmobile.com" or "https://www.t-mobile.com") or "exit"
# Behavior:
#   - Crawls all internal pages with crawl4ai
#   - Uploads markdown to GCS under raw_pages/<site>/<timestamp>/
#   - Reuses a previous crawl if you choose it
#   - Imports the (reused or freshly crawled) GCS prefix into a Vertex RAG corpus
#   - Auto-creates the RAG corpus if missing

import asyncio
import os
import re
import hashlib
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, urldefrag

from google.cloud import storage
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# --- Vertex AI RAG
from vertexai import rag as vx_rag
import vertexai

# =========================
# ---- CONFIG (edit) ------
# =========================
PROJECT_ID = "gen-lang-client-0545494042"
VERTEX_LOCATION = "us-central1"  # RAG lives in a regional Vertex location

# If you already have a corpus, put its full resource name here.
# Else leave empty ("") and the script will create one and print its name.
RAG_CORPUS = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"


# Embedding model used by the RAG index
EMBEDDING_PUBLISHER_MODEL = "publishers/google/models/text-embedding-005"

BUCKET_NAME = "web-assistant-test-bucket-1"
GCS_SUBPATH = "raw_pages"              # final prefix: raw_pages/<site>/<timestamp>/
CRAWL_MAX_DEPTH = 8
CRAWL_MAX_CONCURRENCY = 25
HEADLESS = True

# RAG import (chunking) config
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# =========================
# ---- UTILITIES ----------
# =========================
def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]

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
    return _slugify(urlparse(url).netloc or "site")

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
def upload_markdown_docs_to_gcs(bucket_name: str, base_prefix: str, docs: List[Dict[str, Any]]) -> str:
    """Uploads docs as markdown files to GCS and returns the prefix used (without trailing slash)."""
    if not docs:
        raise ValueError("No docs to upload to GCS.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    first_url = docs[0]["url"]
    netloc = urlparse(first_url).netloc or "site"
    prefix = f"{base_prefix}/{_slugify(netloc)}/{timestamp}"

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

    return prefix  # e.g., raw_pages/example-com/20250814-010203

def list_existing_site_prefixes(bucket_name: str, base_prefix: str, site_url: str) -> List[str]:
    """
    Returns sorted list of GCS prefixes for previous crawls of this site.
    Format: {base_prefix}/{site-slug}/{timestamp}
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    site_slug = _site_slug_from_url(site_url)
    site_root = f"{base_prefix}/{site_slug}/"

    prefixes = set()
    for blob in bucket.list_blobs(prefix=site_root):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[0] == base_prefix and parts[1] == site_slug:
            ts = parts[2]
            if ts:
                prefixes.add(f"{base_prefix}/{site_slug}/{ts}")

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
async def crawl_site_bfs(root_url: str, max_depth: int, max_concurrent: int) -> List[Dict[str, Any]]:
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

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [u for u in current_urls if u not in visited]
            if not urls_to_crawl:
                break

            print(f"[Depth {depth}] Crawling {len(urls_to_crawl)} page(s)...")
            results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
            next_level_urls = set()

            for result in results:
                norm = _normalize_url(result.url)
                visited.add(norm)
                if result.success and result.markdown:
                    all_results.append({"url": result.url, "markdown": result.markdown})
                    for link in result.links.get("internal", []):
                        href = _normalize_url(link.get("href", ""))
                        if href and href not in visited and is_internal(href):
                            next_level_urls.add(href)
            current_urls = next_level_urls

    return all_results

# =========================
# ---- MAIN LOOP ----------
# =========================
def main():
    corpus_name = get_or_create_corpus(PROJECT_ID, VERTEX_LOCATION, RAG_CORPUS)
    print(f"[RAG] Using corpus: {corpus_name}\n")

    print("Crawl → GCS → RAG importer")
    print("Type a website or URL to crawl (e.g., 'tmobile.com' or 'https://www.t-mobile.com').")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Site or URL > ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        url = _ensure_url(user_input)

        try:
            # --- Timing start ---
            start_wall_utc = datetime.utcnow()
            start_perf = time.perf_counter()
            # Reuse?
            existing = list_existing_site_prefixes(BUCKET_NAME, GCS_SUBPATH, url)
            if existing:
                latest = existing[-1]
                ans = input(
                    f"Found {len(existing)} previous crawls.\n"
                    f"Reuse latest gs://{BUCKET_NAME}/{latest}/ ? [Y/n/i=list] "
                ).strip().lower()

                chosen = None
                if ans in ("", "y", "yes"):
                    chosen = latest
                elif ans in ("i", "list", "l"):
                    chosen = choose_prefix_interactively(existing)

                if chosen:
                    print(f"Reusing: gs://{BUCKET_NAME}/{chosen}/")
                    import_gcs_prefix_into_corpus(corpus_name, BUCKET_NAME, chosen)
                    # --- Timing end (reuse path) ---
                    end_wall_utc = datetime.utcnow()
                    elapsed_s = time.perf_counter() - start_perf
                    print("Done.\n")
                    print(f"[TIMING] Start:   {start_wall_utc.isoformat()}Z")
                    print(f"[TIMING] End:     {end_wall_utc.isoformat()}Z")
                    print(f"[TIMING] Elapsed: {elapsed_s:.2f} seconds\n")
                    continue
                else:
                    print("Proceeding to fresh crawl.\n")

            # Fresh crawl
            print(f"Crawling site (BFS): {url}")
            docs = asyncio.run(
                crawl_site_bfs(url, max_depth=CRAWL_MAX_DEPTH, max_concurrent=CRAWL_MAX_CONCURRENCY)
            )
            if not docs:
                print("No pages crawled.\n")
                continue

            crawl_done_perf = time.perf_counter()
            print(f"Crawled {len(docs)} pages. Uploading to gs://{BUCKET_NAME}/{GCS_SUBPATH}/ ...")
            gcs_prefix = upload_markdown_docs_to_gcs(BUCKET_NAME, GCS_SUBPATH, docs)
            upload_done_perf = time.perf_counter()
            print(f"Uploaded to: gs://{BUCKET_NAME}/{gcs_prefix}/")

            # Import into RAG corpus
            print("[RAG] Importing uploaded pages into corpus...")
            import_gcs_prefix_into_corpus(corpus_name, BUCKET_NAME, gcs_prefix)
            # --- Timing end (fresh crawl path) ---
            end_wall_utc = datetime.utcnow()
            import_done_perf = time.perf_counter()
            elapsed_s = import_done_perf - start_perf
            crawl_s = crawl_done_perf - start_perf
            upload_s = upload_done_perf - crawl_done_perf
            import_s = import_done_perf - upload_done_perf
            print("Import complete.\n")
            print(f"[TIMING] Start:   {start_wall_utc.isoformat()}Z")
            print(f"[TIMING] End:     {end_wall_utc.isoformat()}Z")
            print(f"[TIMING] Elapsed: {elapsed_s:.2f} seconds\n")
            print("[TIMING] Breakdown:")
            print(f"  - Crawl:  {crawl_s:.2f} s")
            print(f"  - Upload: {upload_s:.2f} s")
            print(f"  - Import: {import_s:.2f} s\n")

        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
