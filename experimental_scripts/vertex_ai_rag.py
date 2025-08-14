from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import os
import re
import asyncio
from datetime import datetime
import hashlib
from urllib.parse import urldefrag, urlparse
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from google.cloud import storage

# Create a RAG Corpus, Import Files, and Generate a response

# TODO(developer): Update and un-comment below lines
PROJECT_ID = "gen-lang-client-0545494042"
display_name = "test_corpus"
# paths = ["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]  # Supports Google Cloud Storage and Google Drive Links
# BUCKET_NAME = os.getenv("GCS_BUCKET", "YOUR_BUCKET_NAME")
# GCS_SUBPATH = os.getenv("GCS_SUBPATH", "raw_pages")

BUCKET_NAME = "web-assistant-test-bucket-1"
GCS_SUBPATH = "raw_pages"

# if BUCKET_NAME == "YOUR_BUCKET_NAME":
#     raise ValueError(
#         "Set GCS_BUCKET env var to your Google Cloud Storage bucket (e.g., 'my-bucket')."
#     )

paths = [f"gs://{BUCKET_NAME}/{GCS_SUBPATH}/"]

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create RagCorpus
# Configure embedding model, for example "text-embedding-005".
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    ),
)

# Import Files to the RagCorpus
rag.import_files(
    rag_corpus.name,
    paths,
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=1600,
            chunk_overlap=400,
        ),
    ),
    max_embedding_requests_per_min=1000,  # Optional
)

# ---- Crawl + Upload helpers ----
def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "index"

async def crawl_site_bfs(root_url: str, max_depth: int = 5, max_concurrent: int = 20) -> List[Dict[str, Any]]:
    browser_config = BrowserConfig(headless=True, verbose=False)
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
        for _depth in range(max_depth):
            urls_to_crawl = [u for u in current_urls if u not in visited]
            if not urls_to_crawl:
                break

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

def upload_markdown_docs_to_gcs(bucket_name: str, base_prefix: str, docs: List[Dict[str, Any]]) -> str:
    """Uploads docs as markdown files to GCS and returns the prefix used."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    # Organize under raw_pages/<netloc>/<timestamp>/
    first_url = docs[0]["url"] if docs else ""
    netloc = urlparse(first_url).netloc or "site"
    prefix = f"{base_prefix}/{_slugify(netloc)}/{timestamp}"

    for doc in docs:
        url = doc["url"]
        md = doc["markdown"]
        parsed = urlparse(url)
        path_slug = _slugify(parsed.path or "index")
        # Ensure uniqueness using a short hash of the full URL
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"{path_slug or 'index'}-{url_hash}.md"
        blob_name = f"{prefix}/{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(md, content_type="text/markdown")

    return prefix

def import_gcs_prefix_into_corpus(corpus_name: str, bucket_name: str, prefix: str) -> None:
    rag.import_files(
        corpus_name,
        [f"gs://{bucket_name}/{prefix}/"],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
        ),
        max_embedding_requests_per_min=1000,
    )

# Direct context retrieval
rag_retrieval_config = rag.RagRetrievalConfig(
    # top_k=3,  # Optional
    # filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
)
# response = rag.retrieval_query(
#     rag_resources=[
#         rag.RagResource(
#             rag_corpus=rag_corpus.name,
#             # Optional: supply IDs from `rag.list_files()`.
#             # rag_file_ids=["rag-file-1", "rag-file-2", ...],
#         )
#     ],
#     text="What is RAG and why it is helpful?",
#     rag_retrieval_config=rag_retrieval_config,
# )
# print(response)

def debug_retrieval(rag_corpus_name: str, query: str, top_k: int = 12):
    """Print the retrieved contexts safely (SDK-agnostic)."""
    cfg = rag.RagRetrievalConfig(top_k=top_k)  # keep it simple; no vector_distance_threshold
    resp = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=rag_corpus_name)],
        text=query,
        rag_retrieval_config=cfg,
    )

    # --- normalize contexts into a list ---
    ctxs = getattr(resp, "contexts", None)
    try:
        ctx_list = list(ctxs) if ctxs is not None else []
    except TypeError:
        # some versions wrap the list in a .contexts attribute
        inner = getattr(ctxs, "contexts", None)
        ctx_list = list(inner) if inner is not None else []

    # --- pretty demarcation for clarity ---
    print("\n" + "=" * 80)
    print("[RETRIEVED CHUNKS]")
    print(f"- query: {query!r}")
    print(f"- total: {len(ctx_list)}")
    print("=" * 80)

    def _ctx_text(c):
        # text can live at different spots depending on version
        t = getattr(c, "text", None)
        if t: return t
        chunk = getattr(c, "chunk", None)
        if chunk and getattr(chunk, "text", None):
            return chunk.text
        # last resort: repr
        return str(c)

    for i, c in enumerate(ctx_list, 1):
        txt = (_ctx_text(c) or "")[:800]
        print(f"\n---- Chunk {i}/{len(ctx_list)} ----\n{txt}")

    print("\n" + "=" * 80 + "\n")

    return ctx_list, resp


# --- Reuse-existing-crawl helpers ---

def _site_slug_from_url(url: str) -> str:
    return _slugify(urlparse(url).netloc or "site")

def list_existing_site_prefixes(bucket_name: str, base_prefix: str, site_url: str) -> list[str]:
    """
    Returns sorted list of GCS prefixes for previous crawls of this site.
    Format we look for: {base_prefix}/{site-slug}/{timestamp}/...
    e.g., raw_pages/playvalorant-com/20250813-153422/...
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    site_slug = _site_slug_from_url(site_url)
    site_root = f"{base_prefix}/{site_slug}/"

    # Collect unique first-level subfolders under site_root (timestamps)
    prefixes = set()
    for blob in bucket.list_blobs(prefix=site_root):
        # blob.name like: raw_pages/playvalorant-com/20250813-153422/file.md
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[0] == base_prefix and parts[1] == site_slug:
            ts = parts[2]
            if ts:  # "20250813-153422"
                prefixes.add(f"{base_prefix}/{site_slug}/{ts}")

    # Sort by timestamp if possible
    def _ts_key(pref: str) -> tuple:
        # pref ends with ".../{timestamp}"
        ts = pref.rstrip("/").split("/")[-1]
        # try to parse YYYYMMDD-HHMMSS
        try:
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
            return (dt, ts)
        except Exception:
            return (datetime.min, ts)

    return sorted(prefixes, key=_ts_key)

def choose_prefix_interactively(prefixes: list[str]) -> str | None:
    """
    Small interactive picker. Returns chosen prefix or None if user aborts.
    """
    if not prefixes:
        return None
    print("\nFound previous crawls:\n")
    for i, p in enumerate(prefixes, 1):
        print(f"  {i}. gs://{BUCKET_NAME}/{p}/")
    print("  0. Cancel")
    while True:
        choice = input("Pick a number (Enter=latest): ").strip()
        if choice == "":
            return prefixes[-1]  # latest
        if choice == "0":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(prefixes):
                return prefixes[idx - 1]
        print("Invalid choice. Try again.")



# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

# Create a Gemini model instance
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)

# Generate responses interactively from terminal
print("Interactive RAG console: type 'exit', 'quit' or 'q' to leave.")
print("Tip: enter a URL (e.g., https://example.com) to crawl the site, upload to GCS, and index into the current corpus.")
while True:
    try:
        user_query = input("Enter your prompt or URL > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not user_query:
        continue

    if user_query.lower() in ("exit", "quit", "q"):
        print("Goodbye.")
        break

    # If input looks like a URL, perform crawl → upload → import
    if user_query.lower().startswith("http://") or user_query.lower().startswith("https://"):
        try:
            # 1) Check for existing crawls in GCS
            existing = list_existing_site_prefixes(BUCKET_NAME, GCS_SUBPATH, user_query)
            if existing:
                latest = existing[-1]
                ans = input(
                    f"Found {len(existing)} previous crawls for this site.\n"
                    f"Reuse latest prefix gs://{BUCKET_NAME}/{latest}/ ? [Y/n/i=list] "
                ).strip().lower()

                if ans in ("", "y", "yes"):
                    print(f"Reusing {latest} ... importing into corpus.")
                    import_gcs_prefix_into_corpus(rag_corpus.name, BUCKET_NAME, latest)
                    print("Import complete. You can now query against the previously indexed pages.")
                    continue
                elif ans in ("i", "list", "l"):
                    chosen = choose_prefix_interactively(existing)
                    if chosen:
                        print(f"Reusing {chosen} ... importing into corpus.")
                        import_gcs_prefix_into_corpus(rag_corpus.name, BUCKET_NAME, chosen)
                        print("Import complete. You can now query against the selected crawl.")
                        continue
                    else:
                        print("Cancelled reuse; proceeding to fresh crawl.")

            # 2) No reuse (or user chose fresh crawl) → crawl anew
            print(f"Crawling site (BFS): {user_query}")
            crawled_docs = asyncio.run(crawl_site_bfs(user_query, max_depth=8, max_concurrent=25))
            if not crawled_docs:
                print("No pages crawled.")
                continue

            print(f"Crawled {len(crawled_docs)} pages. Uploading to gs://{BUCKET_NAME}/{GCS_SUBPATH}/ ...")
            gcs_prefix = upload_markdown_docs_to_gcs(BUCKET_NAME, GCS_SUBPATH, crawled_docs)
            print(f"Uploaded to gs://{BUCKET_NAME}/{gcs_prefix}/. Importing into Vertex RAG corpus...")
            import_gcs_prefix_into_corpus(rag_corpus.name, BUCKET_NAME, gcs_prefix)
            print("Import complete. You can now query against the newly indexed pages.")
        except Exception as e:
            print(f"Error during crawl/upload/import: {e}")
        continue


    # Otherwise, treat as a normal prompt to the RAG-enabled model
    try:
        # 2a) Inspect what RAG brings back
        _ = debug_retrieval(rag_corpus.name, user_query, top_k=12)

        # 2b) Now generate, grounded on those contexts
        response = rag_model.generate_content(user_query)
        print("\n" + "=" * 80)
        print("[FINAL ANSWER]")
        print("=" * 80)
        print(response.text)
        print("\n" + "=" * 80 + "\n")
    except Exception as e:
        print(f"Error generating response: {e}")
# Example response:
#   RAG stands for Retrieval-Augmented Generation.
#   It's a technique used in AI to enhance the quality of responses
# ...