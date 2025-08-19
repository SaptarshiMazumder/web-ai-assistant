from fastapi import APIRouter, Query
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any
from state import WebsiteMultiHopState
from models import WebAssistantRequest, WebsiteRagRequest
from vertex_rag_eg import run_vertex_rag
import time
from logging_relay import log, smartqa_log_relay
from config import config
import os
from use_cases.qa_usecase import ask_website_use_case, ask_google_use_case
from urllib.parse import urlparse
from crawl_to_gcs import list_existing_site_prefixes, crawl_site_bfs, upload_markdown_docs_to_gcs, CRAWL_MAX_DEPTH, CRAWL_MAX_CONCURRENCY
from typing import Optional

# --- Smart Hop QA API ---
smart_qa_router = APIRouter()

@smart_qa_router.post("/ask-smart")
async def ask_smart(request: WebAssistantRequest):
    return await ask_website_use_case(request)

@smart_qa_router.post("/ask-gemini")
async def ask_gemini(request: WebAssistantRequest):
    return await ask_google_use_case(request)

# --- Website RAG (placeholder) ---
@smart_qa_router.post("/ask-website-rag")
async def ask_website_rag(request: WebsiteRagRequest):
    # Derive domain if not provided
    domain = request.domain
    try:
        if not domain and request.page_url:
            from urllib.parse import urlparse
            domain = urlparse(request.page_url).hostname
    except Exception:
        pass
    print(f"[Website RAG] domain: {domain}")
    # Call Vertex RAG pipeline using the user's question
    result = run_vertex_rag(request.question)
    return result

@smart_qa_router.get("/is-indexed")
async def is_indexed(url: str = Query(..., description="Full page URL to check")):
    """Heuristic check: consider a site indexed if we have any raw page files for its hostname.

    Returns: { indexed: bool, host: str, source: "local_raw_pages" }
    """
    host = ""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = ""
    # Fallback: if caller passed a bare domain
    if not host and url:
        host = url.strip()
    # Normalize and prepare host variants once
    def host_variants(h: str):
        if not h:
            return []
        base = h.split(":")[0].lower()
        nowww = base[4:] if base.startswith("www.") else base
        parts = nowww.split(".")
        apex = nowww if len(parts) < 2 else ".".join(parts[-2:])
        variants = {base, nowww, apex, f"www.{apex}"}
        return [v for v in variants if v]

    variants = host_variants(host)
    try:
        print(f"[is-indexed] url={url}")
        print(f"[is-indexed] host={host}")
        print(f"[is-indexed] variants={variants}")
    except Exception:
        pass

    # 1) Try GCS best-effort; if it fails or finds nothing, silently fall back to local raw_pages
    try:
        bucket_and_prefix = (config.GCS_BUCKET or '').strip('/').split('/', 1)
        if len(bucket_and_prefix) == 2:
            bucket_name, base_prefix = bucket_and_prefix[0], bucket_and_prefix[1]
        else:
            bucket_name, base_prefix = bucket_and_prefix[0], ''
        print(f"[is-indexed] GCS bucket={bucket_name} base_prefix={base_prefix}")
        for hv in variants:
            try:
                prefixes = list_existing_site_prefixes(
                    bucket_name=bucket_name,
                    base_prefix=base_prefix,
                    site_url=f"https://{hv}",
                )
                print(f"[is-indexed] GCS check hv={hv} -> {len(prefixes)} prefix(es)")
                if prefixes:
                    print(f"[is-indexed] DECISION: indexed via GCS for hv={hv}")
                    return {"indexed": True, "host": host, "source": "gcs", "debug": {"hv": hv, "prefix_count": len(prefixes)}}
            except Exception as e:
                print(f"[is-indexed] GCS check error for hv={hv}: {e}")
    except Exception:
        pass

    # 2) Inspect local raw_pages directory (robust host matching: apex, www, subdomains)
    try:
        raw_dir = os.path.join(os.path.dirname(__file__), "raw_pages")
        print(f"[is-indexed] local raw_dir={raw_dir} exists={os.path.isdir(raw_dir)}")
        if os.path.isdir(raw_dir) and variants:
            for fname in os.listdir(raw_dir):
                file_host = fname.split("_", 1)[0].lower()
                for hv in variants:
                    if (
                        file_host == hv
                        or file_host.endswith("." + hv)
                        or hv.endswith("." + file_host)
                        or hv in file_host
                    ):
                        print(f"[is-indexed] DECISION: indexed via local file match file_host={file_host} hv={hv} fname={fname}")
                        return {"indexed": True, "host": host, "source": "local_raw_pages", "debug": {"file_host": file_host, "hv": hv, "fname": fname}}
        print(f"[is-indexed] DECISION: not indexed")
        return {"indexed": False, "host": host, "source": "local_raw_pages"}
    except Exception:
        return {"indexed": False, "host": host, "source": "local_raw_pages"}


@smart_qa_router.post("/index-site")
async def index_site(payload: Dict[str, Any]):
    """Kick off a background crawl of the provided URL and upload results to GCS.

    Request body: { url: string }
    Response: { status: "started" | "error", message?: string }
    """
    url: Optional[str] = (payload or {}).get("url") if isinstance(payload, dict) else None
    if not url:
        return {"status": "error", "message": "Missing 'url' in request body."}

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # Parse bucket and base prefix from config (e.g., "bucket/raw_pages/")
    try:
        bucket_and_prefix = (config.GCS_BUCKET or '').strip('/').split('/', 1)
        if len(bucket_and_prefix) == 2:
            bucket_name, base_prefix = bucket_and_prefix[0], bucket_and_prefix[1]
        else:
            bucket_name, base_prefix = bucket_and_prefix[0], ''
    except Exception:
        return {"status": "error", "message": "Invalid GCS_BUCKET configuration."}

    async def crawl_and_upload():
        try:
            docs = await crawl_site_bfs(url, max_depth=CRAWL_MAX_DEPTH, max_concurrent=CRAWL_MAX_CONCURRENCY)
            if docs:
                upload_markdown_docs_to_gcs(bucket_name=bucket_name, base_prefix=base_prefix, docs=docs)
        except Exception as e:
            try:
                print(f"[index-site] Error while crawling/uploading {url}: {e}")
            except Exception:
                pass

    # Fire-and-forget background task
    asyncio.create_task(crawl_and_upload())
    return {"status": "started"}
