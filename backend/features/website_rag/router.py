from fastapi import APIRouter, Query
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from models import WebsiteRagRequest
from .vertex_rag_eg import run_vertex_rag
from config import config
from .crawl_to_gcs import list_existing_site_prefixes, crawl_site_bfs, upload_markdown_docs_to_gcs, CRAWL_MAX_DEPTH, CRAWL_MAX_CONCURRENCY
import asyncio


router = APIRouter()


@router.post("/ask-website-rag")
async def ask_website_rag(request: WebsiteRagRequest):
    domain = request.domain
    try:
        if not domain and request.page_url:
            domain = urlparse(request.page_url).hostname
    except Exception:
        pass
    print(f"[Website RAG] domain: {domain}")
    result = run_vertex_rag(request.question)
    return result


@router.get("/is-indexed")
async def is_indexed(url: str):
    print("Checking indexed or not...")
    from urllib.parse import urlparse as _urlparse
    host = ""
    try:
        parsed = _urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = ""
    if not host and url:
        host = url.strip()

    def host_variants(h: str):
        if not h:
            return []
        base = h.split(":")[0].lower()
        nowww = base[4:] if base.startswith("www.") else base
        parts = nowww.split(".")
        apex = nowww if len(parts) < 2 else ".".join(parts[-2:])
        return [v for v in {base, nowww, apex, f"www.{apex}"} if v]

    variants = host_variants(host)
    try:
        bucket_and_prefix = (config.GCS_BUCKET or '').strip('/').split('/', 1)
        if len(bucket_and_prefix) == 2:
            bucket_name, base_prefix = bucket_and_prefix[0], bucket_and_prefix[1]
        else:
            bucket_name, base_prefix = bucket_and_prefix[0], ''
        for hv in variants:
            try:
                prefixes = list_existing_site_prefixes(
                    bucket_name=bucket_name,
                    base_prefix=base_prefix,
                    site_url=f"https://{hv}",
                )
                if prefixes:
                    return {"indexed": True, "host": host, "source": "gcs"}
            except Exception:
                pass
    except Exception:
        pass

    # Only use GCS status; do not fall back to local raw_pages
    return {"indexed": False, "host": host, "source": "gcs"}


@router.post("/index-site")
async def index_site(payload: Dict[str, Any]):
    url: Optional[str] = (payload or {}).get("url") if isinstance(payload, dict) else None
    if not url:
        return {"status": "error", "message": "Missing 'url' in request body."}
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

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

    asyncio.create_task(crawl_and_upload())
    return {"status": "started"}


