from fastapi import APIRouter, Query
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
        host = parsed.hostname or ""
    except Exception:
        host = ""
    # Fallback: if caller passed a bare domain
    if not host and url:
        host = url.strip()
    # Inspect local raw_pages directory
    try:
        raw_dir = os.path.join(os.path.dirname(__file__), "raw_pages")
        indexed = False
        if os.path.isdir(raw_dir) and host:
            for fname in os.listdir(raw_dir):
                # Expect filenames like: www.example.com_<hash>.txt
                if fname.startswith(f"{host}_") or fname.startswith(host):
                    indexed = True
                    break
        return {"indexed": indexed, "host": host, "source": "local_raw_pages"}
    except Exception:
        return {"indexed": False, "host": host, "source": "local_raw_pages"}

