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

# --- Chroma debug API ---
class PageData(BaseModel):
    url: str
    html: str
    domain: str

chroma_router = APIRouter()

@chroma_router.get("/chroma_exists")
async def chroma_exists(domain: str = Query(...)):
    print(f"Checking if chroma db exists for domain: {domain}")
    path = f"{config.CHROMA_DB_DIR}{domain}"
    return {"exists": os.path.exists(path)}

@chroma_router.post("/add_page_data")
async def add_page_data(data: PageData):
    print(f"\n--- Received page data ---")
    print(f"Domain: {data.domain}")
    print(f"URL: {data.url}")
    print(f"HTML length: {len(data.html)}")
    print(f"First part of HTML: {data.html[:200]!r}")
    return {"ok": True}
