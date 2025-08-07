from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from state import SmartHopState, SmartQARequest
import time
from logging_relay import log, smartqa_log_relay
from config import config
import os
from use_cases.qa_usecase import ask_smart_service, ask_gemini_service

# --- Smart Hop QA API ---
smart_qa_router = APIRouter()

@smart_qa_router.post("/ask-smart")
async def ask_smart(request: SmartQARequest):
    return await ask_smart_service(request)

@smart_qa_router.post("/ask-gemini")
async def ask_gemini(request: SmartQARequest):
    return await ask_gemini_service(request)

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
