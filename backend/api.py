from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from graph_smart_qa import smart_qa_graph
from state import SmartHopState, SmartQARequest
import time
from logging_relay import log, smartqa_log_relay
from graph_qa import gemini_answer_node, State
from config import config
import os

# --- Smart Hop QA API ---
smart_qa_router = APIRouter()

@smart_qa_router.post("/ask-smart")
async def ask_smart(request: SmartQARequest):
    start_time = time.time()  # ✅ Start timer
    print(f"Received {len(request.links)} links from frontend:")
    for l in request.links:
        print(f"  - {l.get('text', '')[:60]} → {l.get('href')}")
    result = await smart_qa_graph.ainvoke(
        SmartHopState(
            text=request.text,
            question=request.question,
            links=request.links,
            page_url=request.page_url,
            visited_urls=[request.page_url],
            hops=0,
            original_domain=request.page_url.split('/')[2] if '://' in request.page_url else ""
        )
    )

    # Clean print for backend log/debugging
    # log("\n\n===== SMART QA TRACE =====")
    log(f"✅ Sufficient: {result['sufficient']}")
    log(f"🔗 Visited URLs ({len(result['visited_urls'])}):")
    for url in result['visited_urls']:
        log(f"   - {url}")
    print(f"\n🧠 Final Answer (excerpt):\n{result['answer'][:500]}...\n")
    # log("===== END OF TRACE =====\n")

    end_time = time.time()
    duration = round(end_time - start_time, 2)  # seconds with 2 decimals
    log(f"⏱️ Total time taken: {duration} seconds")

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "visited_urls": result["visited_urls"],
        "sufficient": result["sufficient"],
        "confidence": result.get("confidence"),
        "duration_seconds": duration
    }

@smart_qa_router.post("/ask-gemini")
async def ask_gemini(request: SmartQARequest):
    start_time = time.time()
    s = State(
        text=request.text,
        question=request.question,
        page_url=request.page_url
    )
    s = gemini_answer_node(s)
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    return {
        "answer": s.answer,
        "sources": [],
        "visited_urls": [request.page_url],
        "sufficient": s.sufficient,
        "confidence": s.confidence,
        "duration_seconds": duration
    }

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
