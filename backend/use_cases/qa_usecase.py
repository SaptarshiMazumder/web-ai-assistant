from multi_hop_orchestrator import multi_hop_qa_orchestrator
from state import WebsiteMultiHopState
from models import WebAssistantRequest
import time
from logging_relay import log
from state import AssistantState
from web_qa_service import google_search_answer_node
from config import config

async def ask_website_use_case(request: WebAssistantRequest):
    start_time = time.time()
    print(f"Received {len(request.links)} links from frontend:")
    for l in request.links:
        print(f"  - {l.get('text', '')[:60]}  {l.get('href')}")
    result = await multi_hop_qa_orchestrator(
        WebsiteMultiHopState(
            text=request.text,
            question=request.question,
            links=request.links,
            page_url=request.page_url,
            visited_urls=[request.page_url],
            hops=0,
            original_domain=request.page_url.split('/')[2] if '://' in request.page_url else ""
        )
    )
    log(f"✅ Sufficient: {result['sufficient']}")
    log(f"🔗 Visited URLs ({len(result['visited_urls'])}):")
    for url in result['visited_urls']:
        log(f"   - {url}")
    print(f"\n🧠 Final Answer (excerpt):\n{result['answer'][:500]}...\n")
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    log(f"⏱️ Total time taken: {duration} seconds")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "visited_urls": result["visited_urls"],
        "sufficient": result["sufficient"],
        "confidence": result.get("confidence"),
        "duration_seconds": duration
    }

async def ask_google_use_case(request: WebAssistantRequest):
    start_time = time.time()
    s = AssistantState(
        text=request.text,
        question=request.question,
        page_url=request.page_url
    )
    s = google_search_answer_node(s)
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