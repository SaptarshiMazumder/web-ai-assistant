from app.state import SmartHopState, SmartQARequest
import time
from app.domain.qa import smart_qa_graph, gemini_answer_node, State
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.logging_relay import log, smartqa_log_relay

async def ask_smart_use_case(request: SmartQARequest):
    start_time = time.time()
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
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    # Handle both dict and State object
    answer = getattr(result, 'answer', None) or result.get('answer', '')
    visited_urls = getattr(result, 'visited_urls', None) or result.get('visited_urls', [])
    sufficient = getattr(result, 'sufficient', None) or result.get('sufficient', False)
    confidence = getattr(result, 'confidence', None) or result.get('confidence', None)
    # Always provide sources as an empty list (unless you add it to State)
    return {
        "answer": answer,
        "sources": [],
        "visited_urls": visited_urls,
        "sufficient": sufficient,
        "confidence": confidence,
        "duration_seconds": duration
    }

async def ask_gemini_use_case(request: SmartQARequest):
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