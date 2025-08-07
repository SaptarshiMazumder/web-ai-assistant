from app.state import SmartHopState, SmartQARequest
import time
from app.domain.qa import smart_qa_graph, gemini_answer_node, State

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
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "visited_urls": result["visited_urls"],
        "sufficient": result["sufficient"],
        "confidence": result.get("confidence"),
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