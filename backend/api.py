from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from graph_qa import qa_graph, State
from graph_site_qa import ask_site_handler
from graph_smart_qa import smart_qa_graph, SmartQARequest, SmartHopState

# --- Page QA API ---

class QARequest(BaseModel):
    text: str
    question: str

qa_router = APIRouter()

@qa_router.post("/ask")
async def ask(request: QARequest):
    state = State(text=request.text, question=request.question)
    result = qa_graph.invoke(state)
    return {
        "answer": result["answer"],
        "enhanced_query": result["enhanced_query"],
        "sources": result["used_chunks"],
    }

# --- Site QA API ---

class SiteQARequest(BaseModel):
    question: str
    urls: List[str]

site_qa_router = APIRouter()

@site_qa_router.post("/ask-site")
async def ask_site(request: SiteQARequest):
    return await ask_site_handler(request)

# --- Smart Hop QA API ---

smart_qa_router = APIRouter()

@smart_qa_router.post("/ask-smart")
async def ask_smart(request: SmartQARequest):
    result = smart_qa_graph.invoke(
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
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "visited_urls": result["visited_urls"],
        "sufficient": result["sufficient"],
    }
