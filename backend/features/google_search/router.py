from fastapi import APIRouter
from models import WebAssistantRequest
from state import AssistantState
from features.smart_site.web_qa_service import google_search_answer_node


router = APIRouter()


@router.post("/ask-gemini")
async def ask_gemini(request: WebAssistantRequest):
    s = AssistantState(
        text=request.text,
        question=request.question,
        page_url=request.page_url
    )
    s = google_search_answer_node(s)
    return {
        "answer": s.answer,
        "sources": [],
        "visited_urls": [request.page_url],
        "sufficient": s.sufficient,
        "confidence": s.confidence,
    }


