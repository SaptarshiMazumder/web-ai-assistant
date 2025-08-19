from fastapi import APIRouter
from models import WebAssistantRequest
from use_cases.qa_usecase import ask_google_use_case


router = APIRouter()


@router.post("/ask-gemini")
async def ask_gemini(request: WebAssistantRequest):
    return await ask_google_use_case(request)


