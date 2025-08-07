from fastapi import APIRouter
from app.state import SmartQARequest
from app.use_cases.smart_qa import ask_smart_use_case, ask_gemini_use_case

smart_qa_router = APIRouter()

@smart_qa_router.post("/ask-smart")
async def ask_smart(request: SmartQARequest):
    return await ask_smart_use_case(request)

@smart_qa_router.post("/ask-gemini")
async def ask_gemini(request: SmartQARequest):
    return await ask_gemini_use_case(request)