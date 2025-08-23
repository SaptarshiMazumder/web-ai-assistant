from fastapi import APIRouter
from models import WebAssistantRequest
from .qa_usecase import ask_website_use_case


router = APIRouter()


@router.post("/ask-smart")
async def ask_smart(request: WebAssistantRequest):
    return await ask_website_use_case(request)


