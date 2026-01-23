from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PageQAResult(BaseModel):
    url: str
    text: str
    answer: str
    sources: List[Dict[str, Any]]
    sufficient: Optional[bool] = None
    links: List[Dict[str, str]] = []
    confidence: Optional[int] = None


class WebAssistantRequest(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str


class WebsiteRagRequest(WebAssistantRequest):
    domain: Optional[str] = None


# ===========================
# SaaS / Embeddable widget API
# ===========================

class BotCreateRequest(BaseModel):
    display_name: str


class BotCreateResponse(BaseModel):
    bot_id: str
    display_name: str
    publishable_key: str
    secret_key: str


class BotDomainAddRequest(BaseModel):
    hostname: str


class BotDomainAddResponse(BaseModel):
    bot_id: str
    hostname: str
    status: str  # pending|verified
    verification_token: str
    verification_url: str


class BotDomainVerifyResponse(BaseModel):
    bot_id: str
    hostname: str
    status: str  # pending|verified
    verified: bool
    message: str


class BotIndexRequest(BaseModel):
    url: str


class WidgetChatRequest(BaseModel):
    message: str
    # Optional context from the embedding page (helps RAG when user asks vague questions like "this service").
    site_url: Optional[str] = None
    site_title: Optional[str] = None


class Citation(BaseModel):
    url: str = ""
    snippet: str = ""


class WidgetChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []


