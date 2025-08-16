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


