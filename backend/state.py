from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class WebsiteMultiHopState(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str
    answer: str = ""
    sources: List[Any] = []
    sufficient: bool = False
    selected_link: Optional[Dict[str, str]] = None
    visited_urls: List[str] = []
    hops: int = 0
    original_domain: str = ""

class AssistantState(BaseModel):
    text: str
    question: str
    enhanced_query: str = ""
    docs: List[Any] = []
    retrieved_docs: List[Any] = []
    answer: str = ""
    used_chunks: List[Dict[str, Any]] = []
    page_url: str = ""
    sufficient: bool = False
    confidence: Optional[int] = None