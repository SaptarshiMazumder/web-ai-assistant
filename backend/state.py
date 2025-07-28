from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SmartQARequest(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str

class SmartHopState(BaseModel):
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