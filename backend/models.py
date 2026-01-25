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


class BotSummary(BaseModel):
    bot_id: str
    org_id: Optional[str] = None
    display_name: str
    publishable_key: str
    secret_key: str
    created_at: str
    updated_at: str


class BotListResponse(BaseModel):
    bots: List[BotSummary]


class BotDetailResponse(BaseModel):
    bot: BotSummary


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


class BotDomainRecordResponse(BaseModel):
    org_id: Optional[str] = None
    bot_id: str
    hostname: str
    status: str  # pending|verified
    verification_token: str
    verified_at: Optional[str] = None
    created_at: str
    updated_at: str


class BotDomainListResponse(BaseModel):
    bot_id: str
    domains: List[BotDomainRecordResponse] = []


class BotIndexRequest(BaseModel):
    url: str


class BotIndexJobResponse(BaseModel):
    job_id: str
    url: str
    hostname: str
    stage: str
    pages_crawled: int
    docs_count: int
    gcs_prefix: str
    last_error: str
    created_at: str
    updated_at: str


class BotIndexJobListResponse(BaseModel):
    bot_id: str
    jobs: List[BotIndexJobResponse] = []


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


# ===========================
# Admin / Org management
# ===========================

class OrgCreateRequest(BaseModel):
    name: str


class OrgSummary(BaseModel):
    org_id: str
    name: str
    status: str
    plan: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    created_at: str
    updated_at: str


class OrgListResponse(BaseModel):
    orgs: List[OrgSummary]


class OrgMemberAddRequest(BaseModel):
    email: str
    role: str = "org_admin"


class OrgMemberResponse(BaseModel):
    user_id: str
    email: str
    role: str
    created_at: str
    updated_at: str


class OrgMembersListResponse(BaseModel):
    org_id: str
    members: List[OrgMemberResponse] = []


class OrgSelfResponse(BaseModel):
    org_ids: List[str] = []


class OrgUpdateRequest(BaseModel):
    name: str


