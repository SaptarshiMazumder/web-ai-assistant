from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ===========================
# SaaS / Embeddable widget API
# ===========================

class BotCreateRequest(BaseModel):
    display_name: str
    org_id: Optional[str] = None


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
    widget_config: Optional[Dict[str, Any]] = None


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


class BotIndexBatchRequest(BaseModel):
    urls: List[str]


class UrlDiscoveryRequest(BaseModel):
    url: str
    method: str = "auto"  # "auto" (crawl4ai) or "sitemap"


class UrlDiscoveryResponse(BaseModel):
    urls: List[str] = []
    error: Optional[str] = None  # Error message if discovery failed
    method_used: Optional[str] = None  # "auto" or "sitemap" - which method was actually used


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
    site_url: Optional[str] = None
    site_title: Optional[str] = None


class Citation(BaseModel):
    url: str = ""
    snippet: str = ""


class WidgetChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []


class WidgetConfigUpdate(BaseModel):
    """Widget design config stored per bot (widget API shape). All fields optional."""
    position: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None
    size: Optional[str] = None
    welcomeMessage: Optional[str] = None
    placeholder: Optional[str] = None
    footer: Optional[str] = None
    theme: Optional[str] = None
    textColor: Optional[str] = None
    launcherIcon: Optional[str] = None
    launcherText: Optional[str] = None
    headerIcon: Optional[str] = None
    shareIcon: Optional[str] = None
    maxHeight: Optional[int] = None
    fontSize: Optional[str] = None
    headerSize: Optional[str] = None
    autoPopup: Optional[str] = None
    autoScroll: Optional[bool] = None
    displaySources: Optional[bool] = None
    sourcesLabel: Optional[str] = None


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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
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
