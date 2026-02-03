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
    url: str = ""  # required when source_id not provided
    source_id: Optional[str] = None  # when provided, start crawl for this source (url from source.config)


class BotIndexBatchRequest(BaseModel):
    urls: List[str]


class UrlDiscoveryRequest(BaseModel):
    url: str
    method: str = "auto"  # "auto" (crawl4ai) or "sitemap"
    max_duration_sec: Optional[int] = None  # stop discovery after N seconds (e.g. 90 for create-bot)


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
    crawled_urls: List[str] = []  # URLs discovered and indexed by this job
    source_id: Optional[str] = None


class BotIndexJobListResponse(BaseModel):
    bot_id: str
    jobs: List[BotIndexJobResponse] = []


class DiscoveryJobCreateRequest(BaseModel):
    url: str
    method: str = "auto"  # "auto" or "sitemap"


class DiscoveryJobResponse(BaseModel):
    job_id: str
    bot_id: str
    root_url: str
    method: str
    status: str  # queued, running, done, failed
    discovered_urls: List[str] = []
    discovered_count: int = 0
    error: Optional[str] = None
    created_at: str
    updated_at: str


class DiscoveryJobListResponse(BaseModel):
    jobs: List[DiscoveryJobResponse] = []


class BotSourceCreateRequest(BaseModel):
    type: str  # url, drive, docs, ...
    config: Dict[str, Any]  # type-specific, e.g. {"url": "https://..."}
    display_name: Optional[str] = None


class BotSourceResponse(BaseModel):
    source_id: str
    bot_id: str
    type: str
    config: Dict[str, Any]
    display_name: Optional[str] = None
    created_at: str
    updated_at: str


class BotSourceListResponse(BaseModel):
    bot_id: str
    sources: List[BotSourceResponse] = []


class WidgetChatRequest(BaseModel):
    message: str
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    session_id: Optional[str] = None


class Citation(BaseModel):
    url: str = ""
    snippet: str = ""


class WidgetChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    session_id: Optional[str] = None


class AgentConfigPayload(BaseModel):
    model_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None


class AgentConfigResponse(BaseModel):
    model_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None


class TestChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class TestChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    session_id: Optional[str] = None


class ConversationSessionResponse(BaseModel):
    session_id: str
    bot_id: str
    channel: str
    status: str
    title: Optional[str] = None
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    message_count: int = 0
    started_at: str
    last_active_at: str
    ended_at: Optional[str] = None


class ConversationMessageResponse(BaseModel):
    message_id: str
    session_id: str
    bot_id: str
    role: str
    sender_name: Optional[str] = None
    content: str
    citations: List[Citation] = []
    created_at: str


class ConversationListResponse(BaseModel):
    bot_id: str
    sessions: List[ConversationSessionResponse]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


class ConversationDetailResponse(BaseModel):
    bot_id: str
    session_id: str
    messages: List[ConversationMessageResponse]


class ConversationEndResponse(BaseModel):
    session_id: str
    status: str


class EscalationConfigPayload(BaseModel):
    enabled: Optional[bool] = None
    notify_enabled: Optional[bool] = None
    notification_emails: Optional[str] = None


class EscalationConfigResponse(BaseModel):
    enabled: bool = False
    notify_enabled: bool = False
    notification_emails: str = ""


class EscalationCreateRequest(BaseModel):
    visitor_email: str
    details: Optional[str] = None
    site_url: Optional[str] = None
    site_title: Optional[str] = None


class EscalationRecordResponse(BaseModel):
    escalation_id: str
    bot_id: str
    session_id: str
    visitor_email: str
    details: Optional[str] = None
    status: str
    created_at: str
    title: Optional[str] = None
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    last_active_at: Optional[str] = None
    session_status: Optional[str] = None


class EscalationListResponse(BaseModel):
    bot_id: str
    escalations: List[EscalationRecordResponse]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


class EscalationStatusUpdateRequest(BaseModel):
    status: str


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
    suggestedMessages: Optional[List[Dict[str, Any]]] = None


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
