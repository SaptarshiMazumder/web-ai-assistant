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
    headless: Optional[bool] = None  # override headless browser mode for this crawl


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


class AnalyticsSummaryResponse(BaseModel):
    start_day: str
    end_day: str
    conversations: int = 0
    messages_user: int = 0
    messages_bot: int = 0
    escalations: int = 0
    unique_visitors_est: int = 0
    messages_per_conversation: float = 0.0
    escalation_rate: float = 0.0
    positive_feedback: int = 0
    negative_feedback: int = 0


class UsagePoint(BaseModel):
    day: str
    conversations: int = 0
    messages_user: int = 0
    messages_bot: int = 0
    escalations: int = 0
    unique_visitors_est: int = 0


class AnalyticsTimeseriesResponse(BaseModel):
    start_day: str
    end_day: str
    points: List[UsagePoint] = []


class TopSourceItem(BaseModel):
    source_url: str
    count: int = 0


class TopSourcesResponse(BaseModel):
    start_day: str
    end_day: str
    items: List[TopSourceItem] = []


class TopicItem(BaseModel):
    topic: str
    count: int = 0


class TopicsResponse(BaseModel):
    start_day: str
    end_day: str
    items: List[TopicItem] = []


class RecomputeResponse(BaseModel):
    ok: bool = True
    start_day: str
    end_day: str


class ConversationSearchSessionResponse(ConversationSessionResponse):
    snippet: Optional[str] = None


class ConversationSearchResponse(BaseModel):
    bot_id: str
    sessions: List[ConversationSearchSessionResponse]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


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


class EscalationCountsResponse(BaseModel):
    bot_id: str
    total: int
    open: int


class EscalationStatusUpdateRequest(BaseModel):
    status: str


class WidgetConfigUpdate(BaseModel):
    """Widget design config stored per bot (widget API shape). All fields optional."""
    businessType: Optional[str] = None
    allowRealtimeAvailability: Optional[bool] = None
    bookingTestUrl: Optional[str] = None
    bookingUrlPattern: Optional[Dict[str, Any]] = None  # written by availability job
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


# ========== Extracted Topics ==========

class ExtractedTopicItem(BaseModel):
    topic_id: str
    topic: str
    category: Optional[str] = None
    confidence: float = 1.0
    source_urls: List[str] = []
    occurrence_count: int = 1
    is_active: bool = True
    extracted_at: Optional[str] = None
    updated_at: Optional[str] = None


class ExtractedTopicsResponse(BaseModel):
    bot_id: str
    topics: List[ExtractedTopicItem] = []
    total_count: int = 0


class ExtractTopicsRequest(BaseModel):
    clear_existing: bool = False


class ExtractTopicsResponse(BaseModel):
    bot_id: str
    topics_extracted: int = 0
    topics: List[ExtractedTopicItem] = []


class CreateTopicRequest(BaseModel):
    topic: str
    category: Optional[str] = None


class UpdateTopicRequest(BaseModel):
    is_active: Optional[bool] = None
    category: Optional[str] = None


class DeleteTopicResponse(BaseModel):
    ok: bool = True
    topic_id: str


# ========== Topic Extraction Jobs ==========

class TopicJobItem(BaseModel):
    job_id: str
    org_id: str
    bot_id: str
    status: str
    stage: str
    gcs_prefix: Optional[str] = None
    docs_count: int = 0
    topics_count: int = 0
    last_error: Optional[str] = None
    created_at: str
    updated_at: str


class TopicJobsResponse(BaseModel):
    bot_id: str
    jobs: List[TopicJobItem] = []


# ========== Availability Jobs ==========

class AvailabilityRequest(BaseModel):
    url: str
    check_in: Optional[str] = None  # omit to use URL as-is (user pastes full URL with params)
    check_out: Optional[str] = None
    adults: int = 2
    children: int = 0
    rooms: int = 1
    max_seconds: int = 60
    question: Optional[str] = None


class AvailabilityJobItem(BaseModel):
    job_id: str
    org_id: str
    bot_id: str
    url: str
    status: str
    question: Optional[str] = None
    summary: Optional[str] = None
    raw_text_path: Optional[str] = None
    raw_html_path: Optional[str] = None
    last_error: Optional[str] = None
    max_seconds: int = 60
    steps_count: int = 0
    screenshots_dir: Optional[str] = None
    created_at: str
    updated_at: str


class AvailabilityJobsResponse(BaseModel):
    bot_id: str
    jobs: List[AvailabilityJobItem] = []


# ========== Booking Link Jobs ==========

class BookingLinkJobItem(BaseModel):
    job_id: str
    bot_id: str
    index_job_id: Optional[str] = None
    root_url: str
    status: str
    links: List[Dict[str, Any]] = []
    error: Optional[str] = None
    created_at: str
    updated_at: str


class BookingLinkJobsResponse(BaseModel):
    bot_id: str
    jobs: List[BookingLinkJobItem] = []
