from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===========================
# SaaS / Embeddable widget API
# ===========================

class BotCreateRequest(BaseModel):
    display_name: str
    org_id: Optional[str] = None


class BotRenameRequest(BaseModel):
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


class BotSourceSyncSettingsRequest(BaseModel):
    sync_enabled: bool
    sync_frequency: str = "daily"  # daily | weekly | monthly
    sync_time_utc: str = "00:00"  # HH:MM in UTC
    sync_timezone: str = "UTC"  # IANA timezone for display


class BotSourceResponse(BaseModel):
    source_id: str
    bot_id: str
    type: str
    config: Dict[str, Any]
    display_name: Optional[str] = None
    created_at: str
    updated_at: str
    sync_enabled: bool = False
    sync_frequency: str = "daily"
    sync_time_utc: str = "00:00"
    sync_timezone: str = "UTC"
    last_synced_at: Optional[str] = None


class BotSourceListResponse(BaseModel):
    bot_id: str
    sources: List[BotSourceResponse] = []


class PdfSourceUploadItem(BaseModel):
    source: BotSourceResponse
    job_id: str
    status: str = "queued"


class PdfSourceUploadResponse(BaseModel):
    bot_id: str
    items: List[PdfSourceUploadItem] = []


class TextSourceEntry(BaseModel):
    title: Optional[str] = None
    content: str


class TextSourceUploadRequest(BaseModel):
    entries: List[TextSourceEntry]


class TextSourceUploadItem(BaseModel):
    source_id: str
    job_id: str
    status: str = "queued"


class TextSourceUploadResponse(BaseModel):
    bot_id: str
    items: List[TextSourceUploadItem] = []


class DocsSourceUploadItem(BaseModel):
    source_id: str
    job_id: str
    status: str = "queued"


class DocsSourceUploadResponse(BaseModel):
    bot_id: str
    items: List[DocsSourceUploadItem] = []


class WidgetChatRequest(BaseModel):
    message: str
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    session_id: Optional[str] = None
    suggested_message_id: Optional[str] = None


class Citation(BaseModel):
    url: str = ""
    snippet: str = ""


class WidgetChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    assets: List["AssetCard"] = []
    session_id: Optional[str] = None
    suggested_messages: List[Dict[str, Any]] = []


class CustomPersona(BaseModel):
    id: str
    name: str
    emoji: str = ""
    description: str = ""
    system_prompt: str = ""


class AgentConfigPayload(BaseModel):
    model_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    persona_id: Optional[str] = None
    custom_personas: Optional[List[CustomPersona]] = None


class AgentConfigResponse(BaseModel):
    model_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    persona_id: Optional[str] = None
    custom_personas: Optional[List[CustomPersona]] = None


# ── Personas ──────────────────────────────────────────────────────────────────

class PersonaItem(BaseModel):
    id: str
    name: str
    category: str
    description: str
    emoji: str
    system_prompt: str


class PersonaListResponse(BaseModel):
    personas: List[PersonaItem] = []
    categories: List[str] = []


class TestChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class TestChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    assets: List["AssetCard"] = []
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
    assistant_state: str = "bot"
    handoff_active: bool = False
    support_request_id: Optional[str] = None
    support_request_status: Optional[str] = None


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
    assistant_state: str = "bot"
    handoff_active: bool = False
    support_request_id: Optional[str] = None
    support_request_status: Optional[str] = None


class ConversationEndResponse(BaseModel):
    session_id: str
    status: str


class ConversationTakeoverResponse(BaseModel):
    ok: bool = True
    message: str = "Conversation transferred to support. The bot is now paused for this session."
    assistant_state: str = "human_handoff"
    handoff_active: bool = True
    support_request_id: Optional[str] = None
    support_request_status: Optional[str] = None


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
    notify_enabled: Optional[bool] = None
    notify_website: Optional[bool] = None
    notify_instagram: Optional[bool] = None
    notify_line: Optional[bool] = None
    notification_emails: Optional[str] = None


class EscalationConfigResponse(BaseModel):
    notify_enabled: bool = False
    notify_website: bool = False
    notify_instagram: bool = False
    notify_line: bool = False
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
    visitor_name: Optional[str] = None
    details: Optional[str] = None
    status: str
    created_at: str
    title: Optional[str] = None
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    last_active_at: Optional[str] = None
    session_status: Optional[str] = None
    linked_session_id: Optional[str] = None
    notification_read_at: Optional[str] = None
    notification_is_unread: bool = False


class EscalationListResponse(BaseModel):
    bot_id: str
    escalations: List[EscalationRecordResponse]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


class EscalationCountsResponse(BaseModel):
    bot_id: str
    total: int
    open: int
    unread: int = 0


class BotOverviewSetupItemResponse(BaseModel):
    id: str
    label: str
    done: bool = False
    route: str


class BotOverviewSetupResponse(BaseModel):
    bot_id: str
    sections: List[BotOverviewSetupItemResponse] = []


class EscalationStatusUpdateRequest(BaseModel):
    status: str


class WidgetConfigUpdate(BaseModel):
    """Widget design config stored per bot (widget API shape). All fields optional."""
    language: Optional[str] = None  # "en" or "ja" — bot content language
    businessType: Optional[str] = None
    # Where the business content lives. Used to drive UX (own website vs website service).
    contentHosting: Optional[str] = None
    allowRealtimeAvailability: Optional[bool] = None
    allowAutoImageExtraction: Optional[bool] = None
    bookingTestUrl: Optional[str] = None
    bookingUrlPattern: Optional[Dict[str, Any]] = None  # written by availability job
    # Restaurant reservation: one profile per agent (tabelog | hotpepper | tablecheck)
    reservationPlatform: Optional[str] = None
    # Canonical map form (backward-compatible bridge): {platform_id: url}
    reservationLinks: Optional[Dict[str, str]] = None
    reservation_links: Optional[Dict[str, str]] = None
    actionDestinationLinks: Optional[Dict[str, str]] = None
    # URL for the selected platform (use the field matching reservationPlatform)
    tableCheckUrl: Optional[str] = None
    tabelogUrl: Optional[str] = None
    hotPepperUrl: Optional[str] = None
    # Optional custom instruction; use {url} for the link. Overrides platform profile.
    reservationInstruction: Optional[str] = None
    # Labeled links the agent can share in answers (not crawled/trained sources).
    urlBank: Optional[List[Dict[str, Any]]] = None
    position: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None
    size: Optional[str] = None
    welcomeMessage: Optional[str] = None
    welcomeMessagesByChannel: Optional[Dict[str, Dict[str, str]]] = None
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
    suggestedMessagesByLanguage: Optional[Dict[str, List[Dict[str, Any]]]] = None


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
    source_url: Optional[str] = None
    origin: str = "extracted"
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


# ========== Topic Usage / Question Mapping ==========

class TopicUsageItem(BaseModel):
    topic_id: str
    topic: str
    category: str = "other"
    question_count: int = 0
    source_url: Optional[str] = None
    origin: str = "extracted"


class TopicUsageSummaryResponse(BaseModel):
    bot_id: str
    topics: List[TopicUsageItem] = []
    total_questions: int = 0


class TopicQuestionItem(BaseModel):
    id: str
    topic_id: str
    session_id: str
    message_id: Optional[str] = None
    question_text: Optional[str] = None
    asked_at: str
    session_title: Optional[str] = None


class TopicQuestionsResponse(BaseModel):
    topic_id: str
    questions: List[TopicQuestionItem] = []


class SyncUrlBankRequest(BaseModel):
    url_bank: List[dict] = []


class ComputeMappingsResponse(BaseModel):
    bot_id: str
    new_mappings: int = 0


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


# ========== Job Pipeline ==========

class JobPipelineStepItem(BaseModel):
    run_id: str
    step_index: int
    job_id: str
    runner_ref: str
    on_failure: str
    status: str
    progress_pct: int = 0
    current_stage_key: Optional[str] = None
    current_message: Optional[str] = None
    attempt: int = 0
    celery_task_id: Optional[str] = None
    linked_job_type: Optional[str] = None
    linked_job_id: Optional[str] = None
    output: Dict[str, Any] = {}
    last_error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
    updated_at: str


class JobPipelineEventItem(BaseModel):
    event_id: str
    run_id: str
    step_index: Optional[int] = None
    event_type: str
    stage_key: Optional[str] = None
    message: Optional[str] = None
    progress_pct: Optional[int] = None
    details: Dict[str, Any] = {}
    created_at: str


class JobPipelineRunItem(BaseModel):
    run_id: str
    org_id: str
    bot_id: str
    workflow_id: str
    trigger: str
    status: str
    current_step_index: int
    progress_pct: int = 0
    current_step_id: Optional[str] = None
    current_stage_key: Optional[str] = None
    current_message: Optional[str] = None
    context: Dict[str, Any] = {}
    last_error: Optional[str] = None
    created_at: str
    updated_at: str
    steps: List[JobPipelineStepItem] = []
    events: List[JobPipelineEventItem] = []


class JobPipelineLatestResponse(BaseModel):
    bot_id: str
    run: Optional[JobPipelineRunItem] = None


class JobPipelineResumeResponse(BaseModel):
    bot_id: str
    run: JobPipelineRunItem


# ========== LINE Integration ==========

class LineChannelUpsertRequest(BaseModel):
    line_channel_id: str
    line_channel_secret: Optional[str] = None  # blank = keep existing on update
    line_channel_access_token: Optional[str] = None  # blank = keep existing on update
    is_active: bool = True


class LineChannelResponse(BaseModel):
    channel_id: str
    bot_id: str
    org_id: str
    line_channel_id: str
    is_active: bool
    created_at: str
    updated_at: str
    managed_rich_menu_enabled: bool = True
    rich_menu_sync_status: Optional[str] = None
    rich_menu_last_synced_at: Optional[str] = None
    rich_menu_last_error: Optional[str] = None
    rich_menu_variants: Dict[str, str] = Field(default_factory=dict)
    # Secrets are NOT returned


class LineChannelDeleteResponse(BaseModel):
    ok: bool = True
    bot_id: str


class LineChannelTestResponse(BaseModel):
    ok: bool
    message: str
    display_name: Optional[str] = None
    basic_id: Optional[str] = None
    picture_url: Optional[str] = None
    user_id: Optional[str] = None


class LineDesignUpdateRequest(BaseModel):
    suggested_actions: Optional[Dict[str, Any]] = None
    asset_carousel: Optional[Dict[str, Any]] = None
    rich_menu: Optional[Dict[str, Any]] = None


class LineDesignResponse(BaseModel):
    bot_id: str
    effective_config: Dict[str, Any] = Field(default_factory=dict)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[str] = None


# ========== Instagram Integration ==========

class InstagramChannelUpsertRequest(BaseModel):
    ig_page_id: str
    app_secret: Optional[str] = None  # blank = keep existing on update
    page_access_token: Optional[str] = None  # blank = keep existing on update
    is_active: bool = True


class InstagramChannelResponse(BaseModel):
    channel_id: str
    bot_id: str
    org_id: str
    ig_page_id: str
    verify_token: str
    is_active: bool
    created_at: str
    updated_at: str
    # OAuth fields
    ig_user_id: Optional[str] = None
    ig_username: Optional[str] = None
    token_expires_at: Optional[str] = None
    connection_method: str = "manual"
    # Secrets are NOT returned


class InstagramChannelDeleteResponse(BaseModel):
    ok: bool = True
    bot_id: str


class InstagramOAuthUrlResponse(BaseModel):
    auth_url: str


# ── Image Assets ─────────────────────────────────────────────────────────


class AssetCard(BaseModel):
    """Lightweight card included in chat responses."""
    asset_id: str
    name: str
    image_url: str
    link_url: Optional[str] = None
    description: Optional[str] = None
    asset_type: Optional[str] = None


class BotAssetResponse(BaseModel):
    asset_id: str
    bot_id: str
    org_id: str
    name: str
    description: str
    image_url: str
    link_url: Optional[str] = None
    keywords: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    asset_type: str = "image"
    created_at: str
    updated_at: str


class BotAssetListResponse(BaseModel):
    bot_id: str
    assets: List[BotAssetResponse] = []
    count: int = 0
    limit: int = 15
    total_count: int = 0
    page_size: int = 1000
    offset: int = 0
    has_more: bool = False


class BotAssetAutoExtractRequest(BaseModel):
    page_urls: List[str] = []


class BotAssetAutoExtractResponse(BaseModel):
    ok: bool = True
    job_id: Optional[str] = None
    assets_extracted: int = 0
    assets_count: int = 0
    assets_limit: int = 15
    pages_considered: int = 0


class AssetExtractionStatusResponse(BaseModel):
    job_id: str
    status: str  # none, queued, running, done, error, cancelled
    assets_discovered: int = 0
    assets_downloaded: int = 0
    assets_created: int = 0
    assets_total: int = 0
    limit: int = 15
    error: Optional[str] = None


class BotAssetDeleteResponse(BaseModel):
    ok: bool = True
    bot_id: str
    asset_id: str
