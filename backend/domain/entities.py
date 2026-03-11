from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Bot:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str
    widget_config: Optional[str] = None
    agent_config: Optional[str] = None
    escalation_config: Optional[str] = None


@dataclass
class BotRecord:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str
    created_at: str
    updated_at: str
    widget_config: Optional[str] = None
    agent_config: Optional[str] = None
    escalation_config: Optional[str] = None


@dataclass
class BotDomainRecord:
    org_id: str
    bot_id: str
    hostname: str
    status: str
    verification_token: str
    verified_at: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class OrgRecord:
    org_id: str
    name: str
    status: str
    plan: Optional[str]
    stripe_customer_id: Optional[str]
    stripe_subscription_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class OrgMemberRecord:
    user_id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    role: str
    created_at: str
    updated_at: str


@dataclass
class UserRecord:
    user_id: str
    idp_subject: Optional[str]
    email: Optional[str]
    first_name: Optional[str] = None
    last_name: Optional[str] = None


@dataclass
class Document:
    url: str
    content: str
    metadata: Optional[dict] = None


@dataclass
class BotSource:
    source_id: str
    bot_id: str
    type: str  # url, drive, docs, ...
    config: Dict[str, Any]  # type-specific payload, e.g. {"url": "..."}
    display_name: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    sync_enabled: bool = False
    sync_frequency: str = "daily"  # daily | weekly | monthly
    sync_time_utc: str = "00:00"  # HH:MM in UTC
    sync_timezone: str = "UTC"  # IANA timezone for display
    last_synced_at: Optional[str] = None


@dataclass
class IndexJob:
    job_id: str
    bot_id: str
    url: str
    hostname: str
    stage: str  # queued|crawling|uploading|importing|import_submitted|prompt_queued|prompt_generating|cancelled|error|done
    pages_crawled: int
    docs_count: int
    last_crawled_url: str
    last_depth: int
    gcs_prefix: str
    last_error: str
    created_at: str
    updated_at: str
    celery_task_id: Optional[str] = None
    crawled_urls: List[str] = field(default_factory=list)  # URLs discovered and indexed by this job
    source_id: Optional[str] = None  # optional FK to bot_sources
@dataclass
class DiscoveryJob:
    """Background URL discovery job (no time limit). Created when user starts training from create-bot."""
    job_id: str
    bot_id: str
    root_url: str
    method: str  # "auto" or "sitemap"
    status: str  # queued, running, done, failed
    discovered_urls: List[str] = field(default_factory=list)
    error: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BookingLinkJob:
    """Background booking URL extraction job (RAG-based)."""
    job_id: str
    bot_id: str
    index_job_id: Optional[str] = None
    root_url: str = ""
    status: str = "queued"  # queued, running, done, failed
    links: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SuggestedMessagePackSource:
    url: str = ""
    snippet: str = ""
    title: str = ""
    label: str = ""
    source_kind: str = ""


@dataclass
class SuggestedMessagePack:
    pack_id: str
    bot_id: str
    org_id: str
    lang: str
    suggested_message_id: str
    label: str
    prompt: str
    pack_mode: str
    status: str
    version_hash: str
    source_urls: List[str] = field(default_factory=list)
    evidence_snippets: List[Dict[str, Any]] = field(default_factory=list)
    link_targets: List[Dict[str, Any]] = field(default_factory=list)
    instruction: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SuggestedMessageFastPathResult:
    hit: bool
    reason: str
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    pack: Optional[SuggestedMessagePack] = None
    error: Optional[str] = None


@dataclass
class TopicJob:
    job_id: str
    org_id: str
    bot_id: str
    status: str  # queued|running|done|error
    stage: str  # queued|loading_docs|extracting|categorizing|saving|done|error
    gcs_prefix: Optional[str] = None
    docs_count: int = 0
    topics_count: int = 0
    last_error: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
@dataclass
class AvailabilityJob:
    job_id: str
    org_id: str
    bot_id: str
    url: str
    status: str  # queued|running|done|error
    question: Optional[str] = None
    summary: Optional[str] = None
    raw_text_path: Optional[str] = None
    raw_html_path: Optional[str] = None
    last_error: Optional[str] = None
    max_seconds: int = 60
    steps_count: int = 0
    screenshots_dir: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class JobPipelineRun:
    run_id: str
    org_id: str
    bot_id: str
    workflow_id: str
    trigger: str
    status: str  # queued|running|paused|done|error
    current_step_index: int = 0
    progress_pct: int = 0
    current_step_id: Optional[str] = None
    current_stage_key: Optional[str] = None
    current_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class JobPipelineStep:
    run_id: str
    step_index: int
    job_id: str
    runner_ref: str
    on_failure: str  # continue|stop
    status: str  # queued|running|paused|done|error
    progress_pct: int = 0
    current_stage_key: Optional[str] = None
    current_message: Optional[str] = None
    attempt: int = 0
    celery_task_id: Optional[str] = None
    linked_job_type: Optional[str] = None
    linked_job_id: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class JobPipelineStepEvent:
    event_id: str
    run_id: str
    step_index: Optional[int]
    event_type: str  # progress|status|error|info
    stage_key: Optional[str]
    message: Optional[str]
    progress_pct: Optional[int]
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class ConversationSession:
    session_id: str
    bot_id: str
    org_id: str
    channel: str
    status: str
    title: Optional[str]
    site_url: Optional[str]
    site_title: Optional[str]
    message_count: int
    started_at: str
    last_active_at: str
    ended_at: Optional[str] = None
    user_agent: Optional[str] = None
    ip: Optional[str] = None
    assistant_state: str = "bot"
    handoff_active: bool = False
    support_request_id: Optional[str] = None
    support_request_status: Optional[str] = None


@dataclass
class ConversationChannelContact:
    contact_id: str
    bot_id: str
    channel: str
    external_user_id: str
    display_name: Optional[str]
    current_session_id: Optional[str]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSessionHandoff:
    handoff_id: str
    bot_id: str
    session_id: str
    assistant_state: str
    created_at: str
    updated_at: str
    contact_id: Optional[str] = None
    source_channel: Optional[str] = None
    support_request_id: Optional[str] = None
    started_by: Optional[str] = None
    ended_reason: Optional[str] = None
    ended_at: Optional[str] = None


@dataclass
class ConversationMessage:
    message_id: str
    session_id: str
    bot_id: str
    role: str
    content: str
    sender_name: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


@dataclass
class EscalationRecord:
    escalation_id: str
    bot_id: str
    session_id: str
    visitor_email: str
    created_at: str
    status: str
    details: Optional[str] = None
    session_title: Optional[str] = None
    site_url: Optional[str] = None
    site_title: Optional[str] = None
    last_active_at: Optional[str] = None
    session_status: Optional[str] = None
    visitor_name: Optional[str] = None
    linked_session_id: Optional[str] = None
    notification_read_at: Optional[str] = None
    notification_is_unread: bool = False


@dataclass
class LineChannel:
    channel_id: str
    bot_id: str
    org_id: str
    line_channel_id: str
    line_channel_secret: str
    line_channel_access_token: str
    is_active: bool
    created_at: str
    updated_at: str


@dataclass
class LineDesignConfig:
    config_id: str
    bot_id: str
    org_id: str
    config_json: str
    created_at: str
    updated_at: str


@dataclass
class LineRichMenuState:
    state_id: str
    channel_id: str
    bot_id: str
    config_hash: Optional[str]
    default_variant: Optional[str]
    rich_menu_variants: Dict[str, str]
    sync_status: str
    last_synced_at: Optional[str]
    last_error: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class LineUserSession:
    line_user_id: str
    bot_id: str
    session_id: str
    is_escalated: bool
    created_at: str
    updated_at: str
    awaiting_escalation_msg: bool = False
    awaiting_staff_takeover: bool = False
    display_name: Optional[str] = None


@dataclass
class InstagramChannel:
    channel_id: str
    bot_id: str
    org_id: str
    ig_page_id: str
    app_secret: str
    page_access_token: str
    verify_token: str
    is_active: bool
    created_at: str
    updated_at: str
    # OAuth-flow fields (nullable for backward compat with manual setup)
    ig_user_id: Optional[str] = None
    ig_username: Optional[str] = None
    token_expires_at: Optional[str] = None
    connection_method: str = "manual"  # "manual" or "oauth"


@dataclass
class InstagramUserSession:
    ig_user_id: str
    bot_id: str
    session_id: str
    is_escalated: bool
    created_at: str
    updated_at: str
    awaiting_escalation_msg: bool = False
    awaiting_staff_takeover: bool = False


@dataclass
class BotAsset:
    asset_id: str
    bot_id: str
    org_id: str
    name: str
    description: str
    image_gcs_uri: str
    image_public_url: str
    link_url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    asset_type: str = "image"  # "image" or "menu_item"
    created_at: str = ""
    updated_at: str = ""
