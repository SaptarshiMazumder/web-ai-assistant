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


@dataclass
class IndexJob:
    job_id: str
    bot_id: str
    url: str
    hostname: str
    stage: str  # queued|crawling|uploading|importing|import_submitted|cancelled|error|done
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
class LineUserSession:
    line_user_id: str
    bot_id: str
    session_id: str
    is_escalated: bool
    created_at: str
    updated_at: str


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


@dataclass
class InstagramUserSession:
    ig_user_id: str
    bot_id: str
    session_id: str
    is_escalated: bool
    created_at: str
    updated_at: str


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
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""
