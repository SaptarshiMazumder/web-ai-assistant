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
