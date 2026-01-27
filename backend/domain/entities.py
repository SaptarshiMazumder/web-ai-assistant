from dataclasses import dataclass
from typing import Optional


@dataclass
class Bot:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str


@dataclass
class BotRecord:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str
    created_at: str
    updated_at: str


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
