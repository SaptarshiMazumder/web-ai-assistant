from typing import Dict, List, Optional, Protocol, Tuple

from .entities import Bot, BotDomainRecord, BotRecord, OrgMemberRecord, OrgRecord, UserRecord


class BotRepository(Protocol):
    def create_bot(self, display_name: str, org_id: str) -> Bot:
        ...

    def get_bot_by_publishable_key(self, publishable_key: str) -> Optional[Bot]:
        ...

    def get_bot_by_secret_key(self, secret_key: str) -> Optional[Bot]:
        ...

    def get_bot(self, bot_id: str) -> Optional[Bot]:
        ...

    def get_bot_record(self, bot_id: str) -> Optional[BotRecord]:
        ...

    def list_bots(self, org_id: Optional[str] = None) -> List[BotRecord]:
        ...


class BotDomainRepository(Protocol):
    def add_domain(self, bot_id: str, hostname: str) -> Tuple[str, str]:
        ...

    def list_domains(self, bot_id: str) -> List[BotDomainRecord]:
        ...

    def list_verified_hosts(self, bot_id: str) -> List[str]:
        ...

    def mark_domain_verified(self, bot_id: str, hostname: str) -> None:
        ...


class BotCorpusRepository(Protocol):
    def upsert_bot_corpus(self, bot_id: str, corpus_resource: str) -> None:
        ...

    def get_bot_corpus(self, bot_id: str) -> Optional[str]:
        ...


class OrgRepository(Protocol):
    def create_org(self, name: str) -> str:
        ...

    def list_orgs(self) -> List[OrgRecord]:
        ...

    def get_org_by_name(self, name: str) -> Optional[OrgRecord]:
        ...

    def get_org(self, org_id: str) -> Optional[OrgRecord]:
        ...

    def update_org_name(self, org_id: str, name: str) -> None:
        ...

    def set_org_status(self, org_id: str, status: str) -> None:
        ...


class OrgMembershipRepository(Protocol):
    def add_membership(self, org_id: str, user_id: str, role: str) -> None:
        ...

    def get_org_memberships(self, user_id: str) -> List[Dict[str, str]]:
        ...

    def list_org_members(self, org_id: str) -> List[OrgMemberRecord]:
        ...


class UserRepository(Protocol):
    def upsert_user_from_claims(self, *, subject: str, email: str) -> UserRecord:
        ...

    def create_user_placeholder(self, email: str) -> UserRecord:
        ...

    def get_user_by_subject(self, subject: str) -> Optional[UserRecord]:
        ...


class DomainCorpusRepository(Protocol):
    def get_corpus_for_host(self, hostname: str) -> Optional[str]:
        ...

    def upsert_corpus_for_host(self, hostname: str, corpus_resource: str) -> None:
        ...

    def delete_corpus_mapping_for_host(self, hostname: str) -> None:
        ...
