from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Tuple

from .entities import Bot, BotDomainRecord, BotRecord, Document, IndexJob, OrgMemberRecord, OrgRecord, UserRecord


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

    def delete_bot(self, bot_id: str) -> None:
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
    def upsert_user_from_claims(
        self,
        *,
        subject: str,
        email: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> UserRecord:
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


class IndexJobRepository(Protocol):
    def create_job(self, job: IndexJob) -> None:
        ...

    def get_job(self, bot_id: str, job_key: str) -> Optional[IndexJob]:
        ...

    def update_job(self, job: IndexJob) -> None:
        ...

    def list_jobs_for_bot(self, bot_id: str) -> List[IndexJob]:
        ...


class CrawlerRepository(Protocol):
    async def crawl_urls_bfs(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        stop_event: Optional[Any] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Document]:
        ...

    async def crawl_urls_list(
        self,
        urls: List[str],
        *,
        max_concurrent: int,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Document]:
        ...

    async def discover_internal_urls(
        self,
        root_url: str,
        max_depth: int,
        max_concurrent: int,
        *,
        max_urls: int = 2000,
    ) -> List[str]:
        ...


class UrlDiscoveryPort(Protocol):
    """Port for discovering URLs from a site. Swap implementation to use a different service (e.g. replace crawl4ai)."""

    async def discover(self, root_url: str, method: str = "auto") -> List[str]:
        """Discover URLs; method is 'auto' or 'sitemap'. Returns list of URLs."""
        ...

    def discover_stream(
        self, root_url: str, method: str = "auto", *, max_depth: int = 10, max_concurrent: int = 10, max_urls: int = 2000
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream discovery events (discovered, done, error) for UI progress. Same method semantics as discover."""
        ...


class DocumentStorageRepository(Protocol):
    def save_documents(self, bot_id: str, documents: List[Document]) -> str:
        """
        Save documents to storage and return the storage prefix/path.
        """
        ...


class RAGRepository(Protocol):
    def ensure_corpus(self, bot_id: str, *, force_new: bool = False) -> str:
        """
        Ensure a RAG corpus exists for the bot, return corpus resource name.
        """
        ...

    def import_documents(self, corpus_resource: str, storage_prefix: str) -> None:
        """
        Import documents from storage prefix into the RAG corpus.
        """
        ...
