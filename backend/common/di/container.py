from functools import lru_cache

from application.services.bot_service import BotService
from application.services.indexing_service import IndexingService
from application.services.org_service import OrgService
from application.services.conversation_service import ConversationService
from application.services.analytics_service import AnalyticsService
from application.services.user_service import UserService
from infrastructure.db.repositories import (
    PostgresBotCorpusRepository,
    PostgresBotDomainRepository,
    PostgresBotRepository,
    PostgresBotSourceRepository,
    PostgresAnalyticsRepository,
    PostgresIndexJobRepository,
    PostgresOrgMembershipRepository,
    PostgresOrgRepository,
    PostgresUserRepository,
)
from domain.repositories import UrlDiscoveryPort
from infrastructure.rag.url_discovery_adapter import AutoUrlDiscoveryAdapter
from infrastructure.repositories import (
    Crawl4AICrawlerRepository,
    GCSDocumentStorageRepository,
    VertexRAGRepository,
)
from infrastructure.services.indexing_service import _parse_bucket_and_prefix


@lru_cache(maxsize=1)
def bot_service() -> BotService:
    return BotService(
        PostgresBotRepository(),
        PostgresBotDomainRepository(),
        PostgresBotCorpusRepository(),
    )


@lru_cache(maxsize=1)
def org_service() -> OrgService:
    return OrgService(PostgresOrgRepository(), PostgresOrgMembershipRepository())


@lru_cache(maxsize=1)
def user_service() -> UserService:
    return UserService(PostgresUserRepository())


@lru_cache(maxsize=1)
def indexing_service() -> IndexingService:
    bot_repo = PostgresBotRepository()
    domain_repo = PostgresBotDomainRepository()
    corpus_repo = PostgresBotCorpusRepository()
    source_repo = PostgresBotSourceRepository()
    job_repo = PostgresIndexJobRepository()
    crawler_repo = Crawl4AICrawlerRepository()
    
    bucket_name, base_prefix_root = _parse_bucket_and_prefix()
    storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix_root)
    rag_repo = VertexRAGRepository()
    
    return IndexingService(
        bot_repo=bot_repo,
        domain_repo=domain_repo,
        corpus_repo=corpus_repo,
        source_repo=source_repo,
        job_repo=job_repo,
        crawler_repo=crawler_repo,
        storage_repo=storage_repo,
        rag_repo=rag_repo,
    )


@lru_cache(maxsize=1)
def url_discovery() -> UrlDiscoveryPort:
    """URL discovery: try HTTP first (fast); if few URLs, fall back to crawl4ai browser (JS sites)."""
    return AutoUrlDiscoveryAdapter()


@lru_cache(maxsize=1)
def conversation_service() -> ConversationService:
    return ConversationService()


@lru_cache(maxsize=1)
def analytics_service() -> AnalyticsService:
    return AnalyticsService(PostgresAnalyticsRepository())
