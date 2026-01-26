from functools import lru_cache

from application.services.bot_service import BotService
from application.services.org_service import OrgService
from application.services.user_service import UserService
from infrastructure.db.repositories import (
    PostgresBotDomainRepository,
    PostgresBotRepository,
    PostgresOrgMembershipRepository,
    PostgresOrgRepository,
    PostgresUserRepository,
)


@lru_cache(maxsize=1)
def bot_service() -> BotService:
    return BotService(PostgresBotRepository(), PostgresBotDomainRepository())


@lru_cache(maxsize=1)
def org_service() -> OrgService:
    return OrgService(PostgresOrgRepository(), PostgresOrgMembershipRepository())


@lru_cache(maxsize=1)
def user_service() -> UserService:
    return UserService(PostgresUserRepository())
