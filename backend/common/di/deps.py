from application.services.bot_service import BotService
from application.services.org_service import OrgService
from application.services.user_service import UserService
from common.di.container import bot_service, org_service, user_service


def get_bot_service() -> BotService:
    return bot_service()


def get_org_service() -> OrgService:
    return org_service()


def get_user_service() -> UserService:
    return user_service()
