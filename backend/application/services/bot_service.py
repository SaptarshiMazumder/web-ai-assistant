from typing import List, Optional, Tuple

from domain.entities import Bot, BotDomainRecord, BotRecord
from domain.repositories import BotDomainRepository, BotRepository


class BotService:
    def __init__(self, bot_repo: BotRepository, domain_repo: BotDomainRepository) -> None:
        self._bot_repo = bot_repo
        self._domain_repo = domain_repo

    def create_bot(self, display_name: str, org_id: str) -> Bot:
        return self._bot_repo.create_bot(display_name, org_id)

    def list_bots(self, org_id: Optional[str] = None) -> List[BotRecord]:
        return self._bot_repo.list_bots(org_id)

    def get_bot_record(self, bot_id: str) -> Optional[BotRecord]:
        return self._bot_repo.get_bot_record(bot_id)

    def get_bot_by_publishable_key(self, publishable_key: str) -> Optional[Bot]:
        return self._bot_repo.get_bot_by_publishable_key(publishable_key)

    def get_bot_by_secret_key(self, secret_key: str) -> Optional[Bot]:
        return self._bot_repo.get_bot_by_secret_key(secret_key)

    def add_domain(self, bot_id: str, hostname: str) -> Tuple[str, str]:
        return self._domain_repo.add_domain(bot_id, hostname)

    def list_domains(self, bot_id: str) -> List[BotDomainRecord]:
        return self._domain_repo.list_domains(bot_id)

    def list_verified_hosts(self, bot_id: str) -> List[str]:
        return self._domain_repo.list_verified_hosts(bot_id)

    def mark_domain_verified(self, bot_id: str, hostname: str) -> None:
        self._domain_repo.mark_domain_verified(bot_id, hostname)
