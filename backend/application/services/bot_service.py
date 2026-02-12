from typing import List, Optional, Tuple

from domain.entities import Bot, BotDomainRecord, BotRecord
from domain.repositories import BotCorpusRepository, BotDomainRepository, BotRepository


class BotService:
    def __init__(
        self,
        bot_repo: BotRepository,
        domain_repo: BotDomainRepository,
        corpus_repo: Optional[BotCorpusRepository] = None,
    ) -> None:
        self._bot_repo = bot_repo
        self._domain_repo = domain_repo
        self._corpus_repo = corpus_repo

    def create_bot(self, display_name: str, org_id: str) -> Bot:
        return self._bot_repo.create_bot(display_name, org_id)

    def list_bots(self, org_id: Optional[str] = None) -> List[BotRecord]:
        return self._bot_repo.list_bots(org_id)

    def get_bot_record(self, bot_id: str) -> Optional[BotRecord]:
        return self._bot_repo.get_bot_record(bot_id)

    def update_display_name(self, bot_id: str, display_name: str) -> None:
        self._bot_repo.update_display_name(bot_id, display_name)

    def update_widget_config(self, bot_id: str, config_json: str) -> None:
        self._bot_repo.update_widget_config(bot_id, config_json)

    def update_agent_config(self, bot_id: str, config_json: str) -> None:
        self._bot_repo.update_agent_config(bot_id, config_json)

    def update_escalation_config(self, bot_id: str, config_json: str) -> None:
        self._bot_repo.update_escalation_config(bot_id, config_json)

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

    def delete_bot(self, bot_id: str) -> None:
        """Delete a bot and all related data including Vertex AI corpus and GCS objects."""
        # Get corpus resource before deleting database records
        corpus_resource = None
        if self._corpus_repo:
            corpus_resource = self._corpus_repo.get_bot_corpus(bot_id)
        
        # Get bot info for GCS prefix calculation
        bot = self._bot_repo.get_bot(bot_id)
        if not bot:
            raise ValueError(f"Bot {bot_id} not found")
        
        # Delete from database first (this removes the references)
        self._bot_repo.delete_bot(bot_id)
        
        # Delete Vertex AI corpus if it exists
        if corpus_resource:
            try:
                import vertexai
                from vertexai import rag as vx_rag
                from common.config import config
                
                vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
                vx_rag.delete_corpus(corpus_resource)
            except Exception:
                # Best-effort cleanup; continue even if corpus deletion fails
                pass
        
        # Delete GCS objects for this bot
        try:
            from google.cloud import storage
            from common.config import config
            from infrastructure.services.indexing_service import _parse_bucket_and_prefix, _bot_base_prefix
            
            bucket_name, base_prefix_root = _parse_bucket_and_prefix()
            bot_prefix = _bot_base_prefix(base_prefix_root, bot_id)
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            deleted = 0
            for blob in bucket.list_blobs(prefix=bot_prefix):
                blob.delete()
                deleted += 1
        except Exception:
            # Best-effort cleanup; continue even if GCS deletion fails
            pass
