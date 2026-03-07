from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ReservationContext:
    platform_id: str
    domain_key: str
    url: str
    instruction: str
    link_label: str


class PlatformStrategy(Protocol):
    platform_id: str

    def build_reservation_context(self, widget_config: Dict[str, Any], *, lang: str) -> Optional[ReservationContext]:
        ...

    def menu_link(self, category: str, items: List[Any], widget_config: Optional[Dict[str, Any]] = None) -> str:
        ...

    def post_crawl_jobs(self, widget_config: Dict[str, Any]) -> List[str]:
        ...


class ChannelAdapter(Protocol):
    channel_id: str

    def parse_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...

    def format_reply(self, texts: List[str]) -> List[str]:
        ...


class PromptProvider(Protocol):
    def resolve_system_prompt(
        self,
        *,
        agent_config: Dict[str, Any],
        lang: str,
        bot_name: str,
        widget_config: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        ...


class AssetPolicy(Protocol):
    def resolve_assets(
        self,
        *,
        answer: str,
        bot_id: str,
        user_query: Optional[str],
        session_id: Optional[str],
        allowed_asset_types: Optional[set[str]],
        show_assets: Optional[bool],
        asset_term_config: Optional[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, str]]]:
        ...


class FunctionExecutor(Protocol):
    def execute(self, function_id: str, args: Dict[str, Any], context: Dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class JobResult:
    status: str  # done|paused|error
    output: Dict[str, Any] = field(default_factory=dict)
    linked_job_type: Optional[str] = None
    linked_job_id: Optional[str] = None
    error: Optional[str] = None


class JobRunner(Protocol):
    def run(self, context: Dict[str, Any]) -> JobResult:
        ...
