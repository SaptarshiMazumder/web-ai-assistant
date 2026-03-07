from __future__ import annotations

from typing import Any, Dict

from domain.interfaces import JobResult


class PromptGenerationRunner:
    """Prompt generation is still triggered by existing async flow in this phase."""

    def run(self, context: Dict[str, Any]) -> JobResult:
        return JobResult(
            status="done",
            output={"status": "skipped", "reason": "handled_by_existing_prompt_flow"},
        )
