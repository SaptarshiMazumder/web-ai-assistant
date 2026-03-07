from __future__ import annotations

from typing import Any, Dict

from domain.interfaces import JobResult


class DiscoveryRunner:
    """
    Discovery remains user-driven in this phase.
    Pipeline pause/resume is controlled by YAML gate rules.
    """

    def run(self, context: Dict[str, Any]) -> JobResult:
        return JobResult(
            status="done",
            output={
                "status": "skipped",
                "reason": "handled_by_discovery_flow",
            },
        )
