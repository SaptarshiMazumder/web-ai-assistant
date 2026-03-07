from __future__ import annotations

from typing import Any, Dict

from domain.interfaces import JobResult


class CrawlImportRunner:
    """Crawl/import is executed by existing indexing endpoints/tasks in this phase."""

    def run(self, context: Dict[str, Any]) -> JobResult:
        return JobResult(
            status="done",
            output={"status": "skipped", "reason": "handled_by_existing_indexing_flow"},
        )
