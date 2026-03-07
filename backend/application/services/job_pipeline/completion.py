from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class CompletionCheckResult:
    status: str  # running|done|error
    message: str | None = None
    progress_pct: int | None = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class CompletionChecker(Protocol):
    def check(self, context: Dict[str, Any]) -> CompletionCheckResult:
        ...


def _normalize_status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _status_map(context: Dict[str, Any]) -> Dict[str, set[str]]:
    job_cfg = context.get("job_config")
    completion = job_cfg.get("completion") if isinstance(job_cfg, dict) and isinstance(job_cfg.get("completion"), dict) else {}
    status_map = completion.get("status_map") if isinstance(completion.get("status_map"), dict) else {}

    def read_group(key: str) -> set[str]:
        raw = status_map.get(key)
        if not isinstance(raw, list):
            return set()
        return {str(v).strip().lower() for v in raw if str(v).strip()}

    return {
        "running": read_group("running"),
        "done": read_group("done"),
        "error": read_group("error"),
    }


def _map_child_status(raw_status: str, context: Dict[str, Any]) -> str:
    groups = _status_map(context)
    if raw_status in groups["done"]:
        return "done"
    if raw_status in groups["error"]:
        return "error"
    if raw_status in groups["running"]:
        return "running"
    return "running"


class CompletionCheckerRegistry:
    """Config-driven completion checker resolver (checker_ref -> checker object)."""

    def __init__(self) -> None:
        self._cache: Dict[str, CompletionChecker] = {}

    def resolve(self, checker_ref: str) -> CompletionChecker:
        ref = str(checker_ref or "").strip()
        if not ref:
            raise ValueError("completion checker_ref is required")
        cached = self._cache.get(ref)
        if cached is not None:
            return cached
        if ":" in ref:
            module_name, symbol_name = ref.split(":", 1)
        else:
            module_name, symbol_name = ref.rsplit(".", 1)
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
        instance = symbol() if inspect.isclass(symbol) else symbol
        if not hasattr(instance, "check"):
            raise TypeError(f"Resolved completion checker '{ref}' does not implement check(context)")
        self._cache[ref] = instance
        return instance


class BookingLinkCompletionChecker:
    def check(self, context: Dict[str, Any]) -> CompletionCheckResult:
        from infrastructure.db.repositories import PostgresBookingLinkJobRepository

        bot_id = str(context.get("bot_id") or "").strip()
        job_id = str(context.get("linked_job_id") or "").strip()
        if not bot_id or not job_id:
            return CompletionCheckResult(status="error", error="Missing bot_id or linked_job_id for booking_link completion")

        repo = PostgresBookingLinkJobRepository()
        job = repo.get(bot_id, job_id)
        if not job:
            return CompletionCheckResult(status="error", error=f"Booking link job not found: {job_id}")

        raw_status = _normalize_status(getattr(job, "status", ""))
        mapped = _map_child_status(raw_status, context)
        links = list(getattr(job, "links", None) or [])
        details = {"child_status": raw_status, "links_count": len(links)}

        if mapped == "done":
            return CompletionCheckResult(status="done", details=details)
        if mapped == "error":
            return CompletionCheckResult(
                status="error",
                details=details,
                error=str(getattr(job, "error", "") or "").strip() or f"Booking link job failed: {raw_status}",
            )
        return CompletionCheckResult(status="running", details=details)


class MenuExtractionCompletionChecker:
    def check(self, context: Dict[str, Any]) -> CompletionCheckResult:
        from infrastructure.db.repositories import PostgresAssetExtractionJobRepository

        job_id = str(context.get("linked_job_id") or "").strip()
        if not job_id:
            return CompletionCheckResult(status="error", error="Missing linked_job_id for menu_extraction completion")

        repo = PostgresAssetExtractionJobRepository()
        job = repo.get_job(job_id)
        if not job:
            return CompletionCheckResult(status="error", error=f"Menu extraction job not found: {job_id}")

        raw_status = _normalize_status(getattr(job, "status", ""))
        mapped = _map_child_status(raw_status, context)
        discovered = int(getattr(job, "assets_discovered", 0) or 0)
        created = int(getattr(job, "assets_created", 0) or 0)
        details = {
            "child_status": raw_status,
            "assets_discovered": discovered,
            "assets_created": created,
        }

        if mapped == "done":
            return CompletionCheckResult(status="done", details=details)
        if mapped == "error":
            return CompletionCheckResult(
                status="error",
                details=details,
                error=str(getattr(job, "error", "") or "").strip() or f"Menu extraction job failed: {raw_status}",
            )
        return CompletionCheckResult(status="running", details=details)
