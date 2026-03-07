from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from domain.interfaces import JobResult
from domain.platform_profiles import (
    RESERVATION_PLATFORM_CONFIG,
    get_reservation_url_rule,
    normalize_reservation_links,
)
from infrastructure.db.repositories import PostgresBotRepository


def _normalize_http_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return url


def _host_in_allowed_domains(host: str, allowed_domains: Iterable[str]) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    for domain in allowed_domains:
        d = str(domain or "").strip().lower()
        if not d:
            continue
        if h == d or h.endswith("." + d):
            return True
    return False


def pick_reservation_url_candidate(urls: List[str], rule: Dict[str, Any]) -> Optional[str]:
    """
    Pick best reservation URL from discovered/crawled URLs using config-only rule.
    Returns None if no candidate matches the configured constraints.
    """
    if not isinstance(rule, dict):
        return None
    allowed_domains = rule.get("allowed_domains") if isinstance(rule.get("allowed_domains"), list) else []
    include_patterns = rule.get("include_path_patterns") if isinstance(rule.get("include_path_patterns"), list) else []
    exclude_patterns = rule.get("exclude_path_patterns") if isinstance(rule.get("exclude_path_patterns"), list) else []
    keyword_scores = rule.get("path_keyword_scores") if isinstance(rule.get("path_keyword_scores"), dict) else {}

    best: Tuple[int, int, str] | None = None  # score, -length, url
    for raw in urls:
        url = _normalize_http_url(raw)
        if not url:
            continue
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or "/"
        path_lower = path.lower()
        if allowed_domains and not _host_in_allowed_domains(host, allowed_domains):
            continue

        if include_patterns and not any(re.search(str(pat), path, re.IGNORECASE) for pat in include_patterns):
            continue
        if exclude_patterns and any(re.search(str(pat), path, re.IGNORECASE) for pat in exclude_patterns):
            continue

        score = 0
        for key, val in keyword_scores.items():
            key_str = str(key or "").strip().lower()
            if not key_str:
                continue
            if key_str in path_lower:
                try:
                    score += int(val)
                except (TypeError, ValueError):
                    score += 0

        candidate = (score, -len(url), url)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def build_base_reservation_url(raw_url: str, rule: Dict[str, Any]) -> Optional[str]:
    """
    Build canonical base reservation URL from a shop URL.
    Optional rule key: base_path_pattern (regex; first capture group is used if present).
    """
    url = _normalize_http_url(raw_url)
    if not url:
        return None
    parsed = urlparse(url)
    path = parsed.path or "/"
    base_path = path

    base_path_pattern = str(rule.get("base_path_pattern") or "").strip() if isinstance(rule, dict) else ""
    if base_path_pattern:
        match = re.search(base_path_pattern, path, re.IGNORECASE)
        if match:
            if match.lastindex:
                base_path = str(match.group(1) or match.group(0) or path)
            else:
                base_path = str(match.group(0) or path)

    base_path = base_path or "/"
    if not base_path.startswith("/"):
        base_path = "/" + base_path
    if base_path != "/" and not base_path.endswith("/"):
        base_path += "/"

    return urlunparse((parsed.scheme, parsed.netloc, base_path, "", "", ""))


class ReservationUrlRunner:
    """Fill canonical reservation URL map for configured platform, only when missing."""

    def run(self, context: Dict[str, Any]) -> JobResult:
        bot_id = str(context.get("bot_id") or "").strip()
        if not bot_id:
            return JobResult(status="error", error="Missing bot_id for reservation_url runner")

        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        if not bot:
            return JobResult(status="error", error=f"Bot not found: {bot_id}")

        widget_config: Dict[str, Any] = {}
        raw_widget = getattr(bot, "widget_config", None)
        if raw_widget and str(raw_widget).strip():
            try:
                parsed = json.loads(raw_widget)
                if isinstance(parsed, dict):
                    widget_config = parsed
            except Exception:
                widget_config = {}

        platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
        if not platform_id:
            return JobResult(status="done", output={"status": "skipped", "reason": "missing_reservation_platform"})
        if platform_id not in RESERVATION_PLATFORM_CONFIG:
            return JobResult(status="done", output={"status": "skipped", "reason": "unknown_platform"})

        current_links = normalize_reservation_links(widget_config)
        current = str(current_links.get(platform_id) or "").strip()
        if current:
            return JobResult(status="done", output={"status": "skipped", "reason": "url_already_present"})

        rule = get_reservation_url_rule(platform_id)
        if not rule:
            return JobResult(status="done", output={"status": "skipped", "reason": "missing_platform_rule"})

        assignment_mode = str(rule.get("assignment_mode") or "candidate").strip().lower()
        assignment_mode = assignment_mode if assignment_mode in ("candidate", "base_url") else "candidate"

        if assignment_mode == "base_url":
            source_key = str(rule.get("base_url_source_key") or "root_url").strip() or "root_url"
            source_url = str(context.get(source_key) or "").strip()
            if not source_url:
                raw_urls = [str(u or "").strip() for u in list(context.get("crawled_urls") or []) if str(u or "").strip()]
                source_url = raw_urls[0] if raw_urls else ""
            base_candidate = build_base_reservation_url(source_url, rule)
            if not base_candidate:
                return JobResult(status="done", output={"status": "skipped", "reason": "no_base_url_found"})

            current_links[platform_id] = base_candidate
            widget_config["reservationLinks"] = dict(current_links)
            widget_config["reservation_links"] = dict(current_links)
            widget_key, _ = RESERVATION_PLATFORM_CONFIG[platform_id]
            widget_config[widget_key] = base_candidate
            bot_repo.update_widget_config(bot_id, json.dumps(widget_config, ensure_ascii=False))
            return JobResult(
                status="done",
                output={
                    "status": "updated",
                    "platform_id": platform_id,
                    "reservation_url": base_candidate,
                    "assignment_mode": "base_url",
                },
            )

        raw_urls = list(context.get("crawled_urls") or [])
        root_url = str(context.get("root_url") or "").strip()
        if root_url:
            raw_urls.append(root_url)
        candidate = pick_reservation_url_candidate(raw_urls, rule)
        if not candidate:
            return JobResult(status="done", output={"status": "skipped", "reason": "no_candidate_found"})

        current_links[platform_id] = candidate
        widget_config["reservationLinks"] = dict(current_links)
        widget_config["reservation_links"] = dict(current_links)
        widget_key, _ = RESERVATION_PLATFORM_CONFIG[platform_id]
        widget_config[widget_key] = candidate
        bot_repo.update_widget_config(bot_id, json.dumps(widget_config, ensure_ascii=False))

        return JobResult(
            status="done",
            output={
                "status": "updated",
                "platform_id": platform_id,
                "reservation_url": candidate,
                "assignment_mode": "candidate",
            },
        )
