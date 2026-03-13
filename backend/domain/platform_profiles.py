"""
Platform-specific profiles for restaurant booking platforms and other domains.

All platform config (Tabelog, HotPepper, TableCheck, etc.) is loaded from
config/platform_profiles.yml. Edit that file to add or change platforms -
no code changes needed.

Helpers (get_reservation_config_from_widget, get_suggested_messages_for_widget, etc.)
read from the active profile. Web, Line, and Instagram all use these; each channel
renders the result in its own UI (quick replies, flex buttons, etc.).
"""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

# Path to config file (backend/config/platform_profiles.yml)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "platform_profiles.yml"


def _load_platform_config() -> Dict[str, Any]:
    """Load platform profiles from YAML. Returns empty dict if file missing or invalid."""
    if not _CONFIG_PATH.exists():
        logger.warning("Platform config not found: %s", _CONFIG_PATH)
        return {}
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception("Failed to load platform config from %s: %s", _CONFIG_PATH, e)
        return {}


class ConfigValidationError(RuntimeError):
    pass


def _require_dict(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Missing or invalid '{key}' in {_CONFIG_PATH}")
    return value


def _require_list(container: Dict[str, Any], key: str) -> List[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ConfigValidationError(f"Missing or invalid '{key}' in {_CONFIG_PATH}")
    return value


def _require_str(container: Dict[str, Any], key: str) -> str:
    value = str(container.get(key) or "").strip()
    if not value:
        raise ConfigValidationError(f"Missing or invalid '{key}' in {_CONFIG_PATH}")
    return value


def _is_valid_i18n_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        en = str(value.get("en") or "").strip()
        ja = str(value.get("ja") or "").strip()
        return bool(en or ja)
    return False


def _normalize_lang(lang: str) -> str:
    lang_key = (lang or "en").strip().lower()
    return "ja" if lang_key in ("ja", "jp") else "en"


_ALLOWED_CREATE_BOT_COMPONENTS = {
    "details",
    "source_urls",
    "additional_sources",
    "training_progress",
    "suggested_messages",
    "widget_design",
    "embed_install",
    "action_destination_url",
}
_CREATE_BOT_SCREEN_I18N_FIELDS = (
    "title",
    "subtitle",
    "field_label",
    "field_placeholder",
    "field_helper",
    "fallback_notice",
)


def _resolve_i18n_text(value: Any, *, lang: str = "en", fallback: str = "") -> str:
    if isinstance(value, str):
        return value.strip() or fallback
    if isinstance(value, dict):
        lang_key = _normalize_lang(lang)
        text = str(value.get(lang_key) or value.get("en") or value.get("ja") or "").strip()
        if text:
            return text
    return fallback


def _validate_platform_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict) or not cfg:
        raise ConfigValidationError(f"Config file is empty: {_CONFIG_PATH}")

    _require_dict(cfg, "platforms")
    reservation_platform_config = _require_dict(cfg, "reservation_platform_config")
    _require_dict(cfg, "default_asset_rules")
    datc = _require_dict(cfg, "default_asset_term_config")
    for key in (
        "generic_tokens",
        "asset_intent_terms",
        "visual_request_terms",
        "visual_request_many_terms",
        "visual_suppress_terms",
    ):
        _require_list(datc, key)

    dar = _require_dict(cfg, "default_asset_rules")
    _require_str(dar, "marker_rule")
    _require_str(dar, "evidence_template")
    _require_dict(cfg, "default_rag_instruction")
    _require_dict(cfg, "default_menu_texts")
    _require_list(cfg, "default_menu_keywords")
    _require_list(cfg, "default_menu_category_order")
    _require_list(cfg, "default_post_crawl_jobs")
    _require_str(cfg, "line_menu_quick_payload")
    _require_str(cfg, "line_menu_page_payload_prefix")
    _require_str(cfg, "instagram_menu_quick_payload")
    _require_str(cfg, "instagram_menu_page_payload_prefix")
    welcome_messages = _require_dict(cfg, "welcome_messages")
    for channel_key in ("web", "line"):
        channel_messages = _require_dict(welcome_messages, channel_key)
        if not _is_valid_i18n_text(channel_messages):
            raise ConfigValidationError(
                f"Missing or invalid 'welcome_messages.{channel_key}' in {_CONFIG_PATH}"
            )
    support_messages = _require_dict(cfg, "support_messages")
    support_fields = {
        "web": (
            "requested",
            "disabled",
            "modal_title",
            "modal_subtitle",
            "email_placeholder",
            "details_label",
            "details_placeholder",
            "cancel_button",
            "submit_button",
            "invalid_email",
            "submit_failed",
            "submit_success",
        ),
        "line": (
            "prompt",
            "cancel_ack",
            "escalation_ack",
            "takeover_ack",
            "resolved_ack",
            "email_details_no_message",
        ),
        "instagram": (
            "prompt",
            "cancel_ack",
            "escalation_ack",
            "takeover_ack",
            "resolved_ack",
            "email_details_no_message",
        ),
    }
    for channel_key, fields in support_fields.items():
        channel_support = _require_dict(support_messages, channel_key)
        for field in fields:
            if not _is_valid_i18n_text(channel_support.get(field)):
                raise ConfigValidationError(
                    f"Missing or invalid 'support_messages.{channel_key}.{field}' in {_CONFIG_PATH}"
                )
    line_ux = _require_dict(cfg, "line_ux")
    rich_menu = _require_dict(line_ux, "rich_menu")
    _require_dict(rich_menu, "chat_bar_text")
    _require_dict(rich_menu, "actions")
    _require_dict(rich_menu, "layouts")
    design_profile = line_ux.get("design_profile")
    if design_profile is not None and not isinstance(design_profile, dict):
        raise ConfigValidationError(f"Invalid 'line_ux.design_profile' in {_CONFIG_PATH}")
    if not isinstance(line_ux.get("cancel_keywords"), list):
        raise ConfigValidationError(f"Missing or invalid 'line_ux.cancel_keywords' in {_CONFIG_PATH}")
    dashboard = _require_dict(cfg, "dashboard")
    overview_sections = _require_list(dashboard, "overview_setup_sections")
    seen_section_ids = set()
    for idx, section in enumerate(overview_sections):
        if not isinstance(section, dict):
            raise ConfigValidationError(f"Invalid dashboard.overview_setup_sections[{idx}] in {_CONFIG_PATH}")
        section_id = str(section.get("id") or "").strip()
        route = str(section.get("route") or "").strip()
        status_source = str(section.get("status_source") or "").strip()
        if not section_id:
            raise ConfigValidationError(f"Missing dashboard.overview_setup_sections[{idx}].id in {_CONFIG_PATH}")
        if section_id in seen_section_ids:
            raise ConfigValidationError(f"Duplicate dashboard.overview_setup_sections id '{section_id}' in {_CONFIG_PATH}")
        seen_section_ids.add(section_id)
        if not _is_valid_i18n_text(section.get("label")):
            raise ConfigValidationError(
                f"Missing or invalid dashboard.overview_setup_sections[{idx}].label in {_CONFIG_PATH}"
            )
        if not route:
            raise ConfigValidationError(f"Missing dashboard.overview_setup_sections[{idx}].route in {_CONFIG_PATH}")
        if not status_source:
            raise ConfigValidationError(
                f"Missing dashboard.overview_setup_sections[{idx}].status_source in {_CONFIG_PATH}"
            )
        required_tabs = section.get("required_knowledge_tabs")
        if required_tabs is not None and not isinstance(required_tabs, list):
            raise ConfigValidationError(
                f"Invalid dashboard.overview_setup_sections[{idx}].required_knowledge_tabs in {_CONFIG_PATH}"
            )
    create_bot_flow = _require_dict(dashboard, "create_bot_flow")
    step_groups = _require_list(create_bot_flow, "step_groups")
    known_reservation_platform_ids = {
        str(platform_id or "").strip().lower()
        for platform_id in reservation_platform_config.keys()
        if str(platform_id or "").strip()
    }
    seen_step_group_ids = set()
    for idx, group in enumerate(step_groups):
        if not isinstance(group, dict):
            raise ConfigValidationError(f"Invalid dashboard.create_bot_flow.step_groups[{idx}] in {_CONFIG_PATH}")
        group_id = str(group.get("id") or "").strip()
        if not group_id:
            raise ConfigValidationError(
                f"Missing dashboard.create_bot_flow.step_groups[{idx}].id in {_CONFIG_PATH}"
            )
        if group_id in seen_step_group_ids:
            raise ConfigValidationError(
                f"Duplicate dashboard.create_bot_flow.step_groups id '{group_id}' in {_CONFIG_PATH}"
            )
        seen_step_group_ids.add(group_id)
        if not _is_valid_i18n_text(group.get("label")):
            raise ConfigValidationError(
                f"Missing or invalid dashboard.create_bot_flow.step_groups[{idx}].label in {_CONFIG_PATH}"
            )
        if not _is_valid_i18n_text(group.get("description")):
            raise ConfigValidationError(
                f"Missing or invalid dashboard.create_bot_flow.step_groups[{idx}].description in {_CONFIG_PATH}"
            )

    screen_definitions = _require_dict(create_bot_flow, "screen_definitions")
    seen_screen_paths = set()
    for screen_id, screen in screen_definitions.items():
        normalized_screen_id = str(screen_id or "").strip()
        if not normalized_screen_id:
            raise ConfigValidationError(
                f"Missing dashboard.create_bot_flow.screen_definitions id in {_CONFIG_PATH}"
            )
        if not isinstance(screen, dict):
            raise ConfigValidationError(
                f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id} in {_CONFIG_PATH}"
            )
        path = str(screen.get("path") or "").strip()
        step_group = str(screen.get("step_group") or "").strip()
        component = str(screen.get("component") or "").strip()
        if step_group not in seen_step_group_ids:
            raise ConfigValidationError(
                f"Unknown dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.step_group in {_CONFIG_PATH}"
            )
        if component not in _ALLOWED_CREATE_BOT_COMPONENTS:
            raise ConfigValidationError(
                f"Unknown dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.component in {_CONFIG_PATH}"
            )
        if normalized_screen_id != "details" and not path:
            raise ConfigValidationError(
                f"Missing dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.path in {_CONFIG_PATH}"
            )
        if path:
            if path in seen_screen_paths:
                raise ConfigValidationError(
                    f"Duplicate dashboard.create_bot_flow.screen_definitions path '{path}' in {_CONFIG_PATH}"
                )
            seen_screen_paths.add(path)
        visibility = screen.get("visibility")
        if visibility is not None:
            if not isinstance(visibility, dict):
                raise ConfigValidationError(
                    f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility in {_CONFIG_PATH}"
                )
            business_types = visibility.get("business_types")
            if business_types is not None and not isinstance(business_types, list):
                raise ConfigValidationError(
                    f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility.business_types in {_CONFIG_PATH}"
                )
            requires_platform = visibility.get("requires_selected_reservation_platform")
            if requires_platform is not None and not isinstance(requires_platform, bool):
                raise ConfigValidationError(
                    f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility.requires_selected_reservation_platform in {_CONFIG_PATH}"
                )
            reservation_platform_ids = visibility.get("reservation_platform_ids")
            if reservation_platform_ids is not None:
                if not isinstance(reservation_platform_ids, list):
                    raise ConfigValidationError(
                        f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility.reservation_platform_ids in {_CONFIG_PATH}"
                    )
                for raw_platform_id in reservation_platform_ids:
                    normalized_platform_id = str(raw_platform_id or "").strip().lower()
                    if not normalized_platform_id:
                        raise ConfigValidationError(
                            f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility.reservation_platform_ids in {_CONFIG_PATH}"
                        )
                    if normalized_platform_id not in known_reservation_platform_ids:
                        raise ConfigValidationError(
                            f"Unknown dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.visibility.reservation_platform_ids platform '{normalized_platform_id}' in {_CONFIG_PATH}"
                        )
        if component == "action_destination_url":
            action_key = str(screen.get("action_key") or "").strip().lower()
            if not action_key:
                raise ConfigValidationError(
                    f"Missing dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.action_key in {_CONFIG_PATH}"
                )
            for field_name in ("title", "subtitle"):
                if not _is_valid_i18n_text(screen.get(field_name)):
                    raise ConfigValidationError(
                        f"Missing or invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.{field_name} in {_CONFIG_PATH}"
                    )
            for field_name in _CREATE_BOT_SCREEN_I18N_FIELDS[2:]:
                field_value = screen.get(field_name)
                if field_value is not None and not _is_valid_i18n_text(field_value):
                    raise ConfigValidationError(
                        f"Invalid dashboard.create_bot_flow.screen_definitions.{normalized_screen_id}.{field_name} in {_CONFIG_PATH}"
                    )

    screen_order = _require_list(create_bot_flow, "screen_order")
    seen_order_ids = set()
    for idx, screen_id in enumerate(screen_order):
        normalized_screen_id = str(screen_id or "").strip()
        if not normalized_screen_id:
            raise ConfigValidationError(
                f"Missing dashboard.create_bot_flow.screen_order[{idx}] in {_CONFIG_PATH}"
            )
        if normalized_screen_id in seen_order_ids:
            raise ConfigValidationError(
                f"Duplicate dashboard.create_bot_flow.screen_order entry '{normalized_screen_id}' in {_CONFIG_PATH}"
            )
        seen_order_ids.add(normalized_screen_id)
        if normalized_screen_id not in screen_definitions:
            raise ConfigValidationError(
                f"dashboard.create_bot_flow.screen_order references unknown screen '{normalized_screen_id}' in {_CONFIG_PATH}"
            )

    defaults = _require_dict(cfg, "defaults")
    jobs_defaults = _require_dict(defaults, "jobs")
    if not isinstance(jobs_defaults.get("topic_extraction_enabled"), bool):
        raise ConfigValidationError(f"Missing or invalid 'defaults.jobs.topic_extraction_enabled' in {_CONFIG_PATH}")
    _require_str(defaults, "source_language")
    _require_list(defaults, "knowledge_tabs")
    menu_defaults = _require_dict(defaults, "menu")
    _require_dict(menu_defaults, "category_aliases")
    _require_dict(menu_defaults, "view_all_url_tokens")
    prompts_defaults = _require_dict(defaults, "prompts")
    deterministic = _require_dict(prompts_defaults, "deterministic")
    _require_dict(deterministic, "default_business_name")
    _require_dict(deterministic, "personality_with_business_type")
    _require_dict(deterministic, "personality_without_business_type")
    section_titles = _require_dict(deterministic, "section_titles")
    _require_dict(section_titles, "personality")
    _require_dict(section_titles, "response_rules")
    response_rules = _require_dict(deterministic, "response_rules")
    _require_list(response_rules, "en")
    _require_list(response_rules, "ja")

    generation = _require_dict(prompts_defaults, "generation")
    for key in (
        "model",
        "meta_prompt_en",
        "meta_prompt_ja",
        "rag_meta_prompt_en",
        "rag_meta_prompt_ja",
        "standard_response_rules_en",
        "standard_response_rules_ja",
        "identity_instruction_en",
    ):
        _require_str(generation, key)

    fallback = _require_dict(prompts_defaults, "fallback")
    for key in (
        "personality_title_en",
        "about_title_en",
        "personality_title_ja",
        "about_title_ja",
        "personality_en",
        "about_en",
        "personality_ja",
        "about_ja",
    ):
        _require_str(fallback, key)

    functions_defaults = _require_dict(defaults, "functions")
    _require_dict(functions_defaults, "suggested_type_to_function")
    assets_defaults = _require_dict(defaults, "assets")
    _require_list(assets_defaults, "base_stopwords")
    rag_import_defaults = _require_dict(defaults, "rag_import")
    for key in (
        "batch_size",
        "max_embedding_requests_per_min",
        "busy_retry_max_attempts",
        "busy_retry_initial_backoff_sec",
        "busy_retry_max_backoff_sec",
    ):
        if key not in rag_import_defaults:
            raise ConfigValidationError(f"Missing defaults.rag_import.{key} in {_CONFIG_PATH}")
    try:
        if int(rag_import_defaults.get("batch_size")) <= 0:
            raise ValueError
        if int(rag_import_defaults.get("max_embedding_requests_per_min")) <= 0:
            raise ValueError
        if int(rag_import_defaults.get("busy_retry_max_attempts")) <= 0:
            raise ValueError
        if float(rag_import_defaults.get("busy_retry_initial_backoff_sec")) <= 0:
            raise ValueError
        if float(rag_import_defaults.get("busy_retry_max_backoff_sec")) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigValidationError(f"Invalid defaults.rag_import values in {_CONFIG_PATH}")
    logging_defaults = _require_dict(defaults, "logging")
    crawl_logging = _require_dict(logging_defaults, "crawl")
    emit_events = _require_dict(crawl_logging, "emit_events")
    for key in ("stage", "progress", "fetch", "result", "auth", "gcs_prefix", "error"):
        if not isinstance(emit_events.get(key), bool):
            raise ConfigValidationError(
                f"Missing or invalid 'defaults.logging.crawl.emit_events.{key}' in {_CONFIG_PATH}"
            )
    previews = _require_dict(crawl_logging, "previews")
    for key in ("single_page_content", "crawl_raw", "crawl_upload"):
        if not isinstance(previews.get(key), bool):
            raise ConfigValidationError(
                f"Missing or invalid 'defaults.logging.crawl.previews.{key}' in {_CONFIG_PATH}"
            )

    job_pipeline = _require_dict(cfg, "job_pipeline")
    pipeline_defaults = _require_dict(job_pipeline, "defaults")
    _require_str(pipeline_defaults, "on_failure")
    _require_str(pipeline_defaults, "task_queue")
    run_messages = _require_dict(pipeline_defaults, "run_messages")
    for key in ("queued", "running", "paused", "resume_requested", "done", "error"):
        if not _is_valid_i18n_text(run_messages.get(key)):
            raise ConfigValidationError(
                f"Missing or invalid 'job_pipeline.defaults.run_messages.{key}' in {_CONFIG_PATH}"
            )
    jobs_catalog = _require_dict(job_pipeline, "jobs")
    workflows = _require_dict(job_pipeline, "workflows")
    workflow_default = _require_list(workflows, "default")
    for job_id, entry in jobs_catalog.items():
        if not isinstance(entry, dict):
            raise ConfigValidationError(f"Invalid job_pipeline.jobs.{job_id} in {_CONFIG_PATH}")
        _require_str(entry, "runner_ref")
        if "progress_weight" not in entry:
            raise ConfigValidationError(
                f"Missing job_pipeline.jobs.{job_id}.progress_weight in {_CONFIG_PATH}"
            )
        try:
            weight = int(entry.get("progress_weight"))
        except (TypeError, ValueError):
            raise ConfigValidationError(
                f"Invalid job_pipeline.jobs.{job_id}.progress_weight in {_CONFIG_PATH}"
            )
        if weight <= 0:
            raise ConfigValidationError(
                f"Invalid job_pipeline.jobs.{job_id}.progress_weight in {_CONFIG_PATH}"
            )
        if "running_progress_pct" not in entry:
            raise ConfigValidationError(
                f"Missing job_pipeline.jobs.{job_id}.running_progress_pct in {_CONFIG_PATH}"
            )
        try:
            running_pct = int(entry.get("running_progress_pct"))
        except (TypeError, ValueError):
            raise ConfigValidationError(
                f"Invalid job_pipeline.jobs.{job_id}.running_progress_pct in {_CONFIG_PATH}"
            )
        if running_pct < 0 or running_pct > 100:
            raise ConfigValidationError(
                f"Invalid job_pipeline.jobs.{job_id}.running_progress_pct in {_CONFIG_PATH}"
            )
        progress_messages = _require_dict(entry, "progress_messages")
        for status_key in ("queued", "running", "paused", "done", "error"):
            if not _is_valid_i18n_text(progress_messages.get(status_key)):
                raise ConfigValidationError(
                    f"Missing or invalid 'job_pipeline.jobs.{job_id}.progress_messages.{status_key}' in {_CONFIG_PATH}"
                )
        completion = entry.get("completion")
        if completion is not None:
            if not isinstance(completion, dict):
                raise ConfigValidationError(
                    f"Invalid job_pipeline.jobs.{job_id}.completion in {_CONFIG_PATH}"
                )
            enabled = bool(completion.get("enabled"))
            if enabled:
                checker_ref = str(completion.get("checker_ref") or "").strip()
                if not checker_ref:
                    raise ConfigValidationError(
                        f"Missing job_pipeline.jobs.{job_id}.completion.checker_ref in {_CONFIG_PATH}"
                    )
                try:
                    poll_interval = int(completion.get("poll_interval_sec"))
                    max_wait = int(completion.get("max_wait_sec"))
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"Invalid completion poll/max wait config for job_pipeline.jobs.{job_id} in {_CONFIG_PATH}"
                    )
                if poll_interval <= 0 or max_wait <= 0:
                    raise ConfigValidationError(
                        f"Invalid completion poll/max wait config for job_pipeline.jobs.{job_id} in {_CONFIG_PATH}"
                    )
                status_map = completion.get("status_map")
                if not isinstance(status_map, dict):
                    raise ConfigValidationError(
                        f"Missing job_pipeline.jobs.{job_id}.completion.status_map in {_CONFIG_PATH}"
                    )
                for map_key in ("running", "done", "error"):
                    group = status_map.get(map_key)
                    if not isinstance(group, list) or not [str(v).strip() for v in group if str(v).strip()]:
                        raise ConfigValidationError(
                            f"Missing or invalid job_pipeline.jobs.{job_id}.completion.status_map.{map_key} in {_CONFIG_PATH}"
                        )
    for item in workflow_default:
        jid = str(item or "").strip()
        if jid and jid not in jobs_catalog:
            raise ConfigValidationError(
                f"job_pipeline.workflows.default references unknown job id '{jid}' in {_CONFIG_PATH}"
            )
    platform_overrides = workflows.get("platform_overrides")
    if platform_overrides is not None and not isinstance(platform_overrides, dict):
        raise ConfigValidationError(f"Invalid job_pipeline.workflows.platform_overrides in {_CONFIG_PATH}")
    if isinstance(platform_overrides, dict):
        for platform_id, steps in platform_overrides.items():
            if str(platform_id or "").strip().lower() not in cfg.get("reservation_platform_config", {}):
                raise ConfigValidationError(
                    f"job_pipeline.workflows.platform_overrides has unknown platform '{platform_id}' in {_CONFIG_PATH}"
                )
            if not isinstance(steps, list):
                raise ConfigValidationError(
                    f"Invalid workflow override list for platform '{platform_id}' in {_CONFIG_PATH}"
                )
            for item in steps:
                jid = str(item or "").strip()
                if jid and jid not in jobs_catalog:
                    raise ConfigValidationError(
                        f"job_pipeline.workflows.platform_overrides.{platform_id} references unknown job id '{jid}' in {_CONFIG_PATH}"
                    )

    gates = job_pipeline.get("gates")
    if gates is not None and not isinstance(gates, dict):
        raise ConfigValidationError(f"Invalid job_pipeline.gates in {_CONFIG_PATH}")
    if isinstance(gates, dict):
        for gate_id, gate_entry in gates.items():
            if not isinstance(gate_entry, dict):
                raise ConfigValidationError(f"Invalid job_pipeline.gates.{gate_id} in {_CONFIG_PATH}")
            _require_str(gate_entry, "step_id")

    reservation_url_rules = _require_dict(cfg, "reservation_url_rules")
    for platform_id, rule in reservation_url_rules.items():
        if str(platform_id or "").strip().lower() not in cfg.get("reservation_platform_config", {}):
            raise ConfigValidationError(
                f"reservation_url_rules has unknown platform '{platform_id}' in {_CONFIG_PATH}"
            )
        if not isinstance(rule, dict):
            raise ConfigValidationError(f"Invalid reservation_url_rules.{platform_id} in {_CONFIG_PATH}")
        assignment_mode = str(rule.get("assignment_mode") or "candidate").strip().lower()
        if assignment_mode not in ("candidate", "base_url"):
            raise ConfigValidationError(
                f"Invalid reservation_url_rules.{platform_id}.assignment_mode in {_CONFIG_PATH}"
            )
        if "base_url_source_key" in rule and not str(rule.get("base_url_source_key") or "").strip():
            raise ConfigValidationError(
                f"Invalid reservation_url_rules.{platform_id}.base_url_source_key in {_CONFIG_PATH}"
            )
        if "base_path_pattern" in rule and not str(rule.get("base_path_pattern") or "").strip():
            raise ConfigValidationError(
                f"Invalid reservation_url_rules.{platform_id}.base_path_pattern in {_CONFIG_PATH}"
            )
        _require_list(rule, "allowed_domains")


@dataclass
class PlatformProfile:
    """Defines crawling and extraction strategy for a specific domain."""

    domain_pattern: str
    """Regex pattern to match domain (e.g., 'hotpepper\\.jp', 'tabelog\\.com')."""

    include_paths: List[str] = field(default_factory=list)
    """List of regex patterns for URL paths to include in crawl.
    If empty, all paths on the domain are included (subject to exclude_paths)."""

    exclude_paths: List[str] = field(default_factory=list)
    """List of regex patterns for URL paths to exclude from crawl."""

    max_depth: int = 2
    """Maximum link-following depth from seed URL. 0 = don't follow links."""

    priority: int = 0
    """Tiebreaker if multiple profiles match the same domain."""

    strip_query_params: bool = False
    """If True, strip query parameters and fragments before URL matching and deduplication.
    Use for platforms where query params represent UI state (e.g., ?RDT=20260226 on HotPepper),
    not different content. Defaults to False (unknown domains keep all query params)."""

    # ─── Menu Extraction (future) ─────────────────────────────────────────
    menu_url_patterns: List[str] = field(default_factory=list)
    """Regex patterns for URLs containing menu/course data.
    To be used by menu extraction pipeline (not yet implemented)."""

    menu_extraction_rules: Optional[Dict[str, Any]] = None
    """Custom rules for extracting menu items, prices, descriptions.
    Format TBD based on extraction requirements.
    If None, default extraction logic applies."""

    # ─── Image Extraction (future) ────────────────────────────────────────
    image_extraction_enabled: bool = True
    """Whether to extract images from this platform."""

    image_url_patterns: List[str] = field(default_factory=list)
    """Regex patterns for URLs containing images (e.g., /photo/, /gallery/).
    If empty, images from all included URLs are extracted."""

    image_extraction_rules: Optional[Dict[str, Any]] = None
    """Custom rules for filtering/processing images.
    Can specify: min resolution, skip certain image types, alt text extraction, etc.
    If None, default image extraction logic applies."""

    # ─── Metadata ─────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata dict for any platform-specific data."""


# ═══════════════════════════════════════════════════════════════════════════
# LOAD FROM CONFIG (config/platform_profiles.yml)
# ═══════════════════════════════════════════════════════════════════════════


def _dict_to_platform_profile(domain_key: str, data: Dict[str, Any]) -> PlatformProfile:
    """Build PlatformProfile from YAML dict."""
    raw = data or {}
    return PlatformProfile(
        domain_pattern=str(raw.get("domain_pattern") or domain_key.replace(".", r"\.")),
        include_paths=list(raw.get("include_paths") or []),
        exclude_paths=list(raw.get("exclude_paths") or []),
        max_depth=int(raw.get("max_depth", 2)),
        priority=int(raw.get("priority", 0)),
        strip_query_params=bool(raw.get("strip_query_params", False)),
        menu_url_patterns=list(raw.get("menu_url_patterns") or []),
        menu_extraction_rules=raw.get("menu_extraction_rules") if isinstance(raw.get("menu_extraction_rules"), dict) else None,
        image_extraction_enabled=bool(raw.get("image_extraction_enabled", True)),
        image_url_patterns=list(raw.get("image_url_patterns") or []),
        image_extraction_rules=raw.get("image_extraction_rules") if isinstance(raw.get("image_extraction_rules"), dict) else None,
        metadata=dict(raw.get("metadata") or {}),
    )


def _build_platform_registry() -> tuple[
    Dict[str, PlatformProfile],
    Dict[str, Tuple[str, str]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, str]],
    Dict[str, Any],
    Dict[str, str],
    Dict[str, List[str]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[str],
    List[str],
    List[str],
    Dict[str, Any],
    str,
    str,
    str,
    str,
    Optional[Dict[str, Any]],
]:
    """Load config and build platform/profile defaults from YAML."""
    cfg = _load_platform_config()
    _validate_platform_config(cfg)
    profiles: Dict[str, PlatformProfile] = {}
    reservation_config: Dict[str, Tuple[str, str]] = {}
    default_suggested: List[Dict[str, Any]] = []
    welcome_messages: Dict[str, Dict[str, str]] = {}
    support_messages: Dict[str, Any] = {}
    default_asset_rules: Dict[str, str] = {}
    default_asset_term_config: Dict[str, List[str]] = {}

    if isinstance(cfg.get("default_asset_rules"), dict):
        dar = cfg["default_asset_rules"]
        marker = str(dar.get("marker_rule") or "").strip()
        evidence = str(dar.get("evidence_template") or "").strip()
        if marker:
            default_asset_rules["marker_rule"] = marker
        if evidence:
            default_asset_rules["evidence_template"] = evidence

    if isinstance(cfg.get("default_asset_term_config"), dict):
        datc = cfg["default_asset_term_config"]
        for key in ("generic_tokens", "asset_intent_terms", "visual_request_terms", "visual_request_many_terms", "visual_suppress_terms"):
            val = datc.get(key)
            if isinstance(val, list):
                default_asset_term_config[key] = [str(v).strip() for v in val if str(v).strip()]

    # Reservation platform mapping
    rpc = cfg.get("reservation_platform_config") or {}
    for pid, entry in rpc.items():
        if isinstance(entry, dict):
            wk = str(entry.get("widget_key") or "").strip()
            dk = str(entry.get("domain_key") or "").strip()
            if wk and dk:
                reservation_config[str(pid).strip().lower()] = (wk, dk)

    # Default suggested messages
    dsm = cfg.get("default_suggested_messages")
    if isinstance(dsm, list):
        default_suggested = [m for m in dsm if isinstance(m, dict)]

    raw_welcome_messages = cfg.get("welcome_messages")
    if isinstance(raw_welcome_messages, dict):
        for channel_key in ("web", "line"):
            channel_messages = raw_welcome_messages.get(channel_key)
            if not isinstance(channel_messages, dict):
                continue
            welcome_messages[channel_key] = {
                "en": _resolve_i18n_text(channel_messages, lang="en"),
                "ja": _resolve_i18n_text(channel_messages, lang="ja"),
            }

    raw_support_messages = cfg.get("support_messages")
    if isinstance(raw_support_messages, dict):
        support_messages = dict(raw_support_messages)

    # Platform profiles
    platforms = cfg.get("platforms") or {}
    if isinstance(platforms, dict):
        for domain_key, pdata in platforms.items():
            if isinstance(pdata, dict) and domain_key:
                profiles[str(domain_key).strip()] = _dict_to_platform_profile(domain_key, pdata)

    default_json = cfg.get("default_json_response_format") if isinstance(cfg.get("default_json_response_format"), dict) else None
    default_rag = cfg.get("default_rag_instruction") if isinstance(cfg.get("default_rag_instruction"), dict) else None
    default_menu_texts = cfg.get("default_menu_texts") if isinstance(cfg.get("default_menu_texts"), dict) else None
    default_menu_keywords = cfg.get("default_menu_keywords")
    default_menu_category_order = cfg.get("default_menu_category_order")
    default_post_crawl_jobs = cfg.get("default_post_crawl_jobs")
    defaults_cfg = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
    default_menu_request_pattern = str(cfg.get("default_menu_request_pattern") or "").strip()
    line_menu_payload = _require_str(cfg, "line_menu_quick_payload")
    line_menu_prefix = _require_str(cfg, "line_menu_page_payload_prefix")
    ig_menu_payload = _require_str(cfg, "instagram_menu_quick_payload")
    ig_menu_prefix = _require_str(cfg, "instagram_menu_page_payload_prefix")
    line_ux_cfg = cfg.get("line_ux") if isinstance(cfg.get("line_ux"), dict) else None

    return (
        profiles,
        reservation_config,
        default_suggested,
        welcome_messages,
        support_messages,
        default_asset_rules,
        default_asset_term_config,
        default_json,
        default_rag,
        default_menu_texts,
        default_menu_keywords if isinstance(default_menu_keywords, list) else [],
        default_menu_category_order if isinstance(default_menu_category_order, list) else [],
        default_post_crawl_jobs if isinstance(default_post_crawl_jobs, list) else [],
        defaults_cfg if isinstance(defaults_cfg, dict) else {},
        default_menu_request_pattern,
        line_menu_payload,
        line_menu_prefix,
        ig_menu_payload,
        ig_menu_prefix,
        line_ux_cfg,
    )


(
    PLATFORM_PROFILES,
    RESERVATION_PLATFORM_CONFIG,
    DEFAULT_SUGGESTED_MESSAGES,
    DEFAULT_WELCOME_MESSAGES,
    SUPPORT_MESSAGES_CONFIG,
    DEFAULT_ASSET_RULES,
    DEFAULT_ASSET_TERM_CONFIG,
    DEFAULT_JSON_RESPONSE_FORMAT,
    DEFAULT_RAG_INSTRUCTION,
    DEFAULT_MENU_TEXTS,
    DEFAULT_MENU_KEYWORDS,
    DEFAULT_MENU_CATEGORY_ORDER,
    DEFAULT_POST_CRAWL_JOBS,
    DEFAULTS_CONFIG,
    DEFAULT_MENU_REQUEST_PATTERN,
    LINE_MENU_QUICK_PAYLOAD,
    LINE_MENU_PAGE_PAYLOAD_PREFIX,
    INSTAGRAM_MENU_QUICK_PAYLOAD,
    INSTAGRAM_MENU_PAGE_PAYLOAD_PREFIX,
    LINE_UX_CONFIG,
) = _build_platform_registry()


def get_reservation_platforms_list(*, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Return list of reservation platforms from config (for dashboard dropdowns).
    No hardcoding: add platforms in platform_profiles.yml only.
    """
    lang = _normalize_lang(lang)
    cfg = _load_platform_config()
    rpc = cfg.get("reservation_platform_config") or {}
    out: List[Dict[str, Any]] = []
    for pid, (widget_key, domain_key) in RESERVATION_PLATFORM_CONFIG.items():
        label = pid  # fallback
        if domain_key in PLATFORM_PROFILES:
            profile = PLATFORM_PROFILES[domain_key]
            metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
            label = _resolve_i18n_text(metadata.get("service_name"), lang=lang, fallback=pid) or pid
        entry = rpc.get(pid) if isinstance(rpc, dict) else {}
        url_placeholder = str(entry.get("url_placeholder") or "").strip() if isinstance(entry, dict) else ""
        out.append({
            "id": pid,
            "widget_key": widget_key,
            "domain_key": domain_key,
            "label": label,
            "url_placeholder": url_placeholder,
        })
    return out


def get_dashboard_overview_setup_sections(*, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Return bot overview summary pill definitions from config.
    Sections can be gated by required_knowledge_tabs so the dashboard stays YAML-driven.
    """
    lang = _normalize_lang(lang)
    cfg = _load_platform_config()
    dashboard = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
    raw_sections = dashboard.get("overview_setup_sections")
    if not isinstance(raw_sections, list):
        return []

    sections: List[Dict[str, Any]] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            continue
        section_id = str(raw_section.get("id") or "").strip()
        route = str(raw_section.get("route") or "").strip()
        status_source = str(raw_section.get("status_source") or "").strip()
        label = _resolve_i18n_text(raw_section.get("label"), lang=lang, fallback=section_id)
        if not section_id or not route or not status_source or not label:
            continue
        required_tabs_raw = raw_section.get("required_knowledge_tabs")
        required_tabs = (
            [str(tab).strip().lower() for tab in required_tabs_raw if str(tab).strip()]
            if isinstance(required_tabs_raw, list)
            else []
        )
        sections.append({
            "id": section_id,
            "label": label,
            "route": route,
            "status_source": status_source,
            "required_knowledge_tabs": required_tabs,
        })
    return sections


def get_dashboard_create_bot_flow(*, lang: str = "en") -> Dict[str, Any]:
    """
    Return the config-driven create-bot flow definition with localized copy.
    """
    lang = _normalize_lang(lang)
    cfg = _load_platform_config()
    dashboard = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
    raw_flow = dashboard.get("create_bot_flow") if isinstance(dashboard.get("create_bot_flow"), dict) else {}

    raw_step_groups = raw_flow.get("step_groups") if isinstance(raw_flow.get("step_groups"), list) else []
    step_groups: List[Dict[str, Any]] = []
    for raw_group in raw_step_groups:
        if not isinstance(raw_group, dict):
            continue
        group_id = str(raw_group.get("id") or "").strip()
        if not group_id:
            continue
        step_groups.append(
            {
                "id": group_id,
                "label": _resolve_i18n_text(raw_group.get("label"), lang=lang, fallback=group_id),
                "description": _resolve_i18n_text(raw_group.get("description"), lang=lang, fallback=""),
            }
        )

    raw_defs = raw_flow.get("screen_definitions") if isinstance(raw_flow.get("screen_definitions"), dict) else {}
    screen_definitions: Dict[str, Dict[str, Any]] = {}
    for screen_id, raw_screen in raw_defs.items():
        normalized_screen_id = str(screen_id or "").strip()
        if not normalized_screen_id or not isinstance(raw_screen, dict):
            continue
        resolved: Dict[str, Any] = {
            "id": normalized_screen_id,
            "path": str(raw_screen.get("path") or "").strip(),
            "step_group": str(raw_screen.get("step_group") or "").strip(),
            "component": str(raw_screen.get("component") or "").strip(),
        }
        action_key = str(raw_screen.get("action_key") or "").strip().lower()
        if action_key:
            resolved["action_key"] = action_key
        visibility = raw_screen.get("visibility")
        if isinstance(visibility, dict):
            resolved["visibility"] = {
                "business_types": [
                    str(item).strip().lower() for item in visibility.get("business_types", []) if str(item).strip()
                ],
                "requires_selected_reservation_platform": bool(
                    visibility.get("requires_selected_reservation_platform")
                ),
                "reservation_platform_ids": [
                    str(item).strip().lower()
                    for item in visibility.get("reservation_platform_ids", [])
                    if str(item).strip()
                ],
            }
        for field_name in _CREATE_BOT_SCREEN_I18N_FIELDS:
            field_value = raw_screen.get(field_name)
            if field_value is not None:
                resolved[field_name] = _resolve_i18n_text(field_value, lang=lang, fallback="")
        screen_definitions[normalized_screen_id] = resolved

    screen_order = [
        str(item).strip()
        for item in (raw_flow.get("screen_order") if isinstance(raw_flow.get("screen_order"), list) else [])
        if str(item).strip()
    ]
    return {
        "step_groups": step_groups,
        "screen_definitions": screen_definitions,
        "screen_order": screen_order,
    }


def get_defaults_config() -> Dict[str, Any]:
    return dict(DEFAULTS_CONFIG) if isinstance(DEFAULTS_CONFIG, dict) else {}


def get_default_post_crawl_jobs() -> List[str]:
    jobs = [str(j).strip().lower() for j in DEFAULT_POST_CRAWL_JOBS if str(j).strip()]
    if not is_topic_extraction_enabled():
        jobs = [j for j in jobs if j != "topic_extraction"]
    return jobs


def get_job_pipeline_config() -> Dict[str, Any]:
    cfg = _load_platform_config()
    raw = cfg.get("job_pipeline")
    return dict(raw) if isinstance(raw, dict) else {}


def get_job_pipeline_defaults() -> Dict[str, Any]:
    pipeline = get_job_pipeline_config()
    defaults = pipeline.get("defaults")
    return dict(defaults) if isinstance(defaults, dict) else {}


def get_job_pipeline_task_queue() -> str:
    defaults = get_job_pipeline_defaults()
    return str(defaults.get("task_queue") or "").strip()


def get_job_pipeline_jobs() -> Dict[str, Dict[str, Any]]:
    pipeline = get_job_pipeline_config()
    raw = pipeline.get("jobs")
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for job_id, entry in raw.items():
        jid = str(job_id or "").strip()
        if not jid or not isinstance(entry, dict):
            continue
        out[jid] = dict(entry)
    return out


def get_job_pipeline_default_failure_policy() -> str:
    pipeline = get_job_pipeline_config()
    defaults = pipeline.get("defaults")
    if isinstance(defaults, dict):
        policy = str(defaults.get("on_failure") or "").strip().lower()
        if policy in ("continue", "stop"):
            return policy
    return "continue"


def get_job_pipeline_workflow(
    widget_config: Optional[Dict[str, Any]],
    *,
    workflow_id: str = "default",
) -> List[str]:
    """Resolve workflow steps from YAML by platform override, falling back to default workflow."""
    pipeline = get_job_pipeline_config()
    workflows = pipeline.get("workflows") if isinstance(pipeline.get("workflows"), dict) else {}
    default_steps_raw = workflows.get(workflow_id)
    default_steps = [str(v).strip() for v in default_steps_raw if str(v).strip()] if isinstance(default_steps_raw, list) else []
    if not isinstance(widget_config, dict):
        return default_steps

    platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
    if not platform_id:
        cfg = get_reservation_config_from_widget(widget_config)
        platform_id = str((cfg or {}).get("platform_id") or "").strip().lower()
    if not platform_id:
        return default_steps

    platform_overrides = workflows.get("platform_overrides") if isinstance(workflows.get("platform_overrides"), dict) else {}
    steps_raw = platform_overrides.get(platform_id)
    if not isinstance(steps_raw, list):
        return default_steps
    resolved = [str(v).strip() for v in steps_raw if str(v).strip()]
    return resolved if resolved else default_steps


def get_job_pipeline_gates() -> Dict[str, Dict[str, Any]]:
    pipeline = get_job_pipeline_config()
    gates = pipeline.get("gates")
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(gates, dict):
        return out
    for gate_id, entry in gates.items():
        gid = str(gate_id or "").strip()
        if gid and isinstance(entry, dict):
            out[gid] = dict(entry)
    return out


def get_default_source_language() -> str:
    defaults = get_defaults_config()
    return str(defaults.get("source_language") or "").strip().lower()


def get_rag_import_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    rag_import = defaults.get("rag_import")
    return dict(rag_import) if isinstance(rag_import, dict) else {}


def get_rag_import_batch_size() -> int:
    cfg = get_rag_import_config()
    return int(cfg.get("batch_size"))


def get_rag_import_max_embedding_requests_per_min() -> int:
    cfg = get_rag_import_config()
    return int(cfg.get("max_embedding_requests_per_min"))


def get_rag_import_busy_retry_max_attempts() -> int:
    cfg = get_rag_import_config()
    return int(cfg.get("busy_retry_max_attempts"))


def get_rag_import_busy_retry_initial_backoff_sec() -> float:
    cfg = get_rag_import_config()
    return float(cfg.get("busy_retry_initial_backoff_sec"))


def get_rag_import_busy_retry_max_backoff_sec() -> float:
    cfg = get_rag_import_config()
    return float(cfg.get("busy_retry_max_backoff_sec"))


def get_crawl_logging_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    logging_cfg = defaults.get("logging")
    if not isinstance(logging_cfg, dict):
        return {}
    crawl_cfg = logging_cfg.get("crawl")
    return dict(crawl_cfg) if isinstance(crawl_cfg, dict) else {}


def should_emit_crawl_event(event_type: str) -> bool:
    crawl_cfg = get_crawl_logging_config()
    emit_events = crawl_cfg.get("emit_events")
    if not isinstance(emit_events, dict):
        return False
    key = str(event_type or "").strip().lower()
    return bool(emit_events.get(key))


def is_crawl_preview_logging_enabled(kind: str) -> bool:
    crawl_cfg = get_crawl_logging_config()
    previews = crawl_cfg.get("previews")
    if not isinstance(previews, dict):
        return False
    key = str(kind or "").strip().lower()
    return bool(previews.get(key))


def is_topic_extraction_enabled() -> bool:
    defaults = get_defaults_config()
    jobs = defaults.get("jobs") if isinstance(defaults.get("jobs"), dict) else {}
    return bool(jobs.get("topic_extraction_enabled"))


def get_default_knowledge_tabs() -> List[str]:
    defaults = get_defaults_config()
    raw = defaults.get("knowledge_tabs")
    return [str(v).strip().lower() for v in raw] if isinstance(raw, list) else []


def get_menu_category_aliases(*, widget_config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    defaults = get_defaults_config()
    menu_defaults = defaults.get("menu") if isinstance(defaults.get("menu"), dict) else {}
    aliases = menu_defaults.get("category_aliases") if isinstance(menu_defaults.get("category_aliases"), dict) else {}
    out = {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.items() if str(k).strip() and str(v).strip()}

    if widget_config:
        cfg = get_reservation_config_from_widget(widget_config)
        if cfg:
            domain_key = cfg.get("domain_key")
            profile = PLATFORM_PROFILES.get(str(domain_key or ""))
            metadata = profile.metadata if profile and isinstance(getattr(profile, "metadata", None), dict) else {}
            platform_menu = metadata.get("menu") if isinstance(metadata.get("menu"), dict) else {}
            override = platform_menu.get("category_aliases") if isinstance(platform_menu.get("category_aliases"), dict) else {}
            for k, v in override.items():
                key = str(k).strip().lower()
                value = str(v).strip().lower()
                if key and value:
                    out[key] = value
    return out


def get_menu_view_all_url_tokens(*, widget_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    defaults = get_defaults_config()
    menu_defaults = defaults.get("menu") if isinstance(defaults.get("menu"), dict) else {}
    raw = menu_defaults.get("view_all_url_tokens") if isinstance(menu_defaults.get("view_all_url_tokens"), dict) else {}
    out: Dict[str, List[str]] = {}
    for k, values in raw.items():
        key = str(k).strip().lower()
        if not key:
            continue
        if isinstance(values, list):
            out[key] = [str(v).strip() for v in values if str(v).strip()]

    if widget_config:
        cfg = get_reservation_config_from_widget(widget_config)
        if cfg:
            domain_key = cfg.get("domain_key")
            profile = PLATFORM_PROFILES.get(str(domain_key or ""))
            metadata = profile.metadata if profile and isinstance(getattr(profile, "metadata", None), dict) else {}
            platform_menu = metadata.get("menu") if isinstance(metadata.get("menu"), dict) else {}
            override = platform_menu.get("view_all_url_tokens") if isinstance(platform_menu.get("view_all_url_tokens"), dict) else {}
            for k, values in override.items():
                key = str(k).strip().lower()
                if not key:
                    continue
                if isinstance(values, list):
                    out[key] = [str(v).strip() for v in values if str(v).strip()]
    return out


def get_function_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    funcs = defaults.get("functions")
    return dict(funcs) if isinstance(funcs, dict) else {}


def get_deterministic_prompt_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    prompts = defaults.get("prompts") if isinstance(defaults.get("prompts"), dict) else {}
    deterministic = prompts.get("deterministic")
    return dict(deterministic) if isinstance(deterministic, dict) else {}


def get_prompt_generation_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    prompts = defaults.get("prompts") if isinstance(defaults.get("prompts"), dict) else {}
    generation = prompts.get("generation")
    return dict(generation) if isinstance(generation, dict) else {}


def get_prompt_fallback_config() -> Dict[str, Any]:
    defaults = get_defaults_config()
    prompts = defaults.get("prompts") if isinstance(defaults.get("prompts"), dict) else {}
    fallback = prompts.get("fallback")
    return dict(fallback) if isinstance(fallback, dict) else {}


def get_asset_base_stopwords() -> List[str]:
    defaults = get_defaults_config()
    assets = defaults.get("assets") if isinstance(defaults.get("assets"), dict) else {}
    raw = assets.get("base_stopwords")
    return [str(v).strip().lower() for v in raw if str(v).strip()] if isinstance(raw, list) else []


def get_default_asset_term_config() -> Dict[str, List[str]]:
    return {
        key: [str(v).strip() for v in values if str(v).strip()]
        for key, values in (DEFAULT_ASSET_TERM_CONFIG or {}).items()
        if isinstance(values, list)
    }


def normalize_reservation_links(widget_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Canonicalize reservation links from widget_config into {platform_id: url}.
    Supports both new map fields and legacy explicit URL fields.
    """
    links: Dict[str, str] = {}
    if not isinstance(widget_config, dict):
        return links

    for key in ("reservation_links", "reservationLinks"):
        raw = widget_config.get(key)
        if not isinstance(raw, dict):
            continue
        for platform_id, url in raw.items():
            pid = str(platform_id or "").strip().lower()
            val = str(url or "").strip()
            if not pid or not val:
                continue
            if not val.startswith(("http://", "https://")):
                val = f"https://{val}"
            links[pid] = val

    for platform_id, (widget_key, _) in RESERVATION_PLATFORM_CONFIG.items():
        raw_url = str(widget_config.get(widget_key) or "").strip()
        if not raw_url:
            continue
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url}"
        links[platform_id] = raw_url

    return links


def normalize_action_destination_links(widget_config: Dict[str, Any]) -> Dict[str, str]:
    links: Dict[str, str] = {}
    if not isinstance(widget_config, dict):
        return links
    raw = widget_config.get("actionDestinationLinks")
    if not isinstance(raw, dict):
        return links
    for action_key, url in raw.items():
        normalized_key = str(action_key or "").strip().lower()
        normalized_url = str(url or "").strip()
        if not normalized_key or not normalized_url:
            continue
        if not normalized_url.startswith(("http://", "https://")):
            normalized_url = f"https://{normalized_url}"
        parsed = urlparse(normalized_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        links[normalized_key] = normalized_url
    return links


def get_action_destination_url(widget_config: Dict[str, Any], action_key: str) -> str:
    normalized_key = str(action_key or "").strip().lower()
    if not normalized_key:
        return ""
    return str(normalize_action_destination_links(widget_config).get(normalized_key) or "").strip()


def get_reservation_url_for_platform(widget_config: Dict[str, Any], platform_id: str) -> str:
    return str(normalize_reservation_links(widget_config).get(str(platform_id or "").strip().lower()) or "").strip()


def get_reservation_url_rule(platform_id: str) -> Dict[str, Any]:
    cfg = _load_platform_config()
    rules = cfg.get("reservation_url_rules")
    if not isinstance(rules, dict):
        return {}
    pid = str(platform_id or "").strip().lower()
    raw = rules.get(pid)
    return dict(raw) if isinstance(raw, dict) else {}


def get_asset_rules_from_widget(widget_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get asset rules and term config from platform profile or default.
    Config-driven: no hardcoding in asset_resolver.
    Returns dict with marker_rule, evidence_template, asset_term_config.
    """
    out: Dict[str, Any] = dict(DEFAULT_ASSET_RULES)
    out["asset_term_config"] = dict(DEFAULT_ASSET_TERM_CONFIG)

    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return out
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return out
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    rules = metadata.get("asset_rules")
    if isinstance(rules, dict):
        marker = str(rules.get("marker_rule") or "").strip()
        evidence = str(rules.get("evidence_template") or "").strip()
        if marker:
            out["marker_rule"] = marker
        if evidence:
            out["evidence_template"] = evidence
        term_cfg = rules.get("asset_term_config")
        if isinstance(term_cfg, dict):
            for key in ("generic_tokens", "asset_intent_terms", "visual_request_terms", "visual_request_many_terms", "visual_suppress_terms"):
                val = term_cfg.get(key)
                if isinstance(val, list):
                    out["asset_term_config"][key] = [str(v).strip() for v in val if str(v).strip()]
    return out


def get_reservation_config_from_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Get reservation config from widget_config and platform profiles.
    Fully config-driven: presence of reservationPlatform + URL is the only signal.

    Returns:
        Dict with url, instruction, domain_key, link_label, platform_id; or None if not applicable.
    """
    if not isinstance(widget_config, dict):
        return None

    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    # One profile per agent: use reservationPlatform if set, else infer from first URL
    normalized_links = normalize_reservation_links(widget_config)
    platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
    if not platform_id and normalized_links:
        platform_id = next(iter(normalized_links.keys()), "")
    if not platform_id or platform_id not in RESERVATION_PLATFORM_CONFIG:
        return None

    _, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    platform_url = str(normalized_links.get(platform_id) or "").strip()
    if not platform_url:
        return None
    if not platform_url.startswith(("http://", "https://")):
        platform_url = f"https://{platform_url}"

    profile, resolved_domain = resolve_platform_profile(platform_url)
    if profile is None or resolved_domain != domain_key:
        return None

    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation")
    if not isinstance(reservation, dict):
        return None
    if not reservation.get("enabled", True):
        return None

    customer_url = get_action_destination_url(widget_config, "reservation") or platform_url

    # Optional override from widget_config; else use platform profile template
    custom = (widget_config.get("reservationInstruction") or "").strip()
    if custom:
        instruction = custom
    else:
        templates = reservation.get("instruction_template")
        if isinstance(templates, dict):
            instruction = str(templates.get(lang) or templates.get("en") or "").strip()
    if not instruction:
        return None

    try:
        instruction = instruction.format(url=customer_url)
    except (KeyError, ValueError):
        instruction = f"{instruction} {customer_url}"

    labels = reservation.get("link_label")
    if isinstance(labels, dict):
        link_label = str(labels.get(lang) or labels.get("en") or "").strip()
    else:
        link_label = ""
    if not link_label:
        return None

    return {
        "url": customer_url,
        "platform_url": platform_url,
        "instruction": instruction,
        "domain_key": domain_key,
        "link_label": link_label,
        "platform_id": platform_id,
    }


def get_knowledge_tabs_for_widget(widget_config: Dict[str, Any]) -> List[str]:
    """
    Get which knowledge tabs (image-assets, menu-list) to show in the bot dashboard.
    Config-driven via reservation_platform_config.knowledge_tabs in platform_profiles.yml.

    - tabelog, hotpepper: ["menu"] — Menu tab only
    - tablecheck, others: ["image"] — Image tab only (default)
    """
    default_tabs = get_default_knowledge_tabs()
    if not isinstance(widget_config, dict):
        return default_tabs

    normalized_links = normalize_reservation_links(widget_config)
    platform_id = str(widget_config.get("reservationPlatform") or "").strip().lower()
    if not platform_id and normalized_links:
        platform_id = next(iter(normalized_links.keys()), "")
    if not platform_id:
        return default_tabs

    rpc = _load_platform_config().get("reservation_platform_config") or {}
    entry = rpc.get(platform_id) if isinstance(rpc, dict) else {}
    if not isinstance(entry, dict):
        return default_tabs
    tabs = entry.get("knowledge_tabs")
    if isinstance(tabs, list) and tabs:
        out = [str(t).strip().lower() for t in tabs if str(t).strip()]
        if out:
            return out
    return default_tabs


def get_post_crawl_jobs_for_widget(widget_config: Dict[str, Any]) -> List[str]:
    """
    Get list of job names to run automatically after crawl completes.
    Config-driven via reservation_platform_config.post_crawl_jobs or default_post_crawl_jobs.

    Valid job names: topic_extraction, booking_link, menu_extraction
    """
    raw = _load_platform_config()
    default_jobs = get_default_post_crawl_jobs()

    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return default_jobs
    platform_id = cfg.get("platform_id")
    if not platform_id:
        return default_jobs
    rpc = raw.get("reservation_platform_config") or {}
    entry = rpc.get(platform_id) if isinstance(rpc, dict) else {}
    if not isinstance(entry, dict):
        return default_jobs
    jobs = entry.get("post_crawl_jobs")
    if isinstance(jobs, list) and jobs:
        out = [str(j).strip().lower() for j in jobs if str(j).strip()]
        if not is_topic_extraction_enabled():
            out = [j for j in out if j != "topic_extraction"]
        if out:
            return out
    return default_jobs


def get_platform_features_from_widget(widget_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get platform features (menu, suggested_messages) from widget config.
    Fully config-driven: reads from the active platform profile.
    """
    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return None
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    menu_rules = getattr(profile, "menu_extraction_rules", None)
    menu_enabled = (
        isinstance(menu_rules, dict)
        and menu_rules.get("enabled", True)
    )
    suggested = reservation.get("suggested_messages")
    if not isinstance(suggested, list):
        suggested = metadata.get("suggested_messages")  # fallback: top-level
    if not isinstance(suggested, list):
        suggested = None
    return {
        "menu_extraction_enabled": menu_enabled,
        "suggested_messages": suggested,
    }


def _resolve_label_or_prompt(raw: Any, lang: str) -> str:
    """Resolve label/prompt from string or {en, ja} dict."""
    if isinstance(raw, dict):
        return str(raw.get(lang) or raw.get("en") or "").strip()
    return str(raw or "").strip()


_VALID_SUGGESTED_TYPES = ("ai_response", "show_menu", "escalate")
_SUPPORTED_SUGGESTED_LANGS = ("en", "ja")
_SUPPORTED_WELCOME_CHANNELS = ("web", "line")
_SUGGESTED_BINDING_TO_LINE_RICH_MENU_ACTION_ID = {
    "reservation": "reserve",
    "reserve": "reserve",
    "menu": "menu",
    "show_menu": "menu",
    "support": "support",
    "escalate": "support",
}


def _normalize_suggested_lang(lang: Optional[str]) -> str:
    raw = (lang or "en").strip().lower()
    return "ja" if raw in ("ja", "jp") else "en"


def _resolve_suggested_items(items: List[Dict[str, Any]], *, lang: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        label = _resolve_label_or_prompt(raw.get("label"), lang)
        if not label:
            continue
        prompt = _resolve_label_or_prompt(raw.get("prompt"), lang) or label
        raw_type = str(raw.get("type") or "ai_response").strip() or "ai_response"
        if raw_type not in _VALID_SUGGESTED_TYPES:
            raw_type = "ai_response"
        item: Dict[str, Any] = {
            "id": str(raw.get("id") or "").strip() or f"suggest_{len(out) + 1}",
            "label": label,
            "prompt": prompt,
            "type": raw_type,
        }
        urls = raw.get("urls")
        if isinstance(urls, list):
            item["urls"] = [str(url).strip() for url in urls if str(url).strip()]
        message = str(raw.get("message") or "").strip()
        if message:
            item["message"] = message
        for key in ("fastPathBinding", "binding"):
            binding = str(raw.get(key) or "").strip()
            if binding:
                item["fastPathBinding"] = binding
                break
        out.append(item)
    return out


def _get_default_suggested_messages_for_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str,
) -> List[Dict[str, Any]]:
    normalized_lang = _normalize_suggested_lang(lang)
    features = get_platform_features_from_widget(widget_config)
    platform_suggested = features.get("suggested_messages") if features else None
    if isinstance(platform_suggested, list) and platform_suggested:
        return _resolve_suggested_items(platform_suggested, lang=normalized_lang)
    return _resolve_suggested_items(DEFAULT_SUGGESTED_MESSAGES, lang=normalized_lang)


def get_suggested_messages_by_language_for_widget(widget_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    current_lang = _normalize_suggested_lang(
        widget_config.get("language") or widget_config.get("botLanguage") or "en"
    )
    raw_by_lang = widget_config.get("suggestedMessagesByLanguage")
    raw_current = widget_config.get("suggestedMessages")
    resolved: Dict[str, List[Dict[str, Any]]] = {}
    for lang in _SUPPORTED_SUGGESTED_LANGS:
        raw_items = None
        if isinstance(raw_by_lang, dict):
            candidate = raw_by_lang.get(lang)
            if isinstance(candidate, list):
                raw_items = candidate
        if raw_items is None and lang == current_lang and isinstance(raw_current, list):
            raw_items = raw_current
        if isinstance(raw_items, list):
            resolved_items = _resolve_suggested_items(raw_items, lang=lang)
            if resolved_items:
                resolved[lang] = resolved_items
                continue
        resolved[lang] = _get_default_suggested_messages_for_widget(widget_config, lang=lang)
    return resolved


def get_available_suggested_message_types(platform_id: Optional[str] = None) -> List[str]:
    """
    Return the types of suggested messages available for a platform (or default).
    Config-driven: only types present in platform's suggested_messages are allowed.
    Default (no platform): only ai_response.
    """
    if not platform_id or not str(platform_id).strip():
        # Default: only types from default_suggested_messages
        types_seen: set = set()
        for m in DEFAULT_SUGGESTED_MESSAGES:
            if isinstance(m, dict):
                t = str(m.get("type") or "ai_response").strip() or "ai_response"
                if t in _VALID_SUGGESTED_TYPES:
                    types_seen.add(t)
        return list(types_seen) if types_seen else ["ai_response"]

    platform_id = str(platform_id).strip().lower()
    if platform_id not in RESERVATION_PLATFORM_CONFIG:
        return ["ai_response"]
    _, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    if domain_key not in PLATFORM_PROFILES:
        return ["ai_response"]
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    raw = reservation.get("suggested_messages") or metadata.get("suggested_messages")
    if not isinstance(raw, list) or not raw:
        return ["ai_response"]
    types_seen = set()
    for r in raw:
        if isinstance(r, dict):
            t = str(r.get("type") or "ai_response").strip() or "ai_response"
            if t in _VALID_SUGGESTED_TYPES:
                types_seen.add(t)
    return list(types_seen) if types_seen else ["ai_response"]


def get_suggested_messages_for_platform(
    platform_id: str,
    *,
    lang: str = "en",
) -> Optional[List[Dict[str, Any]]]:
    """
    Get platform default suggested messages (for create-bot initial load).
    Returns list of {id, label, type, prompt} or None if platform has no suggested_messages.
    """
    platform_id = str(platform_id or "").strip().lower()
    if platform_id not in RESERVATION_PLATFORM_CONFIG:
        return None
    _, domain_key = RESERVATION_PLATFORM_CONFIG[platform_id]
    if domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    reservation = metadata.get("reservation") if isinstance(metadata.get("reservation"), dict) else {}
    raw = reservation.get("suggested_messages") or metadata.get("suggested_messages")
    if not isinstance(raw, list) or not raw:
        return None
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    def resolve_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _resolve_suggested_items(items, lang=lang)

    resolved = resolve_items(raw)
    return resolved if resolved else None


def get_suggested_messages_for_widget(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Get suggested messages for a bot. Used across Line, Instagram, web widget.

    Priority: DB first, then platform fallback (for first default), then platform_profiles default.
    - If widget_config has suggestedMessages (saved in DB): use those (edits persist)
    - Else: use platform profile (Tabelog, HotPepper, TableCheck) as initial default
    - Else: use default_suggested_messages
    """
    normalized_lang = _normalize_suggested_lang(lang)
    by_lang = get_suggested_messages_by_language_for_widget(widget_config)
    resolved = by_lang.get(normalized_lang) or []
    return resolved if resolved else _get_default_suggested_messages_for_widget(widget_config, lang=normalized_lang)


def get_line_rich_menu_labels_from_suggested_messages(
    widget_config: Dict[str, Any],
    *,
    lang: str = "en",
) -> Dict[str, str]:
    """
    Build rich-menu action label defaults from suggested messages.

    This keeps LINE rich-menu labels consistent with suggested messages for
    supported capabilities:
    - reservation -> reserve
    - show_menu/menu -> menu
    - escalate/support -> support
    """
    labels_by_action_id: Dict[str, str] = {}
    for item in get_suggested_messages_for_widget(widget_config, lang=lang):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        binding = str(item.get("fastPathBinding") or item.get("binding") or "").strip().lower()
        if not binding:
            suggested_type = str(item.get("type") or "").strip().lower()
            if suggested_type == "show_menu":
                binding = "show_menu"
            elif suggested_type == "escalate":
                binding = "escalate"
        action_id = _SUGGESTED_BINDING_TO_LINE_RICH_MENU_ACTION_ID.get(binding)
        if action_id and action_id not in labels_by_action_id:
            labels_by_action_id[action_id] = label
    return labels_by_action_id


def has_support_suggested_message_for_widget(widget_config: Dict[str, Any]) -> bool:
    by_lang = get_suggested_messages_by_language_for_widget(widget_config)
    for items in by_lang.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip() == "escalate":
                return True
    return False


def get_default_welcome_messages() -> Dict[str, Dict[str, str]]:
    resolved: Dict[str, Dict[str, str]] = {}
    for channel in _SUPPORTED_WELCOME_CHANNELS:
        channel_defaults = DEFAULT_WELCOME_MESSAGES.get(channel) if isinstance(DEFAULT_WELCOME_MESSAGES, dict) else {}
        resolved[channel] = {
            "en": str((channel_defaults or {}).get("en") or "").strip(),
            "ja": str((channel_defaults or {}).get("ja") or "").strip(),
        }
    return resolved


def get_support_messages_config() -> Dict[str, Any]:
    return dict(SUPPORT_MESSAGES_CONFIG or {})


def _resolve_support_messages(
    *,
    channel_key: str,
    lang: str,
    field_keys: Tuple[str, ...],
) -> Dict[str, str]:
    cfg = get_support_messages_config()
    messages = cfg.get(channel_key) if isinstance(cfg.get(channel_key), dict) else {}
    normalized_lang = _normalize_lang(lang)
    resolved: Dict[str, str] = {}
    for key in field_keys:
        resolved[key] = _resolve_i18n_text(messages.get(key), lang=normalized_lang)
    return resolved


def get_web_support_messages(*, lang: str = "en") -> Dict[str, str]:
    messages = _resolve_support_messages(
        channel_key="web",
        lang=lang,
        field_keys=(
            "requested",
            "disabled",
            "modal_title",
            "modal_subtitle",
            "email_placeholder",
            "details_label",
            "details_placeholder",
            "cancel_button",
            "submit_button",
            "invalid_email",
            "submit_failed",
            "submit_success",
        ),
    )
    field_map = {
        "requested": "requested",
        "disabled": "disabled",
        "modal_title": "modalTitle",
        "modal_subtitle": "modalSubtitle",
        "email_placeholder": "emailPlaceholder",
        "details_label": "detailsLabel",
        "details_placeholder": "detailsPlaceholder",
        "cancel_button": "cancelButton",
        "submit_button": "submitButton",
        "invalid_email": "invalidEmail",
        "submit_failed": "submitFailed",
        "submit_success": "submitSuccess",
    }
    resolved: Dict[str, str] = {}
    for config_key, public_key in field_map.items():
        resolved[public_key] = str(messages.get(config_key) or "")
    return resolved


def get_welcome_messages_by_channel_for_widget(widget_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    current_lang = _normalize_suggested_lang(
        widget_config.get("language") or widget_config.get("botLanguage") or "en"
    )
    defaults = get_default_welcome_messages()
    raw_by_channel = widget_config.get("welcomeMessagesByChannel")
    legacy_welcome = str(widget_config.get("welcomeMessage") or "").strip()
    resolved: Dict[str, Dict[str, str]] = {}

    for channel in _SUPPORTED_WELCOME_CHANNELS:
        resolved[channel] = {}
        raw_channel = raw_by_channel.get(channel) if isinstance(raw_by_channel, dict) else None
        for lang in _SUPPORTED_SUGGESTED_LANGS:
            value = ""
            if isinstance(raw_channel, dict):
                value = str(raw_channel.get(lang) or "").strip()
            if not value and channel == "web" and lang == current_lang and legacy_welcome:
                value = legacy_welcome
            resolved[channel][lang] = value or defaults.get(channel, {}).get(lang, "")
    return resolved


def get_welcome_message_for_widget(
    widget_config: Dict[str, Any],
    *,
    channel: str = "web",
    lang: str = "en",
) -> str:
    normalized_channel = "line" if str(channel or "").strip().lower() == "line" else "web"
    normalized_lang = _normalize_suggested_lang(lang)
    by_channel = get_welcome_messages_by_channel_for_widget(widget_config)
    channel_messages = by_channel.get(normalized_channel) or {}
    return str(channel_messages.get(normalized_lang) or channel_messages.get("en") or "").strip()


def get_platform_asset_instructions(widget_config: Dict[str, Any], *, lang: str = "en") -> Optional[str]:
    """
    Get asset usage instructions from the active platform profile.
    Config-driven: no platform-specific logic in callers.
    """
    cfg = get_reservation_config_from_widget(widget_config)
    if not cfg:
        return None
    domain_key = cfg.get("domain_key")
    if not domain_key or domain_key not in PLATFORM_PROFILES:
        return None
    profile = PLATFORM_PROFILES[domain_key]
    metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
    instructions = metadata.get("asset_instructions")
    if not isinstance(instructions, dict):
        return None
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"
    text = str(instructions.get(lang) or instructions.get("en") or "").strip()
    return text if text else None


def _build_json_instruction_from_jrf(jrf: Dict[str, Any], lang: str) -> Optional[str]:
    """Build JSON response instruction from a jrf dict (default or platform)."""
    if not isinstance(jrf, dict):
        return None
    schema_dict = jrf.get("schema")
    if not isinstance(schema_dict, dict):
        return None
    if "show_assets" in schema_dict:
        true_rules = jrf.get("show_assets_true_when")
        false_rules = jrf.get("show_assets_false_when")
        true_txt = str((true_rules or {}).get("en") or "").strip() if isinstance(true_rules, dict) else ""
        false_txt = str((false_rules or {}).get("en") or "").strip() if isinstance(false_rules, dict) else ""
        if not true_txt or not false_txt:
            return None

    schema_items = [f'"{k}": {v}' for k, v in schema_dict.items() if isinstance(v, str) and v.strip()]
    if not schema_items:
        return None
    schema_line = "{" + ", ".join(schema_items) + "}"
    parts: List[str] = [
        "\n\nRESPONSE FORMAT (CRITICAL): You MUST respond with valid JSON only. No other text before or after.\n"
        f"Schema: {schema_line}\n"
    ]
    for key, desc in schema_dict.items():
        if isinstance(desc, str) and desc.strip():
            parts.append(f"- {key}: {desc}\n")

    true_rules = jrf.get("show_assets_true_when")
    false_rules = jrf.get("show_assets_false_when")
    true_txt = str((true_rules or {}).get(lang) or (true_rules or {}).get("en") or "").strip() if isinstance(true_rules, dict) else ""
    false_txt = str((false_rules or {}).get(lang) or (false_rules or {}).get("en") or "").strip() if isinstance(false_rules, dict) else ""
    if true_txt and false_txt:
        parts.append(
            "- show_assets: boolean. Follow these rules exactly:\n"
            f"  WHEN TRUE: {true_txt}\n"
            f"  WHEN FALSE: {false_txt}\n"
        )

    intent_when = jrf.get("intent_when")
    if isinstance(intent_when, dict):
        intent_txt = str(intent_when.get(lang) or intent_when.get("en") or "").strip()
        if intent_txt:
            parts.append(f"- intent: Follow these rules:\n{intent_txt}\n")

    parts.append("Output ONLY the JSON object, no markdown code fences.")
    return "".join(parts)


def get_platform_json_response_instruction(widget_config: Dict[str, Any], *, lang: str = "en") -> Optional[str]:
    """
    Get JSON response format instruction. Uses default for all bots; platforms (Tabelog, HotPepper)
    add extra keys like intent via their json_response_format config.
    """
    lang = (lang or "en").strip().lower()
    lang = "ja" if lang in ("ja", "jp") else "en"

    jrf: Optional[Dict[str, Any]] = None
    cfg = get_reservation_config_from_widget(widget_config)
    if cfg:
        domain_key = cfg.get("domain_key")
        if domain_key and domain_key in PLATFORM_PROFILES:
            profile = PLATFORM_PROFILES[domain_key]
            metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
            platform_jrf = metadata.get("json_response_format")
            if isinstance(platform_jrf, dict) and platform_jrf.get("enabled"):
                jrf = platform_jrf

    if jrf is None and DEFAULT_JSON_RESPONSE_FORMAT:
        jrf = DEFAULT_JSON_RESPONSE_FORMAT

    return _build_json_instruction_from_jrf(jrf, lang) if jrf else None


def get_platform_json_response_enabled(widget_config: Dict[str, Any]) -> bool:
    """Return True when JSON response format is configured (default or platform)."""
    return get_platform_json_response_instruction(widget_config, lang="en") is not None


def get_default_rag_instruction() -> Dict[str, Any]:
    """Return default RAG/LLM instruction config (default_system, grounding_suffix, etc.)."""
    return dict(DEFAULT_RAG_INSTRUCTION) if DEFAULT_RAG_INSTRUCTION else {}


def get_default_marker_rule() -> str:
    """Return default marker rule for asset bank (from config)."""
    return str(DEFAULT_ASSET_RULES.get("marker_rule") or "").strip()


def get_menu_texts(lang: str, *, widget_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get menu flow texts for the given lang. Platform can override via metadata.menu_texts."""
    lang = "ja" if (lang or "").strip().lower() in ("ja", "jp") else "en"
    base = (DEFAULT_MENU_TEXTS or {}).get(lang) or (DEFAULT_MENU_TEXTS or {}).get("en") or {}
    if widget_config:
        cfg = get_reservation_config_from_widget(widget_config)
        if cfg:
            domain_key = cfg.get("domain_key")
            if domain_key and domain_key in PLATFORM_PROFILES:
                profile = PLATFORM_PROFILES[domain_key]
                metadata = profile.metadata if isinstance(getattr(profile, "metadata", None), dict) else {}
                platform_texts = metadata.get("menu_texts")
                if isinstance(platform_texts, dict):
                    platform_lang = (platform_texts.get(lang) or platform_texts.get("en") or {})
                    if isinstance(platform_lang, dict):
                        base = dict(base)
                        base.update(platform_lang)
    return base


def get_menu_keywords() -> List[str]:
    """Get menu request keywords from config."""
    return [str(v).strip() for v in (DEFAULT_MENU_KEYWORDS or []) if str(v).strip()]


def get_menu_category_order() -> Tuple[str, ...]:
    """Get menu category order from config."""
    return tuple(str(v).strip().lower() for v in (DEFAULT_MENU_CATEGORY_ORDER or []) if str(v).strip())


def get_menu_request_pattern() -> Optional[str]:
    """Get menu request regex pattern from config."""
    return (DEFAULT_MENU_REQUEST_PATTERN or "").strip() or None


def get_line_menu_quick_payload() -> str:
    """Get LINE menu quick reply payload from config."""
    return str(LINE_MENU_QUICK_PAYLOAD or "").strip()


def get_line_menu_page_payload_prefix() -> str:
    """Get LINE menu page payload prefix from config."""
    return str(LINE_MENU_PAGE_PAYLOAD_PREFIX or "").strip()


def get_instagram_menu_quick_payload() -> str:
    """Get Instagram menu quick reply payload from config."""
    return str(INSTAGRAM_MENU_QUICK_PAYLOAD or "").strip()


def get_instagram_menu_page_payload_prefix() -> str:
    """Get Instagram menu page payload prefix from config."""
    return str(INSTAGRAM_MENU_PAGE_PAYLOAD_PREFIX or "").strip()


def get_line_ux_config() -> Dict[str, Any]:
    return dict(LINE_UX_CONFIG or {})


_HEX_COLOR_6_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_LINE_DESIGN_STYLE_KEYS = (
    "background",
    "text",
    "muted_text",
    "button_background",
    "button_text",
    "border",
    "accent",
)
_LINE_DESIGN_FALLBACK_PROFILE: Dict[str, Any] = {
    "suggested_actions": {
        "defaults": {
            "theme_mode": "light",
            "layout": "column",
            "card_background_color": "#ffffff",
            "card_text_color": "#1f2937",
            "button_background_color": "#f3f4f6",
            "button_text_color": "#374151",
            "button_border_color": "#e5e7eb",
        },
        "options": {
            "theme_modes": ["light", "dark"],
            "layouts": ["column", "row"],
        },
    },
    "asset_carousel": {
        "defaults": {
            "bubble_size": "micro",
            "image_aspect_ratio": "4:3",
            "body_background_color": "#111827",
            "body_text_color": "#ffffff",
            "overlay_background_color": "#111827",
        },
        "options": {
            "bubble_sizes": ["micro", "kilo", "mega"],
            "image_aspect_ratios": ["1:1", "4:3", "16:9", "20:13", "3:4"],
        },
    },
    "rich_menu": {
        "defaults": {
            "styles": {
                "normal": {
                    "background": "#f5f7fb",
                    "text": "#0f172a",
                    "muted_text": "#475569",
                    "button_background": "#ffffff",
                    "button_text": "#0f172a",
                    "border": "#d7dde7",
                    "accent": "#06c755",
                },
                "support": {
                    "background": "#0f172a",
                    "text": "#f8fafc",
                    "muted_text": "#cbd5e1",
                    "button_background": "#fef3c7",
                    "button_text": "#92400e",
                    "border": "#d7dde7",
                    "accent": "#f59e0b",
                },
            },
            "actions": [],
            "layouts": {"normal": [], "support": []},
        },
        "allowed_icon_ids": ["reserve", "menu", "support", "back_to_ai", "chat", "help", "link"],
        "editable_action_ids": [],
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_hex_color(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if not raw.startswith("#") and len(raw) == 6:
        raw = f"#{raw}"
    return raw if _HEX_COLOR_6_RE.fullmatch(raw) else fallback


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _build_line_design_default_actions_from_line_ux(*, lang: str = "en") -> List[Dict[str, Any]]:
    cfg = get_line_ux_config()
    rich_menu = cfg.get("rich_menu") if isinstance(cfg.get("rich_menu"), dict) else {}
    actions_cfg = rich_menu.get("actions") if isinstance(rich_menu.get("actions"), dict) else {}
    defaults: List[Dict[str, Any]] = []
    for action_id, raw_action in actions_cfg.items():
        if not isinstance(raw_action, dict):
            continue
        aid = str(action_id or "").strip()
        if not aid:
            continue
        label_en = _resolve_i18n_text(raw_action.get("label"), lang="en", fallback=aid.replace("_", " ").title())
        label_ja = _resolve_i18n_text(raw_action.get("label"), lang="ja", fallback=label_en)
        defaults.append(
            {
                "id": aid,
                "enabled": True,
                "icon": str(raw_action.get("icon") or aid).strip() or aid,
                "labels": {"en": label_en, "ja": label_ja},
            }
        )
    return defaults


def _build_line_design_default_layouts_from_line_ux() -> Dict[str, List[str]]:
    cfg = get_line_ux_config()
    rich_menu = cfg.get("rich_menu") if isinstance(cfg.get("rich_menu"), dict) else {}
    layouts_cfg = rich_menu.get("layouts") if isinstance(rich_menu.get("layouts"), dict) else {}
    return {
        "normal": _as_str_list(layouts_cfg.get("normal")),
        "support": _as_str_list(layouts_cfg.get("support")),
    }


def get_line_design_profile() -> Dict[str, Any]:
    cfg = get_line_ux_config()
    raw_profile = cfg.get("design_profile") if isinstance(cfg.get("design_profile"), dict) else {}
    profile = _deep_merge_dict(_LINE_DESIGN_FALLBACK_PROFILE, raw_profile if isinstance(raw_profile, dict) else {})

    # Suggested actions
    sa = profile.get("suggested_actions") if isinstance(profile.get("suggested_actions"), dict) else {}
    sa_defaults = sa.get("defaults") if isinstance(sa.get("defaults"), dict) else {}
    sa_options = sa.get("options") if isinstance(sa.get("options"), dict) else {}
    theme_modes = _as_str_list(sa_options.get("theme_modes")) or ["light", "dark"]
    layouts = _as_str_list(sa_options.get("layouts")) or ["column", "row"]
    theme_mode = str(sa_defaults.get("theme_mode") or "light").strip().lower()
    if theme_mode not in theme_modes:
        theme_mode = theme_modes[0]
    layout = str(sa_defaults.get("layout") or "column").strip().lower()
    if layout not in layouts:
        layout = layouts[0]
    profile["suggested_actions"] = {
        "defaults": {
            "theme_mode": theme_mode,
            "layout": layout,
            "card_background_color": _normalize_hex_color(sa_defaults.get("card_background_color"), "#ffffff"),
            "card_text_color": _normalize_hex_color(sa_defaults.get("card_text_color"), "#1f2937"),
            "button_background_color": _normalize_hex_color(sa_defaults.get("button_background_color"), "#f3f4f6"),
            "button_text_color": _normalize_hex_color(sa_defaults.get("button_text_color"), "#374151"),
            "button_border_color": _normalize_hex_color(sa_defaults.get("button_border_color"), "#e5e7eb"),
        },
        "options": {
            "theme_modes": theme_modes,
            "layouts": layouts,
        },
    }

    # Asset carousel
    ac = profile.get("asset_carousel") if isinstance(profile.get("asset_carousel"), dict) else {}
    ac_defaults = ac.get("defaults") if isinstance(ac.get("defaults"), dict) else {}
    ac_options = ac.get("options") if isinstance(ac.get("options"), dict) else {}
    bubble_sizes = _as_str_list(ac_options.get("bubble_sizes")) or ["micro", "kilo", "mega"]
    image_ratios = _as_str_list(ac_options.get("image_aspect_ratios")) or ["1:1", "4:3", "16:9", "20:13", "3:4"]
    bubble_size = str(ac_defaults.get("bubble_size") or "micro").strip().lower()
    if bubble_size not in bubble_sizes:
        bubble_size = bubble_sizes[0]
    image_aspect_ratio = str(ac_defaults.get("image_aspect_ratio") or "4:3").strip()
    if image_aspect_ratio not in image_ratios:
        image_aspect_ratio = image_ratios[0]
    profile["asset_carousel"] = {
        "defaults": {
            "bubble_size": bubble_size,
            "image_aspect_ratio": image_aspect_ratio,
            "body_background_color": _normalize_hex_color(ac_defaults.get("body_background_color"), "#111827"),
            "body_text_color": _normalize_hex_color(ac_defaults.get("body_text_color"), "#ffffff"),
            "overlay_background_color": _normalize_hex_color(ac_defaults.get("overlay_background_color"), "#111827"),
        },
        "options": {
            "bubble_sizes": bubble_sizes,
            "image_aspect_ratios": image_ratios,
        },
    }

    # Rich menu profile
    rm = profile.get("rich_menu") if isinstance(profile.get("rich_menu"), dict) else {}
    rm_defaults = rm.get("defaults") if isinstance(rm.get("defaults"), dict) else {}
    rm_styles = rm_defaults.get("styles") if isinstance(rm_defaults.get("styles"), dict) else {}
    rm_actions = rm_defaults.get("actions") if isinstance(rm_defaults.get("actions"), list) else []
    rm_layouts = rm_defaults.get("layouts") if isinstance(rm_defaults.get("layouts"), dict) else {}
    allowed_icon_ids = _as_str_list(rm.get("allowed_icon_ids"))
    editable_action_ids = _as_str_list(rm.get("editable_action_ids"))

    if not rm_actions:
        rm_actions = _build_line_design_default_actions_from_line_ux()
    if not rm_layouts or (not _as_str_list(rm_layouts.get("normal")) and not _as_str_list(rm_layouts.get("support"))):
        rm_layouts = _build_line_design_default_layouts_from_line_ux()
    if not editable_action_ids:
        editable_action_ids = [str(item.get("id") or "").strip() for item in rm_actions if isinstance(item, dict)]
        editable_action_ids = [item for item in editable_action_ids if item]
    if not allowed_icon_ids:
        allowed_icon_ids = ["reserve", "menu", "support", "back_to_ai", "chat", "help", "link"]
    for aid in editable_action_ids:
        if aid not in allowed_icon_ids:
            allowed_icon_ids.append(aid)

    normalized_actions: List[Dict[str, Any]] = []
    for raw_action in rm_actions:
        if not isinstance(raw_action, dict):
            continue
        action_id = str(raw_action.get("id") or "").strip()
        if not action_id or action_id not in editable_action_ids:
            continue
        labels = raw_action.get("labels") if isinstance(raw_action.get("labels"), dict) else {}
        label_en = str(labels.get("en") or "").strip() or action_id.replace("_", " ").title()
        label_ja = str(labels.get("ja") or "").strip() or label_en
        icon_id = str(raw_action.get("icon") or action_id).strip()
        if icon_id not in allowed_icon_ids:
            icon_id = action_id if action_id in allowed_icon_ids else allowed_icon_ids[0]
        normalized_actions.append(
            {
                "id": action_id,
                "enabled": bool(raw_action.get("enabled", True)),
                "icon": icon_id,
                "labels": {"en": label_en, "ja": label_ja},
            }
        )

    normalized_styles: Dict[str, Dict[str, str]] = {}
    for state_name in ("normal", "support"):
        raw_state = rm_styles.get(state_name) if isinstance(rm_styles.get(state_name), dict) else {}
        fallback_state = _LINE_DESIGN_FALLBACK_PROFILE["rich_menu"]["defaults"]["styles"][state_name]
        normalized_styles[state_name] = {
            key: _normalize_hex_color(raw_state.get(key), fallback_state[key])
            for key in _LINE_DESIGN_STYLE_KEYS
        }

    normalized_layouts: Dict[str, List[str]] = {}
    for state_name in ("normal", "support"):
        raw_layout = _as_str_list(rm_layouts.get(state_name))
        filtered = [aid for aid in raw_layout if aid in editable_action_ids]
        if not filtered:
            filtered = [action["id"] for action in normalized_actions if action["id"] in editable_action_ids]
        normalized_layouts[state_name] = filtered

    profile["rich_menu"] = {
        "defaults": {
            "styles": normalized_styles,
            "actions": normalized_actions,
            "layouts": normalized_layouts,
        },
        "allowed_icon_ids": allowed_icon_ids,
        "editable_action_ids": editable_action_ids,
    }
    return profile


def normalize_line_design_overrides(
    overrides: Optional[Dict[str, Any]],
    *,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(overrides, dict):
        return {}
    cfg = profile if isinstance(profile, dict) else get_line_design_profile()
    out: Dict[str, Any] = {}

    # Suggested actions
    raw_sa = overrides.get("suggested_actions")
    if isinstance(raw_sa, dict):
        sa_cfg = cfg.get("suggested_actions") if isinstance(cfg.get("suggested_actions"), dict) else {}
        sa_defaults = sa_cfg.get("defaults") if isinstance(sa_cfg.get("defaults"), dict) else {}
        sa_options = sa_cfg.get("options") if isinstance(sa_cfg.get("options"), dict) else {}
        theme_modes = _as_str_list(sa_options.get("theme_modes")) or ["light", "dark"]
        layouts = _as_str_list(sa_options.get("layouts")) or ["column", "row"]
        sa_out: Dict[str, Any] = {}
        theme_mode = str(raw_sa.get("theme_mode") or "").strip().lower()
        if theme_mode in theme_modes:
            sa_out["theme_mode"] = theme_mode
        layout = str(raw_sa.get("layout") or "").strip().lower()
        if layout in layouts:
            sa_out["layout"] = layout
        for key, fallback in (
            ("card_background_color", sa_defaults.get("card_background_color") or "#ffffff"),
            ("card_text_color", sa_defaults.get("card_text_color") or "#1f2937"),
            ("button_background_color", sa_defaults.get("button_background_color") or "#f3f4f6"),
            ("button_text_color", sa_defaults.get("button_text_color") or "#374151"),
            ("button_border_color", sa_defaults.get("button_border_color") or "#e5e7eb"),
        ):
            if key in raw_sa:
                sa_out[key] = _normalize_hex_color(raw_sa.get(key), fallback)
        if sa_out:
            out["suggested_actions"] = sa_out

    # Asset carousel
    raw_ac = overrides.get("asset_carousel")
    if isinstance(raw_ac, dict):
        ac_cfg = cfg.get("asset_carousel") if isinstance(cfg.get("asset_carousel"), dict) else {}
        ac_defaults = ac_cfg.get("defaults") if isinstance(ac_cfg.get("defaults"), dict) else {}
        ac_options = ac_cfg.get("options") if isinstance(ac_cfg.get("options"), dict) else {}
        bubble_sizes = _as_str_list(ac_options.get("bubble_sizes")) or ["micro", "kilo", "mega"]
        image_ratios = _as_str_list(ac_options.get("image_aspect_ratios")) or ["1:1", "4:3", "16:9", "20:13", "3:4"]
        ac_out: Dict[str, Any] = {}
        bubble_size = str(raw_ac.get("bubble_size") or "").strip().lower()
        if bubble_size in bubble_sizes:
            ac_out["bubble_size"] = bubble_size
        image_aspect_ratio = str(raw_ac.get("image_aspect_ratio") or "").strip()
        if image_aspect_ratio in image_ratios:
            ac_out["image_aspect_ratio"] = image_aspect_ratio
        for key, fallback in (
            ("body_background_color", ac_defaults.get("body_background_color") or "#111827"),
            ("body_text_color", ac_defaults.get("body_text_color") or "#ffffff"),
            ("overlay_background_color", ac_defaults.get("overlay_background_color") or "#111827"),
        ):
            if key in raw_ac:
                ac_out[key] = _normalize_hex_color(raw_ac.get(key), fallback)
        if ac_out:
            out["asset_carousel"] = ac_out

    # Rich menu
    raw_rm = overrides.get("rich_menu")
    if isinstance(raw_rm, dict):
        rm_cfg = cfg.get("rich_menu") if isinstance(cfg.get("rich_menu"), dict) else {}
        rm_defaults = rm_cfg.get("defaults") if isinstance(rm_cfg.get("defaults"), dict) else {}
        default_styles = rm_defaults.get("styles") if isinstance(rm_defaults.get("styles"), dict) else {}
        editable_action_ids = _as_str_list(rm_cfg.get("editable_action_ids"))
        allowed_icon_ids = _as_str_list(rm_cfg.get("allowed_icon_ids"))
        rm_out: Dict[str, Any] = {}

        raw_styles = raw_rm.get("styles")
        if isinstance(raw_styles, dict):
            styles_out: Dict[str, Any] = {}
            for state_name in ("normal", "support"):
                state_raw = raw_styles.get(state_name)
                if not isinstance(state_raw, dict):
                    continue
                fallback_state = (
                    default_styles.get(state_name)
                    if isinstance(default_styles.get(state_name), dict)
                    else _LINE_DESIGN_FALLBACK_PROFILE["rich_menu"]["defaults"]["styles"][state_name]
                )
                state_out: Dict[str, str] = {}
                for key in _LINE_DESIGN_STYLE_KEYS:
                    if key in state_raw:
                        state_out[key] = _normalize_hex_color(state_raw.get(key), fallback_state.get(key) or "#000000")
                if state_out:
                    styles_out[state_name] = state_out
            if styles_out:
                rm_out["styles"] = styles_out

        raw_layouts = raw_rm.get("layouts")
        if isinstance(raw_layouts, dict):
            layouts_out: Dict[str, List[str]] = {}
            for state_name in ("normal", "support"):
                raw_order = _as_str_list(raw_layouts.get(state_name))
                filtered = [aid for aid in raw_order if aid in editable_action_ids]
                if filtered:
                    layouts_out[state_name] = filtered
            if layouts_out:
                rm_out["layouts"] = layouts_out

        raw_actions = raw_rm.get("actions")
        if isinstance(raw_actions, list):
            actions_out: List[Dict[str, Any]] = []
            for raw_action in raw_actions:
                if not isinstance(raw_action, dict):
                    continue
                aid = str(raw_action.get("id") or "").strip()
                if not aid or aid not in editable_action_ids:
                    continue
                action_out: Dict[str, Any] = {"id": aid}
                if "enabled" in raw_action:
                    action_out["enabled"] = bool(raw_action.get("enabled"))
                icon_id = str(raw_action.get("icon") or "").strip()
                if icon_id and icon_id in allowed_icon_ids:
                    action_out["icon"] = icon_id
                labels_raw = raw_action.get("labels")
                if isinstance(labels_raw, dict):
                    labels_out: Dict[str, str] = {}
                    for lang_key in ("en", "ja"):
                        label = str(labels_raw.get(lang_key) or "").strip()
                        if label:
                            labels_out[lang_key] = label
                    if labels_out:
                        action_out["labels"] = labels_out
                actions_out.append(action_out)
            if actions_out:
                rm_out["actions"] = actions_out

        if rm_out:
            out["rich_menu"] = rm_out

    return out


def build_line_design_effective(
    overrides: Optional[Dict[str, Any]],
    *,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = profile if isinstance(profile, dict) else get_line_design_profile()
    normalized = normalize_line_design_overrides(overrides or {}, profile=cfg)
    effective: Dict[str, Any] = {
        "suggested_actions": copy.deepcopy((cfg.get("suggested_actions") or {}).get("defaults") or {}),
        "asset_carousel": copy.deepcopy((cfg.get("asset_carousel") or {}).get("defaults") or {}),
        "rich_menu": copy.deepcopy((cfg.get("rich_menu") or {}).get("defaults") or {}),
    }

    sa_override = normalized.get("suggested_actions")
    if isinstance(sa_override, dict):
        effective["suggested_actions"] = {
            **(effective.get("suggested_actions") or {}),
            **sa_override,
        }

    ac_override = normalized.get("asset_carousel")
    if isinstance(ac_override, dict):
        effective["asset_carousel"] = {
            **(effective.get("asset_carousel") or {}),
            **ac_override,
        }

    rich_override = normalized.get("rich_menu")
    rich_effective = effective.get("rich_menu") if isinstance(effective.get("rich_menu"), dict) else {}
    if isinstance(rich_override, dict):
        # styles
        rich_styles = rich_effective.get("styles") if isinstance(rich_effective.get("styles"), dict) else {}
        override_styles = rich_override.get("styles") if isinstance(rich_override.get("styles"), dict) else {}
        for state_name in ("normal", "support"):
            state_current = rich_styles.get(state_name) if isinstance(rich_styles.get(state_name), dict) else {}
            state_override = override_styles.get(state_name) if isinstance(override_styles.get(state_name), dict) else {}
            if state_override:
                rich_styles[state_name] = {**state_current, **state_override}
        rich_effective["styles"] = rich_styles

        # actions
        default_actions = rich_effective.get("actions") if isinstance(rich_effective.get("actions"), list) else []
        action_map: Dict[str, Dict[str, Any]] = {}
        for item in default_actions:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip()
            if aid:
                action_map[aid] = copy.deepcopy(item)
        override_actions = rich_override.get("actions") if isinstance(rich_override.get("actions"), list) else []
        for item in override_actions:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip()
            if not aid or aid not in action_map:
                continue
            current = action_map[aid]
            if "enabled" in item:
                current["enabled"] = bool(item.get("enabled"))
            icon = str(item.get("icon") or "").strip()
            if icon:
                current["icon"] = icon
            labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
            current_labels = current.get("labels") if isinstance(current.get("labels"), dict) else {}
            for lang_key in ("en", "ja"):
                text = str(labels.get(lang_key) or "").strip()
                if text:
                    current_labels[lang_key] = text
            current["labels"] = current_labels
            action_map[aid] = current
        rich_effective["actions"] = list(action_map.values())

        # layouts
        current_layouts = rich_effective.get("layouts") if isinstance(rich_effective.get("layouts"), dict) else {}
        override_layouts = rich_override.get("layouts") if isinstance(rich_override.get("layouts"), dict) else {}
        for state_name in ("normal", "support"):
            raw = _as_str_list(override_layouts.get(state_name))
            if raw:
                current_layouts[state_name] = raw
        rich_effective["layouts"] = current_layouts

    effective["rich_menu"] = rich_effective
    return effective


def get_line_cancel_keywords() -> Tuple[str, ...]:
    cfg = get_line_ux_config()
    raw = cfg.get("cancel_keywords")
    if not isinstance(raw, list):
        return ("cancel", "キャンセル")
    resolved = tuple(str(item).strip() for item in raw if str(item).strip())
    return resolved if resolved else ("cancel", "キャンセル")


def get_line_support_messages(*, lang: str = "en") -> Dict[str, str]:
    return _resolve_support_messages(
        channel_key="line",
        lang=lang,
        field_keys=(
            "prompt",
            "cancel_ack",
            "escalation_ack",
            "takeover_ack",
            "resolved_ack",
            "email_details_no_message",
        ),
    )


def get_instagram_support_messages(*, lang: str = "en") -> Dict[str, str]:
    return _resolve_support_messages(
        channel_key="instagram",
        lang=lang,
        field_keys=(
            "prompt",
            "cancel_ack",
            "escalation_ack",
            "takeover_ack",
            "resolved_ack",
            "email_details_no_message",
        ),
    )


def get_line_rich_menu_definition(*, state: str = "normal", lang: str = "en") -> Dict[str, Any]:
    cfg = get_line_ux_config()
    rich_menu = cfg.get("rich_menu") if isinstance(cfg.get("rich_menu"), dict) else {}
    normalized_state = "support" if str(state or "").strip().lower() == "support" else "normal"
    normalized_lang = _normalize_lang(lang)
    chat_bar = rich_menu.get("chat_bar_text") if isinstance(rich_menu.get("chat_bar_text"), dict) else {}
    actions_cfg = rich_menu.get("actions") if isinstance(rich_menu.get("actions"), dict) else {}
    layouts_cfg = rich_menu.get("layouts") if isinstance(rich_menu.get("layouts"), dict) else {}
    layout_ids = layouts_cfg.get(normalized_state)
    if not isinstance(layout_ids, list):
        layout_ids = layouts_cfg.get("normal") if isinstance(layouts_cfg.get("normal"), list) else []
    actions: List[Dict[str, Any]] = []
    for raw_action_id in layout_ids:
        action_id = str(raw_action_id or "").strip()
        action_cfg = actions_cfg.get(action_id) if action_id else None
        if not action_id or not isinstance(action_cfg, dict):
            continue
        actions.append(
            {
                "id": action_id,
                "label": _resolve_i18n_text(action_cfg.get("label"), lang=normalized_lang, fallback=action_id.replace("_", " ").title()),
                "capability": str(action_cfg.get("capability") or action_id).strip() or action_id,
                "icon": str(action_cfg.get("icon") or action_id).strip() or action_id,
                "postback_data": str(action_cfg.get("postback_data") or "").strip() or None,
                "fallback_order": [
                    str(item).strip()
                    for item in (action_cfg.get("fallback_order") or [])
                    if str(item).strip()
                ],
                "uri_fallback_order": [
                    str(item).strip()
                    for item in (action_cfg.get("uri_fallback_order") or [])
                    if str(item).strip()
                ],
            }
        )
    return {
        "state": normalized_state,
        "chat_bar_text": _resolve_i18n_text(
            chat_bar.get(normalized_state),
            lang=normalized_lang,
            fallback="Quick actions" if normalized_state == "normal" else "Support options",
        ),
        "size": rich_menu.get("size") if isinstance(rich_menu.get("size"), dict) else {},
        "styles": rich_menu.get("styles") if isinstance(rich_menu.get("styles"), dict) else {},
        "actions": actions,
    }


def ensure_canonical_reservation_url_in_text(text: str, canonical_url: str, domain_key: str) -> str:
    """Replace any URLs from the given domain in text with the canonical URL."""
    import re
    if not text or not canonical_url or not domain_key:
        return text
    escaped = re.escape(domain_key)
    pattern = rf"https?://[^\s\)\]\"\']*{escaped}[^\s\)\]\"\']*"
    return re.sub(pattern, canonical_url, text, flags=re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
# RESOLVER + URL FILTERS
# Imported by both infrastructure (discovery) and application (crawl pipeline)
# so filtering rules are defined once and applied everywhere.
# ═══════════════════════════════════════════════════════════════════════════

import re
from typing import Tuple
from urllib.parse import unquote, urlparse


def resolve_platform_profile(url: str) -> Tuple[Optional[PlatformProfile], Optional[str]]:
    """
    Find a matching platform profile for the given URL.

    Args:
        url: The URL to match against registered profiles

    Returns:
        Tuple of (profile, domain_key) where:
        - profile: The matching PlatformProfile, or None if no match
        - domain_key: The key in PLATFORM_PROFILES (e.g. 'hotpepper.jp'), or None

    Behavior:
        - Matches domain_pattern as a regex against the URL
        - If multiple profiles match, returns the one with highest priority
        - If no profile matches, returns (None, None) → default crawl behavior applies
    """
    matches: List[Tuple[PlatformProfile, str, int]] = []

    for domain_key, profile in PLATFORM_PROFILES.items():
        if re.search(profile.domain_pattern, url):
            matches.append((profile, domain_key, profile.priority))

    if not matches:
        return None, None

    # Return highest priority match (or first if tied)
    profile, domain_key, _ = max(matches, key=lambda x: x[2])
    return profile, domain_key


def normalize_url_for_crawl(url: str) -> str:
    """Remove query parameters and fragments from a URL for pattern matching."""
    if '#' in url:
        url = url.split('#')[0]
    if '?' in url:
        url = url.split('?')[0]
    return url


def _decoded_path_segments(path: str) -> List[str]:
    """
    Return path segments decoded up to two rounds to handle mixed/double encoding.
    """
    out: List[str] = []
    for raw in (path or "").split("/"):
        seg = (raw or "").strip()
        if not seg:
            continue
        decoded = seg
        for _ in range(2):
            next_decoded = unquote(decoded)
            if next_decoded == decoded:
                break
            decoded = next_decoded
        out.append(decoded)
    return out


def is_junk_url(url: str) -> bool:
    """
    Detect obviously broken crawler artifacts that should never be discovered or crawled.

    Catches redirect URLs embedded in paths, malformed joined URLs, and CSS/tracking
    artifacts that get scraped as links. Applied globally regardless of platform.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
    except Exception:
        return True

    # URL-encoded URL in path (e.g., /strJ000.../https%3A%2F%2Fwww.hotpepper.jp%2F...)
    if "%3A%2F%2F" in path or "%3a%2f%2f" in path:
        return True

    # Path starts with /http:// or /https:// — malformed joined URL
    lower_path = path.lower()
    if lower_path.startswith("/http://") or lower_path.startswith("/https://"):
        return True

    # Comma in path segment — CSS media query or tracking artifact (e.g. /o,i.media=)
    if re.search(r"/[^/]*,[^/]*", path):
        return True

    # Segments that decode to quoted, space-containing payload text are usually
    # crawler artifacts (e.g., encoded UI labels), not real pages.
    for seg in _decoded_path_segments(path):
        s = seg.strip()
        has_double_quote = '"' in s or "“" in s or "”" in s
        if has_double_quote and (
            any(ch.isspace() for ch in s)
            or s.startswith(('"', "“", "”"))
            or s.endswith(('"', "“", "”"))
        ):
            return True

    return False


def should_allow_url(url: str) -> bool:
    """
    Single gate for both URL discovery and crawl indexing.

    Returns True if the URL should be:
    - Added to the discovery list shown in the dashboard
    - Followed during BFS link traversal
    - Indexed and crawled

    Applies in order:
    1. Global junk detection (broken artifacts, malformed paths)
    2. Platform-specific include/exclude rules from the matching profile
       (using normalized URL if profile.strip_query_params is True)

    Unknown domains (no matching profile) pass through by default.
    """
    if is_junk_url(url):
        return False

    profile, _ = resolve_platform_profile(url)
    if profile is None:
        return True  # unknown domain: allow

    # Normalize for matching only when the profile requests it
    check_url = normalize_url_for_crawl(url) if profile.strip_query_params else url

    # Exclude patterns take priority
    for pattern in profile.exclude_paths:
        if re.search(pattern, check_url):
            return False

    # If include patterns are defined, URL must match at least one
    if profile.include_paths:
        return any(re.search(p, check_url) for p in profile.include_paths)

    return True
