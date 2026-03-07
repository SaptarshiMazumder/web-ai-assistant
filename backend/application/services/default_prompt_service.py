"""
Deterministic default system-instruction builder (no LLM).
"""

from typing import Any, Dict, Optional

from domain.platform_profiles import get_deterministic_prompt_config


def extract_business_type_from_widget_config(widget_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return normalized business type label from widget config, or None."""
    if not isinstance(widget_config, dict):
        return None
    raw = str(widget_config.get("businessType") or "").strip()
    if not raw:
        return None
    normalized = raw.replace("_", " ").replace("-", " ").strip().lower()
    return normalized or None


def build_default_system_instruction(
    *,
    bot_name: str,
    business_type: Optional[str],
    lang: str = "en",
) -> str:
    """
    Build a deterministic system instruction using bot name + optional business type.
    Config-driven from platform_profiles.yml (defaults.prompts.deterministic).
    """
    normalized_lang = "ja" if str(lang or "").strip().lower() in ("ja", "jp") else "en"
    cfg = get_deterministic_prompt_config()

    default_name_cfg = cfg.get("default_business_name") if isinstance(cfg.get("default_business_name"), dict) else {}
    fallback_name = str(default_name_cfg.get(normalized_lang) or default_name_cfg.get("en") or "").strip()
    name = (bot_name or "").strip() or fallback_name
    business = (business_type or "").strip()

    with_bt_cfg = cfg.get("personality_with_business_type") if isinstance(cfg.get("personality_with_business_type"), dict) else {}
    without_bt_cfg = cfg.get("personality_without_business_type") if isinstance(cfg.get("personality_without_business_type"), dict) else {}
    with_bt_tpl = str(with_bt_cfg.get(normalized_lang) or with_bt_cfg.get("en") or "").strip()
    without_bt_tpl = str(without_bt_cfg.get(normalized_lang) or without_bt_cfg.get("en") or "").strip()

    personality = with_bt_tpl.format(bot_name=name, business_type=business) if business else without_bt_tpl.format(bot_name=name)

    section_titles = cfg.get("section_titles") if isinstance(cfg.get("section_titles"), dict) else {}
    personality_title_cfg = section_titles.get("personality") if isinstance(section_titles.get("personality"), dict) else {}
    response_title_cfg = section_titles.get("response_rules") if isinstance(section_titles.get("response_rules"), dict) else {}
    personality_title = str(personality_title_cfg.get(normalized_lang) or personality_title_cfg.get("en") or "").strip()
    response_title = str(response_title_cfg.get(normalized_lang) or response_title_cfg.get("en") or "").strip()

    rules_cfg = cfg.get("response_rules") if isinstance(cfg.get("response_rules"), dict) else {}
    rule_lines = rules_cfg.get(normalized_lang) or rules_cfg.get("en") or []
    if not isinstance(rule_lines, list):
        rule_lines = []
    rendered_rules = "\n".join(f"- {str(rule).strip()}" for rule in rule_lines if str(rule).strip())

    return f"{personality_title}\n{personality}\n\n{response_title}\n{rendered_rules}".strip()
