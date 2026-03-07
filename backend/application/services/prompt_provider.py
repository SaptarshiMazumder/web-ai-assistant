from __future__ import annotations

from typing import Any, Dict, Optional

from application.services.default_prompt_service import (
    build_default_system_instruction,
    extract_business_type_from_widget_config,
)
from domain.personas import get_default_persona_id, get_persona_system_prompt


def _normalize_prompt(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


class ConfigPromptProvider:
    """Prompt selection strategy used by route handlers."""

    def resolve_system_prompt(
        self,
        *,
        agent_config: Dict[str, Any],
        lang: str,
        bot_name: str,
        widget_config: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        default_persona_id = get_default_persona_id()
        explicit = agent_config.get("instructions") if isinstance(agent_config, dict) else None
        if explicit and str(explicit).strip():
            default_prompt = get_persona_system_prompt(default_persona_id, lang=lang) or ""
            if _normalize_prompt(str(explicit)) != _normalize_prompt(default_prompt):
                return str(explicit)

        persona_id_raw = agent_config.get("persona_id") if isinstance(agent_config, dict) else None
        persona_id = str(persona_id_raw).strip() if persona_id_raw else default_persona_id
        has_explicit_non_default = bool(persona_id_raw and persona_id != default_persona_id)

        if has_explicit_non_default:
            builtin = get_persona_system_prompt(persona_id, lang=lang)
            if builtin:
                return builtin
            for cp in (agent_config.get("custom_personas") or []):
                if isinstance(cp, dict) and cp.get("id") == persona_id and cp.get("system_prompt"):
                    return str(cp["system_prompt"])

        return build_default_system_instruction(
            bot_name=(bot_name or "").strip(),
            business_type=extract_business_type_from_widget_config(widget_config),
            lang=lang,
        )
