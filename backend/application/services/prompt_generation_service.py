"""
Prompt Generation Service

Analyzes crawled content using Gemini to generate a system prompt in the business language.
All templates and defaults are loaded from config/platform_profiles.yml.
"""

import logging
import os
from typing import Optional

from domain.platform_profiles import get_prompt_generation_config

logger = logging.getLogger(__name__)


def _generation_config() -> dict:
    cfg = get_prompt_generation_config()
    return dict(cfg) if isinstance(cfg, dict) else {}


def _cfg_str(cfg: dict, key: str) -> str:
    return str(cfg.get(key) or "").strip()


def _prompt_model() -> str:
    cfg = _generation_config()
    return (
        os.environ.get("VERTEX_PROMPT_GEN_MODEL")
        or os.environ.get("VERTEX_ASSET_MODEL")
        or _cfg_str(cfg, "model")
    ).strip()


def _standard_response_rules(lang: str) -> str:
    cfg = _generation_config()
    if lang == "ja":
        return _cfg_str(cfg, "standard_response_rules_ja")
    return _cfg_str(cfg, "standard_response_rules_en")


def _meta_prompt_for_homepage(lang: str) -> str:
    cfg = _generation_config()
    if lang == "ja":
        return _cfg_str(cfg, "meta_prompt_ja")
    return _cfg_str(cfg, "meta_prompt_en")


def _meta_prompt_for_rag(lang: str) -> str:
    cfg = _generation_config()
    if lang == "ja":
        return _cfg_str(cfg, "rag_meta_prompt_ja")
    return _cfg_str(cfg, "rag_meta_prompt_en")


def generate_prompt_from_rag_content(
    rag_snippets: list[str],
    business_name: str = "",
    lang: str = "en",
) -> Optional[str]:
    """
    Use Gemini to generate a professional system prompt from RAG snippets.

    Returns the generated prompt string (with response rules appended), or None on failure.
    """
    if not rag_snippets:
        logger.warning("[PromptGen] No RAG snippets provided")
        return None

    project = (os.environ.get("PROJECT_ID") or "").strip()
    if not project:
        logger.warning("[PromptGen] PROJECT_ID not set, cannot generate prompt")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("[PromptGen] google-genai not available")
        return None

    normalized_lang = "ja" if str(lang or "").strip().lower() in ("ja", "jp") else "en"
    context_block = "\n\n".join(rag_snippets[:4])
    template = _meta_prompt_for_rag(normalized_lang)
    if not template:
        logger.warning("[PromptGen] Missing rag prompt template for lang=%s", normalized_lang)
        return None

    prompt = template.format(
        business_name=business_name or ("ビジネス" if normalized_lang == "ja" else "the business"),
        content=context_block,
    )

    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    client = genai.Client(vertexai=True, project=project, location=location)

    try:
        response = client.models.generate_content(
            model=_prompt_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        generated = (response.text or "").strip()
        if not generated:
            logger.warning("[PromptGen] LLM returned empty response for RAG prompt")
            return None

        if generated.startswith("```"):
            lines = generated.splitlines()
            if lines and lines[-1].strip() == "```":
                generated = "\n".join(lines[1:-1]).strip()
            else:
                generated = "\n".join(lines[1:]).strip()

        rules = _standard_response_rules(normalized_lang)
        generated = generated + rules

        logger.info("[PromptGen] Generated RAG-based prompt (%d chars, lang=%s) for %s", len(generated), normalized_lang, business_name)
        return generated

    except Exception as e:
        logger.warning(
            "[PromptGen] LLM call failed for RAG prompt: %s: %s",
            type(e).__name__,
            str(e)[:200],
        )
        return None


def generate_prompt_from_content(
    homepage_content: str,
    root_url: str = "",
    business_name: str = "",
    lang: str = "en",
) -> Optional[str]:
    """
    Use Gemini to generate a professional system prompt from homepage content.

    Returns the generated prompt string, or None on failure.
    """
    if not homepage_content or not homepage_content.strip():
        logger.warning("[PromptGen] No homepage content provided")
        return None

    normalized_lang = "ja" if str(lang or "").strip().lower() in ("ja", "jp") else "en"

    project = (os.environ.get("PROJECT_ID") or "").strip()
    if not project:
        logger.warning("[PromptGen] PROJECT_ID not set, cannot generate prompt")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("[PromptGen] google-genai not available")
        return None

    max_content = int(os.environ.get("PROMPT_GEN_MAX_CONTENT_CHARS", "12000"))
    content = homepage_content[:max_content]

    if normalized_lang == "ja":
        template = _meta_prompt_for_homepage("ja") or _meta_prompt_for_rag("ja")
        if not template:
            logger.warning("[PromptGen] Missing homepage/rag prompt template for Japanese")
            return None
        prompt = template.format(
            business_name=business_name or "ビジネス",
            content=content,
            url=root_url or "unknown",
        )
    else:
        template = _meta_prompt_for_homepage("en")
        if not template:
            logger.warning("[PromptGen] Missing homepage prompt template for English")
            return None
        prompt = template.format(
            url=root_url or "unknown",
            content=content,
            business_name=business_name or "the business",
        )

        if business_name:
            identity_instruction_template = _cfg_str(_generation_config(), "identity_instruction_en")
            identity_instruction = identity_instruction_template.format(business_name=business_name) if identity_instruction_template else ""
            if identity_instruction:
                prompt = identity_instruction + "\n\n" + prompt

    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    client = genai.Client(vertexai=True, project=project, location=location)

    try:
        response = client.models.generate_content(
            model=_prompt_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        generated = (response.text or "").strip()
        if not generated:
            logger.warning("[PromptGen] LLM returned empty response")
            return None

        if generated.startswith("```"):
            lines = generated.splitlines()
            if lines and lines[-1].strip() == "```":
                generated = "\n".join(lines[1:-1]).strip()
            else:
                generated = "\n".join(lines[1:]).strip()

        rules = _standard_response_rules(normalized_lang)
        generated = generated + rules

        logger.info("[PromptGen] Generated prompt (%d chars, lang=%s) for %s", len(generated), normalized_lang, root_url)
        return generated

    except Exception as e:
        logger.warning(
            "[PromptGen] LLM call failed: %s: %s",
            type(e).__name__,
            str(e)[:200],
        )
        return None


_STANDARD_RESPONSE_RULES = _standard_response_rules("en")
_STANDARD_RESPONSE_RULES_JA = _standard_response_rules("ja")
