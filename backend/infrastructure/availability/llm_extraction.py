"""LLM extraction of availability summary from page text."""

import os
from typing import Any, Optional

from openai import OpenAI

from infrastructure.availability.logging_utils import get_logger


def _extract_text_output(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        return response.get("output_text") or response.get("text") or ""
    text = getattr(response, "output_text", None)
    if text:
        return text
    text = getattr(response, "text", None)
    if text:
        return text
    try:
        dumped = response.model_dump()  # type: ignore[attr-defined]
        return dumped.get("output_text") or dumped.get("text") or ""
    except Exception:
        return ""


def _truncate_text(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def extract_availability_summary(
    *,
    question: str,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    extracted_text: str,
) -> Optional[str]:
    """Use LLM to extract a structured availability summary from page text."""
    model = (os.environ.get("AVAILABILITY_SUMMARY_MODEL") or "gpt-4o-mini").strip()
    if not model or model.lower() in ("none", "disabled", "off"):
        return None
    if not extracted_text:
        return None
    log = get_logger()
    user_question = question.strip() or "Summarize the availability and pricing for the selected dates."
    prompt = (
        "You are a hotel availability assistant. Answer the user's question using ONLY the extracted text.\n"
        "Extract room names, room types, and prices (including currency) if present. "
        "Look for room cards, rate plans, totals, or any pricing/availability info. "
        "If the page has login prompts, region selectors, or cookie banners, note that. "
        "If you find partial info (e.g. room names without prices), include it. "
        "Only say 'could not find' if the text has no room/rate/availability content at all.\n"
        "Keep the answer concise and structured.\n\n"
        f"User question: {user_question}\n"
        f"URL: {url}\n"
        f"Check-in: {check_in}\n"
        f"Check-out: {check_out}\n"
        f"Adults: {adults}\n"
        f"Children: {children}\n"
        f"Rooms: {rooms}\n\n"
        "Extracted text:\n"
        f"{_truncate_text(extracted_text)}\n"
    )
    log.debug("llm_extraction prompt len=%d, extracted_text len=%d", len(prompt), len(extracted_text))
    log.debug("llm_extraction prompt (first 600 chars): %s", prompt[:600])
    try:
        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            truncation="auto",
        )
        answer = _extract_text_output(response).strip()
        log.info("llm_extraction result (first 300 chars): %s", (answer or "")[:300])
        return answer or None
    except Exception as e:
        log.warning("llm_extraction failed: %s", e)
        return None
