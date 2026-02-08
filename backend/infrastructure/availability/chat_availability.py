"""Chat widget availability: intent detection and sync check."""

import asyncio
import concurrent.futures
import json
import os
import re
from datetime import date, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

# Skip reasons for chat availability (for debugging)
SKIP_NOT_HOTEL = "businessType is not hotel"
SKIP_REALTIME_DISABLED = "allowRealtimeAvailability is false or missing"
SKIP_NO_BOOKING_URL = "bookingUrlPattern and bookingTestUrl both missing"
SKIP_NO_INTENT = "message does not look like availability/booking intent"

from openai import OpenAI

from common.di.container import bot_service
from infrastructure.availability.extraction import run_generic_extraction
from infrastructure.availability.llm_extraction import extract_availability_summary
from infrastructure.availability.url_pattern import build_url, get_default_pattern, infer_pattern

# Keywords that suggest availability/booking intent
AVAILABILITY_KEYWORDS = frozenset({
    "available", "availability", "rooms", "room", "book", "booking", "reservation",
    "reserve", "check", "vacancy", "vacancies", "stay", "accommodation", "prices",
    "rate", "rates", "tomorrow", "tonight", "weekend", "next week", "dates",
})


def _normalize(msg: str) -> str:
    return msg.lower().strip()


def detect_availability_intent(message: str) -> bool:
    """Heuristic: does the message ask about room availability or booking?"""
    norm = _normalize(message)
    if len(norm) < 10:
        return False
    words = set(re.findall(r"\b\w+\b", norm))
    return bool(words & AVAILABILITY_KEYWORDS)


# Regex for structured format from availability form: "Check room availability: check-in YYYY-MM-DD, check-out YYYY-MM-DD, N adults, N room(s)"
_STRUCTURED_PATTERN = re.compile(
    r"check-in\s+(\d{4}-\d{2}-\d{2})\s*,\s*check-out\s+(\d{4}-\d{2}-\d{2})\s*,\s*(\d+)\s+adults?\s*,\s*(\d+)\s+rooms?",
    re.IGNORECASE,
)


def _parse_structured_availability(message: str) -> Optional[Tuple[str, str, int, int]]:
    """
    Parse structured availability message from the widget form.
    Returns (check_in, check_out, adults, rooms) or None if no match.
    """
    match = _STRUCTURED_PATTERN.search(message)
    if not match:
        return None
    check_in = match.group(1).strip()
    check_out = match.group(2).strip()
    adults = max(1, int(match.group(3)))
    rooms = max(1, int(match.group(4)))
    return (check_in, check_out, adults, rooms)


def _extract_json_text(response: Any) -> str:
    """Extract text from OpenAI response for JSON parsing."""
    if response is None:
        return ""
    if isinstance(response, dict):
        return response.get("output_text") or response.get("text") or ""
    text = getattr(response, "output_text", None) or getattr(response, "text", None)
    if text:
        return str(text)
    try:
        dumped = response.model_dump()
        return dumped.get("output_text") or dumped.get("text") or ""
    except Exception:
        return ""


def _parse_dates_from_message(message: str) -> Tuple[str, str, int, int]:
    """
    Try to extract check_in, check_out, rooms, adults from message using LLM.
    Returns (check_in, check_out, rooms, adults), using defaults on failure.
    """
    today = date.today()
    default_checkin = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    default_checkout = (today + timedelta(days=9)).strftime("%Y-%m-%d")
    defaults = (default_checkin, default_checkout, 1, 2)

    model = (os.environ.get("AVAILABILITY_SUMMARY_MODEL") or "gpt-4o-mini").strip()
    if not model:
        return defaults

    prompt = (
        "Extract from this user message about hotel/room availability:\n"
        "- check_in: date as YYYY-MM-DD (default 7 days from today)\n"
        "- check_out: date as YYYY-MM-DD (default 2 days after check_in)\n"
        "- rooms: integer (default 1)\n"
        "- adults: integer (default 2)\n"
        f"Today is {today.isoformat()}.\n\n"
        f'User message: "{message}"\n\n'
        "Reply with ONLY valid JSON: "
        '{"check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "rooms": 1, "adults": 2}'
    )
    try:
        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            truncation="auto",
        )
        text = _extract_json_text(response)
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            data = json.loads(match.group())
            ci = str(data.get("check_in") or default_checkin).strip()
            co = str(data.get("check_out") or default_checkout).strip()
            r = max(1, int(data.get("rooms", 1)))
            a = max(1, int(data.get("adults", 2)))
            return (ci, co, r, a)
    except Exception:
        pass
    return defaults


def run_availability_check_sync(
    bot_id: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    rooms: int = 1,
    timeout: int = 30,
) -> Optional[str]:
    """
    Run availability check synchronously for chat. Returns summary string or None on failure.
    """
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        return None

    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass

    booking_pattern = widget_config.get("bookingUrlPattern") if isinstance(widget_config, dict) else None
    booking_test_url = (
        widget_config.get("bookingTestUrl")
        if isinstance(widget_config, dict) else None
    )
    base_url = (booking_test_url or "").strip()
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url if base_url else ""

    if not base_url:
        return None

    if booking_pattern and isinstance(booking_pattern, dict) and booking_pattern.get("param_mapping"):
        resolved_url = build_url(
            booking_pattern,
            check_in=check_in,
            check_out=check_out,
            adults=adults,
            children=0,
            rooms=rooms,
        )
    else:
        inferred = infer_pattern(base_url)
        if inferred:
            resolved_url = build_url(
                inferred,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=0,
                rooms=rooms,
            )
        else:
            resolved_url = build_url(
                get_default_pattern(base_url),
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=0,
                rooms=rooms,
            )

    def _run_in_thread() -> Tuple[str, int, str, str]:
        # Run in a thread so asyncio.run() works when called from FastAPI's async handler (which has a running event loop).
        return asyncio.run(
            run_generic_extraction(
                url=resolved_url,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=0,
                rooms=rooms,
                max_seconds=timeout,
                write_screenshot=None,
            )
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_thread)
            summary, _, raw_text, _ = future.result(timeout=timeout + 15)
    except concurrent.futures.TimeoutError:
        return "The availability check is taking longer than expected. Please try again or check the Testing tab."
    except asyncio.TimeoutError:
        return "The availability check is taking longer than expected. Please try again or check the Testing tab."
    except Exception as e:
        return f"Could not complete availability check: {str(e)[:200]}"

    llm_summary = extract_availability_summary(
        question="Summarize availability and pricing for the user.",
        url=resolved_url,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        children=0,
        rooms=rooms,
        extracted_text=raw_text or summary,
    )
    return llm_summary or summary


def maybe_run_chat_availability(
    bot_id: str,
    message: str,
    widget_config: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    If message asks about availability and bot has booking config, run check and return summary.
    Returns (summary, skip_reason). summary is set when check ran; skip_reason is set when we skipped (for debugging).
    """
    if widget_config.get("businessType") != "hotel":
        return (None, SKIP_NOT_HOTEL)
    if not widget_config.get("allowRealtimeAvailability"):
        return (None, SKIP_REALTIME_DISABLED)
    if not (widget_config.get("bookingUrlPattern") or widget_config.get("bookingTestUrl")):
        return (None, SKIP_NO_BOOKING_URL)
    if not detect_availability_intent(message):
        return (None, SKIP_NO_INTENT)

    parsed = _parse_structured_availability(message)
    if parsed is not None:
        check_in, check_out, adults, rooms = parsed
    else:
        check_in, check_out, rooms, adults = _parse_dates_from_message(message)

    summary = run_availability_check_sync(
        bot_id=bot_id,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        rooms=rooms,
        timeout=30,
    )
    if summary:
        # Prepend dates so the RAG model knows this availability is for the user's requested dates
        # (avoids confusion when cancellation dates like 2026/02/26 appear in the summary)
        summary = f"**Live availability for check-in {check_in}, check-out {check_out}:**\n\n{summary}"
    return (summary, None)