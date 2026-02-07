"""URL pattern inference and construction for generic booking sites."""

import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from openai import OpenAI

from infrastructure.availability.logging_utils import get_logger


# Canonical keys used internally
CANONICAL_CHECKIN = "checkin"
CANONICAL_CHECKOUT = "checkout"
CANONICAL_ADULTS = "adults"
CANONICAL_CHILDREN = "children"
CANONICAL_ROOMS = "rooms"

# Fallback when LLM fails or URL has no query params
DEFAULT_PARAM_MAPPING: Dict[str, str] = {
    CANONICAL_CHECKIN: "checkin",
    CANONICAL_CHECKOUT: "checkout",
    CANONICAL_ADULTS: "group_adults",
    CANONICAL_CHILDREN: "group_children",
    CANONICAL_ROOMS: "no_rooms",
}


def get_default_pattern(base_url: str) -> Dict[str, Any]:
    """Return a pattern with Booking.com-style param mapping for fallback when LLM cannot deduce."""
    return {
        "base_url": base_url,
        "param_mapping": dict(DEFAULT_PARAM_MAPPING),
    }


def infer_pattern(url: str) -> Optional[Dict[str, Any]]:
    """
    Use LLM to deduce the booking URL pattern from the test URL.
    Returns { "base_url": str, "param_mapping": { canonical_key -> site_param_name } }
    or None if URL cannot be parsed or LLM fails.
    """
    log = get_logger()
    log.info("infer_pattern input url=%s", url[:120] + "..." if len(url) > 120 else url)
    if not url or not url.strip():
        log.warning("infer_pattern: empty url")
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    model = (os.environ.get("AVAILABILITY_SUMMARY_MODEL") or "gpt-4o-mini").strip()
    log.debug("infer_pattern using model=%s", model)
    if not model or model.lower() in ("none", "disabled", "off"):
        log.info("infer_pattern: LLM disabled, using fallback")
        return _infer_pattern_fallback(url)

    log.debug("infer_pattern prompt (abbreviated): %s", "Analyze booking URL...")
    prompt = (
        "You are analyzing a hotel/booking website URL to extract its query parameter structure.\n\n"
        "Given this URL, identify which query parameters correspond to:\n"
        "- checkin: check-in / arrival date (YYYY-MM-DD or similar)\n"
        "- checkout: check-out / departure date\n"
        "- adults: number of adults\n"
        "- children: number of children (omit if not present)\n"
        "- rooms: number of rooms\n\n"
        "Return a JSON object with:\n"
        '1. "base_url": the URL with ONLY the fixed parts (path, domain) and any non-date/guest params (lang, etc). '
        "Strip the date and guest params - we will add them when building.\n"
        '2. "param_mapping": object mapping canonical keys to the exact param names used by this site. '
        'Keys: "checkin", "checkout", "adults", "children" (optional), "rooms". '
        'Values: the exact query param name as it appears in the URL (e.g. "checkIn", "group_adults").\n\n'
        f"URL:\n{url}\n\n"
        "Reply with ONLY valid JSON, no markdown or other text. Example:\n"
        '{"base_url": "https://example.com/hotel/123?lang=en", "param_mapping": {"checkin": "checkIn", "checkout": "checkOut", "adults": "guests", "rooms": "roomCount"}}'
    )

    try:
        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            truncation="auto",
        )
        text = _extract_response_text(response)
        log.debug("infer_pattern LLM raw response (first 500 chars): %s", (text or "")[:500])
        pattern = _parse_llm_pattern_response(text, url)
        if pattern and pattern.get("param_mapping"):
            log.info("infer_pattern result: base_url=%s param_mapping=%s",
                     pattern.get("base_url", "")[:80], pattern.get("param_mapping"))
            return pattern
    except Exception as e:
        log.warning("infer_pattern LLM failed: %s", e)

    log.info("infer_pattern: falling back to alias matching")
    return _infer_pattern_fallback(url)


def _extract_response_text(response: Any) -> str:
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


def _parse_llm_pattern_response(text: str, original_url: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response and validate against original URL params."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    base_url = (data.get("base_url") or "").strip()
    mapping = data.get("param_mapping")
    if not isinstance(mapping, dict):
        return None

    # Validate: mapped params should exist in original URL (case-insensitive match)
    try:
        parsed = urlparse(original_url)
        url_param_list = list(parse_qs(parsed.query, keep_blank_values=True).keys())
        url_params_lower = {p.lower(): p for p in url_param_list}
    except Exception:
        return None

    validated: Dict[str, str] = {}
    for canonical in (CANONICAL_CHECKIN, CANONICAL_CHECKOUT, CANONICAL_ADULTS, CANONICAL_CHILDREN, CANONICAL_ROOMS):
        site_param = mapping.get(canonical)
        if not site_param:
            continue
        sp = str(site_param).strip()
        # Use exact match first, else case-insensitive
        if sp in url_param_list:
            validated[canonical] = sp
        elif sp.lower() in url_params_lower:
            validated[canonical] = url_params_lower[sp.lower()]

    if not validated:
        return None

    if not base_url.startswith(("http://", "https://")):
        base_url = original_url.split("?")[0] if "?" in original_url else original_url

    return {"base_url": base_url, "param_mapping": validated}


def _infer_pattern_fallback(url: str) -> Optional[Dict[str, Any]]:
    """
    Fallback when LLM is disabled: parse URL and match known param names.
    Only used when LLM cannot run; supports common booking sites.
    """
    KNOWN_ALIASES: Dict[str, str] = {
        "checkin": CANONICAL_CHECKIN,
        "check_in": CANONICAL_CHECKIN,
        "checkin_date": CANONICAL_CHECKIN,
        "arrival": CANONICAL_CHECKIN,
        "datein": CANONICAL_CHECKIN,
        "checkout": CANONICAL_CHECKOUT,
        "check_out": CANONICAL_CHECKOUT,
        "checkout_date": CANONICAL_CHECKOUT,
        "departure": CANONICAL_CHECKOUT,
        "dateout": CANONICAL_CHECKOUT,
        "adults": CANONICAL_ADULTS,
        "group_adults": CANONICAL_ADULTS,
        "guests": CANONICAL_ADULTS,
        "children": CANONICAL_CHILDREN,
        "group_children": CANONICAL_CHILDREN,
        "rooms": CANONICAL_ROOMS,
        "no_rooms": CANONICAL_ROOMS,
        "numrooms": CANONICAL_ROOMS,
    }
    try:
        parsed = urlparse(url)
        original_params = parse_qs(parsed.query, keep_blank_values=True)
    except Exception:
        return None

    param_mapping: Dict[str, str] = {}
    params_to_strip: List[str] = []

    for qs_key in original_params:
        key_lower = qs_key.lower().replace("-", "_")
        canonical = KNOWN_ALIASES.get(key_lower)
        if canonical:
            param_mapping[canonical] = qs_key
            params_to_strip.append(qs_key)

    if not param_mapping:
        return None

    base_params: Dict[str, List[str]] = {}
    for k, v in original_params.items():
        if k not in params_to_strip and v:
            base_params[k] = [v[0]]

    base_query = urlencode(base_params, doseq=True) if base_params else ""
    base_url = urlunparse(parsed._replace(query=base_query, fragment=""))

    return {"base_url": base_url, "param_mapping": param_mapping}


def build_url(
    pattern: Dict[str, Any],
    check_in: str,
    check_out: str,
    adults: int = 2,
    children: int = 0,
    rooms: int = 1,
) -> str:
    """
    Build a booking URL from a saved pattern and date/guest values.
    Dates should be YYYY-MM-DD. Adds or replaces params on base_url.
    """
    base_url = pattern.get("base_url", "")
    mapping = pattern.get("param_mapping") or {}
    if not base_url or not mapping:
        return base_url

    try:
        parsed = urlparse(base_url)
        original_params = parse_qs(parsed.query, keep_blank_values=True)
    except Exception:
        return base_url

    # Map canonical -> value
    values: Dict[str, str] = {
        CANONICAL_CHECKIN: check_in,
        CANONICAL_CHECKOUT: check_out,
        CANONICAL_ADULTS: str(max(1, adults)),
        CANONICAL_CHILDREN: str(max(0, children)),
        CANONICAL_ROOMS: str(max(1, rooms)),
    }

    for canonical, site_param in mapping.items():
        if canonical in values:
            original_params[site_param] = [values[canonical]]

    query = urlencode(original_params, doseq=True)
    return urlunparse(parsed._replace(query=query, fragment=""))
