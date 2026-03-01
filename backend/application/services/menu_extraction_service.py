"""
Menu Extraction Service

Extracts structured menu items (courses, dishes, food items) from crawled
restaurant pages using Gemini LLM. Stores results as BotAsset records with
asset_type="menu_item", reusing the same image download, GCS storage, and
chat injection infrastructure as image assets.

Platform-aware: uses menu_url_patterns from PlatformProfile to identify
which crawled pages contain menu data.
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import html as html_lib
import httpx

from application.services.asset_extraction_service import (
    _chunk_content,
    _download_and_store_image,
    _load_docs_from_gcs,
    _normalize_url_for_match,
    _parse_llm_json,
    _scrape_image_urls,
)
from domain.entities import BotAsset
from domain.platform_profiles import resolve_platform_profile
from infrastructure.db.repositories import PostgresBotAssetRepository

logger = logging.getLogger(__name__)

_HTML_DIV_TAG_RE = re.compile(r"<div\b[^>]*>|</div\s*>", re.IGNORECASE)
_HTML_CLASS_ATTR_RE = re.compile(r'class\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PRICE_TEXT_RE = re.compile(
    r"(?:[\u00A5\uFFE5]\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:[-~\u301C]|to)\s*[\u00A5\uFFE5]?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|\d[\d,]*(?:\.\d+)?\s*\u5186(?:\s*(?:[-~\u301C]|to)\s*\d[\d,]*(?:\.\d+)?\s*\u5186)?"
    r"|\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:[-~]|to)\s*\$?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|\u20AC\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:[-~]|to)\s*\u20AC?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|\u00A3\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:[-~]|to)\s*\u00A3?\s*\d[\d,]*(?:\.\d+)?)?)",
    re.IGNORECASE,
)
_PRICE_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = (os.environ.get(name) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))

_MENU_EXTRACTION_PROMPT = """You are analyzing a restaurant webpage to extract menu items.
Extract every distinct menu item, course, set meal, or food/drink offering on this page.

CRITICAL: Keep all extracted text in the ORIGINAL language of the page.
Do NOT translate anything.

For each menu item found, return:
- name: Dish/course/item name
- price_text: Price text if available (for example "¥3,500", "$12.99", "1,200円"), else empty string
- details: Item details without repeating the price
- description: Combined AI-friendly text that includes both price_text and details when available
- category: One of "course", "dish", "drink", "lunch", or "menu"
- link_url: The page URL or a more specific URL for this item, or null
- image_url: Pick the best matching dish image URL from the image list below, or null
- keywords: 3-6 keywords in original language

Rules:
- Extract every distinct menu item/course/set meal; do not skip items.
- Only use image URLs from "Available images". Do not invent URLs.
- Return [] if no menu items exist on the page.
- Do not extract non-menu information (address, opening hours, maps, etc.).

Page URL: {page_url}

Available images on this page:
{image_list}

Page content:
{page_content}

Return ONLY a JSON array. No markdown fences. No commentary.
Example: [{{"name":"Special Course","price_text":"¥5,500","details":"8 dishes full course","description":"¥5,500 8 dishes full course","category":"course","link_url":null,"image_url":"https://example.com/img/course.jpg","keywords":["course","dinner","8 dishes"]}}]"""

_MENU_PROMPT_IMAGE_LIMIT = _bounded_env_int(
    "MENU_PROMPT_IMAGE_LIMIT",
    default=120,
    minimum=20,
    maximum=400,
)


def _has_class_token(open_div_tag: str, class_token: str) -> bool:
    if not open_div_tag or not class_token:
        return False
    m = _HTML_CLASS_ATTR_RE.search(open_div_tag)
    if not m:
        return False
    tokens = [t.strip() for t in m.group(1).split() if t.strip()]
    return class_token in tokens


def _iter_div_blocks_by_class(html_text: str, class_token: str) -> List[str]:
    if not html_text or not class_token:
        return []

    blocks: List[str] = []
    stack: List[Tuple[int, bool]] = []

    for m in _HTML_DIV_TAG_RE.finditer(html_text):
        tag = m.group(0)
        if tag.lower().startswith("</div"):
            if not stack:
                continue
            start_idx, is_target = stack.pop()
            if is_target:
                blocks.append(html_text[start_idx:m.end()])
            continue

        stack.append((m.start(), _has_class_token(tag, class_token)))

    return blocks


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = _HTML_BR_RE.sub(" ", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html_lib.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_first_match(text: str, pattern: str) -> str:
    if not text:
        return ""
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _extract_text_by_class(html_text: str, class_token: str, tag_name: str) -> str:
    if not html_text:
        return ""
    pattern = (
        rf"<{tag_name}\b[^>]*class\s*=\s*['\"][^'\"]*\b{re.escape(class_token)}\b[^'\"]*['\"][^>]*>"
        rf"(.*?)</{tag_name}>"
    )
    raw = _extract_first_match(html_text, pattern)
    return _strip_html(raw)


def _extract_href_by_class(html_text: str, class_token: str) -> str:
    if not html_text:
        return ""
    anchor_tags = re.findall(r"<a\b[^>]*>", html_text, re.IGNORECASE)
    class_re = re.compile(
        rf"\bclass\s*=\s*['\"][^'\"]*\b{re.escape(class_token)}\b[^'\"]*['\"]",
        re.IGNORECASE,
    )
    href_re = re.compile(r"\bhref\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    for tag in anchor_tags:
        if not class_re.search(tag):
            continue
        m = href_re.search(tag)
        if m:
            return html_lib.unescape((m.group(1) or "").strip())
    return ""


def _extract_first_img_src(html_text: str) -> str:
    if not html_text:
        return ""
    for attr in ("src", "data-src", "data-original", "data-lazy-src"):
        pattern = rf"<img\b[^>]*\b{attr}\s*=\s*['\"]([^'\"]+)['\"]"
        value = _extract_first_match(html_text, pattern)
        if value:
            return html_lib.unescape(value)
    return ""


def _extract_tabelog_menu_items_with_bs4(page_url: str, html_text: str) -> List[Dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return []

    soup = BeautifulSoup(html_text or "", "html.parser")
    if not soup:
        return []

    items: List[Dict[str, Any]] = []

    # Course blocks (root page and /party pages)
    for block in soup.select("div.rstdtl-course-list"):
        name_el = block.select_one(".rstdtl-course-list__course-title-text")
        if name_el is None:
            name_el = block.select_one(".rstdtl-course-list__course-title")
        name = _strip_html(name_el.get_text(" ", strip=True) if name_el else "")
        if not name:
            continue

        trigger = block.select_one(".js-show-yoyaku-modal-trigger-course")
        price = ""
        if trigger is not None:
            real_price = _strip_html(str(trigger.get("data-real-price") or ""))
            tax = _strip_html(str(trigger.get("data-tax") or ""))
            if real_price:
                if not any(u in real_price for u in ("\u5186", "\u00A5", "\uFFE5")):
                    real_price = f"{real_price}\u5186"
                price = _format_menu_description(real_price, tax)

        if not price:
            price_el = block.select_one(".rstdtl-course-list__price")
            if price_el is not None:
                price = _strip_html(price_el.get_text(" ", strip=True))

        desc_el = block.select_one(".rstdtl-course-list__desc")
        body = _strip_html(desc_el.get_text(" ", strip=True) if desc_el else "")
        description = _format_menu_description(price, body)

        link_el = block.select_one("a.rstdtl-course-list__target[href]")
        if link_el is None:
            link_el = block.select_one("a.rstdtl-course-list__img-target[href]")
        href = str(link_el.get("href") or "").strip() if link_el is not None else ""
        resolved_link = urljoin(page_url, href) if href else page_url

        img_el = block.select_one("img")
        image_src = ""
        if img_el is not None:
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                candidate = str(img_el.get(attr) or "").strip()
                if candidate:
                    image_src = candidate
                    break
        resolved_image = urljoin(page_url, image_src) if image_src else None

        items.append(
            {
                "name": name,
                "price_text": price,
                "details": body,
                "description": description,
                "link_url": resolved_link,
                "image_url": resolved_image,
                "category": "course",
                "keywords": _keywords_for_menu_item(name, description),
            }
        )

    # Dishes blocks (/dtlmenu pages)
    for block in soup.select("div.rstdtl-menu-lst__contents"):
        name_el = block.select_one(".rstdtl-menu-lst__menu-title")
        name = _strip_html(name_el.get_text(" ", strip=True) if name_el else "")
        if not name:
            continue

        price_el = block.select_one(".rstdtl-menu-lst__price")
        body_el = block.select_one(".rstdtl-menu-lst__ex")
        price = _strip_html(price_el.get_text(" ", strip=True) if price_el else "")
        body = _strip_html(body_el.get_text(" ", strip=True) if body_el else "")
        description = _format_menu_description(price, body)

        img_el = block.select_one("img")
        image_src = ""
        if img_el is not None:
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                candidate = str(img_el.get(attr) or "").strip()
                if candidate:
                    image_src = candidate
                    break
        resolved_image = urljoin(page_url, image_src) if image_src else None

        items.append(
            {
                "name": name,
                "price_text": price,
                "details": body,
                "description": description,
                "link_url": page_url,
                "image_url": resolved_image,
                "category": "dish",
                "keywords": _keywords_for_menu_item(name, description),
            }
        )

    return items


def _dedupe_menu_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        link = (item.get("link_url") or "").strip().rstrip("/")
        key = (name.lower(), link.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _keywords_for_menu_item(name: str, description: str) -> List[str]:
    text = f"{name} {description}".strip()
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9\u3040-\u30FF\u3400-\u9FFF]+", text.lower())
    out: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        if len(token) < 2 and not re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", token):
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 6:
            break
    return out


def _format_menu_description(price: str, body: str) -> str:
    parts = [p for p in [price.strip(), body.strip()] if p and p.strip()]
    if not parts:
        return ""
    return " ".join(parts).strip()


def _split_price_from_text(text: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    match = _PRICE_TEXT_RE.search(raw)
    if not match:
        return "", raw
    price_text = (match.group(0) or "").strip()
    details = (raw[: match.start()] + " " + raw[match.end() :]).strip(" -|:;/,")
    details = re.sub(r"\s+", " ", details).strip()
    return price_text, details


def _currency_from_price_text(price_text: str) -> str:
    text = (price_text or "").strip()
    if not text:
        return ""
    if any(ch in text for ch in ("\u00A5", "\uFFE5", "\u5186")):
        return "JPY"
    if "$" in text:
        return "USD"
    if "\u20AC" in text:
        return "EUR"
    if "\u00A3" in text:
        return "GBP"
    return ""


def _normalize_price_value(value: float) -> Any:
    if abs(value - round(value)) < 0.000001:
        return int(round(value))
    return round(value, 2)


def _price_metadata(price_text: str) -> Dict[str, Any]:
    text = (price_text or "").strip()
    if not text:
        return {}
    raw_values = _PRICE_NUMBER_RE.findall(text)
    numbers: List[float] = []
    for raw in raw_values:
        candidate = (raw or "").replace(",", "")
        if not candidate:
            continue
        try:
            numbers.append(float(candidate))
        except Exception:
            continue
    amount_min: Optional[Any] = None
    amount_max: Optional[Any] = None
    if numbers:
        amount_min = _normalize_price_value(numbers[0])
        if len(numbers) >= 2 and re.search(r"(?:-|~|〜|\bto\b)", text, re.IGNORECASE):
            amount_max = _normalize_price_value(numbers[-1])
        else:
            amount_max = amount_min
    meta: Dict[str, Any] = {
        "text": text,
        "currency": _currency_from_price_text(text),
    }
    if amount_min is not None:
        meta["amount_min"] = amount_min
    if amount_max is not None:
        meta["amount_max"] = amount_max
    return meta


def _infer_menu_category(
    *,
    page_url: str,
    name: str,
    details: str,
    keywords: List[str],
) -> str:
    path = (urlparse(page_url).path or "").lower()
    text = " ".join([name, details, " ".join(keywords)]).lower()
    if "/party" in path or any(token in text for token in ("course", "\u30b3\u30fc\u30b9", "plan", "\u30d7\u30e9\u30f3", "set")):
        return "course"
    if any(token in text for token in ("drink", "beverage", "cocktail", "\u30c9\u30ea\u30f3\u30af", "\u98f2\u307f\u7269", "\u9152")):
        return "drink"
    if "/dtlmenu/lunch" in path or any(token in text for token in ("lunch", "\u30e9\u30f3\u30c1")):
        return "lunch"
    if "/dtlmenu" in path or any(token in text for token in ("dish", "menu", "\u6599\u7406", "\u30e1\u30cb\u30e5\u30fc", "dinner")):
        return "dish"
    return "menu"


def _normalize_menu_fields(
    *,
    price_text: str,
    details: str,
    description: str,
) -> Tuple[str, str, str]:
    normalized_price = (price_text or "").strip()
    normalized_details = (details or "").strip()
    normalized_description = (description or "").strip()

    if normalized_description and not normalized_price:
        split_price, split_details = _split_price_from_text(normalized_description)
        if split_price:
            normalized_price = split_price
        if split_details and not normalized_details:
            normalized_details = split_details

    if normalized_description and not normalized_details:
        if normalized_price:
            stripped = normalized_description.replace(normalized_price, " ").strip(" -|:;/,")
            normalized_details = re.sub(r"\s+", " ", stripped).strip()
        else:
            normalized_details = normalized_description

    if not normalized_description:
        normalized_description = _format_menu_description(normalized_price, normalized_details)

    return normalized_price, normalized_details, normalized_description


def _build_menu_metadata(
    *,
    page_url: str,
    name: str,
    keywords: List[str],
    price_text: str,
    details: str,
    category: str,
    extraction_method: str,
) -> Dict[str, Any]:
    normalized_category = (category or "").strip().lower()
    if normalized_category not in {"course", "dish", "drink", "lunch", "menu"}:
        normalized_category = _infer_menu_category(
            page_url=page_url,
            name=name,
            details=details,
            keywords=keywords,
        )

    metadata: Dict[str, Any] = {
        "schema": "menu_item.v1",
        "category": normalized_category,
        "details": (details or "").strip(),
        "price_text": (price_text or "").strip(),
        "source_url": (page_url or "").strip(),
        "extraction_method": (extraction_method or "").strip() or "unknown",
    }
    if keywords:
        metadata["keywords"] = keywords

    price_meta = _price_metadata(price_text)
    if price_meta:
        metadata["price"] = price_meta

    return metadata


def _normalize_menu_item(
    raw_item: Dict[str, Any],
    *,
    source_url: str,
    extraction_method: str,
    default_category: str = "",
    forced_category: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_item, dict):
        return None

    name = str(raw_item.get("name") or "").strip()
    if not name:
        return None

    raw_price = raw_item.get("price_text")
    if raw_price is None:
        raw_price = raw_item.get("price")
    if isinstance(raw_price, dict):
        raw_price = raw_price.get("text") or raw_price.get("display") or ""
    price_text = str(raw_price or "").strip()

    details = str(raw_item.get("details") or raw_item.get("detail") or "").strip()
    description = str(raw_item.get("description") or "").strip()
    price_text, details, description = _normalize_menu_fields(
        price_text=price_text,
        details=details,
        description=description,
    )

    raw_keywords = raw_item.get("keywords")
    if isinstance(raw_keywords, list):
        keywords = [str(k).strip() for k in raw_keywords if str(k).strip()][:10]
    else:
        keywords = []
    if not keywords:
        keywords = _keywords_for_menu_item(name, f"{price_text} {details}".strip())

    link_url = str(raw_item.get("link_url") or "").strip() or source_url
    image_url = str(raw_item.get("image_url") or "").strip()
    category = str(forced_category or raw_item.get("category") or default_category or "").strip().lower()
    metadata = _build_menu_metadata(
        page_url=source_url,
        name=name,
        keywords=keywords,
        price_text=price_text,
        details=details,
        category=category,
        extraction_method=extraction_method,
    )

    return {
        "name": name,
        "description": description,
        "link_url": link_url,
        "image_url": image_url or None,
        "keywords": keywords,
        "metadata": metadata,
    }


def _menu_rules(profile: Any) -> Dict[str, Any]:
    raw = profile.menu_extraction_rules if profile else None
    return raw if isinstance(raw, dict) else {}


def _is_menu_extraction_enabled(profile: Any) -> bool:
    rules = _menu_rules(profile)
    return bool(rules) and bool(rules.get("enabled"))


def _path_matches_patterns(path: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(str(pattern), path):
                return True
        except re.error:
            continue
    return False


def _is_profile_menu_candidate_url(url: str, profile: Any) -> bool:
    if not _is_menu_extraction_enabled(profile):
        return False
    rules = _menu_rules(profile)
    raw_patterns = rules.get("allowed_path_patterns")
    patterns = [str(p) for p in raw_patterns] if isinstance(raw_patterns, list) else []
    if patterns:
        return _path_matches_patterns(urlparse(url).path or "", patterns)
    if profile and profile.menu_url_patterns:
        return _path_matches_patterns(url, profile.menu_url_patterns)
    return True


def _configured_category_for_url(page_url: str, rules: Dict[str, Any]) -> str:
    path = urlparse(page_url).path or ""
    raw_rules = rules.get("path_category_patterns") if isinstance(rules, dict) else None
    path_rules = raw_rules if isinstance(raw_rules, list) else []
    for entry in path_rules:
        if not isinstance(entry, dict):
            continue
        pattern = str(entry.get("pattern") or "").strip()
        category = str(entry.get("category") or "").strip().lower()
        if not pattern or not category:
            continue
        try:
            if re.search(pattern, path):
                return category
        except re.error:
            continue
    return ""


def _fetch_html(url: str) -> str:
    if not url:
        return ""
    raw_timeout = (os.environ.get("MENU_EXTRACTION_HTTP_TIMEOUT_SEC") or "15").strip() or "15"
    try:
        timeout_sec = float(raw_timeout)
    except ValueError:
        timeout_sec = 15.0
    try:
        with httpx.Client(
            timeout=max(5.0, min(timeout_sec, 60.0)),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; WebAI/1.0)",
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            },
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text or ""
    except Exception as e:
        logger.warning("[MenuExtraction] Failed to fetch HTML for %s: %s", url, type(e).__name__)
        return ""


def _extract_tabelog_course_price(block_html: str) -> str:
    if not block_html:
        return ""

    real_price = _extract_first_match(
        block_html,
        r"\bdata-real-price\s*=\s*['\"]([^'\"]+)['\"]",
    )
    tax = _extract_first_match(
        block_html,
        r"\bdata-tax\s*=\s*['\"]([^'\"]+)['\"]",
    )
    if real_price:
        price = _strip_html(real_price)
        if price and not any(u in price for u in ("\u5186", "\u00A5", "\uFFE5")):
            price = f"{price}\u5186"
        tax_clean = _strip_html(tax)
        return _format_menu_description(price, tax_clean)

    price_blocks = _iter_div_blocks_by_class(block_html, "rstdtl-course-list__price")
    for pb in price_blocks:
        text = _strip_html(pb)
        if text:
            return text
    return ""


def _extract_tabelog_course_items(page_url: str, html_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    blocks = _iter_div_blocks_by_class(html_text, "rstdtl-course-list")
    for block in blocks:
        name = _extract_text_by_class(block, "rstdtl-course-list__course-title-text", "span")
        if not name:
            name = _extract_text_by_class(block, "rstdtl-course-list__course-title", "p")
        if not name:
            continue

        price = _extract_tabelog_course_price(block)
        body = _extract_text_by_class(block, "rstdtl-course-list__desc", "p")
        description = _format_menu_description(price, body)

        link_url = _extract_href_by_class(block, "rstdtl-course-list__target")
        if not link_url:
            link_url = _extract_href_by_class(block, "rstdtl-course-list__img-target")
        resolved_link = urljoin(page_url, link_url) if link_url else page_url

        image_src = _extract_first_img_src(block)
        resolved_image = urljoin(page_url, image_src) if image_src else None

        items.append(
            {
                "name": name,
                "price_text": price,
                "details": body,
                "description": description,
                "link_url": resolved_link,
                "image_url": resolved_image,
                "category": "course",
                "keywords": _keywords_for_menu_item(name, description),
            }
        )
    return items


def _extract_tabelog_dtlmenu_items(page_url: str, html_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    blocks = _iter_div_blocks_by_class(html_text, "rstdtl-menu-lst__contents")
    for block in blocks:
        name = _extract_text_by_class(block, "rstdtl-menu-lst__menu-title", "p")
        if not name:
            continue

        price = _extract_text_by_class(block, "rstdtl-menu-lst__price", "p")
        body = _extract_text_by_class(block, "rstdtl-menu-lst__ex", "p")
        description = _format_menu_description(price, body)

        image_src = _extract_first_img_src(block)
        resolved_image = urljoin(page_url, image_src) if image_src else None

        items.append(
            {
                "name": name,
                "price_text": price,
                "details": body,
                "description": description,
                "link_url": page_url,
                "image_url": resolved_image,
                "category": "dish",
                "keywords": _keywords_for_menu_item(name, description),
            }
        )
    return items


def _extract_tabelog_menu_items(page_url: str) -> List[Dict[str, Any]]:
    html_text = _fetch_html(page_url)
    if not html_text:
        return []
    # Prefer parser-based extraction when bs4 is available; keep regex fallback.
    bs4_items = _extract_tabelog_menu_items_with_bs4(page_url, html_text)
    if bs4_items:
        return _dedupe_menu_items(bs4_items)

    # Root and /party pages expose course blocks. /dtlmenu pages expose dish blocks.
    course_items = _extract_tabelog_course_items(page_url, html_text)
    menu_items = _extract_tabelog_dtlmenu_items(page_url, html_text)
    return _dedupe_menu_items(course_items + menu_items)


_DETERMINISTIC_MENU_EXTRACTORS: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
    "tabelog_v1": _extract_tabelog_menu_items,
}


def _select_menu_pages(
    documents: List[Dict[str, Any]],
    page_urls: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Select which documents to process for menu extraction.

    Uses platform profile's menu_url_patterns to identify menu pages.
    If page_urls are explicitly provided, those take priority.
    The home page (root URL) is always included as it often has menu info.
    """
    if not documents:
        return []

    allowed_pages = {
        _normalize_url_for_match(u)
        for u in (page_urls or [])
        if (u or "").strip()
    }

    def _allow_doc_url(raw_url: str) -> bool:
        profile, _ = resolve_platform_profile(raw_url)
        if not _is_profile_menu_candidate_url(raw_url, profile):
            return False
        rules = _menu_rules(profile)
        if isinstance(rules, dict) and str(rules.get("mode") or "").strip().lower() == "deterministic":
            return True
        if profile and profile.menu_url_patterns:
            return any(re.search(pattern, raw_url) for pattern in profile.menu_url_patterns)
        return True

    # If explicit page_urls provided, use those
    if allowed_pages:
        return [
            doc for doc in documents
            if _normalize_url_for_match(doc.get("url", "")) in allowed_pages
            and _allow_doc_url(doc.get("url", ""))
        ]

    # Otherwise, use platform profile menu_url_patterns + home page
    selected: List[Dict[str, Any]] = []
    for doc in documents:
        url = doc.get("url", "")
        if not url:
            continue

        profile, _ = resolve_platform_profile(url)

        if not _is_profile_menu_candidate_url(url, profile):
            continue

        # Always include home page (path is / or empty)
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if not path or path == "":
            selected.append(doc)
            continue

        rules = _menu_rules(profile)
        if isinstance(rules, dict) and str(rules.get("mode") or "").strip().lower() == "deterministic":
            selected.append(doc)
            continue

        # Match against menu_url_patterns from profile
        if profile and profile.menu_url_patterns:
            for pattern in profile.menu_url_patterns:
                if re.search(pattern, url):
                    selected.append(doc)
                    break
            continue

        # No explicit menu patterns: include page by default.
        selected.append(doc)

    return selected


def _extract_menu_from_page(
    page_url: str,
    page_content: str,
    image_urls: List[str],
) -> List[Dict[str, Any]]:
    """Use Gemini to extract menu items from a page's content + image list."""
    from google import genai
    from google.genai import types

    project = (os.environ.get("PROJECT_ID") or "").strip()
    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    model = (
        os.environ.get("VERTEX_ASSET_MODEL")
        or os.environ.get("VERTEX_RAG_MODEL")
        or "gemini-2.0-flash-001"
    ).strip()

    if not project:
        return []

    chunk = page_content

    if image_urls:
        image_list = "\n".join(f"- {url}" for url in image_urls[:_MENU_PROMPT_IMAGE_LIMIT])
    else:
        image_list = "(no images found)"

    prompt = _MENU_EXTRACTION_PROMPT.format(
        page_url=page_url,
        page_content=chunk,
        image_list=image_list,
    )

    timeout_sec = _bounded_env_int(
        "MENU_EXTRACTION_LLM_TIMEOUT_SEC",
        default=60,
        minimum=10,
        maximum=300,
    )

    result_holder: List[List[Dict[str, Any]]] = []
    error_holder: List[Exception] = []
    completed = Event()

    def _run_llm_call() -> None:
        client = genai.Client(vertexai=True, project=project, location=location)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            text = (response.text or "").strip()
            result_holder.append(_parse_llm_json(text))
        except Exception as exc:
            error_holder.append(exc)
        finally:
            completed.set()

    try:
        worker = Thread(
            target=_run_llm_call,
            name=f"menu-llm-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        worker.start()
        finished = completed.wait(timeout=timeout_sec)
        if not finished:
            raise TimeoutError(f"LLM call exceeded {timeout_sec}s")
        if error_holder:
            raise error_holder[0]
        return result_holder[0] if result_holder else []
    except TimeoutError:
        logger.warning(
            "[MenuExtraction] LLM timeout for %s after %ss",
            page_url,
            timeout_sec,
        )
    except Exception as e:
        logger.warning(
            "[MenuExtraction] LLM extraction failed for %s: %s: %s",
            page_url,
            type(e).__name__,
            str(e)[:200],
        )
    return []


class MenuExtractionService:
    """Extracts menu items from crawled restaurant pages using LLM."""

    def __init__(self):
        self._repo = PostgresBotAssetRepository()

    def extract_menu_from_documents(
        self,
        *,
        org_id: str,
        bot_id: str,
        documents: List[Dict[str, Any]],
        max_items: int = 50,
        page_urls: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> List[BotAsset]:
        """Core logic: select menu pages -> extract items -> download images -> save."""
        job_repo = None
        current_job = None
        if job_id:
            try:
                from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
                job_repo = PostgresAssetExtractionJobRepository()
                current_job = job_repo.get_job(job_id)
            except Exception as e:
                logger.warning(f"Failed to load job {job_id}: {e}")

        def _is_cancelled() -> bool:
            if not job_repo or not job_id:
                return False
            try:
                latest = job_repo.get_job(job_id)
                return bool(latest and latest.status == "cancelled")
            except Exception:
                return False

        # Get existing menu items to avoid duplicates
        existing = self._repo.list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item")
        existing_names = {a.name.strip().lower() for a in existing}

        # Select menu-relevant pages
        menu_docs = _select_menu_pages(documents, page_urls)
        max_docs = _bounded_env_int(
            "MENU_EXTRACTION_MAX_DOCS",
            default=60,
            minimum=1,
            maximum=500,
        )
        if len(menu_docs) > max_docs:
            menu_docs = menu_docs[:max_docs]
        logger.info(
            "[MenuExtraction] Selected %d menu pages from %d total documents for bot %s",
            len(menu_docs),
            len(documents),
            bot_id,
        )

        all_extracted: List[Dict[str, Any]] = []
        last_progress_emit = 0.0

        def _emit_progress(force: bool = False) -> None:
            nonlocal last_progress_emit
            if not job_repo or not current_job:
                return
            now_monotonic = time.monotonic()
            if not force and now_monotonic - last_progress_emit < 1.5:
                return
            current_job.assets_discovered = len(all_extracted)
            try:
                job_repo.update_job(current_job)
                last_progress_emit = now_monotonic
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

        _emit_progress(force=True)

        for doc in menu_docs:
            if _is_cancelled():
                logger.info("[MenuExtraction] Job %s cancelled while extracting", job_id)
                break
            if len(all_extracted) >= max_items:
                break
            _emit_progress()

            content = doc.get("content", "")
            url = doc.get("url", "")

            profile, _ = resolve_platform_profile(url)
            rules = _menu_rules(profile)
            extraction_mode = str(rules.get("mode") or "").strip().lower() if isinstance(rules, dict) else ""

            if extraction_mode == "deterministic":
                extractor_key = str(rules.get("extractor") or "").strip().lower()
                extractor = _DETERMINISTIC_MENU_EXTRACTORS.get(extractor_key)
                if extractor is None:
                    logger.warning(
                        "[MenuExtraction] Deterministic mode set but extractor is missing for %s: %s",
                        url,
                        extractor_key or "(empty)",
                    )
                    continue
                page_items = extractor(url)
                forced_category = _configured_category_for_url(url, rules)
                logger.info(
                    "[MenuExtraction] Deterministic extraction (%s) on %s: %d items",
                    extractor_key,
                    url,
                    len(page_items),
                )
                for item in page_items:
                    if len(all_extracted) >= max_items:
                        break
                    normalized_item = _normalize_menu_item(
                        item,
                        source_url=url,
                        extraction_method=f"deterministic_{extractor_key}",
                        forced_category=forced_category,
                    )
                    if not normalized_item:
                        continue
                    name = (normalized_item.get("name") or "").strip()
                    if not name:
                        continue
                    if name.lower() in existing_names:
                        continue
                    existing_names.add(name.lower())
                    all_extracted.append(normalized_item)
                _emit_progress()
                continue

            if not content or len(content.strip()) < 100:
                continue

            image_urls = _scrape_image_urls(url)
            chunks = _chunk_content(content)
            logger.info(
                "[MenuExtraction] Processing %s: %d chunks, %d image candidates",
                url,
                len(chunks),
                len(image_urls),
            )

            for chunk in chunks[:2]:
                if _is_cancelled():
                    break
                if len(all_extracted) >= max_items:
                    break
                page_items = _extract_menu_from_page(url, chunk, image_urls)
                for item in page_items:
                    if len(all_extracted) >= max_items:
                        break
                    normalized_item = _normalize_menu_item(
                        item,
                        source_url=url,
                        extraction_method="llm_gemini",
                    )
                    if not normalized_item:
                        continue
                    name = (normalized_item.get("name") or "").strip()
                    if not name:
                        continue
                    if name.lower() in existing_names:
                        continue
                    existing_names.add(name.lower())
                    all_extracted.append(normalized_item)
                _emit_progress()

        # Update job with discovered count
        _emit_progress(force=True)

        # Save to database
        created: List[BotAsset] = []
        now = datetime.now(timezone.utc).isoformat()
        limit = _bounded_env_int(
            "MENU_MAX_PER_BOT",
            default=_bounded_env_int("ASSET_MAX_PER_BOT", default=500, minimum=1, maximum=1000),
            minimum=1,
            maximum=1000,
        )

        for item in all_extracted:
            if _is_cancelled():
                logger.info("[MenuExtraction] Job %s cancelled while saving", job_id)
                break

            current_count = len(self._repo.list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item"))
            if current_count >= limit:
                logger.info(
                    "Skipping remaining menu items for bot %s: limit reached (%d/%d)",
                    bot_id, current_count, limit,
                )
                break

            asset_id = "asset_" + uuid.uuid4().hex[:16]
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).strip() for k in keywords if k][:10]

            link_url = (item.get("link_url") or item.get("source_url") or "").strip() or None
            description = (item.get("description") or "").strip()
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if not metadata:
                metadata = _build_menu_metadata(
                    page_url=link_url or "",
                    name=str(item.get("name") or "").strip(),
                    keywords=keywords,
                    price_text="",
                    details=description,
                    category="",
                    extraction_method="unknown",
                )

            # Download and store image (optional for menu items â€” some may not have images)
            image_gcs_uri = ""
            image_public_url = ""
            image_url = item.get("image_url")
            if image_url:
                try:
                    image_gcs_uri, image_public_url = _download_and_store_image(
                        image_url, bot_id, asset_id
                    )
                except Exception as e:
                    logger.warning(f"Menu image download failed for {image_url}: {e}")

            # Menu items can exist without images (unlike image assets)
            # but we still need a name
            asset = BotAsset(
                asset_id=asset_id,
                bot_id=bot_id,
                org_id=org_id,
                name=item.get("name") or "Untitled",
                description=description,
                image_gcs_uri=image_gcs_uri,
                image_public_url=image_public_url,
                link_url=link_url,
                keywords=keywords,
                metadata=metadata,
                is_active=True,
                asset_type="menu_item",
                created_at=now,
                updated_at=now,
            )

            try:
                self._repo.create_asset(asset)
                created.append(asset)

                if job_repo and current_job:
                    current_job.assets_downloaded += 1
                    current_job.assets_created += 1
                    try:
                        job_repo.update_job(current_job)
                    except Exception as e:
                        logger.warning(f"Failed to update job progress: {e}")

            except Exception as e:
                logger.warning(
                    "[MenuExtraction] Failed to save menu item '%s': %s",
                    asset.name,
                    str(e)[:200],
                )

        logger.info(
            "Extracted %d menu items for bot %s from %d documents",
            len(created), bot_id, len(menu_docs),
        )
        return created

    def extract_from_gcs_prefix(
        self,
        *,
        org_id: str,
        bot_id: str,
        gcs_prefix: str,
        max_items: int = 50,
        page_urls: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Load documents from GCS and extract menu items. Returns count of created items."""
        documents = _load_docs_from_gcs(gcs_prefix)
        if not documents:
            return 0
        items = self.extract_menu_from_documents(
            org_id=org_id,
            bot_id=bot_id,
            documents=documents,
            max_items=max_items,
            page_urls=page_urls,
            job_id=job_id,
        )
        return len(items)


_instance: Optional[MenuExtractionService] = None


def menu_extraction_service() -> MenuExtractionService:
    global _instance
    if _instance is None:
        _instance = MenuExtractionService()
    return _instance

