"""
Asset Extraction Service

Analyzes crawled page content using Gemini LLM to automatically extract
business assets (products, services, rooms, menu items, etc.) with
name, description, link_url, image_url, and keywords.

Images are scraped from the live page HTML (<img> tags), passed to the LLM
so it can match each asset to the best image, then downloaded and stored in GCS.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from application.services.asset_image_service import optimize_asset_image
from domain.entities import BotAsset
from infrastructure.db.repositories import PostgresBotAssetRepository

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are analyzing a business website page to extract business assets.
Business assets are distinct products, services, offerings, rooms, menu items, packages,
facilities, amenities, or any concrete item/service that a business offers to customers.

CRITICAL: Keep ALL text (name, description, keywords) in the ORIGINAL language of the page.
Do NOT translate anything. If the page is in Japanese, output Japanese. If Thai, output Thai. etc.

For each asset found, extract:
- name: Short, clear name of the asset in the ORIGINAL language
- description: 1 sentence description for AI chatbot context in the ORIGINAL language
- link_url: The page URL or a more specific URL for this asset, or null
- image_url: Pick the best matching image URL from the image list below, or null
- keywords: 3-6 keywords in the ORIGINAL language that a customer might use

Rules:
- Only extract CONCRETE offerings (not generic pages like "About Us", "Contact")
- Each asset should be a distinct item — extract ALL distinct items you can find on the page
- If a page lists multiple items (menu, products, courses, rooms), extract EVERY one separately
- For image_url: ONLY use URLs from the "Available images" list below. Pick the most relevant image per asset. Do NOT invent image URLs.
- Return empty array [] if no business assets found

Page URL: {page_url}

Available images on this page:
{image_list}

Page content:
{page_content}

Return ONLY a JSON array, no markdown fences, no commentary.
Example: [{{"name":"Room A","description":"A nice room.","link_url":null,"image_url":"https://example.com/img/room-a.jpg","keywords":["room","suite"]}}]"""

_ASSET_IMAGE_CANDIDATE_LIMIT = max(50, min(int(os.environ.get("ASSET_IMAGE_CANDIDATE_LIMIT", "400")), 1000))
_ASSET_PROMPT_IMAGE_LIMIT = max(20, min(int(os.environ.get("ASSET_PROMPT_IMAGE_LIMIT", "120")), 400))
_ASSET_EXTRACTION_MODE = (os.environ.get("ASSET_EXTRACTION_MODE") or "llm").strip().lower()
_ASSET_FILTER_MODEL = (
    os.environ.get("VERTEX_ASSET_FILTER_MODEL")
    or os.environ.get("VERTEX_ASSET_MODEL")
    or "gemini-2.0-flash-lite-001"
).strip()
_ASSET_FILTER_BATCH_SIZE = max(20, min(int(os.environ.get("ASSET_FILTER_BATCH_SIZE", "80")), 200))

_GENERIC_SECTION_NAMES = {
    "about",
    "about us",
    "contact",
    "contact us",
    "home",
    "services",
    "products",
    "product",
    "service",
    "menu",
    "rooms",
    "room",
    "gallery",
    "portfolio",
    "blog",
    "news",
    "faq",
    "careers",
    "privacy",
    "terms",
    "login",
    "sign in",
    "signup",
    "register",
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "your",
    "our",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "into",
    "over",
    "under",
    "each",
    "per",
    "all",
    "any",
    "item",
    "items",
}


def _scrape_image_urls(page_url: str) -> List[str]:
    """Fetch raw HTML from a page URL and extract all <img> src URLs."""
    if not page_url:
        return []
    try:
        with httpx.Client(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebAI/1.0)"},
        ) as client:
            resp = client.get(page_url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.debug("[AssetExtraction] Failed to fetch HTML for images from %s: %s", page_url, type(e).__name__)
        return []

    # Extract img src from HTML
    img_urls: List[str] = []
    seen = set()
    for match in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE):
        src = match.group(1).strip()
        if not src or src.startswith("data:"):
            continue
        # Make absolute
        absolute = urljoin(page_url, src)
        # Skip only obvious tracking/pixel assets.
        lower = absolute.lower()
        if any(skip in lower for skip in ("favicon", "tracking", "spacer", "blank.gif", "1x1")):
            continue
        if absolute not in seen:
            seen.add(absolute)
            img_urls.append(absolute)

    # Also extract from srcset, lazy-load attributes
    for match in re.finditer(r'(?:data-src|data-original|data-lazy-src|srcset)=["\']([^"\']+)["\']', html, re.IGNORECASE):
        val = match.group(1).strip()
        # srcset can have multiple URLs with descriptors
        for part in val.split(","):
            src = part.strip().split()[0] if part.strip() else ""
            if not src or src.startswith("data:"):
                continue
            absolute = urljoin(page_url, src)
            if absolute not in seen:
                seen.add(absolute)
                img_urls.append(absolute)

    return img_urls


def _download_and_store_image(
    image_url: str,
    bot_id: str,
    asset_id: str,
) -> Tuple[str, str]:
    """
    Download image from URL and store in GCS.
    Returns (gcs_uri, public_url) tuple. Returns ("","") on failure.
    """
    from google.cloud import storage as gcs_storage

    try:
        from common.config import config
    except Exception:
        return "", ""

    bucket_raw = (config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")).strip()
    if not bucket_raw:
        return "", ""
    bucket_name = bucket_raw.strip("/").split("/", 1)[0]
    base_prefix = bucket_raw.strip("/").split("/", 1)[1] if "/" in bucket_raw else ""

    # Download image with timeout
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(image_url)
            response.raise_for_status()
            data = response.content
            resp_content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    except Exception as e:
        logger.debug("[AssetExtraction] Image download failed %s: %s", image_url, type(e).__name__)
        return "", ""

    if not data or len(data) < 1000:  # Skip tiny images (likely icons/pixels)
        return "", ""
    if len(data) > 10 * 1024 * 1024:  # Skip > 10MB
        return "", ""

    # Detect content type from HTTP response header first, then URL extension
    content_type = resp_content_type if resp_content_type.startswith("image/") else ""
    if not content_type:
        url_lower = image_url.lower().split("?")[0]
        if url_lower.endswith(".png"):
            content_type = "image/png"
        elif url_lower.endswith(".gif"):
            content_type = "image/gif"
        elif url_lower.endswith(".webp"):
            content_type = "image/webp"
        elif url_lower.endswith(".svg"):
            content_type = "image/svg+xml"
        else:
            content_type = "image/jpeg"

    # Resize/compress to keep asset storage + client load fast.
    data, content_type, ext = optimize_asset_image(data, content_type)

    # Upload to GCS
    try:
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"{base_prefix}/assets/{bot_id}/{asset_id}.{ext}".strip("/")
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        public_url = f"/v1/assets/{asset_id}/image"
        return gcs_uri, public_url
    except Exception as e:
        logger.warning("[AssetExtraction] GCS upload failed: %s", type(e).__name__)
        return "", ""


def _parse_llm_json(text: str) -> List[Dict[str, Any]]:
    """Parse JSON from LLM response, handling truncated output and markdown fences."""
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json) and last line (```)
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()

    # Try parsing as-is first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict) and item.get("name")]
        return []
    except json.JSONDecodeError:
        pass

    # Handle truncated JSON: try to recover partial array
    # Find the last complete object by looking for the last '}' before truncation
    # Then close the array
    try:
        last_brace = text.rfind("}")
        if last_brace > 0:
            candidate = text[:last_brace + 1]
            # Ensure it starts with [ and close with ]
            if not candidate.strip().startswith("["):
                candidate = "[" + candidate
            if not candidate.strip().endswith("]"):
                candidate = candidate + "]"
            result = json.loads(candidate)
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict) and item.get("name")]
    except json.JSONDecodeError:
        pass

    return []


_CHUNK_SIZE = 8000    # characters per LLM call
_CHUNK_OVERLAP = 600  # overlap so items near chunk boundaries aren't missed


def _chunk_content(content: str) -> List[str]:
    """Split content into overlapping chunks so long pages are fully covered."""
    if len(content) <= _CHUNK_SIZE:
        return [content]
    chunks: List[str] = []
    start = 0
    while start < len(content):
        end = min(start + _CHUNK_SIZE, len(content))
        chunks.append(content[start:end])
        if end >= len(content):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{2,6})\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_INLINE_SPLIT_RE = re.compile(r"^(.{2,120}?)(?:\s(?:-|\u2013|\u2014|:)\s)(.{6,300})$")


def _strip_markdown(text: str) -> str:
    if not text:
        return ""
    out = text
    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", out)
    out = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", r"\1", out)
    out = re.sub(r"<[^>]+>", " ", out)
    out = out.replace("`", " ")
    out = re.sub(r"[*_~>|#]", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip(" \t\r\n-:")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9\u00C0-\u024F\u3040-\u30FF\u3400-\u9FFF]+", text.lower())


def _is_cjk_token(token: str) -> bool:
    return bool(re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", token or ""))


def _looks_like_asset_name(name: str, *, allow_generic: bool = False) -> bool:
    cleaned = _strip_markdown(name)
    if len(cleaned) < 2 or len(cleaned) > 120:
        return False

    lowered = cleaned.lower()
    if not allow_generic and lowered in _GENERIC_SECTION_NAMES:
        return False
    if lowered.startswith(("http://", "https://", "www.")):
        return False

    tokens = _tokenize(cleaned)
    if not tokens:
        return False
    if (not allow_generic) and all(t in _STOPWORDS or t in _GENERIC_SECTION_NAMES for t in tokens):
        return False
    if len(tokens) > 20:
        return False

    # Reject mostly punctuation.
    alpha_num = sum(1 for ch in cleaned if ch.isalnum())
    if alpha_num < max(2, len(cleaned) // 8):
        return False

    return True


def _resolve_link_url(page_url: str, raw_link: str) -> Optional[str]:
    link = (raw_link or "").strip()
    if not link:
        return None
    if link.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    return urljoin(page_url, link)


def _extract_first_link_url(text: str, page_url: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
    if match:
        return _resolve_link_url(page_url, match.group(1))
    raw = re.search(r"(https?://[^\s)]+)", text)
    if raw:
        return _resolve_link_url(page_url, raw.group(1))
    return None


def _extract_links_from_markdown(markdown_text: str, page_url: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for anchor, href in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", markdown_text or ""):
        link = _resolve_link_url(page_url, href)
        anchor_text = _strip_markdown(anchor)
        if link and anchor_text:
            out.append((anchor_text, link))
    return out


def _extract_markdown_image_urls(markdown_text: str, page_url: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for href in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text or ""):
        resolved = _resolve_link_url(page_url, href)
        if not resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _split_name_description(raw_text: str) -> Tuple[str, str]:
    cleaned = _strip_markdown(raw_text)
    if not cleaned:
        return "", ""

    for delim in (" - ", " – ", " — ", ": "):
        if delim in cleaned:
            left, right = cleaned.split(delim, 1)
            left = left.strip()
            right = right.strip()
            if _looks_like_asset_name(left, allow_generic=True):
                return left, right

    return cleaned.strip(), ""


def _keywords_for_asset(name: str, description: str) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()

    for token in _tokenize(f"{name} {description}"):
        if token in seen:
            continue
        if token in _STOPWORDS or token in _GENERIC_SECTION_NAMES:
            continue
        if len(token) < 2 and not _is_cjk_token(token):
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 6:
            break

    return result[:6]


def _image_url_tokens(url: str) -> Set[str]:
    parsed = urlparse(url or "")
    material = f"{parsed.netloc} {parsed.path} {parsed.query}"
    return set(_tokenize(material))


def _pick_best_image_url(name: str, description: str, image_urls: List[str]) -> Optional[str]:
    if not image_urls:
        return None

    query_tokens = set(_tokenize(f"{name} {description}"))
    fallback = None
    best_url = None
    best_score = float("-inf")

    for image_url in image_urls:
        lower = (image_url or "").lower()
        if not fallback and not any(x in lower for x in ("icon", "favicon", "tracking", "spacer", "1x1")):
            fallback = image_url

        score = 0.0
        if any(x in lower for x in ("icon", "favicon", "tracking", "spacer", "1x1", "sprite")):
            score -= 4.0
        if lower.endswith(".svg"):
            score -= 0.5

        image_tokens = _image_url_tokens(image_url)
        overlap = len(query_tokens & image_tokens) if query_tokens else 0
        score += overlap * 3.0

        if query_tokens and any(tok in lower for tok in query_tokens):
            score += 1.5
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            score += 0.2

        if score > best_score:
            best_score = score
            best_url = image_url

    if best_url and best_score > 0:
        return best_url
    return fallback or image_urls[0]


def _extract_table_candidates(lines: List[str], page_url: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in lines:
        stripped = (line or "").strip()
        if "|" not in stripped:
            continue
        if _TABLE_SEPARATOR_RE.match(stripped):
            continue

        cells = [_strip_markdown(c).strip() for c in stripped.strip("|").split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue

        name = cells[0]
        if not _looks_like_asset_name(name, allow_generic=True):
            continue
        description = " ".join(cells[1:3]).strip()
        link_url = _extract_first_link_url(stripped, page_url)
        out.append({"name": name, "description": description, "link_url": link_url or ""})
    return out


def _extract_list_candidates(lines: List[str], page_url: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in lines:
        m = _LIST_ITEM_RE.match(line or "")
        if not m:
            continue

        body = m.group(1).strip()
        if not body:
            continue
        name, description = _split_name_description(body)
        if not _looks_like_asset_name(name, allow_generic=True):
            continue
        link_url = _extract_first_link_url(body, page_url)
        out.append({"name": name, "description": description, "link_url": link_url or ""})
    return out


def _extract_heading_candidates(lines: List[str], page_url: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line or "")
        if not m:
            continue

        level = len(m.group(1))
        if level < 3:
            continue
        name = _strip_markdown(m.group(2))
        if not _looks_like_asset_name(name, allow_generic=True):
            continue

        description = ""
        for lookahead in range(idx + 1, min(len(lines), idx + 5)):
            nxt = (lines[lookahead] or "").strip()
            if not nxt:
                continue
            if _HEADING_RE.match(nxt) or _LIST_ITEM_RE.match(nxt) or "|" in nxt:
                break
            description = _strip_markdown(nxt)
            break

        link_url = _extract_first_link_url(line, page_url)
        out.append({"name": name, "description": description, "link_url": link_url or ""})
    return out


def _extract_inline_candidates(lines: List[str], page_url: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in lines:
        stripped = (line or "").strip()
        if not stripped:
            continue
        if _HEADING_RE.match(stripped) or _LIST_ITEM_RE.match(stripped) or "|" in stripped:
            continue

        normalized = _strip_markdown(stripped)
        m = _INLINE_SPLIT_RE.match(normalized)
        if not m:
            continue
        name = m.group(1).strip()
        description = m.group(2).strip()
        if not _looks_like_asset_name(name, allow_generic=True):
            continue
        link_url = _extract_first_link_url(stripped, page_url)
        out.append({"name": name, "description": description, "link_url": link_url or ""})
    return out


def _extract_anchor_candidates(markdown_text: str, page_url: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for anchor, link in _extract_links_from_markdown(markdown_text, page_url):
        if not _looks_like_asset_name(anchor, allow_generic=True):
            continue
        out.append({"name": anchor, "description": "", "link_url": link})
    return out


def _extract_assets_from_page_deterministic(
    page_url: str,
    page_content: str,
    image_urls: List[str],
) -> List[Dict[str, Any]]:
    if not page_content:
        return []

    content = page_content
    if content.startswith("Source URL:"):
        parts = content.splitlines()
        content = "\n".join(parts[1:]) if len(parts) > 1 else ""

    lines = content.splitlines()
    if not lines:
        return []

    image_candidates: List[str] = []
    image_seen: Set[str] = set()
    for u in image_urls + _extract_markdown_image_urls(content, page_url):
        if not u or u in image_seen:
            continue
        image_seen.add(u)
        image_candidates.append(u)

    candidates: List[Dict[str, str]] = []
    candidates.extend(_extract_table_candidates(lines, page_url))
    candidates.extend(_extract_list_candidates(lines, page_url))
    candidates.extend(_extract_heading_candidates(lines, page_url))
    candidates.extend(_extract_inline_candidates(lines, page_url))
    candidates.extend(_extract_anchor_candidates(content, page_url))

    assets: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()
    for cand in candidates:
        name = _strip_markdown(cand.get("name", ""))
        if not _looks_like_asset_name(name, allow_generic=True):
            continue

        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)

        description = _strip_markdown(cand.get("description", ""))[:280]
        link_url = (cand.get("link_url") or "").strip() or page_url
        image_url = _pick_best_image_url(name, description, image_candidates)
        keywords = _keywords_for_asset(name, description)

        assets.append(
            {
                "name": name,
                "description": description,
                "link_url": link_url,
                "image_url": image_url,
                "keywords": keywords,
            }
        )

    return assets


def _filter_business_assets_heuristic(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for item in candidates:
        name = _strip_markdown(str(item.get("name") or ""))
        if not _looks_like_asset_name(name, allow_generic=False):
            continue
        kept.append(item)
    return kept


def _parse_non_business_ids(text: str) -> Tuple[Set[int], bool]:
    raw = (text or "").strip()
    if not raw:
        return set(), False

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
        else:
            raw = "\n".join(lines[1:]).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return set(), False
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return set(), False

    ids_raw: Any = None
    if isinstance(data, dict):
        ids_raw = (
            data.get("non_business_ids")
            or data.get("exclude_ids")
            or data.get("excluded_ids")
            or []
        )
    elif isinstance(data, list):
        ids_raw = data
    else:
        ids_raw = []

    if not isinstance(ids_raw, list):
        return set(), False

    out: Set[int] = set()
    for value in ids_raw:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out, True


def _filter_business_assets_with_llm(page_url: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    project = (os.environ.get("PROJECT_ID") or "").strip()
    if not project:
        return _filter_business_assets_heuristic(candidates)

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return _filter_business_assets_heuristic(candidates)

    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    client = genai.Client(vertexai=True, project=project, location=location)
    kept: List[Dict[str, Any]] = []

    for offset in range(0, len(candidates), _ASSET_FILTER_BATCH_SIZE):
        batch = candidates[offset: offset + _ASSET_FILTER_BATCH_SIZE]
        payload = []
        for idx, item in enumerate(batch):
            payload.append(
                {
                    "id": idx,
                    "name": str(item.get("name") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "link_url": str(item.get("link_url") or "").strip() or None,
                }
            )

        prompt = (
            "You are filtering candidate items extracted from a business webpage.\n"
            "Identify which candidate IDs are NOT concrete business assets offered to customers.\n"
            "Non-business examples: About, Contact, Blog posts, Careers, Policy pages, Login/Register, generic site sections.\n"
            "Business examples: products, services, rooms, menu items, plans, packages, bookable offerings, facilities sold/offered.\n\n"
            f"Page URL: {page_url}\n\n"
            "Candidates JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Return ONLY JSON object: {\"non_business_ids\": [<id>, ...]}"
        )

        try:
            response = client.models.generate_content(
                model=_ASSET_FILTER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            ids, ok = _parse_non_business_ids((response.text or "").strip())
            if not ok:
                # If parsing fails, use strict local heuristic for this batch.
                kept.extend(_filter_business_assets_heuristic(batch))
                continue
            for idx, item in enumerate(batch):
                if idx in ids:
                    continue
                kept.append(item)
        except Exception as e:
            logger.warning(
                "[AssetExtraction] LLM filter failed for %s: %s: %s",
                page_url,
                type(e).__name__,
                str(e)[:200],
            )
            kept.extend(_filter_business_assets_heuristic(batch))

    return kept


def _normalize_url_for_match(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.rstrip("/").lower()

    if not parsed.scheme and not parsed.netloc:
        return raw.rstrip("/").lower()

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            parsed.query,
            "",
        )
    )


def _extract_assets_from_page(
    page_url: str,
    page_content: str,
    image_urls: List[str],
) -> List[Dict[str, Any]]:
    """Use Gemini to extract business assets from a page's content + image list."""
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

    # Content is already chunked by the caller – use as-is
    chunk = page_content

    # Format image list for the prompt
    if image_urls:
        image_list = "\n".join(f"- {url}" for url in image_urls[:_ASSET_PROMPT_IMAGE_LIMIT])
    else:
        image_list = "(no images found)"

    prompt = _EXTRACTION_PROMPT.format(
        page_url=page_url,
        page_content=chunk,
        image_list=image_list,
    )

    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        text = (response.text or "").strip()
        return _parse_llm_json(text)
    except Exception as e:
        logger.warning(
            "[AssetExtraction] LLM extraction failed for %s: %s: %s",
            page_url,
            type(e).__name__,
            str(e)[:200],
        )
    return []


class AssetExtractionService:
    """Extracts business assets from crawled documents using LLM."""

    def __init__(self):
        self._repo = PostgresBotAssetRepository()

    def extract_assets_from_documents(
        self,
        *,
        org_id: str,
        bot_id: str,
        documents: List[Dict[str, Any]],
        max_assets: int = 50,
        page_urls: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> List[BotAsset]:
        """Core logic: parse docs -> extract candidates -> download images -> save."""
        # Setup job repo if needed
        job_repo = None
        current_job = None
        if job_id:
            try:
                from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
                job_repo = PostgresAssetExtractionJobRepository()
                current_job = job_repo.get_job(job_id)
            except Exception as e:
                logger.warning(f"Failed to load job {job_id}: {e}")

        # Get existing assets to avoid duplicates
        existing = self._repo.list_assets_for_bot(bot_id, active_only=False)
        existing_names = {a.name.strip().lower() for a in existing}
        allowed_pages = {
            _normalize_url_for_match(u)
            for u in (page_urls or [])
            if (u or "").strip()
        }

        all_extracted: List[Dict[str, Any]] = []

        for doc in documents:
            if len(all_extracted) >= max_assets:
                break

            content = doc.get("content", "")
            url = doc.get("url", "")
            if not content or len(content.strip()) < 100:
                continue
            if allowed_pages and _normalize_url_for_match(url) not in allowed_pages:
                continue

            # Scrape image URLs from the live page HTML
            image_urls = _scrape_image_urls(url)
            chunks = _chunk_content(content)
            logger.info(
                "[AssetExtraction] Processing %s: mode=%s, %d chunks, %d image candidates",
                url,
                _ASSET_EXTRACTION_MODE,
                len(chunks),
                len(image_urls),
            )

            def _append_page_assets(page_assets: List[Dict[str, Any]]) -> None:
                for asset_data in page_assets:
                    if len(all_extracted) >= max_assets:
                        return
                    name = (asset_data.get("name") or "").strip()
                    if not name:
                        continue
                    # Skip duplicates (case-insensitive name match).
                    if name.lower() in existing_names:
                        continue
                    existing_names.add(name.lower())
                    all_extracted.append({
                        **asset_data,
                        "source_url": url,
                    })

            extraction_mode = _ASSET_EXTRACTION_MODE
            if extraction_mode not in ("deterministic", "llm", "hybrid"):
                extraction_mode = "deterministic"

            if extraction_mode in ("llm", "hybrid") and len(all_extracted) < max_assets:
                # Limit to first 2 chunks per page to avoid excessive LLM calls
                for chunk in chunks[:2]:
                    if len(all_extracted) >= max_assets:
                        break
                    page_assets = _extract_assets_from_page(url, chunk, image_urls)
                    _append_page_assets(page_assets)

            if extraction_mode in ("deterministic", "hybrid") and len(all_extracted) < max_assets:
                deterministic_assets = _extract_assets_from_page_deterministic(url, content, image_urls)
                filtered_assets = _filter_business_assets_with_llm(url, deterministic_assets)
                logger.info(
                    "[AssetExtraction] Deterministic candidates on %s: %d -> kept %d",
                    url,
                    len(deterministic_assets),
                    len(filtered_assets),
                )
                _append_page_assets(filtered_assets)

        # Update job with discovered count
        if job_repo and current_job:
            current_job.assets_discovered = len(all_extracted)
            # We treat 'discovered' as the total candidates we will try to download
            try:
                job_repo.update_job(current_job)
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

        # Save to database
        created: List[BotAsset] = []
        now = datetime.now(timezone.utc).isoformat()
        limit = int((os.environ.get("ASSET_MAX_PER_BOT") or "15").strip() or 15)

        for item in all_extracted:
            # ── Re-check limit before EACH insert ──
            current_count = len(self._repo.list_assets_for_bot(bot_id, active_only=False))
            if current_count >= limit:
                logger.info(
                    "Skipping remaining assets for bot %s: limit reached (%d/%d)",
                    bot_id,
                    current_count,
                    limit,
                )
                break

            asset_id = "asset_" + uuid.uuid4().hex[:16]
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).strip() for k in keywords if k][:10]

            link_url = (item.get("link_url") or item.get("source_url") or "").strip() or None
            description = (item.get("description") or "").strip()

            # Must have valid image URL to be worth saving
            image_url = item.get("image_url")
            if not image_url:
                continue
            
            # Download and store image
            image_gcs_uri = ""
            image_public_url = ""
            try:
                # Use helper in this file if available or import
                # The file has local _download_and_store_image helper? 
                # Checking file content... yes, lines 1120+ typically.
                # But wait, looking at line 1007 of previous file content, it seemed to call _download_and_store_image.
                # I'll check if I need to use self or global. 
                # It seems to be a standalone function at bottom of file usually.
                # Assuming `_download_and_store_image(image_url, bot_id, asset_id)` signature based on previous code.
                image_gcs_uri, image_public_url = _download_and_store_image(
                     image_url, bot_id, asset_id
                )
            except Exception as e:
                logger.warning(f"Download failed for {image_url}: {e}")

            # Skip if image download failed
            if not image_gcs_uri:
                logger.debug("[AssetExtraction] Skipping '%s': image download failed", item.get("name", ""))
                continue

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
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            
            try:
                self._repo.create_asset(asset)
                created.append(asset)
                
                # Update job progress
                if job_repo and current_job:
                    current_job.assets_downloaded += 1
                    current_job.assets_created += 1
                    try:
                        job_repo.update_job(current_job)
                    except Exception as e:
                        logger.warning(f"Failed to update job progress: {e}")

            except Exception as e:
                logger.warning(
                    "[AssetExtraction] Failed to save asset '%s': %s",
                    asset.name,
                    str(e)[:200],
                )

        logger.info(
            "Extracted %d assets for bot %s from %d documents",
            len(created),
            bot_id,
            len(documents),
        )
        return created

    def extract_from_gcs_prefix(
        self,
        *,
        org_id: str,
        bot_id: str,
        gcs_prefix: str,
        max_assets: int = 50,
        page_urls: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Load documents from GCS and extract assets. Returns count of created assets."""
        documents = _load_docs_from_gcs(gcs_prefix)
        if not documents:
            return 0
        assets = self.extract_assets_from_documents(
            org_id=org_id,
            bot_id=bot_id,
            documents=documents,
            max_assets=max_assets,
            page_urls=page_urls,
            job_id=job_id,
        )
        return len(assets)


def _load_docs_from_gcs(gcs_prefix: str) -> List[Dict[str, Any]]:
    """Load markdown documents from GCS prefix."""
    if not gcs_prefix:
        return []
    try:
        from common.config import config
    except Exception:
        return []
    bucket_raw = (config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")).strip()
    if not bucket_raw:
        return []
    bucket_name = bucket_raw.strip("/").split("/", 1)[0]

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    documents: List[Dict[str, Any]] = []
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        try:
            content = blob.download_as_text(encoding="utf-8")
        except Exception:
            continue
        url = ""
        if content.startswith("Source URL:"):
            first_line = content.split("\n")[0]
            url = first_line.replace("Source URL:", "").strip()
        if content:
            documents.append({"content": content, "url": url})
    return documents


_instance: Optional[AssetExtractionService] = None


def asset_extraction_service() -> AssetExtractionService:
    global _instance
    if _instance is None:
        _instance = AssetExtractionService()
    return _instance
