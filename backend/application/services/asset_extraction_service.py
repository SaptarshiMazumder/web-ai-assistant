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
from typing import Any, Dict, List, Optional, Tuple
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

    return img_urls[:_ASSET_IMAGE_CANDIDATE_LIMIT]


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
    ) -> List[BotAsset]:
        """
        Analyze crawled documents and extract business assets.

        Args:
            org_id: Organization ID
            bot_id: Bot ID
            documents: List of {"content": str, "url": str} dicts
            max_assets: Maximum number of new assets to create
            page_urls: Optional page URL allow-list

        Returns:
            List of created BotAsset entities
        """
        if not documents or max_assets <= 0:
            return []

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
                "[AssetExtraction] Processing %s: %d chunks, %d image candidates",
                url,
                len(chunks),
                len(image_urls),
            )

            for chunk in chunks:
                if len(all_extracted) >= max_assets:
                    break

                page_assets = _extract_assets_from_page(url, chunk, image_urls)
                for asset_data in page_assets:
                    if len(all_extracted) >= max_assets:
                        break
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

        # Save to database
        created: List[BotAsset] = []
        now = datetime.now(timezone.utc).isoformat()

        for item in all_extracted:
            asset_id = "asset_" + uuid.uuid4().hex[:16]
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).strip() for k in keywords if k][:10]

            link_url = (item.get("link_url") or item.get("source_url") or "").strip() or None
            description = (item.get("description") or "").strip()

            # Try to download and store image from extracted URL
            image_gcs_uri = ""
            image_public_url = ""
            image_url = (item.get("image_url") or "").strip()
            if image_url and image_url.startswith("http"):
                try:
                    image_gcs_uri, image_public_url = _download_and_store_image(
                        image_url, bot_id, asset_id
                    )
                    if image_gcs_uri:
                        logger.info("[AssetExtraction] Downloaded image for '%s': %s", item["name"].strip(), image_url)
                except Exception as e:
                    logger.warning(
                        "[AssetExtraction] Image download failed for '%s': %s",
                        item["name"].strip(),
                        str(e)[:200],
                    )

            asset = BotAsset(
                asset_id=asset_id,
                bot_id=bot_id,
                org_id=org_id,
                name=item["name"].strip(),
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
            except Exception as e:
                logger.warning(
                    "[AssetExtraction] Failed to save asset '%s': %s",
                    asset.name,
                    str(e)[:200],
                )

        logger.info(
            "[AssetExtraction] Extracted %d assets for bot %s from %d documents",
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
