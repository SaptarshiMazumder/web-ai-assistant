"""
Menu Items API – list, create, update, delete menu item assets for a bot.
Mirrors the image-assets API but filters by asset_type="menu_item".
Reuses the same BotAsset storage, GCS, and schemas.
"""

import asyncio
import logging
import os
import re
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, Response as FastAPIResponse, UploadFile
from google.cloud import storage  # type: ignore[import-untyped]

from application.services.asset_image_service import optimize_asset_image
from api.deps.auth import get_current_user, is_super_admin
from api.schemas import (
    AssetExtractionStatusResponse,
    BotAssetAutoExtractRequest,
    BotAssetAutoExtractResponse,
    BotAssetDeleteResponse,
    BotAssetListResponse,
    BotAssetResponse,
)
from common.di.container import asset_repo, bot_service, line_rich_menu_service
from domain.entities import BotAsset
from infrastructure.celery_app import celery_app
from infrastructure.services.indexing_service import _parse_bucket_and_prefix
from infrastructure.tasks.crawl_tasks import menu_extraction_task

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
}
_PAGE_SIZE_DEFAULT = 1000
_PAGE_SIZE_MAX = 1000
_LIST_CACHE_CONTROL = "private, max-age=30, stale-while-revalidate=60"

_PRICE_PATTERN = re.compile(
    r"(?:[¥￥]\s*\d[\d,]*(?:\.\d+)?(?:\s*[-~〜]\s*[¥￥]?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|\$\s*\d[\d,]*(?:\.\d+)?(?:\s*[-~]\s*\$?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|€\s*\d[\d,]*(?:\.\d+)?(?:\s*[-~]\s*€?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|£\s*\d[\d,]*(?:\.\d+)?(?:\s*[-~]\s*£?\s*\d[\d,]*(?:\.\d+)?)?"
    r"|\d[\d,]*(?:\.\d+)?\s*円(?:\s*[-~〜]\s*\d[\d,]*(?:\.\d+)?\s*円)?)",
    re.IGNORECASE,
)


def _split_price_and_details(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    match = _PRICE_PATTERN.search(raw)
    if not match:
        return "", raw
    price_text = match.group(0).strip()
    details = (raw[:match.start()] + " " + raw[match.end():]).strip(" -|:;/,")
    details = re.sub(r"\s+", " ", details).strip()
    return price_text, details


def _build_manual_menu_metadata(
    *,
    name: str,
    description: str,
    link_url: Optional[str],
    keywords: list[str],
) -> Dict[str, Any]:
    price_text, details = _split_price_and_details(description)
    category = "menu"
    joined = f"{name} {description} {' '.join(keywords)}".lower()
    if any(token in joined for token in ("course", "set", "plan", "\u30b3\u30fc\u30b9", "\u30d7\u30e9\u30f3")):
        category = "course"
    elif any(token in joined for token in ("drink", "beverage", "cocktail", "\u30c9\u30ea\u30f3\u30af", "\u98f2\u307f\u7269")):
        category = "drink"
    elif any(token in joined for token in ("lunch", "\u30e9\u30f3\u30c1")):
        category = "lunch"
    elif any(token in joined for token in ("dish", "food", "menu", "\u6599\u7406", "\u30e1\u30cb\u30e5\u30fc", "dinner")):
        category = "dish"
    metadata: Dict[str, Any] = {
        "schema": "menu_item.v1",
        "display_name": (name or "").strip(),
        "category": category,
        "details": details,
        "price_text": price_text,
        "source_url": (link_url or "").strip() or "",
        "keywords": [k for k in keywords if k],
    }
    if price_text:
        metadata["price"] = {"text": price_text}
    return metadata


def _menu_limit() -> int:
    raw = (os.environ.get("MENU_MAX_PER_BOT") or os.environ.get("ASSET_MAX_PER_BOT") or "500").strip()
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 500


def _normalize_page_size(value: int) -> int:
    try:
        return max(1, min(int(value), _PAGE_SIZE_MAX))
    except Exception:
        return _PAGE_SIZE_DEFAULT


def _normalize_offset(value: int) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _asset_list_etag(
    *,
    bot_id: str,
    asset_type: str,
    page_size: int,
    offset: int,
    total_count: int,
    latest_updated: Optional[str],
) -> str:
    raw = f"{bot_id}|{asset_type}|{page_size}|{offset}|{total_count}|{latest_updated or ''}"
    return '"' + hashlib.sha1(raw.encode("utf-8")).hexdigest() + '"'


def _matches_if_none_match(request: Request, etag: str) -> bool:
    header = (request.headers.get("if-none-match") or "").strip()
    if not header:
        return False
    values = [part.strip() for part in header.split(",") if part.strip()]
    return "*" in values or etag in values


def _menu_status_stale_seconds() -> int:
    raw = (os.environ.get("MENU_EXTRACTION_STALE_SECONDS") or "600").strip()
    try:
        return max(120, min(int(raw), 7200))
    except ValueError:
        return 600


def _parse_iso_utc(value: Optional[object]) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _reconcile_menu_job_status(job, repo):
    """
    Reconcile DB status with Celery task state and mark stale jobs as error.
    Prevents infinite queued/running polling when worker/task gets stuck.
    """
    if not job or job.status not in ("queued", "running"):
        return job

    task_state = ""
    task_error = ""
    if job.celery_task_id:
        try:
            task = celery_app.AsyncResult(job.celery_task_id)
            task_state = str(task.state or "").upper()
            if task_state == "FAILURE":
                task_error = str(task.result or "")[:200]
        except Exception:
            task_state = ""

    # Sync terminal Celery states back into DB if worker couldn't persist it.
    if task_state == "SUCCESS":
        job.status = "done"
        repo.update_job(job)
        return repo.get_job(job.job_id) or job
    if task_state == "FAILURE":
        job.status = "error"
        job.error = task_error or job.error or "Menu extraction task failed."
        repo.update_job(job)
        return repo.get_job(job.job_id) or job
    if task_state == "REVOKED":
        job.status = "cancelled"
        job.error = job.error or "Cancelled."
        repo.update_job(job)
        return repo.get_job(job.job_id) or job

    # If a queued/running job stops heartbeating for too long, mark as error.
    now = datetime.now(timezone.utc)
    updated = _parse_iso_utc(getattr(job, "updated_at", None)) or _parse_iso_utc(getattr(job, "created_at", None)) or now
    age_sec = max(0, int((now - updated).total_seconds()))
    if age_sec >= _menu_status_stale_seconds():
        hint = f" celery_state={task_state.lower()}" if task_state else ""
        job.status = "error"
        job.error = job.error or f"Menu extraction timed out after {age_sec}s.{hint}"
        repo.update_job(job)
        return repo.get_job(job.job_id) or job

    return job


def _resolve_org_id(user_ctx, org_id: Optional[str]) -> str:
    if org_id:
        if org_id in user_ctx.org_ids or is_super_admin(user_ctx.claims):
            return org_id
        raise HTTPException(status_code=403, detail="Org membership required")
    if len(user_ctx.org_ids) == 1:
        return user_ctx.org_ids[0]
    if is_super_admin(user_ctx.claims):
        raise HTTPException(status_code=400, detail="org_id is required for admin")
    raise HTTPException(status_code=403, detail="Org membership required")


def _assert_bot_org(bot_id: str, org_id: str) -> None:
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    if bot.org_id != org_id:
        raise HTTPException(status_code=403, detail="Bot does not belong to this org")


def _schedule_line_rich_menu_sync(bot_id: str, *, force: bool = False) -> None:
    async def _runner() -> None:
        try:
            await line_rich_menu_service().sync_for_bot(bot_id, force=force)
        except Exception:
            logger.exception("LINE rich menu sync failed after menu-item change bot_id=%s", bot_id)

    asyncio.create_task(_runner())


def _asset_to_response(a: BotAsset) -> BotAssetResponse:
    return BotAssetResponse(
        asset_id=a.asset_id,
        bot_id=a.bot_id,
        org_id=a.org_id,
        name=a.name,
        description=a.description,
        image_url=a.image_public_url,
        link_url=a.link_url,
        keywords=a.keywords,
        metadata=a.metadata if isinstance(a.metadata, dict) else {},
        is_active=a.is_active,
        asset_type=a.asset_type,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/v1/org/bots/{bot_id}/menu-items", response_model=BotAssetListResponse)
async def list_menu_items(
    bot_id: str,
    request: Request,
    response: FastAPIResponse,
    page_size: int = _PAGE_SIZE_DEFAULT,
    offset: int = 0,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    safe_page_size = _normalize_page_size(page_size)
    safe_offset = _normalize_offset(offset)
    assets, total_count, latest_updated = asset_repo().list_assets_for_bot_paginated(
        bot_id,
        asset_type="menu_item",
        page_size=safe_page_size,
        offset=safe_offset,
    )
    etag = _asset_list_etag(
        bot_id=bot_id,
        asset_type="menu_item",
        page_size=safe_page_size,
        offset=safe_offset,
        total_count=total_count,
        latest_updated=latest_updated,
    )
    headers = {
        "ETag": etag,
        "Cache-Control": _LIST_CACHE_CONTROL,
        "Vary": "Authorization",
    }
    if request is not None and _matches_if_none_match(request, etag):
        return FastAPIResponse(status_code=304, headers=headers)
    if response is not None:
        for key, value in headers.items():
            response.headers[key] = value

    menu_limit = _menu_limit()
    return BotAssetListResponse(
        bot_id=bot_id,
        assets=[_asset_to_response(a) for a in assets],
        count=total_count,
        limit=menu_limit,
        total_count=total_count,
        page_size=safe_page_size,
        offset=safe_offset,
        has_more=(safe_offset + len(assets)) < total_count,
    )


@router.post("/v1/org/bots/{bot_id}/menu-items", response_model=BotAssetResponse)
async def create_menu_item(
    bot_id: str,
    name: str = Form(...),
    description: str = Form(default=""),
    link_url: Optional[str] = Form(default=None),
    keywords: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
    org_id: Optional[str] = Form(default=None),
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    limit = _menu_limit()
    existing = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item")
    if len(existing) >= limit:
        raise HTTPException(status_code=400, detail=f"Menu item limit reached ({limit}).")

    asset_id = "asset_" + uuid.uuid4().hex[:16]
    gcs_uri = ""
    public_url = ""

    # Image is optional for menu items
    if file and file.filename:
        content_type = (getattr(file, "content_type", None) or "").strip().lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {content_type}")
        data = await file.read()
        if data:
            data, content_type, ext = optimize_asset_image(data, content_type)
            bucket_name, base_prefix = _parse_bucket_and_prefix()
            blob_name = f"{base_prefix}/assets/{bot_id}/{asset_id}.{ext}".strip("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data, content_type=content_type)
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            public_url = f"/v1/image-assets/{asset_id}/image"

    now = datetime.now(timezone.utc).isoformat()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    item_name = name.strip()
    item_description = description.strip()
    item_link_url = link_url.strip() if link_url else None

    asset = BotAsset(
        asset_id=asset_id,
        bot_id=bot_id,
        org_id=resolved_org,
        name=item_name,
        description=item_description,
        image_gcs_uri=gcs_uri,
        image_public_url=public_url,
        link_url=item_link_url,
        keywords=kw_list,
        metadata=_build_manual_menu_metadata(
            name=item_name,
            description=item_description,
            link_url=item_link_url,
            keywords=kw_list,
        ),
        is_active=True,
        asset_type="menu_item",
        created_at=now,
        updated_at=now,
    )
    asset_repo().create_asset(asset)
    _schedule_line_rich_menu_sync(bot_id, force=True)
    return _asset_to_response(asset)


@router.put("/v1/org/bots/{bot_id}/menu-items/{asset_id}", response_model=BotAssetResponse)
async def update_menu_item(
    bot_id: str,
    asset_id: str,
    name: str = Form(...),
    description: str = Form(default=""),
    link_url: Optional[str] = Form(default=None),
    keywords: str = Form(default=""),
    is_active: bool = Form(default=True),
    file: Optional[UploadFile] = File(default=None),
    org_id: Optional[str] = Form(default=None),
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    existing = asset_repo().get_asset(asset_id)
    if not existing or existing.bot_id != bot_id or existing.asset_type != "menu_item":
        raise HTTPException(status_code=404, detail="Menu item not found")

    if file and file.filename:
        content_type = (getattr(file, "content_type", None) or "").strip().lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {content_type}")
        data = await file.read()
        if data:
            data, content_type, ext = optimize_asset_image(data, content_type)
            bucket_name, base_prefix = _parse_bucket_and_prefix()
            blob_name = f"{base_prefix}/assets/{bot_id}/{asset_id}.{ext}".strip("/")
            client = storage.Client()
            bucket_obj = client.bucket(bucket_name)
            blob = bucket_obj.blob(blob_name)
            blob.upload_from_string(data, content_type=content_type)
            existing.image_gcs_uri = f"gs://{bucket_name}/{blob_name}"
            existing.image_public_url = f"/v1/image-assets/{asset_id}/image"

    now = datetime.now(timezone.utc).isoformat()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    item_name = name.strip()
    item_description = description.strip()
    item_link_url = link_url.strip() if link_url else None

    existing.name = item_name
    existing.description = item_description
    existing.link_url = item_link_url
    existing.keywords = kw_list
    existing.metadata = _build_manual_menu_metadata(
        name=item_name,
        description=item_description,
        link_url=item_link_url,
        keywords=kw_list,
    )
    existing.is_active = is_active
    existing.updated_at = now
    asset_repo().update_asset(existing)
    _schedule_line_rich_menu_sync(bot_id, force=True)
    return _asset_to_response(existing)


@router.delete("/v1/org/bots/{bot_id}/menu-items/{asset_id}", response_model=BotAssetDeleteResponse)
async def delete_menu_item(
    bot_id: str,
    asset_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    existing = asset_repo().get_asset(asset_id)
    if not existing or existing.bot_id != bot_id or existing.asset_type != "menu_item":
        raise HTTPException(status_code=404, detail="Menu item not found")

    # Delete from GCS
    try:
        bucket_name, _ = _parse_bucket_and_prefix()
        gcs_uri = existing.image_gcs_uri or ""
        if gcs_uri.startswith("gs://"):
            parts = gcs_uri.replace("gs://", "").split("/", 1)
            if len(parts) == 2:
                client = storage.Client()
                bucket_obj = client.bucket(parts[0])
                blob = bucket_obj.blob(parts[1])
                blob.delete()
    except Exception as e:
        logger.warning("Failed to delete GCS blob for menu item %s: %s", asset_id, e)

    asset_repo().delete_asset(bot_id, asset_id)
    _schedule_line_rich_menu_sync(bot_id, force=True)
    return BotAssetDeleteResponse(bot_id=bot_id, asset_id=asset_id)


# ---------------------------------------------------------------------------
# Auto-extract menu items from training data
# ---------------------------------------------------------------------------


@router.post("/v1/org/bots/{bot_id}/menu-items/auto-extract", response_model=BotAssetAutoExtractResponse)
async def auto_extract_menu_items(
    bot_id: str,
    payload: BotAssetAutoExtractRequest = Body(default_factory=BotAssetAutoExtractRequest),
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Extract menu items from the bot's crawled training data using LLM."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    selected_pages = []
    seen_pages = set()
    for raw in payload.page_urls or []:
        cleaned = (raw or "").strip()
        if not cleaned or cleaned in seen_pages:
            continue
        seen_pages.add(cleaned)
        selected_pages.append(cleaned)

    from infrastructure.db.repositories import PostgresIndexJobRepository, PostgresAssetExtractionJobRepository, AssetExtractionJob
    job_repo = PostgresIndexJobRepository()
    jobs = job_repo.list_jobs_for_bot(bot_id)
    gcs_prefix = ""
    if jobs:
        for j in sorted(jobs, key=lambda x: x.created_at or "", reverse=True):
            if j.gcs_prefix:
                gcs_prefix = j.gcs_prefix
                break

    if not gcs_prefix:
        raise HTTPException(status_code=400, detail="No training data found. Train the bot first.")

    limit = _menu_limit()
    current_items = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item")
    current_count = len(current_items)
    remaining = max(0, limit - current_count)

    if remaining <= 0:
        return BotAssetAutoExtractResponse(
            ok=True,
            job_id=None,
            assets_extracted=0,
            assets_count=current_count,
            assets_limit=limit,
            pages_considered=len(selected_pages),
        )

    extract_repo = PostgresAssetExtractionJobRepository()
    job_id = "menuext_" + uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    job = AssetExtractionJob(
        job_id=job_id,
        bot_id=bot_id,
        org_id=resolved_org,
        status="queued",
        created_at=now,
        updated_at=now,
        gcs_prefix=gcs_prefix,
        page_urls=selected_pages,
        assets_discovered=0,
        assets_downloaded=0,
        assets_created=0,
    )
    extract_repo.create_job(job)

    task = menu_extraction_task.delay(
        bot_id=bot_id,
        org_id=resolved_org,
        gcs_prefix=gcs_prefix,
        job_id=job_id,
    )

    job.celery_task_id = task.id
    extract_repo.update_job(job)

    return BotAssetAutoExtractResponse(
        ok=True,
        job_id=job_id,
        assets_extracted=0,
        assets_count=current_count,
        assets_limit=limit,
        pages_considered=len(selected_pages),
    )


@router.get("/v1/org/bots/{bot_id}/menu-items/extract-status", response_model=AssetExtractionStatusResponse)
async def get_menu_extraction_status(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get status of the latest menu extraction job."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
    repo = PostgresAssetExtractionJobRepository()
    # Find latest menu extraction job (prefixed with menuext_)
    job = repo.get_latest_job_for_bot(bot_id, prefix="menuext_")
    job = _reconcile_menu_job_status(job, repo)

    current_items = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="menu_item")
    limit = _menu_limit()

    if not job:
        return AssetExtractionStatusResponse(
            job_id="",
            status="none",
            assets_total=len(current_items),
            limit=limit,
        )
    if job.status == "done":
        _schedule_line_rich_menu_sync(bot_id, force=True)

    return AssetExtractionStatusResponse(
        job_id=job.job_id,
        status=job.status,
        assets_discovered=job.assets_discovered,
        assets_downloaded=job.assets_downloaded,
        assets_created=job.assets_created,
        assets_total=len(current_items),
        limit=limit,
        error=job.error,
    )


@router.post("/v1/org/bots/{bot_id}/menu-items/extract-cancel")
async def cancel_menu_extraction(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Cancel the latest queued/running menu extraction job for this bot."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
    repo = PostgresAssetExtractionJobRepository()
    job = repo.get_latest_job_for_bot(bot_id, prefix="menuext_")
    if not job:
        return {"status": "none", "job_id": ""}
    if job.status not in ("queued", "running"):
        return {"status": "already_done", "job_id": job.job_id}

    if job.celery_task_id:
        try:
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception as e:
            logger.warning("Failed to revoke menu extraction task %s for bot %s: %s", job.celery_task_id, bot_id, e)

    job.status = "cancelled"
    job.error = "Cancelled by user."
    repo.update_job(job)
    return {"status": "cancelled", "job_id": job.job_id}


