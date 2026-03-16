"""
Image Assets API – upload, list, update, delete asset cards for a bot.
Also exposes a **public** image proxy endpoint so LINE / Instagram can fetch images.
"""

import logging
import os
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, Response as FastAPIResponse, UploadFile
from fastapi.responses import Response
from google.cloud import storage  # type: ignore[import-untyped]

from application.services.asset_image_service import optimize_asset_image
from infrastructure.assets.asset_resolver import invalidate_asset_bank_cache
from api.deps.auth import get_current_user, is_super_admin
from api.schemas import (
    AssetExtractionStatusResponse,
    BotAssetAutoExtractRequest,
    BotAssetAutoExtractResponse,
    BotAssetDeleteResponse,
    BotAssetListResponse,
    BotAssetResponse,
)
from common.di.container import asset_repo, bot_service
from domain.entities import BotAsset
from infrastructure.celery_app import celery_app
from infrastructure.services.indexing_service import _parse_bucket_and_prefix
from infrastructure.tasks.crawl_tasks import asset_extraction_task

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers (mirrors saas.py helpers)
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


def _asset_limit() -> int:
    raw = (os.environ.get("ASSET_MAX_PER_BOT") or "15").strip()
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 15


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


def _pick_image_extraction_job(repo, bot_id: str, *, prefer_active: bool = False):
    """Return image extraction job across manual + pipeline prefixes.

    If prefer_active is True, queued/running jobs are prioritized first.
    """
    jobs = []
    for prefix in ("extract_", "assetext_"):
        job = repo.get_latest_job_for_bot(bot_id, prefix=prefix)
        if job:
            jobs.append(job)
    if not jobs:
        return None

    def sort_key(job):
        return (
            str(getattr(job, "updated_at", "") or ""),
            str(getattr(job, "created_at", "") or ""),
            str(getattr(job, "job_id", "") or ""),
        )

    if prefer_active:
        active_jobs = [job for job in jobs if str(getattr(job, "status", "") or "").strip().lower() in ("queued", "running")]
        if active_jobs:
            return max(active_jobs, key=sort_key)

    return max(jobs, key=sort_key)


def _active_pipeline_asset_extraction_job_id(bot_id: str) -> Optional[str]:
    """Return linked asset extraction job id for the active pipeline step, if any."""
    from infrastructure.db.repositories import PostgresJobPipelineRepository

    pipeline_repo = PostgresJobPipelineRepository()
    run = pipeline_repo.get_latest_run_for_bot(bot_id)
    if not run:
        return None
    run_status = str(getattr(run, "status", "") or "").strip().lower()
    if run_status in ("done", "error"):
        return None

    steps = pipeline_repo.list_steps(run.run_id)
    active_statuses = {"queued", "running", "paused"}
    current_step_index = int(getattr(run, "current_step_index", 0) or 0)

    # Prefer the active step at run.current_step_index.
    for step in steps:
        if int(getattr(step, "step_index", -1) or -1) != current_step_index:
            continue
        if str(getattr(step, "job_id", "") or "").strip().lower() != "asset_extraction":
            continue
        if str(getattr(step, "status", "") or "").strip().lower() not in active_statuses:
            continue
        linked_job_id = str(getattr(step, "linked_job_id", "") or "").strip()
        if linked_job_id:
            return linked_job_id

    # Fallback: any active asset_extraction step in this run.
    candidates = []
    for step in steps:
        if str(getattr(step, "job_id", "") or "").strip().lower() != "asset_extraction":
            continue
        if str(getattr(step, "status", "") or "").strip().lower() not in active_statuses:
            continue
        linked_job_id = str(getattr(step, "linked_job_id", "") or "").strip()
        if linked_job_id:
            candidates.append((int(getattr(step, "step_index", 0) or 0), linked_job_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _mark_pipeline_asset_extraction_cancelled(bot_id: str, linked_job_id: str) -> None:
    """Best effort: immediately reflect extraction cancellation in pipeline run/steps."""
    if not bot_id or not linked_job_id:
        return
    from infrastructure.db.repositories import PostgresJobPipelineRepository

    pipeline_repo = PostgresJobPipelineRepository()
    run = pipeline_repo.get_latest_run_for_bot(bot_id)
    if not run:
        return
    run_status = str(getattr(run, "status", "") or "").strip().lower()
    if run_status in ("done", "error", "cancelled"):
        return

    steps = pipeline_repo.list_steps(run.run_id)
    target_step = None
    for step in steps:
        if str(getattr(step, "job_id", "") or "").strip().lower() != "asset_extraction":
            continue
        if str(getattr(step, "linked_job_id", "") or "").strip() != linked_job_id:
            continue
        if str(getattr(step, "status", "") or "").strip().lower() not in ("queued", "running", "paused"):
            continue
        if target_step is None or int(getattr(step, "step_index", -1) or -1) > int(getattr(target_step, "step_index", -1) or -1):
            target_step = step
    if target_step is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    cancel_message = "Image extraction cancelled by user."

    target_step.status = "cancelled"
    target_step.progress_pct = 100
    target_step.current_stage_key = "cancelled"
    target_step.current_message = cancel_message
    target_step.last_error = None
    target_step.completed_at = now
    target_step.updated_at = now
    pipeline_repo.update_step(target_step)

    # Mark remaining queued steps as cancelled so UI treats run as terminal immediately.
    for step in steps:
        if int(getattr(step, "step_index", -1) or -1) <= int(getattr(target_step, "step_index", -1) or -1):
            continue
        if str(getattr(step, "status", "") or "").strip().lower() != "queued":
            continue
        step.status = "cancelled"
        step.progress_pct = 100
        step.current_stage_key = "cancelled"
        step.current_message = "Skipped after cancellation."
        step.last_error = None
        step.completed_at = now
        step.updated_at = now
        pipeline_repo.update_step(step)

    run.status = "cancelled"
    run.current_step_index = int(getattr(target_step, "step_index", 0) or 0)
    run.current_step_id = target_step.job_id
    run.current_stage_key = "cancelled"
    run.current_message = cancel_message
    run.last_error = None
    run.progress_pct = max(int(getattr(run, "progress_pct", 0) or 0), 100)
    run.updated_at = now
    pipeline_repo.update_run(run)


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
# CRUD endpoints (authenticated)
# ---------------------------------------------------------------------------


@router.post("/v1/org/bots/{bot_id}/image-assets", response_model=BotAssetResponse)
async def create_asset(
    bot_id: str,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
    link_url: Optional[str] = Form(default=None),
    keywords: str = Form(default=""),  # comma-separated
    org_id: Optional[str] = Form(default=None),
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    limit = _asset_limit()
    existing_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="image")
    if len(existing_assets) >= limit:
        raise HTTPException(status_code=400, detail=f"Asset limit reached ({limit}).")

    content_type = (getattr(file, "content_type", None) or "").strip().lower()
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    data, content_type, ext = optimize_asset_image(data, content_type)

    # Upload to GCS
    bucket_name, base_prefix = _parse_bucket_and_prefix()
    asset_id = "asset_" + uuid.uuid4().hex[:16]
    blob_name = f"{base_prefix}/assets/{bot_id}/{asset_id}.{ext}".strip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    gcs_uri = f"gs://{bucket_name}/{blob_name}"

    # Public URL goes through our proxy endpoint (using new image-assets path)
    public_url = f"/v1/image-assets/{asset_id}/image"

    now = datetime.now(timezone.utc).isoformat()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    asset = BotAsset(
        asset_id=asset_id,
        bot_id=bot_id,
        org_id=resolved_org,
        name=name.strip(),
        description=description.strip(),
        image_gcs_uri=gcs_uri,
        image_public_url=public_url,
        link_url=link_url.strip() if link_url else None,
        keywords=kw_list,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    asset_repo().create_asset(asset)
    invalidate_asset_bank_cache(bot_id)
    return _asset_to_response(asset)


@router.get("/v1/org/bots/{bot_id}/image-assets", response_model=BotAssetListResponse)
async def list_assets(
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
        asset_type="image",
        page_size=safe_page_size,
        offset=safe_offset,
    )
    etag = _asset_list_etag(
        bot_id=bot_id,
        asset_type="image",
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

    asset_limit = _asset_limit()
    return BotAssetListResponse(
        bot_id=bot_id,
        assets=[_asset_to_response(a) for a in assets],
        count=total_count,
        limit=asset_limit,
        total_count=total_count,
        page_size=safe_page_size,
        offset=safe_offset,
        has_more=(safe_offset + len(assets)) < total_count,
    )


@router.put("/v1/org/bots/{bot_id}/image-assets/{asset_id}", response_model=BotAssetResponse)
async def update_asset(
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
    if not existing or existing.bot_id != bot_id:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Optionally replace image
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
            # Update to new public URL format
            existing.image_public_url = f"/v1/image-assets/{asset_id}/image"

    now = datetime.now(timezone.utc).isoformat()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    existing.name = name.strip()
    existing.description = description.strip()
    existing.link_url = link_url.strip() if link_url else None
    existing.keywords = kw_list
    existing.is_active = is_active
    existing.updated_at = now
    asset_repo().update_asset(existing)
    invalidate_asset_bank_cache(bot_id)
    return _asset_to_response(existing)


@router.delete("/v1/org/bots/{bot_id}/image-assets/{asset_id}", response_model=BotAssetDeleteResponse)
async def delete_asset(
    bot_id: str,
    asset_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    existing = asset_repo().get_asset(asset_id)
    if not existing or existing.bot_id != bot_id:
        raise HTTPException(status_code=404, detail="Asset not found")

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
        logger.warning("Failed to delete GCS blob for asset %s: %s", asset_id, e)

    asset_repo().delete_asset(bot_id, asset_id)
    invalidate_asset_bank_cache(bot_id)
    return BotAssetDeleteResponse(bot_id=bot_id, asset_id=asset_id)


# ---------------------------------------------------------------------------
# Auto-extract assets from training data (authenticated)
# ---------------------------------------------------------------------------


@router.post("/v1/org/bots/{bot_id}/image-assets/auto-extract", response_model=BotAssetAutoExtractResponse)
async def auto_extract_assets(
    bot_id: str,
    payload: BotAssetAutoExtractRequest = Body(default_factory=BotAssetAutoExtractRequest),
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Extract image assets from the bot's crawled training data using LLM."""
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

    # Find the latest GCS prefix from index jobs
    from infrastructure.db.repositories import PostgresIndexJobRepository, PostgresAssetExtractionJobRepository, AssetExtractionJob
    job_repo = PostgresIndexJobRepository()
    jobs = job_repo.list_jobs_for_bot(bot_id)
    gcs_prefix = ""
    if jobs:
        # Get latest completed job with a gcs_prefix
        for j in sorted(jobs, key=lambda x: x.created_at or "", reverse=True):
            if j.gcs_prefix:
                gcs_prefix = j.gcs_prefix
                break

    if not gcs_prefix:
        raise HTTPException(
            status_code=400,
            detail="No training data found. Train the bot first.",
        )

    limit = _asset_limit()
    current_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="image")
    current_count = len(current_assets)
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

    # Create extraction job
    extract_repo = PostgresAssetExtractionJobRepository()
    job_id = "extract_" + uuid.uuid4().hex
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

    # Convert sync task call to async
    task = asset_extraction_task.delay(
        bot_id=bot_id,
        org_id=resolved_org,
        gcs_prefix=gcs_prefix,
        job_id=job_id,
    )
    
    # Update with task ID
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


@router.get("/v1/org/bots/{bot_id}/image-assets/extract-status", response_model=AssetExtractionStatusResponse)
async def get_asset_extraction_status(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get status of the latest asset extraction job."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
    repo = PostgresAssetExtractionJobRepository()
    preferred_job_id = _active_pipeline_asset_extraction_job_id(bot_id)
    job = repo.get_job(preferred_job_id) if preferred_job_id else None
    if not job:
        job = _pick_image_extraction_job(repo, bot_id, prefer_active=True)

    current_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False, asset_type="image")
    limit = _asset_limit()

    if not job:
        return AssetExtractionStatusResponse(
            job_id="",
            status="none",
            assets_total=len(current_assets),
            limit=limit,
        )

    return AssetExtractionStatusResponse(
        job_id=job.job_id,
        status=job.status,
        assets_discovered=job.assets_discovered,
        assets_downloaded=job.assets_downloaded,
        assets_created=job.assets_created,
        assets_total=len(current_assets),
        limit=limit,
        error=job.error,
    )


@router.post("/v1/org/bots/{bot_id}/image-assets/extract-cancel")
async def cancel_asset_extraction(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Cancel the latest queued/running image extraction job for this bot."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    from infrastructure.db.repositories import PostgresAssetExtractionJobRepository
    repo = PostgresAssetExtractionJobRepository()
    preferred_job_id = _active_pipeline_asset_extraction_job_id(bot_id)
    job = repo.get_job(preferred_job_id) if preferred_job_id else None
    if not job:
        job = _pick_image_extraction_job(repo, bot_id, prefer_active=True)
    if not job:
        return {"status": "none", "job_id": ""}
    if job.status not in ("queued", "running"):
        return {"status": "already_done", "job_id": job.job_id}

    if job.celery_task_id:
        try:
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception as e:
            logger.warning("Failed to revoke extraction task %s for bot %s: %s", job.celery_task_id, bot_id, e)

    job.status = "cancelled"
    job.error = "Cancelled by user."
    repo.update_job(job)
    try:
        _mark_pipeline_asset_extraction_cancelled(bot_id, job.job_id)
    except Exception as e:
        logger.warning("Failed to mark pipeline run cancelled for asset extraction job %s: %s", job.job_id, e)
    return {"status": "cancelled", "job_id": job.job_id}


# ---------------------------------------------------------------------------
# Public image proxy (no auth – LINE / Instagram / widget fetch this)
# ---------------------------------------------------------------------------


@router.get("/v1/assets/{asset_id}/image")
async def public_asset_image_legacy(asset_id: str):
    """(Legacy) Serve image under old path."""
    return await public_asset_image(asset_id)


@router.get("/v1/image-assets/{asset_id}/image")
async def public_asset_image(asset_id: str):
    """Serve asset image from GCS – publicly accessible for external channels."""
    asset = asset_repo().get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    gcs_uri = asset.image_gcs_uri or ""
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=404, detail="No image stored")

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=404, detail="Invalid GCS URI")

    try:
        client = storage.Client()
        bucket = client.bucket(parts[0])
        blob = bucket.blob(parts[1])
        data = blob.download_as_bytes()
        ct = blob.content_type or "image/png"
    except Exception as e:
        logger.error("Failed to fetch asset image %s: %s", asset_id, e)
        raise HTTPException(status_code=502, detail="Failed to fetch image")

    return Response(
        content=data,
        media_type=ct,
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": "inline",
        },
    )
