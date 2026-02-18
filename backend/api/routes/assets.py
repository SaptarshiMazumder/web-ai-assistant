"""
Image Assets API – upload, list, update, delete asset cards for a bot.
Also exposes a **public** image proxy endpoint so LINE / Instagram can fetch images.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
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
from common.di.container import asset_repo, bot_service
from domain.entities import BotAsset
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


def _asset_limit() -> int:
    raw = (os.environ.get("ASSET_MAX_PER_BOT") or "15").strip()
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 15


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
        is_active=a.is_active,
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
    existing_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False)
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
    return _asset_to_response(asset)


@router.get("/v1/org/bots/{bot_id}/image-assets", response_model=BotAssetListResponse)
async def list_assets(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    assets = asset_repo().list_assets_for_bot(bot_id)
    limit = _asset_limit()
    return BotAssetListResponse(
        bot_id=bot_id,
        assets=[_asset_to_response(a) for a in assets],
        count=len(assets),
        limit=limit,
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
    current_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False)
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
    job = repo.get_latest_job_for_bot(bot_id)
    
    current_assets = asset_repo().list_assets_for_bot(bot_id, active_only=False)
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
