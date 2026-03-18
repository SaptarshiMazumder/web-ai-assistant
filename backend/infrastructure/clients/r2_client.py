"""
Cloudflare R2 storage client (S3-compatible).

Uploads image assets to R2 and returns public CDN URLs for direct serving.
Falls back gracefully: if R2 is not configured, callers should use GCS.
"""

import logging
from typing import Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig

from common.config import config

logger = logging.getLogger(__name__)

_client = None


def _is_r2_configured() -> bool:
    return bool(
        config.R2_ACCOUNT_ID
        and config.R2_ACCESS_KEY_ID
        and config.R2_SECRET_ACCESS_KEY
        and config.R2_BUCKET_NAME
        and config.R2_PUBLIC_URL
    )


def _get_client():
    global _client
    if _client is not None:
        return _client

    _client = boto3.client(
        "s3",
        endpoint_url=f"https://{config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=BotoConfig(
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )
    return _client


def upload_image(
    data: bytes,
    key: str,
    content_type: str = "image/jpeg",
) -> Tuple[str, str]:
    """
    Upload image bytes to R2.

    Args:
        data: Image bytes to upload.
        key: Object key (e.g. "assets/bot_id/asset_id.jpg").
        content_type: MIME type of the image.

    Returns:
        (r2_key, public_url) tuple.
    """
    client = _get_client()
    client.put_object(
        Bucket=config.R2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=31536000, immutable",
    )
    public_url = f"{config.R2_PUBLIC_URL}/{key}"
    return key, public_url


def delete_image(key: str) -> None:
    """Delete an image from R2 by key."""
    try:
        client = _get_client()
        client.delete_object(
            Bucket=config.R2_BUCKET_NAME,
            Key=key,
        )
    except Exception as e:
        logger.warning("Failed to delete R2 object %s: %s", key, e)


def is_available() -> bool:
    """Check if R2 is configured and should be used for new uploads."""
    return _is_r2_configured()
