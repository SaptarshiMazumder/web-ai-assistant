"""
Helpers for normalizing asset images before storage.
"""

from __future__ import annotations

import io
import os
from typing import Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}

_DEFAULT_MAX_DIMENSION = max(256, min(int(os.environ.get("ASSET_IMAGE_MAX_DIMENSION", "1280")), 4096))
_DEFAULT_JPEG_QUALITY = max(40, min(int(os.environ.get("ASSET_IMAGE_JPEG_QUALITY", "82")), 95))


def extension_for_content_type(content_type: str) -> str:
    return _CONTENT_TYPE_EXTENSIONS.get((content_type or "").lower(), "jpg")


def optimize_asset_image(
    image_bytes: bytes,
    content_type: str,
    *,
    max_dimension: int = _DEFAULT_MAX_DIMENSION,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> Tuple[bytes, str, str]:
    """
    Resize + compress asset images for faster load and smaller storage.

    Returns: (optimized_bytes, optimized_content_type, file_extension)
    """
    if not image_bytes:
        return image_bytes, "image/jpeg", "jpg"

    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type == "image/svg+xml":
        return image_bytes, normalized_type, "svg"

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        fallback_type = normalized_type if normalized_type.startswith("image/") else "image/jpeg"
        return image_bytes, fallback_type, extension_for_content_type(fallback_type)
    except Exception:
        fallback_type = normalized_type if normalized_type.startswith("image/") else "image/jpeg"
        return image_bytes, fallback_type, extension_for_content_type(fallback_type)

    try:
        img = ImageOps.exif_transpose(img)

        # Never upscale.
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img.thumbnail((max_dimension, max_dimension), resample)

        has_alpha = img.mode in ("RGBA", "LA")
        if img.mode == "P" and "transparency" in img.info:
            has_alpha = True

        out = io.BytesIO()

        if has_alpha:
            img.convert("RGBA").save(out, format="PNG", optimize=True)
            payload = out.getvalue()
            return payload, "image/png", "png"

        img.convert("RGB").save(
            out,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
        payload = out.getvalue()
        return payload, "image/jpeg", "jpg"
    except Exception:
        fallback_type = normalized_type if normalized_type.startswith("image/") else "image/jpeg"
        return image_bytes, fallback_type, extension_for_content_type(fallback_type)
