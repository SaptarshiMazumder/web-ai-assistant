import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from google.cloud import storage


logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "document.pdf"
    # Keep it readable but safe for GCS object names.
    n = os.path.basename(n)
    n = re.sub(r"[^a-zA-Z0-9._-]+", "_", n).strip("_")
    if not n.lower().endswith(".pdf"):
        n += ".pdf"
    return n[:180]


@dataclass(frozen=True)
class UploadedSourceFile:
    bucket: str
    blob_name: str
    gcs_uri: str
    filename: str
    bytes: int


class GcsSourceFileRepository:
    """
    Store raw uploaded source files (e.g., PDFs) in GCS.

    This is separate from extracted markdown documents (handled by GCSDocumentStorageRepository).
    """

    def __init__(
        self,
        bucket_name: str,
        base_prefix: str,
        *,
        storage_client: Optional[storage.Client] = None,
    ) -> None:
        self._bucket_name = (bucket_name or "").strip()
        self._base_prefix = (base_prefix or "").strip("/")
        self._client = storage_client

    def _get_client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    def upload_pdf(self, *, bot_id: str, source_id: str, filename: str, data: bytes) -> UploadedSourceFile:
        if not self._bucket_name:
            raise RuntimeError("GCS bucket is not configured")
        if not bot_id or not source_id:
            raise ValueError("Missing bot_id/source_id")
        if not data:
            raise ValueError("Empty PDF")

        safe = _safe_name(filename)
        prefix = f"{self._base_prefix}/source-files/{source_id}".strip("/")
        blob_name = f"{prefix}/{safe}"

        client = self._get_client()
        bucket = client.bucket(self._bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type="application/pdf")

        gcs_uri = f"gs://{self._bucket_name}/{blob_name}"
        logger.info("Uploaded PDF source to %s (%d bytes)", gcs_uri, len(data))
        return UploadedSourceFile(
            bucket=self._bucket_name,
            blob_name=blob_name,
            gcs_uri=gcs_uri,
            filename=safe,
            bytes=len(data),
        )

    def download(self, *, blob_name: str) -> bytes:
        if not self._bucket_name:
            raise RuntimeError("GCS bucket is not configured")
        name = (blob_name or "").strip().lstrip("/")
        if not name:
            raise ValueError("Missing blob_name")
        client = self._get_client()
        bucket = client.bucket(self._bucket_name)
        blob = bucket.blob(name)
        return blob.download_as_bytes()
