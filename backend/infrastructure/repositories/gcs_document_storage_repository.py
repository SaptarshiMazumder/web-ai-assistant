import hashlib
import re
import logging
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from google.cloud import storage

from domain.entities import Document
from domain.repositories import DocumentStorageRepository

from infrastructure.rag.crawl_service import host_prefix_from_url


logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "index"


class GCSDocumentStorageRepository(DocumentStorageRepository):
    def __init__(self, bucket_name: str, base_prefix: str, storage_client: Optional[storage.Client] = None) -> None:
        self._bucket_name = bucket_name
        self._base_prefix = base_prefix
        self._client = storage_client

    def _get_client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    def save_documents(self, bot_id: str, documents: List[Document]) -> str:
        if not documents:
            raise ValueError("No documents to upload to GCS.")
        client = self._get_client()
        bucket = client.bucket(self._bucket_name)

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        first_url = documents[0].url
        host_prefix = host_prefix_from_url(first_url)
        prefix = f"{self._base_prefix}/{host_prefix}/{timestamp}"

        max_log_docs = 5
        for idx, doc in enumerate(documents):
            url = doc.url
            content = doc.content
            parsed = urlparse(url)
            path_slug = _slugify(parsed.path or "index")
            url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            filename = f"{path_slug or 'index'}-{url_hash}.md"
            blob_name = f"{prefix}/{filename}"
            blob = bucket.blob(blob_name)
            content_bytes = content.encode("utf-8") if isinstance(content, str) else content
            blob.upload_from_string(content_bytes, content_type="text/markdown; charset=utf-8")
            if idx < max_log_docs:
                try:
                    saved = blob.download_as_text(encoding="utf-8")
                    preview = saved if len(saved) <= 4000 else saved[:4000] + "\n...[truncated]"
                    logger.info("GCS saved preview [%s] %s:\n%s", idx + 1, url, preview)
                except Exception:
                    logger.warning("GCS saved preview failed for %s", url)

        return prefix
