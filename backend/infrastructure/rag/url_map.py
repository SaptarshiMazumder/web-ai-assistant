"""
Resolve GCS chunk URIs to original page URLs for chat citations.
At index time we write url_map.json (filename -> page URL) per GCS prefix.
At retrieval we load the map and replace evidence[].url so the widget shows page links.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from google.cloud import storage

logger = logging.getLogger(__name__)

URL_MAP_FILENAME = "url_map.json"

# In-memory cache per (bucket, prefix) to avoid repeated GCS reads in the same process.
_prefix_cache: Dict[Tuple[str, str], Dict[str, str]] = {}


def _parse_gcs_uri(gcs_uri: str) -> Optional[Tuple[str, str, str]]:
    """Return (bucket, prefix, filename) or None if not a valid gs:// URI."""
    if not (gcs_uri or isinstance(gcs_uri, str)) or not gcs_uri.strip().lower().startswith("gs://"):
        return None
    uri = gcs_uri.strip()
    rest = uri[5:]  # after "gs://"
    if "/" not in rest:
        return None
    bucket, path = rest.split("/", 1)
    if not path:
        return None
    parts = path.rstrip("/").split("/")
    filename = parts[-1] if parts else ""
    prefix = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return (bucket, prefix, filename)


def load_url_map(bucket_name: str, prefix: str) -> Dict[str, str]:
    """Load url_map.json from gs://bucket_name/prefix/url_map.json. Returns filename -> page URL."""
    cache_key = (bucket_name, prefix)
    if cache_key in _prefix_cache:
        return _prefix_cache[cache_key]
    result: Dict[str, str] = {}
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{prefix}/{URL_MAP_FILENAME}" if prefix else URL_MAP_FILENAME)
        if not blob.exists():
            _prefix_cache[cache_key] = result
            return result
        data = blob.download_as_text(encoding="utf-8")
        loaded = json.loads(data)
        if isinstance(loaded, dict):
            result = {str(k): str(v) for k, v in loaded.items()}
    except Exception as e:
        logger.debug("url_map load failed for gs://%s/%s: %s", bucket_name, prefix, e)
    _prefix_cache[cache_key] = result
    return result


def resolve_evidence_urls(evidence: List[Dict[str, str]], bucket_name: str) -> None:
    """
    In-place: replace evidence[].url when it is a GCS URI with the original page URL
    from url_map.json in that prefix. If bucket_name is empty or no map exists, leave url unchanged.
    """
    if not bucket_name or not evidence:
        return
    for e in evidence:
        url = e.get("url") or ""
        if not url.strip().lower().startswith("gs://"):
            continue
        parsed = _parse_gcs_uri(url)
        if not parsed:
            continue
        b, prefix, filename = parsed
        if b != bucket_name:
            continue
        # Strip fragment (#) and query (?) — Vertex RAG may return gs://.../file.md#chunk-0
        base_filename = filename.split("#")[0].split("?")[0]
        url_map = load_url_map(bucket_name, prefix)
        page_url = url_map.get(base_filename) or url_map.get(filename)
        if page_url:
            e["url"] = page_url
