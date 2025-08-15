# save as: inspect_corpus.py
# Run: python inspect_corpus.py

from vertexai import rag
import vertexai
from urllib.parse import urlparse
from collections import Counter
from datetime import datetime, timezone

# --------------------
# CONSTANTS
# --------------------
PROJECT_ID  = "gen-lang-client-0545494042"
LOCATION    = "us-central1"

RAG_CORPUS  = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"

# Show the latest N files (by update/create time)
LATEST_N    = 100

# If you only care about files from a specific GCS prefix or domain, set a filter (optional)
FILTER_CONTAINS = ""   # e.g. "gs://web-assistant-test-bucket-1/raw_pages/playvalorant-com/"

# --------------------
# HELPERS
# --------------------
def _get_uri(f) -> str:
    # Try common fields across SDK versions
    for attr in ("source_uri", "gcs_uri", "uri"):
        v = getattr(f, attr, None)
        if v:
            return v
    # Sometimes metadata may carry it
    md = getattr(f, "metadata", None)
    if isinstance(md, dict):
        for k in ("source_uri", "gcs_uri", "uri"):
            if k in md and md[k]:
                return md[k]
    # Fallback: display_name can still help
    return getattr(f, "display_name", "") or ""

def _get_domain_from_uri(uri: str) -> str:
    if not uri:
        return "(unknown)"
    if uri.startswith("gs://"):
        # gs://bucket/path → domain = bucket
        try:
            return uri.split("/", 3)[2]  # bucket name
        except Exception:
            return "(gcs)"
    # http(s)
    try:
        netloc = urlparse(uri).netloc
        return netloc or "(unknown)"
    except Exception:
        return "(unknown)"

def _to_dt(x):
    # Try to turn SDK timestamp fields into datetime for sorting
    # Common attributes: update_time, create_time
    for attr in ("update_time", "create_time"):
        v = getattr(x, attr, None)
        if not v:
            continue
        # v might already be a datetime, string, or protobuf-like with seconds/nanos
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try RFC 3339/ISO
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                pass
        # Protobuf Timestamp-ish with seconds/nanos
        sec = getattr(v, "seconds", None)
        nanos = getattr(v, "nanos", 0)
        if isinstance(sec, (int, float)):
            try:
                return datetime.fromtimestamp(sec + nanos / 1e9, tz=timezone.utc)
            except Exception:
                pass
    return None

def _fmt_dt(dt: datetime) -> str:
    if not dt:
        return "-"
    # Display in UTC ISO short form
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _get_size_bytes(f) -> int:
    # Some SDKs expose size via content_length or file_size or metadata
    for attr in ("content_length", "file_size", "size_bytes"):
        v = getattr(f, attr, None)
        if isinstance(v, int):
            return v
    md = getattr(f, "metadata", None)
    if isinstance(md, dict):
        for k in ("content_length", "file_size", "size_bytes"):
            if isinstance(md.get(k), int):
                return md[k]
    return -1

def _get_status(f) -> str:
    # Many SDKs expose status/state like PROCESSING, READY, FAILED
    for attr in ("state", "status"):
        v = getattr(f, attr, None)
        if v:
            # Some objects are enums; convert to str
            return str(v)
    return "-"

# --------------------
# MAIN
# --------------------
def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print(f"[RAG] Inspecting corpus:\n  {RAG_CORPUS}\n")
    files = list(rag.list_files(corpus_name=RAG_CORPUS))
    total = len(files)
    print(f"[RAG] Total files in corpus: {total}")

    # Optional filter
    if FILTER_CONTAINS:
        files = [f for f in files if FILTER_CONTAINS in _get_uri(f)]
        print(f"[RAG] After filter ({FILTER_CONTAINS!r}): {len(files)} files")

    # Group by domain
    domain_counts = Counter()
    for f in files:
        uri = _get_uri(f)
        domain = _get_domain_from_uri(uri)
        domain_counts[domain] += 1

    if domain_counts:
        print("\n[RAG] Files by domain (top 50):")
        for domain, count in domain_counts.most_common(50):
            print(f"  {domain:50}  {count}")
    else:
        print("\n[RAG] No domain information found.")

    # Sort by update/create time (desc) and show latest N
    files_with_time = []
    for f in files:
        dt = _to_dt(f)
        files_with_time.append((dt, f))
    files_with_time.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)

    latest = files_with_time[:LATEST_N]
    if latest:
        print(f"\n[RAG] Latest {len(latest)} file(s):\n")
        for i, (dt, f) in enumerate(latest, 1):
            name = getattr(f, "name", "-")
            display = getattr(f, "display_name", "") or "-"
            uri = _get_uri(f)
            status = _get_status(f)
            size = _get_size_bytes(f)
            size_s = f"{size} B" if size >= 0 else "-"
            print(f"{i:3}. name: {name}")
            print(f"     time:   {_fmt_dt(dt)}")
            print(f"     status: {status}")
            print(f"     size:   {size_s}")
            print(f"     title:  {display}")
            print(f"     uri:    {uri}\n")
    else:
        print("\n[RAG] No files to list for 'latest' view.\n")

    print("[RAG] Done.\n")

if __name__ == "__main__":
    main()
