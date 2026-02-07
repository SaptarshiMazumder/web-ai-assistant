"""
Delete ALL bots everywhere: Postgres, Vertex RAG corpora, and GCS.

Run from backend dir. Loads .env for PROJECT_ID, LOCATION, DATABASE_URL, GCS_BUCKET.

  cd backend
  python scripts/reset_all_bots.py

On Windows, if the DB step fails with "no pq wrapper available", install:
  pip install "psycopg[binary]"

Order: 1) Vertex RAG corpora, 2) GCS objects (under GCS_BUCKET bucket/prefix),
3) DB tables (index_jobs, bot_domains, bot_corpora, bots). Organizations and
users are left intact.
"""
import os
import sys
from typing import Tuple

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
os.chdir(_backend_dir)

from dotenv import load_dotenv
load_dotenv()

# Use config after path is set
from common.config import config


def _delete_all_vertex_corpora() -> int:
    import vertexai
    from vertexai import rag as vx_rag

    project = (getattr(config, "PROJECT_ID", None) or os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (getattr(config, "LOCATION", None) or os.environ.get("LOCATION") or "us-central1").strip()
    if not project:
        raise RuntimeError("Set PROJECT_ID or GOOGLE_CLOUD_PROJECT in .env")

    vertexai.init(project=project, location=location)
    corpora = list(vx_rag.list_corpora())
    if not corpora:
        return 0
    deleted = 0
    for c in corpora:
        name = getattr(c, "name", None) or str(c)
        try:
            vx_rag.delete_corpus(name=name)
            deleted += 1
            print(f"  Deleted RAG corpus: {name}")
        except Exception as e:
            print(f"  Failed to delete {name}: {e}")
    return deleted


def _parse_bucket_and_prefix() -> Tuple[str, str]:
    """Parse GCS_BUCKET (bucket or bucket/prefix) without importing DB layer."""
    bucket_cfg = (getattr(config, "GCS_BUCKET", None) or os.environ.get("GCS_BUCKET", "") or "").strip()
    if not bucket_cfg:
        raise ValueError("GCS_BUCKET is not set")
    parts = bucket_cfg.strip("/").split("/", 1)
    bucket_name = parts[0]
    base_prefix = parts[1] if len(parts) == 2 else ""
    return bucket_name, base_prefix


def _delete_gcs_under_prefix() -> Tuple[str, str, int]:
    """Delete all GCS objects under the app bucket/prefix. No DB imports."""
    from google.cloud import storage
    bucket_name, base_prefix = _parse_bucket_and_prefix()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    deleted = 0
    for blob in bucket.list_blobs(prefix=base_prefix):
        blob.delete()
        deleted += 1
    return bucket_name, base_prefix, deleted


def _reset_bots_in_db() -> None:
    """Delete all bot-related rows. Requires psycopg with libpq (e.g. psycopg[binary])."""
    try:
        from infrastructure.db.connection import get_connection
    except ImportError as e:
        if "pq" in str(e).lower() or "psycopg" in str(e).lower():
            print(
                "   Database step requires psycopg with libpq. On Windows, run:\n"
                "   pip install \"psycopg[binary]\""
            )
        raise
    con = get_connection()
    try:
        con.execute("DELETE FROM booking_link_jobs")
        con.execute("DELETE FROM index_jobs")
        con.execute("DELETE FROM bot_sources")
        con.execute("DELETE FROM bot_domains")
        con.execute("DELETE FROM bot_corpora")
        con.execute("DELETE FROM bots")
        con.commit()
    finally:
        con.close()


def main() -> None:
    print("Reset all bots (DB + Vertex RAG + GCS)...")
    if not (os.environ.get("DATABASE_URL") or getattr(config, "DATABASE_URL", "")):
        print("DATABASE_URL is not set. Aborting.")
        sys.exit(1)
    bucket_cfg = (os.environ.get("GCS_BUCKET") or getattr(config, "GCS_BUCKET", "") or "").strip()
    if not bucket_cfg:
        print("GCS_BUCKET is not set. Skipping GCS deletion.")

    # 1) Vertex RAG
    print("\n1) Vertex RAG corpora...")
    n = _delete_all_vertex_corpora()
    print(f"   Deleted {n} corpora.")

    # 2) GCS
    print("\n2) GCS objects under app prefix...")
    if bucket_cfg:
        try:
            bucket_name, prefix, deleted = _delete_gcs_under_prefix()
            print(f"   Deleted {deleted} objects under gs://{bucket_name}/{prefix or '(root)'}")
        except Exception as e:
            print(f"   GCS error: {e}")
    else:
        print("   Skipped (no GCS_BUCKET).")

    # 3) DB
    print("\n3) Database (index_jobs, bot_domains, bot_corpora, bots)...")
    _reset_bots_in_db()
    print("   Done.")

    print("\nAll bots removed. You can create new bots from scratch.")


if __name__ == "__main__":
    main()
