from datetime import datetime, timezone
from typing import Iterable, List, Tuple

import vertexai
from google.cloud import storage
from vertexai import rag as vx_rag

from common.config import config
from infrastructure.db.connection import  get_connection


def _parse_bucket_and_prefix() -> Tuple[str, str]:
    bucket_and_prefix = (config.GCS_BUCKET or "").strip("/").split("/", 1)
    if len(bucket_and_prefix) == 2:
        return bucket_and_prefix[0], bucket_and_prefix[1]
    if len(bucket_and_prefix) == 1 and bucket_and_prefix[0]:
        return bucket_and_prefix[0], ""
    raise ValueError("Invalid GCS_BUCKET configuration")


def _list_corpora() -> List[str]:
    con = get_connection()
    try:
        rows = con.execute("SELECT DISTINCT corpus_resource FROM bot_corpora").fetchall()
        return [r[0] for r in rows if r and r[0]]
    finally:
        con.close()


def _clear_corpora_table() -> None:
    con = get_connection()
    try:
        con.execute("DELETE FROM bot_corpora")
        con.commit()
    finally:
        con.close()


def delete_gcs_objects(*, allow_root: bool = False) -> Tuple[str, str, int]:
    bucket_name, base_prefix = _parse_bucket_and_prefix()
    if not base_prefix and not allow_root:
        raise ValueError("Refusing to delete entire bucket without allow_root")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    deleted = 0
    for blob in bucket.list_blobs(prefix=base_prefix):
        blob.delete()
        deleted += 1
    return bucket_name, base_prefix, deleted


def delete_rag_corpora(corpora: Iterable[str] | None = None) -> int:
    to_delete = list(corpora) if corpora is not None else _list_corpora()
    if not to_delete:
        _clear_corpora_table()
        return 0
    deleted = 0
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    for corpus_resource in to_delete:
        try:
            vx_rag.delete_corpus(corpus_resource)
            deleted += 1
        except Exception:
            # Best-effort cleanup; keep going.
            pass
    _clear_corpora_table()
    return deleted


def reset_postgres_data() -> None:
    """
    Fresh reset for Postgres-backed local dev. Truncates core tables and
    recreates the default org.
    """
    con = get_connection()
    try:
        con.execute(
            """
            TRUNCATE TABLE
              index_jobs,
              booking_link_jobs,
              bot_sources,
              bot_domains,
              bot_corpora,
              bots,
              org_memberships,
              users,
              organizations,
              domain_corpora
            """
        )
        con.commit()
        
    finally:
        con.close()
