import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import vertexai
from vertexai import rag as vx_rag

from config import config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify_hostname(hostname: str) -> str:
    """
    Safe slug for display names. Hostnames can contain dots and hyphens; we normalize
    to a stable, collision-resistant slug for corpus display_name purposes.
    """
    h = (hostname or "").strip().lower()
    h = re.sub(r"[^a-z0-9]+", "-", h)
    h = re.sub(r"-+", "-", h).strip("-")
    return h or "unknown-host"


def _db_path() -> str:
    # Local persistent registry. For production you’d swap this to Postgres/etc.
    root = os.path.join(os.path.dirname(__file__), "_data")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "domain_registry.sqlite3")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path())
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS domain_corpora (
          hostname TEXT PRIMARY KEY,
          corpus_resource TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    return con


def get_corpus_for_host(hostname: str) -> Optional[str]:
    h = (hostname or "").strip().lower()
    if not h:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT corpus_resource FROM domain_corpora WHERE hostname = ?",
            (h,),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def upsert_corpus_for_host(hostname: str, corpus_resource: str) -> None:
    h = (hostname or "").strip().lower()
    if not h:
        raise ValueError("hostname required")
    now = _utc_now()
    con = _connect()
    try:
        con.execute(
            """
            INSERT INTO domain_corpora(hostname, corpus_resource, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(hostname) DO UPDATE SET
              corpus_resource=excluded.corpus_resource,
              updated_at=excluded.updated_at
            """,
            (h, corpus_resource, now, now),
        )
        con.commit()
    finally:
        con.close()


def delete_corpus_mapping_for_host(hostname: str) -> None:
    h = (hostname or "").strip().lower()
    if not h:
        return
    con = _connect()
    try:
        con.execute("DELETE FROM domain_corpora WHERE hostname = ?", (h,))
        con.commit()
    finally:
        con.close()


def get_or_create_corpus_for_host(hostname: str) -> str:
    """
    Returns the Vertex RAG corpus resource name for this hostname, creating a new corpus
    if it doesn’t exist in our registry yet.
    """
    h = (hostname or "").strip().lower()
    if not h:
        raise ValueError("hostname required")

    existing = get_corpus_for_host(h)
    if existing:
        return existing

    # Create new corpus
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    display_name = f"web-rag-host-{_slugify_hostname(h)}"
    emb_cfg = vx_rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=vx_rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
    corpus = vx_rag.create_corpus(
        display_name=display_name,
        backend_config=vx_rag.RagVectorDbConfig(rag_embedding_model_config=emb_cfg),
    )
    upsert_corpus_for_host(h, corpus.name)
    return corpus.name

