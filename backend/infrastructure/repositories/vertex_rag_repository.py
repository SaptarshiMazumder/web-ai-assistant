import hashlib
import os
import time
from typing import Optional

import vertexai
from vertexai import rag as vx_rag

from common.config import config
from domain.repositories import RAGRepository

from infrastructure.rag.crawl_service import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_PUBLISHER_MODEL

PROJECT_ID = (config.PROJECT_ID or "").strip()
VERTEX_LOCATION = (config.LOCATION or "us-central1").strip()


class VertexRAGRepository(RAGRepository):
    def ensure_corpus(self, bot_id: str, *, force_new: bool = False) -> str:
        """Ensure a RAG corpus exists for the bot, return corpus resource name."""
        from infrastructure.db.repositories import PostgresBotCorpusRepository

        corpus_repo = PostgresBotCorpusRepository()
        existing = corpus_repo.get_bot_corpus(bot_id)
        if existing and not force_new:
            try:
                vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
                try:
                    vx_rag.get_corpus(existing)
                    return existing
                except Exception:
                    pass
            except Exception:
                pass

        vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
        from infrastructure.rag.crawl_service import _slugify
        from infrastructure.services.indexing_service import _rag_display_name

        display_name = _rag_display_name(bot_id)
        emb_cfg = vx_rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=vx_rag.VertexPredictionEndpoint(
                publisher_model=EMBEDDING_PUBLISHER_MODEL
            )
        )
        corpus = vx_rag.create_corpus(
            display_name=display_name,
            backend_config=vx_rag.RagVectorDbConfig(rag_embedding_model_config=emb_cfg),
        )
        corpus_repo.upsert_bot_corpus(bot_id, corpus.name)
        return corpus.name

    def import_documents(self, corpus_resource: str, storage_prefix: str) -> None:
        """Import documents from storage prefix into the RAG corpus."""
        from infrastructure.rag.crawl_service import _parse_bucket_and_prefix
        from infrastructure.db.connection import get_connection

        bucket_name, _ = _parse_bucket_and_prefix()
        gcs_uri = f"gs://{bucket_name}/{storage_prefix}/"
        # Vertex RAG corpora reject concurrent import operations (FailedPrecondition).
        # We serialize imports per corpus across workers using a Postgres advisory lock,
        # and add a small retry loop for any lingering in-flight operations.
        lock_key = int.from_bytes(hashlib.sha1((corpus_resource or "").encode("utf-8")).digest()[:8], "big") % (2**63 - 1)

        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
            con.commit()

            attempt = 0
            backoff_s = 2.0
            max_attempts = 10
            while True:
                try:
                    vx_rag.import_files(
                        corpus_resource,
                        [gcs_uri],
                        transformation_config=vx_rag.TransformationConfig(
                            chunking_config=vx_rag.ChunkingConfig(
                                chunk_size=CHUNK_SIZE,
                                chunk_overlap=CHUNK_OVERLAP,
                            )
                        ),
                        max_embedding_requests_per_min=1000,
                    )
                    return
                except Exception as e:
                    msg = str(e) or ""
                    busy = ("There are other operations running on the RagCorpus" in msg) or ("FailedPrecondition" in msg and "RagCorpus" in msg)
                    attempt += 1
                    if busy and attempt < max_attempts:
                        time.sleep(backoff_s)
                        backoff_s = min(backoff_s * 1.8, 30.0)
                        continue
                    raise RuntimeError(
                        f"RAG import failed for corpus {corpus_resource}, GCS URI {gcs_uri}: {e}"
                    ) from e
        finally:
            try:
                with con.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                con.commit()
            except Exception:
                pass
            try:
                con.close()
            except Exception:
                pass
