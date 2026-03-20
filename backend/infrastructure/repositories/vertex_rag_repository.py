import logging
import os
import time
from typing import Optional

import vertexai
from vertexai import rag as vx_rag
from google.cloud import storage

from common.config import config
from domain.platform_profiles import (
    get_rag_import_batch_size,
    get_rag_import_busy_retry_initial_backoff_sec,
    get_rag_import_busy_retry_max_attempts,
    get_rag_import_busy_retry_max_backoff_sec,
    get_rag_import_max_embedding_requests_per_min,
)
from domain.repositories import RAGRepository

from infrastructure.rag.crawl_service import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_PUBLISHER_MODEL

PROJECT_ID = (config.PROJECT_ID or "").strip()
VERTEX_LOCATION = (config.LOCATION or "us-central1").strip()
_VERTEX_IMPORT_MAX_GCS_URIS = 25

logger = logging.getLogger(__name__)


class VertexRAGRepository(RAGRepository):

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        """Check if an exception indicates the resource genuinely does not exist (404/NOT_FOUND)."""
        msg = str(exc).lower()
        return "404" in msg or "not_found" in msg or "not found" in msg

    def ensure_corpus(self, bot_id: str, *, force_new: bool = False) -> str:
        """Ensure a RAG corpus exists for the bot, return corpus resource name.

        SAFETY: Only creates a new corpus if:
          - No corpus reference exists in the DB, OR
          - force_new=True, OR
          - The existing corpus was genuinely deleted (404 from Vertex AI).

        Transient errors (network, auth, timeout) are retried with backoff.
        If retries are exhausted, the existing ref is returned — we NEVER
        silently create a new empty corpus and lose indexed data.
        """
        import logging
        _log = logging.getLogger("VertexRAGRepository.ensure_corpus")

        from infrastructure.db.repositories import PostgresBotCorpusRepository

        corpus_repo = PostgresBotCorpusRepository()
        existing = corpus_repo.get_bot_corpus(bot_id)

        if existing and not force_new:
            max_retries = 3
            backoff = 2.0
            last_err: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
                    vx_rag.get_corpus(existing)
                    return existing  # ✅ Corpus exists and is accessible
                except Exception as exc:
                    last_err = exc
                    if self._is_not_found_error(exc):
                        _log.warning(
                            "Corpus %s for bot %s was deleted (404). Will create a new one.",
                            existing, bot_id,
                        )
                        break
                    _log.warning(
                        "ensure_corpus: transient error verifying corpus %s for bot %s "
                        "(attempt %d/%d): %s",
                        existing, bot_id, attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2
            else:
                # All retries exhausted and NOT a 404 — refuse to recreate.
                _log.error(
                    "ensure_corpus: REFUSING to create new corpus for bot %s. "
                    "Existing corpus ref %s could not be verified after %d attempts. "
                    "Last error: %s. Returning existing ref to avoid data loss.",
                    bot_id, existing, max_retries, last_err,
                )
                return existing

        _log.info("Creating new RAG corpus for bot %s (force_new=%s, had_existing=%s)",
                  bot_id, force_new, bool(existing))
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
        _log.info("Created new corpus %s for bot %s", corpus.name, bot_id)
        return corpus.name

    def import_documents(self, corpus_resource: str, storage_prefix: str) -> None:
        """Import documents from storage prefix into the RAG corpus."""
        from infrastructure.rag.crawl_service import _parse_bucket_and_prefix

        bucket_name, _ = _parse_bucket_and_prefix()
        if not bucket_name:
            raise RuntimeError("GCS bucket is not configured.")

        # Import only markdown source documents.
        # Avoid importing helper artifacts (e.g. url_map.json) which degrade retrieval quality.
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_paths = [
            blob.name
            for blob in bucket.list_blobs(prefix=storage_prefix)
            if blob.name.endswith(".md")
        ]
        if not blob_paths:
            raise RuntimeError(f"No markdown files found under prefix: gs://{bucket_name}/{storage_prefix}/")
        gcs_uris = [f"gs://{bucket_name}/{name}" for name in blob_paths]
        configured_batch_size = get_rag_import_batch_size()
        batch_size = max(1, min(configured_batch_size, _VERTEX_IMPORT_MAX_GCS_URIS))
        max_embedding_requests_per_min = get_rag_import_max_embedding_requests_per_min()
        busy_retry_max_attempts = get_rag_import_busy_retry_max_attempts()
        busy_retry_initial_backoff_sec = get_rag_import_busy_retry_initial_backoff_sec()
        busy_retry_max_backoff_sec = get_rag_import_busy_retry_max_backoff_sec()
        batches = [gcs_uris[i:i + batch_size] for i in range(0, len(gcs_uris), batch_size)]
        logger.info(
            "RAG import batching: corpus=%s files=%d configured_batch_size=%d effective_batch_size=%d batches=%d",
            corpus_resource,
            len(gcs_uris),
            configured_batch_size,
            batch_size,
            len(batches),
        )
        # Vertex RAG corpora reject concurrent import operations (FailedPrecondition).
        # No advisory lock — the Vertex busy-retry loop below handles concurrency
        # with exponential backoff. Advisory locks caused zombie-lock hangs.
        for batch_index, batch_uris in enumerate(batches, start=1):
            attempt = 0
            backoff_s = busy_retry_initial_backoff_sec
            while True:
                try:
                    logger.info("RAG import: calling vx_rag.import_files batch %d/%d (%d files) ...", batch_index, len(batches), len(batch_uris))
                    import_result = vx_rag.import_files(
                        corpus_resource,
                        batch_uris,
                        transformation_config=vx_rag.TransformationConfig(
                            chunking_config=vx_rag.ChunkingConfig(
                                chunk_size=CHUNK_SIZE,
                                chunk_overlap=CHUNK_OVERLAP,
                            )
                        ),
                        max_embedding_requests_per_min=max_embedding_requests_per_min,
                    )
                    # Log import result to detect silent partial failures
                    partial_failures = getattr(import_result, "partial_failures_count", None)
                    skipped = getattr(import_result, "skipped_rag_files_count", None)
                    imported = getattr(import_result, "imported_rag_files_count", None)
                    logger.info(
                        "RAG import: batch %d/%d completed — imported=%s skipped=%s partial_failures=%s result=%s",
                        batch_index, len(batches), imported, skipped, partial_failures,
                        str(import_result)[:500] if import_result else "None",
                    )
                    if partial_failures and int(partial_failures) > 0:
                        logger.warning(
                            "RAG import: batch %d/%d had %s partial failures! Some files may not have been embedded.",
                            batch_index, len(batches), partial_failures,
                        )
                    # Fail loudly if nothing was actually imported
                    if imported is not None and int(imported) == 0:
                        raise RuntimeError(
                            f"RAG import batch {batch_index}/{len(batches)} imported 0 files "
                            f"(skipped={skipped}, partial_failures={partial_failures}). "
                            f"Content was NOT embedded into the corpus."
                        )
                    break
                except Exception as e:
                    msg = str(e) or ""
                    busy = ("There are other operations running on the RagCorpus" in msg) or ("FailedPrecondition" in msg and "RagCorpus" in msg)
                    attempt += 1
                    if busy and attempt < busy_retry_max_attempts:
                        logger.info("RAG import: corpus busy (attempt %d/%d), retrying in %.1fs ...", attempt, busy_retry_max_attempts, backoff_s)
                        time.sleep(backoff_s)
                        backoff_s = min(backoff_s * 1.8, busy_retry_max_backoff_sec)
                        continue
                    raise RuntimeError(
                        "RAG import failed for corpus "
                        f"{corpus_resource}, prefix gs://{bucket_name}/{storage_prefix}/, "
                        f"batch {batch_index}/{len(batches)} ({len(batch_uris)} files): {e}"
                    ) from e
