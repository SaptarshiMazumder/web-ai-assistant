# save as: sync_gcs_to_corpus.py
# Run: python sync_gcs_to_corpus.py

import vertexai
from vertexai import rag

# --------------------
# CONSTANTS
# --------------------
PROJECT_ID = "gen-lang-client-0545494042"
LOCATION = "us-central1"
RAG_CORPUS = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"

# GCS paths to import — can be a list of one or more prefixes
GCS_PATHS = [
    "gs://web-assistant-test-bucket-1/raw_pages/"
]

# Chunking + rate limit settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
MAX_EMBED_RPM = 1000  # embedding requests per minute

# --------------------
# MAIN LOGIC
# --------------------
def main():
    # Initialize Vertex in the same region as your corpus
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print(f"[RAG] Updating corpus:\n  {RAG_CORPUS}")
    print("[RAG] From GCS paths:")
    for p in GCS_PATHS:
        print(f"  - {p}")

    # Import (or re-import) the files
    rag.import_files(
        RAG_CORPUS,
        GCS_PATHS,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        ),
        max_embedding_requests_per_min=MAX_EMBED_RPM,
    )

    print("\n[RAG] Import request submitted.")
    print("[RAG] Unchanged files will be skipped; new/updated files will be indexed.")
    print("[RAG] To remove files, delete them from the corpus explicitly.\n")

if __name__ == "__main__":
    main()
