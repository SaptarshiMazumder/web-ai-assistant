from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import os

# Create a RAG Corpus, Import Files, and Generate a response

# TODO(developer): Update and un-comment below lines
PROJECT_ID = "gen-lang-client-0545494042"
display_name = "test_corpus"
# paths = ["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]  # Supports Google Cloud Storage and Google Drive Links
# BUCKET_NAME = os.getenv("GCS_BUCKET", "YOUR_BUCKET_NAME")
# GCS_SUBPATH = os.getenv("GCS_SUBPATH", "raw_pages")

BUCKET_NAME = "web-assistant-test-bucket-1"
GCS_SUBPATH = "raw_pages"

# if BUCKET_NAME == "YOUR_BUCKET_NAME":
#     raise ValueError(
#         "Set GCS_BUCKET env var to your Google Cloud Storage bucket (e.g., 'my-bucket')."
#     )

paths = [f"gs://{BUCKET_NAME}/{GCS_SUBPATH}/"]

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create RagCorpus
# Configure embedding model, for example "text-embedding-005".
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    ),
)

# Import Files to the RagCorpus
rag.import_files(
    rag_corpus.name,
    paths,
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=100,
        ),
    ),
    max_embedding_requests_per_min=1000,  # Optional
)

# Direct context retrieval
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,  # Optional
    filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
)
# response = rag.retrieval_query(
#     rag_resources=[
#         rag.RagResource(
#             rag_corpus=rag_corpus.name,
#             # Optional: supply IDs from `rag.list_files()`.
#             # rag_file_ids=["rag-file-1", "rag-file-2", ...],
#         )
#     ],
#     text="What is RAG and why it is helpful?",
#     rag_retrieval_config=rag_retrieval_config,
# )
# print(response)

# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

# Create a Gemini model instance
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)

# Generate responses interactively from terminal
print("Interactive RAG console: type 'exit', 'quit' or 'q' to leave.")
while True:
    try:
        user_query = input("Enter your prompt > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not user_query:
        continue

    if user_query.lower() in ("exit", "quit", "q"):
        print("Goodbye.")
        break

    try:
        response = rag_model.generate_content(user_query)
        print(response.text)
    except Exception as e:
        print(f"Error generating response: {e}")
# Example response:
#   RAG stands for Retrieval-Augmented Generation.
#   It's a technique used in AI to enhance the quality of responses
# ...