from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
from google.cloud import aiplatform, storage
from vertexai.preview import rag as preview_rag
from app.config import config
import uuid

aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)

# --- Create corpus once ---
def get_or_create_rag_corpus():
    embedding_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="projects/google/models/text-embedding-005"
        )
    )
    return rag.create_corpus(
        display_name="web-ai-dynamic-corpus",
        backend_config=rag.RagVectorDbConfig(rag_embedding_model_config=embedding_config)
    )

def generate_rag_answer_from_vertex_ai(question: str) -> str:
    rag_corpus = get_or_create_rag_corpus()
    retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)]
            )
        )
    )
    model = GenerativeModel(
        model_name="gemini-2.0-flash-001",
        tools=[retrieval_tool]
    )
    response = model.generate_content(question)
    return response.text

def ingest_content_to_vertex_ai(text: str, url: str):
    bucket_name = config.GCS_BUCKET
    file_name = f"scraped_pages/{uuid.uuid4()}.txt"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(text)
    gcs_path = f"gs://{bucket_name}/{file_name}"
    preview_rag.import_files(
        get_or_create_rag_corpus().name,
        paths=[gcs_path],
        transformation_config=preview_rag.TransformationConfig(
            chunking_config=preview_rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
        )
    )
    return gcs_path