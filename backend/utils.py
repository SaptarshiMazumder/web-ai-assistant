import re, datetime

def extract_json_from_text(text):
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return arr_match.group(0)
    return None


def clean_markdown(md: str) -> str:
    # Remove [text](url) links, keep just the text
    md = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md)
    # Remove naked URLs
    md = re.sub(r'http[s]?://\S+', '', md)
    # Optionally, remove extra whitespace
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()




PROMPT_LOG_PATH = "llm_prompt_log.txt"

def log_llm_prompt(prompt: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("="*40 + "\n")
        f.write(f"[{now}] LLM PROMPT SENT\n")
        f.write("="*40 + "\n")
        f.write(prompt)
        f.write("\n" + "-"*40 + "\n\n")


from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

from google.cloud import aiplatform
from vertexai.preview import rag

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

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