import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")

    # Google Gemini/Vertex
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not GOOGLE_APPLICATION_CREDENTIALS:
        # Optionally set a default or raise error
        # GOOGLE_APPLICATION_CREDENTIALS = r"C:\Users\googler\Downloads\tour-proj-451201-f03b91fdf3d7.json"
        GOOGLE_APPLICATION_CREDENTIALS = r"C:\Users\googler\Downloads\gen-lang-client-0545494042-b36c2aa59869.json"

    # PROJECT_ID = os.environ.get("PROJECT_ID", "tour-proj-451201")
    PROJECT_ID = os.environ.get("PROJECT_ID", "gen-lang-client-0545494042")

    LOCATION = os.environ.get("LOCATION", "us-central1")
    # Chroma DB path
    CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "backend/chroma_db/")
    # GCS bucket for RAG
    GCS_BUCKET = os.environ.get("GCS_BUCKET", "web-ai-dynamic-corpus-bucket")
    # Prompt log path
    PROMPT_LOG_PATH = os.environ.get("PROMPT_LOG_PATH", "llm_prompt_log.txt")

config = Config()