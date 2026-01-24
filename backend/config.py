import os
from dotenv import load_dotenv

load_dotenv()

_BACKEND_DIR = os.path.dirname(__file__)
_DEFAULT_WORKER_PYTHON = ""
try:
    if os.name == "nt":
        cand = os.path.join(_BACKEND_DIR, "venv", "Scripts", "python.exe")
    else:
        cand = os.path.join(_BACKEND_DIR, "venv", "bin", "python")
    if os.path.exists(cand):
        _DEFAULT_WORKER_PYTHON = cand
except Exception:
    _DEFAULT_WORKER_PYTHON = ""

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

    # Ensure all Google SDKs (GCS, Vertex) use this credential file, even if
    # Application Default Credentials (ADC) is set to a different user account.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

    # PROJECT_ID = os.environ.get("PROJECT_ID", "tour-proj-451201")
    PROJECT_ID = os.environ.get("PROJECT_ID", "gen-lang-client-0545494042")

    LOCATION = os.environ.get("LOCATION", "us-central1")
    # Chroma DB path
    CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "backend/chroma_db/")
    # GCS bucket for RAG (default prefix uses the new multi-tenant layout)
    GCS_BUCKET = os.environ.get("GCS_BUCKET", "web-assistant-test-bucket-1/saas/")
    # Prompt log path
    PROMPT_LOG_PATH = os.environ.get("PROMPT_LOG_PATH", "llm_prompt_log.txt")

    # Optional: protects bot creation endpoint. If unset, /v1/bots is open (dev bootstrap).
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

    # If true, enforce verified-domain checks + tight widget CORS.
    # For now (per request), default is OFF.
    REQUIRE_DOMAIN_VERIFICATION = os.environ.get("REQUIRE_DOMAIN_VERIFICATION", "").strip().lower() in ("1", "true", "yes", "y")

    # Optional override for the Python executable used to spawn the indexing worker.
    # Useful on Windows when `uvicorn` is launched from a different interpreter than your venv.
    # If unset, we default to backend/venv python when present.
    WORKER_PYTHON = (os.environ.get("WORKER_PYTHON") or _DEFAULT_WORKER_PYTHON).strip()

config = Config()