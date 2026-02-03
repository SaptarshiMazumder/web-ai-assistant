import os
from dotenv import load_dotenv

load_dotenv()

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
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
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if GOOGLE_APPLICATION_CREDENTIALS:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

    PROJECT_ID = os.environ.get("PROJECT_ID", "").strip()

    LOCATION = os.environ.get("LOCATION", "us-central1")
    # Chroma DB path
    CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "backend/chroma_db/")
    # GCS bucket for RAG (default prefix uses the new multi-tenant layout)
    GCS_BUCKET = os.environ.get("GCS_BUCKET", "").strip()
    # Prompt log path
    PROMPT_LOG_PATH = os.environ.get(
        "PROMPT_LOG_PATH",
        os.path.join(_BACKEND_DIR, "common", "logs", "llm_prompt_log.txt"),
    )

    # Optional: protects bot creation endpoint. If unset, /v1/bots is open (dev bootstrap).
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

    # Auth (managed IdP / OIDC)
    AUTH_ISSUER = os.environ.get("AUTH_ISSUER", "")
    AUTH_AUDIENCE = os.environ.get("AUTH_AUDIENCE", "")
    AUTH_JWKS_URL = os.environ.get("AUTH_JWKS_URL", "")
    AUTH_ROLES_CLAIM = os.environ.get("AUTH_ROLES_CLAIM", "")
    AUTH_ORG_CLAIM = os.environ.get("AUTH_ORG_CLAIM", "")
    SUPER_ADMIN_EMAILS = os.environ.get("SUPER_ADMIN_EMAILS", "")

    # Postgres
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

    # If true, enforce verified-domain checks + tight widget CORS.
    REQUIRE_DOMAIN_VERIFICATION = os.environ.get("REQUIRE_DOMAIN_VERIFICATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    # Optional override for the Python executable used to spawn the indexing worker.
    WORKER_PYTHON = (os.environ.get("WORKER_PYTHON") or _DEFAULT_WORKER_PYTHON).strip()

    # Celery configuration
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0").strip()
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "").strip() or CELERY_BROKER_URL
    CELERY_WORKER_CONCURRENCY = int(os.environ.get("CELERY_WORKER_CONCURRENCY", "10"))

    # Redis (pubsub for conversations). Defaults to Celery broker if not set.
    REDIS_URL = os.environ.get("REDIS_URL", "").strip()

    # Conversation stream (Redis Streams)
    CONVERSATION_STREAM_KEY = os.environ.get("CONVERSATION_STREAM_KEY", "webai:conversation_events").strip()
    CONVERSATION_STREAM_MAXLEN = os.environ.get("CONVERSATION_STREAM_MAXLEN", "10000").strip()


config = Config()
