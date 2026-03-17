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
    AUTH_JWKS_JSON = os.environ.get("AUTH_JWKS_JSON", "")
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

    # Redis-first chat runtime (conversation/session/message hot path)
    CHAT_RUNTIME_REDIS_ENABLED = os.environ.get("CHAT_RUNTIME_REDIS_ENABLED", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_RUNTIME_STRICT_NO_DB_PRE_RESPONSE = os.environ.get(
        "CHAT_RUNTIME_STRICT_NO_DB_PRE_RESPONSE",
        "false",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_RUNTIME_REDIS_OUTAGE_DB_FALLBACK = os.environ.get(
        "CHAT_RUNTIME_REDIS_OUTAGE_DB_FALLBACK",
        "true",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_RUNTIME_ASYNC_PERSIST_ENABLED = os.environ.get(
        "CHAT_RUNTIME_ASYNC_PERSIST_ENABLED",
        "false",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_RUNTIME_REDIS_NAMESPACE = os.environ.get(
        "CHAT_RUNTIME_REDIS_NAMESPACE",
        "webai:chatruntime:v1",
    ).strip()

    _CHAT_RUNTIME_HISTORY_MAX_MESSAGES_RAW = (os.environ.get("CHAT_RUNTIME_HISTORY_MAX_MESSAGES") or "100").strip()
    try:
        _CHAT_RUNTIME_HISTORY_MAX_MESSAGES = int(_CHAT_RUNTIME_HISTORY_MAX_MESSAGES_RAW)
    except ValueError:
        _CHAT_RUNTIME_HISTORY_MAX_MESSAGES = 100
    CHAT_RUNTIME_HISTORY_MAX_MESSAGES = max(20, min(_CHAT_RUNTIME_HISTORY_MAX_MESSAGES, 300))

    _CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS_RAW = (os.environ.get("CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS") or "30").strip()
    try:
        _CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS = int(_CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS_RAW)
    except ValueError:
        _CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS = 30
    CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS = max(5, min(_CHAT_RUNTIME_TURN_LOCK_TTL_SECONDS, 300))

    _CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS_RAW = (
        os.environ.get("CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS") or "8000"
    ).strip()
    try:
        _CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS = int(_CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS_RAW)
    except ValueError:
        _CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS = 8000
    CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS = max(100, min(_CHAT_RUNTIME_TURN_LOCK_WAIT_TIMEOUT_MS, 60000))

    CHAT_RUNTIME_PERSIST_STREAM_KEY = os.environ.get(
        "CHAT_RUNTIME_PERSIST_STREAM_KEY",
        "webai:chatruntime:persist:v1",
    ).strip()
    _CHAT_RUNTIME_PERSIST_STREAM_MAXLEN_RAW = (os.environ.get("CHAT_RUNTIME_PERSIST_STREAM_MAXLEN") or "200000").strip()
    try:
        _CHAT_RUNTIME_PERSIST_STREAM_MAXLEN = int(_CHAT_RUNTIME_PERSIST_STREAM_MAXLEN_RAW)
    except ValueError:
        _CHAT_RUNTIME_PERSIST_STREAM_MAXLEN = 200000
    CHAT_RUNTIME_PERSIST_STREAM_MAXLEN = max(1000, min(_CHAT_RUNTIME_PERSIST_STREAM_MAXLEN, 2_000_000))

    CHAT_RUNTIME_PERSIST_CONSUMER_GROUP = os.environ.get(
        "CHAT_RUNTIME_PERSIST_CONSUMER_GROUP",
        "webai:chatruntime:persist:group:v1",
    ).strip()
    CHAT_RUNTIME_PERSIST_CONSUMER_NAME = os.environ.get("CHAT_RUNTIME_PERSIST_CONSUMER_NAME", "").strip()
    _CHAT_RUNTIME_PERSIST_MAX_RETRIES_RAW = (os.environ.get("CHAT_RUNTIME_PERSIST_MAX_RETRIES") or "8").strip()
    try:
        _CHAT_RUNTIME_PERSIST_MAX_RETRIES = int(_CHAT_RUNTIME_PERSIST_MAX_RETRIES_RAW)
    except ValueError:
        _CHAT_RUNTIME_PERSIST_MAX_RETRIES = 8
    CHAT_RUNTIME_PERSIST_MAX_RETRIES = max(1, min(_CHAT_RUNTIME_PERSIST_MAX_RETRIES, 100))
    CHAT_RUNTIME_PERSIST_DLQ_STREAM_KEY = os.environ.get(
        "CHAT_RUNTIME_PERSIST_DLQ_STREAM_KEY",
        "webai:chatruntime:persist:dlq:v1",
    ).strip()
    CHAT_STREAM_STRUCTURED_RESPONSE_ENABLED = os.environ.get(
        "CHAT_STREAM_STRUCTURED_RESPONSE_ENABLED",
        "false",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_STORAGE_TRACE_LOGS = os.environ.get(
        "CHAT_STORAGE_TRACE_LOGS",
        "false",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    # Chat hot-path cache (L1 in-process + L2 Redis cache-aside)
    CHAT_CACHE_ENABLED = os.environ.get("CHAT_CACHE_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_CACHE_L2_ENABLED = os.environ.get("CHAT_CACHE_L2_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    CHAT_CACHE_NAMESPACE = os.environ.get("CHAT_CACHE_NAMESPACE", "webai:chatcache:v1").strip()
    CHAT_CACHE_INVALIDATION_CHANNEL = os.environ.get(
        "CHAT_CACHE_INVALIDATION_CHANNEL",
        "webai:chatcache:v1:invalidate",
    ).strip()

    _CHAT_CACHE_L1_MAX_ITEMS_RAW = (os.environ.get("CHAT_CACHE_L1_MAX_ITEMS") or "5000").strip()
    try:
        _CHAT_CACHE_L1_MAX_ITEMS = int(_CHAT_CACHE_L1_MAX_ITEMS_RAW)
    except ValueError:
        _CHAT_CACHE_L1_MAX_ITEMS = 5000
    CHAT_CACHE_L1_MAX_ITEMS = max(200, min(_CHAT_CACHE_L1_MAX_ITEMS, 20000))

    _CHAT_CACHE_TTL_BOT_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_BOT_SEC") or "300").strip()
    try:
        _CHAT_CACHE_TTL_BOT_SEC = int(_CHAT_CACHE_TTL_BOT_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_BOT_SEC = 300
    CHAT_CACHE_TTL_BOT_SEC = max(30, min(_CHAT_CACHE_TTL_BOT_SEC, 3600))

    _CHAT_CACHE_TTL_CHANNEL_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_CHANNEL_SEC") or "300").strip()
    try:
        _CHAT_CACHE_TTL_CHANNEL_SEC = int(_CHAT_CACHE_TTL_CHANNEL_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_CHANNEL_SEC = 300
    CHAT_CACHE_TTL_CHANNEL_SEC = max(30, min(_CHAT_CACHE_TTL_CHANNEL_SEC, 3600))

    _CHAT_CACHE_TTL_DESIGN_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_DESIGN_SEC") or "300").strip()
    try:
        _CHAT_CACHE_TTL_DESIGN_SEC = int(_CHAT_CACHE_TTL_DESIGN_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_DESIGN_SEC = 300
    CHAT_CACHE_TTL_DESIGN_SEC = max(30, min(_CHAT_CACHE_TTL_DESIGN_SEC, 3600))

    _CHAT_CACHE_TTL_CORPUS_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_CORPUS_SEC") or "900").strip()
    try:
        _CHAT_CACHE_TTL_CORPUS_SEC = int(_CHAT_CACHE_TTL_CORPUS_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_CORPUS_SEC = 900
    CHAT_CACHE_TTL_CORPUS_SEC = max(60, min(_CHAT_CACHE_TTL_CORPUS_SEC, 7200))

    _CHAT_CACHE_TTL_SESSION_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_SESSION_SEC") or "180").strip()
    try:
        _CHAT_CACHE_TTL_SESSION_SEC = int(_CHAT_CACHE_TTL_SESSION_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_SESSION_SEC = 180
    CHAT_CACHE_TTL_SESSION_SEC = max(30, min(_CHAT_CACHE_TTL_SESSION_SEC, 1800))

    _CHAT_CACHE_TTL_HISTORY_SEC_RAW = (os.environ.get("CHAT_CACHE_TTL_HISTORY_SEC") or "120").strip()
    try:
        _CHAT_CACHE_TTL_HISTORY_SEC = int(_CHAT_CACHE_TTL_HISTORY_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_TTL_HISTORY_SEC = 120
    CHAT_CACHE_TTL_HISTORY_SEC = max(30, min(_CHAT_CACHE_TTL_HISTORY_SEC, 900))

    _CHAT_CACHE_HISTORY_MAX_MESSAGES_RAW = (os.environ.get("CHAT_CACHE_HISTORY_MAX_MESSAGES") or "100").strip()
    try:
        _CHAT_CACHE_HISTORY_MAX_MESSAGES = int(_CHAT_CACHE_HISTORY_MAX_MESSAGES_RAW)
    except ValueError:
        _CHAT_CACHE_HISTORY_MAX_MESSAGES = 100
    CHAT_CACHE_HISTORY_MAX_MESSAGES = max(20, min(_CHAT_CACHE_HISTORY_MAX_MESSAGES, 200))

    _CHAT_CACHE_NEGATIVE_TTL_SEC_RAW = (os.environ.get("CHAT_CACHE_NEGATIVE_TTL_SEC") or "20").strip()
    try:
        _CHAT_CACHE_NEGATIVE_TTL_SEC = int(_CHAT_CACHE_NEGATIVE_TTL_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_NEGATIVE_TTL_SEC = 20
    CHAT_CACHE_NEGATIVE_TTL_SEC = max(5, min(_CHAT_CACHE_NEGATIVE_TTL_SEC, 120))

    _CHAT_CACHE_IDEMPOTENCY_TTL_SEC_RAW = (os.environ.get("CHAT_CACHE_IDEMPOTENCY_TTL_SEC") or "86400").strip()
    try:
        _CHAT_CACHE_IDEMPOTENCY_TTL_SEC = int(_CHAT_CACHE_IDEMPOTENCY_TTL_SEC_RAW)
    except ValueError:
        _CHAT_CACHE_IDEMPOTENCY_TTL_SEC = 86400
    CHAT_CACHE_IDEMPOTENCY_TTL_SEC = max(60, min(_CHAT_CACHE_IDEMPOTENCY_TTL_SEC, 604800))

    # Escalation email notifications (SMTP)
    SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ.get("SMTP_USER", "").strip()
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
    SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "").strip()
    SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").strip().lower() in ("1", "true", "yes", "y")

    # Conversation stream (Redis Streams)
    CONVERSATION_STREAM_KEY = os.environ.get("CONVERSATION_STREAM_KEY", "webai:conversation_events").strip()
    CONVERSATION_STREAM_MAXLEN = os.environ.get("CONVERSATION_STREAM_MAXLEN", "10000").strip()
    _CONV_TTL_RAW = (os.environ.get("CONVERSATION_SESSION_TTL_MINUTES") or "30").strip()
    try:
        _CONV_TTL = int(_CONV_TTL_RAW)
    except ValueError:
        _CONV_TTL = 30
    CONVERSATION_SESSION_TTL_MINUTES = max(15, min(_CONV_TTL, 10080))


config = Config()
