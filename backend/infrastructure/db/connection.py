import time
from typing import Iterable, Optional

from psycopg import connect
from psycopg.errors import OperationalError
from psycopg.rows import tuple_row

from common.config import config

_SCHEMA_SQL: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS organizations (
      org_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      status TEXT NOT NULL,
      plan TEXT,
      stripe_customer_id TEXT,
      stripe_subscription_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
      user_id TEXT PRIMARY KEY,
      idp_subject TEXT UNIQUE,
      email TEXT UNIQUE,
      first_name TEXT,
      last_name TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT",
    """
    CREATE TABLE IF NOT EXISTS org_memberships (
      user_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      role TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (user_id, org_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bots (
      bot_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      display_name TEXT NOT NULL,
      publishable_key TEXT NOT NULL UNIQUE,
      secret_key TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE bots ADD COLUMN IF NOT EXISTS widget_config TEXT",
    "ALTER TABLE bots ADD COLUMN IF NOT EXISTS agent_config TEXT",
    """
    CREATE TABLE IF NOT EXISTS bot_domains (
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      hostname TEXT NOT NULL,
      status TEXT NOT NULL,
      verification_token TEXT NOT NULL,
      verified_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (bot_id, hostname)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bot_corpora (
      bot_id TEXT PRIMARY KEY,
      corpus_resource TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS domain_corpora (
      hostname TEXT PRIMARY KEY,
      corpus_resource TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bot_sources (
      source_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      type TEXT NOT NULL,
      config TEXT NOT NULL DEFAULT '{}',
      display_name TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_sources_bot_id ON bot_sources (bot_id)",
    """
    CREATE TABLE IF NOT EXISTS index_jobs (
      job_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      source_id TEXT,
      url TEXT NOT NULL,
      hostname TEXT NOT NULL,
      celery_task_id TEXT,
      stage TEXT NOT NULL,
      pages_crawled INTEGER NOT NULL DEFAULT 0,
      docs_count INTEGER NOT NULL DEFAULT 0,
      last_crawled_url TEXT NOT NULL DEFAULT '',
      last_depth INTEGER NOT NULL DEFAULT -1,
      gcs_prefix TEXT NOT NULL DEFAULT '',
      last_error TEXT NOT NULL DEFAULT '',
      crawled_urls TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS crawled_urls TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS source_id TEXT",
    "CREATE INDEX IF NOT EXISTS index_jobs_bot_id ON index_jobs (bot_id)",
    "CREATE INDEX IF NOT EXISTS index_jobs_bot_hostname ON index_jobs (bot_id, hostname)",
    "CREATE INDEX IF NOT EXISTS index_jobs_stage ON index_jobs (stage)",
    "CREATE INDEX IF NOT EXISTS index_jobs_updated_at ON index_jobs (updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS index_jobs_source_id ON index_jobs (source_id)",
    """
    CREATE TABLE IF NOT EXISTS discovery_jobs (
      job_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      root_url TEXT NOT NULL,
      method TEXT NOT NULL,
      status TEXT NOT NULL,
      discovered_urls TEXT NOT NULL DEFAULT '[]',
      error TEXT,
      celery_task_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS discovery_jobs_bot_id ON discovery_jobs (bot_id)",
    """
    CREATE TABLE IF NOT EXISTS conversation_sessions (
      session_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      channel TEXT NOT NULL,
      status TEXT NOT NULL,
      title TEXT,
      site_url TEXT,
      site_title TEXT,
      message_count INTEGER NOT NULL DEFAULT 0,
      started_at TEXT NOT NULL,
      last_active_at TEXT NOT NULL,
      ended_at TEXT,
      user_agent TEXT,
      ip TEXT
    )
    """,
    "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS title TEXT",
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
      message_id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      role TEXT NOT NULL,
      sender_name TEXT,
      content TEXT NOT NULL,
      citations TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS sender_name TEXT",
    "CREATE INDEX IF NOT EXISTS conversation_sessions_bot_id ON conversation_sessions (bot_id)",
    "CREATE INDEX IF NOT EXISTS conversation_sessions_last_active ON conversation_sessions (bot_id, last_active_at DESC)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_session_id ON conversation_messages (session_id)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_bot_id ON conversation_messages (bot_id)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_created_at ON conversation_messages (created_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS organizations_name_unique ON organizations (lower(name))",
)

_SCHEMA_INITIALIZED = False


def _database_url() -> str:
    url = (config.DATABASE_URL or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


def _connect_with_retry() -> "Connection":
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            return connect(_database_url(), row_factory=tuple_row)
        except OperationalError as exc:
            last_exc = exc
            time.sleep(0.4 * (2**attempt))
    raise last_exc or RuntimeError("Unable to connect to Postgres")


def _ensure_schema(con: "Connection") -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with con.cursor() as cur:
        for stmt in _SCHEMA_SQL:
            cur.execute(stmt)
    con.commit()
    _SCHEMA_INITIALIZED = True


def get_connection() -> "Connection":
    con = _connect_with_retry()
    _ensure_schema(con)
    return con
