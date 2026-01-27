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
