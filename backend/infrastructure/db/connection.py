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
    "ALTER TABLE bots ADD COLUMN IF NOT EXISTS escalation_config TEXT",
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
    CREATE TABLE IF NOT EXISTS booking_link_jobs (
      job_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      index_job_id TEXT,
      root_url TEXT NOT NULL,
      status TEXT NOT NULL,
      links TEXT NOT NULL DEFAULT '[]',
      error TEXT,
      celery_task_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS booking_link_jobs_bot_id ON booking_link_jobs (bot_id)",
    "CREATE INDEX IF NOT EXISTS booking_link_jobs_updated_at ON booking_link_jobs (updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS topic_jobs (
      job_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      status TEXT NOT NULL,
      stage TEXT NOT NULL,
      gcs_prefix TEXT,
      docs_count INTEGER NOT NULL DEFAULT 0,
      topics_count INTEGER NOT NULL DEFAULT 0,
      last_error TEXT,
      celery_task_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS topic_jobs_bot_id ON topic_jobs (bot_id)",
    "CREATE INDEX IF NOT EXISTS topic_jobs_updated_at ON topic_jobs (updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS availability_jobs (
      job_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      url TEXT NOT NULL,
      status TEXT NOT NULL,
      question TEXT,
      summary TEXT,
      raw_text_path TEXT,
      raw_html_path TEXT,
      last_error TEXT,
      max_seconds INTEGER NOT NULL DEFAULT 60,
      steps_count INTEGER NOT NULL DEFAULT 0,
      screenshots_dir TEXT,
      celery_task_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE availability_jobs ADD COLUMN IF NOT EXISTS question TEXT",
    "ALTER TABLE availability_jobs ADD COLUMN IF NOT EXISTS raw_text_path TEXT",
    "ALTER TABLE availability_jobs ADD COLUMN IF NOT EXISTS raw_html_path TEXT",
    "CREATE INDEX IF NOT EXISTS availability_jobs_bot_id ON availability_jobs (bot_id)",
    "CREATE INDEX IF NOT EXISTS availability_jobs_updated_at ON availability_jobs (updated_at DESC)",
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
    """
    CREATE TABLE IF NOT EXISTS conversation_escalations (
      escalation_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      visitor_email TEXT NOT NULL,
      details TEXT,
      status TEXT NOT NULL DEFAULT 'open',
      created_at TEXT NOT NULL
    )
    """,
    "ALTER TABLE conversation_escalations ADD COLUMN IF NOT EXISTS details TEXT",
    "ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS sender_name TEXT",
    "CREATE INDEX IF NOT EXISTS conversation_sessions_bot_id ON conversation_sessions (bot_id)",
    "CREATE INDEX IF NOT EXISTS conversation_sessions_last_active ON conversation_sessions (bot_id, last_active_at DESC)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_session_id ON conversation_messages (session_id)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_bot_id ON conversation_messages (bot_id)",
    "CREATE INDEX IF NOT EXISTS conversation_messages_created_at ON conversation_messages (created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS conversation_feedback (
      feedback_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      message_id TEXT,
      rating INTEGER NOT NULL,
      comment TEXT,
      created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS conversation_feedback_bot_day ON conversation_feedback (org_id, bot_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS bot_usage_daily (
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      day TEXT NOT NULL,
      conversations INTEGER NOT NULL DEFAULT 0,
      messages_user INTEGER NOT NULL DEFAULT 0,
      messages_bot INTEGER NOT NULL DEFAULT 0,
      escalations INTEGER NOT NULL DEFAULT 0,
      unique_visitors_est INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (org_id, bot_id, day)
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_usage_daily_bot_day ON bot_usage_daily (org_id, bot_id, day)",
    """
    CREATE TABLE IF NOT EXISTS bot_sources_daily (
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      day TEXT NOT NULL,
      source_url TEXT NOT NULL,
      count INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (org_id, bot_id, day, source_url)
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_sources_daily_bot_day ON bot_sources_daily (org_id, bot_id, day)",
    """
    CREATE TABLE IF NOT EXISTS bot_topics_daily (
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      day TEXT NOT NULL,
      topic TEXT NOT NULL,
      count INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (org_id, bot_id, day, topic)
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_topics_daily_bot_day ON bot_topics_daily (org_id, bot_id, day)",
    """
    CREATE TABLE IF NOT EXISTS bot_extracted_topics (
      topic_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      topic TEXT NOT NULL,
      category TEXT,
      confidence REAL DEFAULT 1.0,
      source_urls TEXT DEFAULT '[]',
      occurrence_count INTEGER DEFAULT 1,
      is_active BOOLEAN DEFAULT TRUE,
      extracted_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_extracted_topics_bot ON bot_extracted_topics (org_id, bot_id)",
    """
    CREATE TABLE IF NOT EXISTS rollup_watermarks (
      bot_id TEXT PRIMARY KEY,
      last_processed_at TEXT NOT NULL
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS organizations_name_unique ON organizations (lower(name))",
    # ── LINE integration ──
    """
    CREATE TABLE IF NOT EXISTS line_channels (
      channel_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL UNIQUE,
      org_id TEXT NOT NULL,
      line_channel_id TEXT NOT NULL,
      line_channel_secret TEXT NOT NULL,
      line_channel_access_token TEXT NOT NULL,
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS line_channels_bot_id ON line_channels (bot_id)",
    """
    CREATE TABLE IF NOT EXISTS line_user_sessions (
      line_user_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      is_escalated BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (line_user_id, bot_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS line_user_sessions_session_id ON line_user_sessions (session_id)",
    # ── Instagram integration ──
    """
    CREATE TABLE IF NOT EXISTS instagram_channels (
      channel_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL UNIQUE,
      org_id TEXT NOT NULL,
      ig_page_id TEXT NOT NULL,
      app_secret TEXT NOT NULL,
      page_access_token TEXT NOT NULL,
      verify_token TEXT NOT NULL,
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS instagram_channels_bot_id ON instagram_channels (bot_id)",
    "CREATE INDEX IF NOT EXISTS instagram_channels_ig_page_id ON instagram_channels (ig_page_id)",
    # ── Instagram OAuth migration ──
    "ALTER TABLE instagram_channels ADD COLUMN IF NOT EXISTS ig_user_id TEXT",
    "ALTER TABLE instagram_channels ADD COLUMN IF NOT EXISTS ig_username TEXT",
    "ALTER TABLE instagram_channels ADD COLUMN IF NOT EXISTS token_expires_at TEXT",
    "ALTER TABLE instagram_channels ADD COLUMN IF NOT EXISTS connection_method TEXT DEFAULT 'manual'",
    "CREATE INDEX IF NOT EXISTS instagram_channels_ig_user_id ON instagram_channels (ig_user_id)",
    """
    CREATE TABLE IF NOT EXISTS instagram_user_sessions (
      ig_user_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      is_escalated BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (ig_user_id, bot_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS instagram_user_sessions_session_id ON instagram_user_sessions (session_id)",
    # ── Topic system v2 ──
    "ALTER TABLE bot_extracted_topics ADD COLUMN IF NOT EXISTS source_url TEXT",
    "ALTER TABLE bot_extracted_topics ADD COLUMN IF NOT EXISTS origin TEXT DEFAULT 'extracted'",
    "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS page_title TEXT",
    """
    CREATE TABLE IF NOT EXISTS topic_question_mappings (
      id TEXT PRIMARY KEY,
      topic_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      message_id TEXT,
      question_text TEXT,
      matched_at TEXT NOT NULL,
      match_method TEXT DEFAULT 'keyword'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tqm_topic ON topic_question_mappings (topic_id)",
    "CREATE INDEX IF NOT EXISTS idx_tqm_session ON topic_question_mappings (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_tqm_bot ON topic_question_mappings (org_id, bot_id)",
    # ── Business assets ──
    """
    CREATE TABLE IF NOT EXISTS bot_assets (
      asset_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      name TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      image_gcs_uri TEXT NOT NULL,
      image_public_url TEXT NOT NULL DEFAULT '',
      link_url TEXT,
      keywords TEXT NOT NULL DEFAULT '[]',
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_assets_bot_id ON bot_assets (bot_id)",
    "CREATE INDEX IF NOT EXISTS bot_assets_org_bot ON bot_assets (org_id, bot_id)",
    """
    CREATE TABLE IF NOT EXISTS asset_extraction_jobs (
      job_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      status TEXT NOT NULL,
      gcs_prefix TEXT,
      page_urls TEXT DEFAULT '[]',
      assets_discovered INTEGER DEFAULT 0,
      assets_downloaded INTEGER DEFAULT 0,
      assets_created INTEGER DEFAULT 0,
      error TEXT,
      celery_task_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS asset_extraction_jobs_bot_updated ON asset_extraction_jobs (bot_id, updated_at DESC)",
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
