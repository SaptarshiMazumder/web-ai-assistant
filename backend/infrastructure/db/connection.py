import os
import queue
import threading
import time
from typing import Any, Iterable, Optional

from psycopg import connect
from psycopg.errors import OperationalError
from psycopg.pq import TransactionStatus
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
    CREATE TABLE IF NOT EXISTS bot_suggested_message_packs (
      pack_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      org_id TEXT NOT NULL,
      lang TEXT NOT NULL,
      suggested_message_id TEXT NOT NULL,
      label TEXT NOT NULL,
      prompt TEXT NOT NULL,
      pack_mode TEXT NOT NULL,
      status TEXT NOT NULL,
      version_hash TEXT NOT NULL,
      source_urls TEXT NOT NULL DEFAULT '[]',
      evidence_snippets TEXT NOT NULL DEFAULT '[]',
      link_targets TEXT NOT NULL DEFAULT '[]',
      instruction TEXT NOT NULL DEFAULT '',
      citations TEXT NOT NULL DEFAULT '[]',
      error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_suggested_message_packs_bot_lang ON bot_suggested_message_packs (bot_id, lang)",
    "CREATE UNIQUE INDEX IF NOT EXISTS bot_suggested_message_packs_version ON bot_suggested_message_packs (bot_id, lang, suggested_message_id, version_hash)",
    "CREATE INDEX IF NOT EXISTS bot_suggested_message_packs_updated_at ON bot_suggested_message_packs (bot_id, updated_at DESC)",
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
    """
    CREATE TABLE IF NOT EXISTS conversation_escalation_reads (
      escalation_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      read_at TEXT NOT NULL,
      PRIMARY KEY (escalation_id, user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_channel_contacts (
      contact_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      channel TEXT NOT NULL,
      external_user_id TEXT NOT NULL,
      display_name TEXT,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      current_session_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_session_handoffs (
      handoff_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL,
      session_id TEXT NOT NULL UNIQUE,
      contact_id TEXT,
      source_channel TEXT,
      assistant_state TEXT NOT NULL DEFAULT 'bot',
      support_request_id TEXT,
      started_by TEXT,
      ended_reason TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      ended_at TEXT
    )
    """,
    "ALTER TABLE conversation_escalations ADD COLUMN IF NOT EXISTS details TEXT",
    "CREATE INDEX IF NOT EXISTS conversation_escalation_reads_user_id ON conversation_escalation_reads (user_id, read_at DESC)",
    "CREATE INDEX IF NOT EXISTS conversation_escalation_reads_escalation_id ON conversation_escalation_reads (escalation_id)",
    "ALTER TABLE conversation_channel_contacts ADD COLUMN IF NOT EXISTS display_name TEXT",
    "ALTER TABLE conversation_channel_contacts ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE conversation_channel_contacts ADD COLUMN IF NOT EXISTS current_session_id TEXT",
    "CREATE UNIQUE INDEX IF NOT EXISTS conversation_channel_contacts_identity ON conversation_channel_contacts (bot_id, channel, external_user_id)",
    "CREATE INDEX IF NOT EXISTS conversation_channel_contacts_session_id ON conversation_channel_contacts (current_session_id)",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS contact_id TEXT",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS source_channel TEXT",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS assistant_state TEXT NOT NULL DEFAULT 'bot'",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS support_request_id TEXT",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS started_by TEXT",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS ended_reason TEXT",
    "ALTER TABLE conversation_session_handoffs ADD COLUMN IF NOT EXISTS ended_at TEXT",
    "CREATE INDEX IF NOT EXISTS conversation_session_handoffs_bot_state ON conversation_session_handoffs (bot_id, assistant_state, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS conversation_session_handoffs_contact_id ON conversation_session_handoffs (contact_id)",
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
    CREATE TABLE IF NOT EXISTS line_design_configs (
      config_id TEXT PRIMARY KEY,
      bot_id TEXT NOT NULL UNIQUE,
      org_id TEXT NOT NULL,
      config_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS line_design_configs_bot_id ON line_design_configs (bot_id)",
    """
    CREATE TABLE IF NOT EXISTS line_rich_menu_states (
      state_id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL UNIQUE,
      bot_id TEXT NOT NULL UNIQUE,
      config_hash TEXT,
      default_variant TEXT,
      rich_menu_variants_json TEXT NOT NULL DEFAULT '{}',
      sync_status TEXT NOT NULL DEFAULT 'pending',
      last_synced_at TEXT,
      last_error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS line_rich_menu_states_bot_id ON line_rich_menu_states (bot_id)",
    "CREATE INDEX IF NOT EXISTS line_rich_menu_states_channel_id ON line_rich_menu_states (channel_id)",
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
      metadata_json TEXT NOT NULL DEFAULT '{}',
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS bot_assets_bot_id ON bot_assets (bot_id)",
    "CREATE INDEX IF NOT EXISTS bot_assets_org_bot ON bot_assets (org_id, bot_id)",
    "ALTER TABLE bot_assets ADD COLUMN IF NOT EXISTS asset_type TEXT NOT NULL DEFAULT 'image'",
    "ALTER TABLE bot_assets ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}'",
    "CREATE INDEX IF NOT EXISTS bot_assets_type ON bot_assets (bot_id, asset_type)",
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
    # ── Config-first job pipeline ──
    """
    CREATE TABLE IF NOT EXISTS job_pipeline_runs (
      run_id TEXT PRIMARY KEY,
      org_id TEXT NOT NULL,
      bot_id TEXT NOT NULL,
      workflow_id TEXT NOT NULL,
      trigger TEXT NOT NULL,
      status TEXT NOT NULL,
      current_step_index INTEGER NOT NULL DEFAULT 0,
      progress_pct INTEGER NOT NULL DEFAULT 0,
      current_step_id TEXT,
      current_stage_key TEXT,
      current_message TEXT,
      context_json TEXT NOT NULL DEFAULT '{}',
      last_error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS job_pipeline_runs_bot_updated ON job_pipeline_runs (bot_id, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS job_pipeline_steps (
      run_id TEXT NOT NULL,
      step_index INTEGER NOT NULL,
      job_id TEXT NOT NULL,
      runner_ref TEXT NOT NULL,
      on_failure TEXT NOT NULL DEFAULT 'continue',
      status TEXT NOT NULL,
      progress_pct INTEGER NOT NULL DEFAULT 0,
      current_stage_key TEXT,
      current_message TEXT,
      attempt INTEGER NOT NULL DEFAULT 0,
      celery_task_id TEXT,
      linked_job_type TEXT,
      linked_job_id TEXT,
      output_json TEXT NOT NULL DEFAULT '{}',
      last_error TEXT,
      started_at TEXT,
      completed_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (run_id, step_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS job_pipeline_steps_run ON job_pipeline_steps (run_id, step_index)",
    "CREATE INDEX IF NOT EXISTS job_pipeline_steps_status ON job_pipeline_steps (status, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS job_pipeline_step_events (
      event_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      step_index INTEGER,
      event_type TEXT NOT NULL,
      stage_key TEXT,
      message TEXT,
      progress_pct INTEGER,
      details_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS job_pipeline_step_events_run ON job_pipeline_step_events (run_id, created_at DESC)",
    "ALTER TABLE job_pipeline_runs ADD COLUMN IF NOT EXISTS progress_pct INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE job_pipeline_runs ADD COLUMN IF NOT EXISTS current_step_id TEXT",
    "ALTER TABLE job_pipeline_runs ADD COLUMN IF NOT EXISTS current_stage_key TEXT",
    "ALTER TABLE job_pipeline_runs ADD COLUMN IF NOT EXISTS current_message TEXT",
    "ALTER TABLE job_pipeline_steps ADD COLUMN IF NOT EXISTS progress_pct INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE job_pipeline_steps ADD COLUMN IF NOT EXISTS current_stage_key TEXT",
    "ALTER TABLE job_pipeline_steps ADD COLUMN IF NOT EXISTS current_message TEXT",
    "ALTER TABLE job_pipeline_steps ADD COLUMN IF NOT EXISTS attempt INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE job_pipeline_steps ADD COLUMN IF NOT EXISTS celery_task_id TEXT",
    # ── Escalation optional-message flow ──
    "ALTER TABLE instagram_user_sessions ADD COLUMN IF NOT EXISTS awaiting_escalation_msg BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE line_user_sessions ADD COLUMN IF NOT EXISTS awaiting_escalation_msg BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE instagram_user_sessions ADD COLUMN IF NOT EXISTS awaiting_staff_takeover BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE line_user_sessions ADD COLUMN IF NOT EXISTS awaiting_staff_takeover BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE line_user_sessions ADD COLUMN IF NOT EXISTS display_name TEXT",
    """
    INSERT INTO conversation_channel_contacts(
      contact_id, bot_id, channel, external_user_id, display_name, current_session_id, created_at, updated_at
    )
    SELECT
      'cc_' || md5(
        lus.bot_id || ':line:' || lus.line_user_id || ':' || lus.session_id || ':' ||
        clock_timestamp()::text || ':' || random()::text
      ),
      lus.bot_id,
      'line',
      lus.line_user_id,
      lus.display_name,
      lus.session_id,
      lus.created_at,
      lus.updated_at
    FROM line_user_sessions lus
    ON CONFLICT (bot_id, channel, external_user_id)
    DO UPDATE SET
      display_name = COALESCE(EXCLUDED.display_name, conversation_channel_contacts.display_name),
      current_session_id = COALESCE(EXCLUDED.current_session_id, conversation_channel_contacts.current_session_id),
      updated_at = EXCLUDED.updated_at
    """,
    """
    INSERT INTO conversation_channel_contacts(
      contact_id, bot_id, channel, external_user_id, display_name, current_session_id, created_at, updated_at
    )
    SELECT
      'cc_' || md5(
        ius.bot_id || ':instagram:' || ius.ig_user_id || ':' || ius.session_id || ':' ||
        clock_timestamp()::text || ':' || random()::text
      ),
      ius.bot_id,
      'instagram',
      ius.ig_user_id,
      NULL,
      ius.session_id,
      ius.created_at,
      ius.updated_at
    FROM instagram_user_sessions ius
    ON CONFLICT (bot_id, channel, external_user_id)
    DO UPDATE SET
      current_session_id = COALESCE(EXCLUDED.current_session_id, conversation_channel_contacts.current_session_id),
      updated_at = EXCLUDED.updated_at
    """,
    """
    INSERT INTO conversation_session_handoffs(
      handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state, started_by, created_at, updated_at, ended_at
    )
    SELECT
      'hof_' || md5(lus.bot_id || ':' || lus.session_id),
      lus.bot_id,
      lus.session_id,
      c.contact_id,
      'line',
      CASE
        WHEN COALESCE(lus.awaiting_escalation_msg, FALSE) THEN 'awaiting_support_details'
        WHEN COALESCE(lus.is_escalated, FALSE) THEN 'human_handoff'
        ELSE 'bot'
      END,
      'legacy_bridge',
      lus.created_at,
      lus.updated_at,
      NULL
    FROM line_user_sessions lus
    LEFT JOIN conversation_channel_contacts c
      ON c.bot_id = lus.bot_id AND c.channel = 'line' AND c.external_user_id = lus.line_user_id
    WHERE COALESCE(lus.awaiting_escalation_msg, FALSE) OR COALESCE(lus.is_escalated, FALSE)
    ON CONFLICT (session_id) DO NOTHING
    """,
    """
    INSERT INTO conversation_session_handoffs(
      handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state, started_by, created_at, updated_at, ended_at
    )
    SELECT
      'hof_' || md5(ius.bot_id || ':' || ius.session_id),
      ius.bot_id,
      ius.session_id,
      c.contact_id,
      'instagram',
      CASE
        WHEN COALESCE(ius.awaiting_escalation_msg, FALSE) THEN 'awaiting_support_details'
        WHEN COALESCE(ius.is_escalated, FALSE) OR COALESCE(ius.awaiting_staff_takeover, FALSE) THEN 'human_handoff'
        ELSE 'bot'
      END,
      'legacy_bridge',
      ius.created_at,
      ius.updated_at,
      NULL
    FROM instagram_user_sessions ius
    LEFT JOIN conversation_channel_contacts c
      ON c.bot_id = ius.bot_id AND c.channel = 'instagram' AND c.external_user_id = ius.ig_user_id
    WHERE COALESCE(ius.awaiting_escalation_msg, FALSE)
       OR COALESCE(ius.is_escalated, FALSE)
       OR COALESCE(ius.awaiting_staff_takeover, FALSE)
    ON CONFLICT (session_id) DO NOTHING
    """,
  )

_SCHEMA_INITIALIZED = False
_SCHEMA_LOCK = threading.Lock()


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(minimum, val)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        val = float(raw)
    except ValueError:
        return default
    return max(minimum, val)


_POOL_MIN_SIZE = _env_int("DB_POOL_MIN_SIZE", default=2, minimum=1)
_POOL_MAX_SIZE = _env_int("DB_POOL_MAX_SIZE", default=20, minimum=_POOL_MIN_SIZE)
_POOL_ACQUIRE_TIMEOUT_SEC = _env_float("DB_POOL_ACQUIRE_TIMEOUT_SEC", default=10.0, minimum=0.1)
_POOL_EXECUTE_RETRIES = _env_int("DB_POOL_EXECUTE_RETRIES", default=4, minimum=1)
_POOL_IDLE: "queue.LifoQueue[Any]" = queue.LifoQueue(maxsize=_POOL_MAX_SIZE)
_POOL_LOCK = threading.Lock()
_POOL_TOTAL_CONNECTIONS = 0


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


def _borrow_raw_connection() -> "Connection":
    global _POOL_TOTAL_CONNECTIONS

    try:
        con = _POOL_IDLE.get_nowait()
        if getattr(con, "closed", False):
            with _POOL_LOCK:
                _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
            return _borrow_raw_connection()
        return con
    except queue.Empty:
        pass

    with _POOL_LOCK:
        can_grow = _POOL_TOTAL_CONNECTIONS < _POOL_MAX_SIZE
        if can_grow:
            _POOL_TOTAL_CONNECTIONS += 1

    if can_grow:
        try:
            return _connect_with_retry()
        except Exception:
            with _POOL_LOCK:
                _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
            raise

    try:
        con = _POOL_IDLE.get(timeout=_POOL_ACQUIRE_TIMEOUT_SEC)
    except queue.Empty as exc:
        raise RuntimeError(
            f"Timed out waiting for DB connection from pool after {_POOL_ACQUIRE_TIMEOUT_SEC:.1f}s "
            f"(max={_POOL_MAX_SIZE})"
        ) from exc

    if getattr(con, "closed", False):
        with _POOL_LOCK:
            _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
        return _borrow_raw_connection()
    return con


def _release_raw_connection(con: "Connection") -> None:
    global _POOL_TOTAL_CONNECTIONS
    if con is None:
        return

    if getattr(con, "closed", False):
        with _POOL_LOCK:
            _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
        return

    try:
        tx_status = con.info.transaction_status
        if tx_status != TransactionStatus.IDLE:
            con.rollback()
    except Exception:
        try:
            con.close()
        finally:
            with _POOL_LOCK:
                _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
        return

    try:
        _POOL_IDLE.put_nowait(con)
    except queue.Full:
        try:
            con.close()
        finally:
            with _POOL_LOCK:
                _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)


def _discard_raw_connection(con: "Connection") -> None:
    global _POOL_TOTAL_CONNECTIONS
    if con is None:
        return
    try:
        con.close()
    except Exception:
        pass
    with _POOL_LOCK:
        _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)


def _is_retryable_operational_error(exc: OperationalError) -> bool:
    msg = str(exc or "").lower()
    needles = (
        "unexpected eof",
        "consuming input failed",
        "server closed the connection unexpectedly",
        "connection not open",
        "ssl error",
    )
    return any(n in msg for n in needles)


class _PooledConnection:
    def __init__(self, con: "Connection") -> None:
        self._con = con
        self._released = False

    def __getattr__(self, item: str) -> Any:
        return getattr(self._con, item)

    def _replace_connection_after_failure(self) -> None:
        _discard_raw_connection(self._con)
        self._con = _borrow_raw_connection()

    def execute(self, *args, **kwargs):
        for attempt in range(_POOL_EXECUTE_RETRIES):
            try:
                cursor = self._con.execute(*args, **kwargs)
                return _RetryingResultCursor(self, args, kwargs, cursor)
            except OperationalError as exc:
                if not _is_retryable_operational_error(exc):
                    raise
                if attempt >= _POOL_EXECUTE_RETRIES - 1:
                    raise
                self._replace_connection_after_failure()

    def cursor(self, *args, **kwargs):
        return _PooledCursor(self, self._con.cursor(*args, **kwargs))

    def close(self) -> None:
        if self._released:
            return
        _release_raw_connection(self._con)
        self._released = True

    def __enter__(self) -> "_PooledConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _PooledCursor:
    def __init__(self, pooled_connection: "_PooledConnection", cursor: Any) -> None:
        self._pooled_connection = pooled_connection
        self._cursor = cursor

    def __getattr__(self, item: str) -> Any:
        return getattr(self._cursor, item)

    def _replace_connection_after_failure(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass
        self._pooled_connection._replace_connection_after_failure()
        self._cursor = self._pooled_connection._con.cursor()

    def execute(self, *args, **kwargs):
        for attempt in range(_POOL_EXECUTE_RETRIES):
            try:
                return self._cursor.execute(*args, **kwargs)
            except OperationalError as exc:
                if not _is_retryable_operational_error(exc):
                    raise
                if attempt >= _POOL_EXECUTE_RETRIES - 1:
                    raise
                self._replace_connection_after_failure()

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass

    def __enter__(self) -> "_PooledCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _RetryingResultCursor:
    """Wrap psycopg cursor so fetch failures can replay query once on a fresh connection."""

    def __init__(self, pooled_connection: "_PooledConnection", execute_args: tuple, execute_kwargs: dict, cursor: Any) -> None:
        self._pooled_connection = pooled_connection
        self._execute_args = execute_args
        self._execute_kwargs = execute_kwargs
        self._cursor = cursor
        self._replayed = False

    def __getattr__(self, item: str) -> Any:
        return getattr(self._cursor, item)

    def _replay_once(self) -> bool:
        if self._replayed:
            return False
        self._replayed = True
        try:
            self._cursor.close()
        except Exception:
            pass
        self._pooled_connection._replace_connection_after_failure()
        self._cursor = self._pooled_connection._con.execute(*self._execute_args, **self._execute_kwargs)
        return True

    def _call_with_retry(self, method_name: str, *args, **kwargs):
        try:
            return getattr(self._cursor, method_name)(*args, **kwargs)
        except OperationalError as exc:
            if not _is_retryable_operational_error(exc):
                raise
            if not self._replay_once():
                raise
            return getattr(self._cursor, method_name)(*args, **kwargs)

    def fetchone(self):
        return self._call_with_retry("fetchone")

    def fetchmany(self, size: Optional[int] = None):
        if size is None:
            return self._call_with_retry("fetchmany")
        return self._call_with_retry("fetchmany", size)

    def fetchall(self):
        return self._call_with_retry("fetchall")

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass

    def __iter__(self):
        return iter(self._cursor)

    def __enter__(self) -> "_RetryingResultCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _ensure_schema(con: "Connection") -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with con.cursor() as cur:
        for idx, stmt in enumerate(_SCHEMA_SQL, start=1):
            try:
                cur.execute(stmt)
            except Exception as exc:
                con.rollback()
                snippet = " ".join(stmt.split())
                if len(snippet) > 220:
                    snippet = f"{snippet[:217]}..."
                raise RuntimeError(
                    f"Schema bootstrap failed at statement #{idx}: {snippet}"
                ) from exc
    con.commit()
    _SCHEMA_INITIALIZED = True


def initialize_connection_pool() -> None:
    global _POOL_TOTAL_CONNECTIONS
    while True:
        with _POOL_LOCK:
            if _POOL_TOTAL_CONNECTIONS >= _POOL_MIN_SIZE:
                return
            _POOL_TOTAL_CONNECTIONS += 1
        try:
            con = _connect_with_retry()
            _POOL_IDLE.put_nowait(con)
        except Exception:
            with _POOL_LOCK:
                _POOL_TOTAL_CONNECTIONS = max(0, _POOL_TOTAL_CONNECTIONS - 1)
            raise


def close_connection_pool() -> None:
    global _POOL_TOTAL_CONNECTIONS
    while True:
        try:
            con = _POOL_IDLE.get_nowait()
        except queue.Empty:
            break
        try:
            con.close()
        except Exception:
            pass
    with _POOL_LOCK:
        _POOL_TOTAL_CONNECTIONS = 0


def _alembic_manages_schema(con) -> bool:
    """Check if alembic_version table exists and has a revision stamped."""
    try:
        with con.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'alembic_version')"
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return False
            cur.execute("SELECT COUNT(*) FROM alembic_version")
            count_row = cur.fetchone()
            return bool(count_row and count_row[0] > 0)
    except Exception:
        return False


def ensure_schema_once() -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALIZED:
            return
        con = _connect_with_retry()
        try:
            if _alembic_manages_schema(con):
                _SCHEMA_INITIALIZED = True
                return
            _ensure_schema(con)
        finally:
            con.close()


def get_connection() -> "_PooledConnection":
    return _PooledConnection(_borrow_raw_connection())
