"""Initial schema baseline — captures all existing tables.

Revision ID: 001
Revises: None
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Core tables ──
    op.execute("""
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
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS organizations_name_unique ON organizations (lower(name))")

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            idp_subject TEXT UNIQUE,
            email TEXT UNIQUE,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS org_memberships (
            user_id TEXT NOT NULL,
            org_id TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, org_id)
        )
    """)

    # ── Bots ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS bots (
            bot_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            publishable_key TEXT NOT NULL UNIQUE,
            secret_key TEXT NOT NULL UNIQUE,
            widget_config TEXT,
            agent_config TEXT,
            escalation_config TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    op.execute("""
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
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS bot_corpora (
            bot_id TEXT PRIMARY KEY,
            corpus_resource TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS domain_corpora (
            hostname TEXT PRIMARY KEY,
            corpus_resource TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # ── Sources ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS bot_sources (
            source_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL,
            type TEXT NOT NULL,
            config TEXT NOT NULL DEFAULT '{}',
            display_name TEXT,
            page_title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_sources_bot_id ON bot_sources (bot_id)")

    # ── Jobs ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS index_jobs_bot_id ON index_jobs (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS index_jobs_bot_hostname ON index_jobs (bot_id, hostname)")
    op.execute("CREATE INDEX IF NOT EXISTS index_jobs_stage ON index_jobs (stage)")
    op.execute("CREATE INDEX IF NOT EXISTS index_jobs_updated_at ON index_jobs (updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS index_jobs_source_id ON index_jobs (source_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS discovery_jobs_bot_id ON discovery_jobs (bot_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS booking_link_jobs_bot_id ON booking_link_jobs (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS booking_link_jobs_updated_at ON booking_link_jobs (updated_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_suggested_message_packs_bot_lang ON bot_suggested_message_packs (bot_id, lang)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS bot_suggested_message_packs_version ON bot_suggested_message_packs (bot_id, lang, suggested_message_id, version_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS bot_suggested_message_packs_updated_at ON bot_suggested_message_packs (bot_id, updated_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS topic_jobs_bot_id ON topic_jobs (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS topic_jobs_updated_at ON topic_jobs (updated_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS availability_jobs_bot_id ON availability_jobs (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS availability_jobs_updated_at ON availability_jobs (updated_at DESC)")

    # ── Conversations ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS conversation_sessions_bot_id ON conversation_sessions (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_sessions_last_active ON conversation_sessions (bot_id, last_active_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS conversation_messages_session_id ON conversation_messages (session_id)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_messages_bot_id ON conversation_messages (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_messages_created_at ON conversation_messages (created_at DESC)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversation_escalations (
            escalation_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            visitor_email TEXT NOT NULL,
            details TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversation_escalation_reads (
            escalation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            read_at TEXT NOT NULL,
            PRIMARY KEY (escalation_id, user_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS conversation_escalation_reads_user_id ON conversation_escalation_reads (user_id, read_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_escalation_reads_escalation_id ON conversation_escalation_reads (escalation_id)")

    op.execute("""
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
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS conversation_channel_contacts_identity ON conversation_channel_contacts (bot_id, channel, external_user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_channel_contacts_session_id ON conversation_channel_contacts (current_session_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS conversation_session_handoffs_bot_state ON conversation_session_handoffs (bot_id, assistant_state, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS conversation_session_handoffs_contact_id ON conversation_session_handoffs (contact_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS conversation_feedback_bot_day ON conversation_feedback (org_id, bot_id, created_at DESC)")

    # ── Analytics ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_usage_daily_bot_day ON bot_usage_daily (org_id, bot_id, day)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bot_sources_daily (
            org_id TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            day TEXT NOT NULL,
            source_url TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (org_id, bot_id, day, source_url)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_sources_daily_bot_day ON bot_sources_daily (org_id, bot_id, day)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bot_topics_daily (
            org_id TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            day TEXT NOT NULL,
            topic TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (org_id, bot_id, day, topic)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_topics_daily_bot_day ON bot_topics_daily (org_id, bot_id, day)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bot_extracted_topics (
            topic_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            category TEXT,
            confidence REAL DEFAULT 1.0,
            source_urls TEXT DEFAULT '[]',
            source_url TEXT,
            origin TEXT DEFAULT 'extracted',
            occurrence_count INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT TRUE,
            extracted_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_extracted_topics_bot ON bot_extracted_topics (org_id, bot_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS rollup_watermarks (
            bot_id TEXT PRIMARY KEY,
            last_processed_at TEXT NOT NULL
        )
    """)

    # ── Topic system v2 ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tqm_topic ON topic_question_mappings (topic_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tqm_session ON topic_question_mappings (session_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tqm_bot ON topic_question_mappings (org_id, bot_id)")

    # ── LINE integration ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS line_channels_bot_id ON line_channels (bot_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS line_design_configs (
            config_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL UNIQUE,
            org_id TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS line_design_configs_bot_id ON line_design_configs (bot_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS line_rich_menu_states_bot_id ON line_rich_menu_states (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS line_rich_menu_states_channel_id ON line_rich_menu_states (channel_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS line_user_sessions (
            line_user_id TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            is_escalated BOOLEAN NOT NULL DEFAULT FALSE,
            awaiting_escalation_msg BOOLEAN NOT NULL DEFAULT FALSE,
            awaiting_staff_takeover BOOLEAN NOT NULL DEFAULT FALSE,
            display_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (line_user_id, bot_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS line_user_sessions_session_id ON line_user_sessions (session_id)")

    # ── Instagram integration ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS instagram_channels (
            channel_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL UNIQUE,
            org_id TEXT NOT NULL,
            ig_page_id TEXT NOT NULL,
            app_secret TEXT NOT NULL,
            page_access_token TEXT NOT NULL,
            verify_token TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            ig_user_id TEXT,
            ig_username TEXT,
            token_expires_at TEXT,
            connection_method TEXT DEFAULT 'manual',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS instagram_channels_bot_id ON instagram_channels (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS instagram_channels_ig_page_id ON instagram_channels (ig_page_id)")
    op.execute("CREATE INDEX IF NOT EXISTS instagram_channels_ig_user_id ON instagram_channels (ig_user_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS instagram_user_sessions (
            ig_user_id TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            is_escalated BOOLEAN NOT NULL DEFAULT FALSE,
            awaiting_escalation_msg BOOLEAN NOT NULL DEFAULT FALSE,
            awaiting_staff_takeover BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ig_user_id, bot_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS instagram_user_sessions_session_id ON instagram_user_sessions (session_id)")

    # ── Business assets ──
    op.execute("""
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
            asset_type TEXT NOT NULL DEFAULT 'image',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS bot_assets_bot_id ON bot_assets (bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS bot_assets_org_bot ON bot_assets (org_id, bot_id)")
    op.execute("CREATE INDEX IF NOT EXISTS bot_assets_type ON bot_assets (bot_id, asset_type)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS asset_extraction_jobs_bot_updated ON asset_extraction_jobs (bot_id, updated_at DESC)")

    # ── Job pipeline ──
    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS job_pipeline_runs_bot_updated ON job_pipeline_runs (bot_id, updated_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS job_pipeline_steps_run ON job_pipeline_steps (run_id, step_index)")
    op.execute("CREATE INDEX IF NOT EXISTS job_pipeline_steps_status ON job_pipeline_steps (status, updated_at DESC)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS job_pipeline_step_events_run ON job_pipeline_step_events (run_id, created_at DESC)")


def downgrade() -> None:
    tables = [
        "job_pipeline_step_events", "job_pipeline_steps", "job_pipeline_runs",
        "asset_extraction_jobs", "bot_assets",
        "instagram_user_sessions", "instagram_channels",
        "line_user_sessions", "line_rich_menu_states", "line_design_configs", "line_channels",
        "topic_question_mappings",
        "rollup_watermarks", "bot_extracted_topics",
        "bot_topics_daily", "bot_sources_daily", "bot_usage_daily",
        "conversation_feedback",
        "conversation_session_handoffs", "conversation_channel_contacts",
        "conversation_escalation_reads", "conversation_escalations",
        "conversation_messages", "conversation_sessions",
        "availability_jobs", "topic_jobs",
        "bot_suggested_message_packs", "booking_link_jobs",
        "discovery_jobs", "index_jobs",
        "bot_sources", "domain_corpora", "bot_corpora", "bot_domains",
        "bots", "org_memberships", "users", "organizations",
    ]
    for table in tables:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
