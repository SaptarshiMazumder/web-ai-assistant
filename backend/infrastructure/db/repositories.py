import json
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg import errors as pg_errors

from domain.entities import (
    AvailabilityJob,
    BookingLinkJob,
    Bot,
    BotAsset,
    BotDomainRecord,
    BotRecord,
    ConversationChannelContact,
    BotSource,
    ConversationMessage,
    ConversationSession,
    ConversationSessionHandoff,
    EscalationRecord,
    DiscoveryJob,
    IndexJob,
    JobPipelineRun,
    JobPipelineStep,
    JobPipelineStepEvent,
    InstagramChannel,
    InstagramUserSession,
    LineChannel,
    LineRichMenuState,
    LineUserSession,
    OrgMemberRecord,
    OrgRecord,
    SuggestedMessagePack,
    TopicJob,
    UserRecord,
)
from domain.repositories import (
    AvailabilityJobRepository,
    BookingLinkJobRepository,
    BotSourceRepository,
    DiscoveryJobRepository,
    IndexJobRepository,
    JobPipelineRepository,
    SuggestedMessagePackRepository,
    TopicJobRepository,
)
from infrastructure.db.connection import get_connection
from infrastructure.clients.rag_client import extract_topics_from_titles


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    return get_connection()


def _normalize_hostname(hostname: str) -> str:
    h = (hostname or "").strip().lower()
    h = re.sub(r"^https?://", "", h)
    h = h.split("/")[0]
    h = h.split(":")[0]
    return h


def _new_bot_id() -> str:
    return "bot_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_publishable_key() -> str:
    return "pk_" + secrets.token_urlsafe(24)


def _new_secret_key() -> str:
    return "sk_" + secrets.token_urlsafe(32)


def _new_conversation_id() -> str:
    return "conv_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_message_id() -> str:
    return "msg_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_escalation_id() -> str:
    return "esc_" + secrets.token_urlsafe(24).replace("-", "_").replace(".", "_")


def _new_contact_id() -> str:
    return "cc_" + secrets.token_urlsafe(18).replace("-", "_").replace(".", "_")


def _new_handoff_id() -> str:
    return "hof_" + secrets.token_urlsafe(18).replace("-", "_").replace(".", "_")


def _new_pipeline_event_id() -> str:
    return "jpe_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


class PostgresBotRepository:
    def create_bot(self, display_name: str, org_id: str) -> Bot:
        bot_id = _new_bot_id()
        pk = _new_publishable_key()
        sk = _new_secret_key()
        now = _utc_now()
        oid = (org_id or "").strip()
        if not oid:
            raise ValueError("org_id is required")
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO bots(bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at, widget_config)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                """,
                (bot_id, oid, display_name, pk, sk, now, now),
            )
            con.commit()
            return Bot(bot_id=bot_id, org_id=oid, display_name=display_name, publishable_key=pk, secret_key=sk, widget_config=None)
        finally:
            con.close()

    def get_bot_by_publishable_key(self, publishable_key: str) -> Optional[Bot]:
        pk = (publishable_key or "").strip()
        if not pk:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key, widget_config, agent_config, escalation_config FROM bots WHERE publishable_key = %s",
                (pk,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4], widget_config=row[5] if len(row) > 5 else None, agent_config=row[6] if len(row) > 6 else None, escalation_config=row[7] if len(row) > 7 else None)
        finally:
            con.close()

    def get_bot_by_secret_key(self, secret_key: str) -> Optional[Bot]:
        sk = (secret_key or "").strip()
        if not sk:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key, widget_config, agent_config, escalation_config FROM bots WHERE secret_key = %s",
                (sk,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4], widget_config=row[5] if len(row) > 5 else None, agent_config=row[6] if len(row) > 6 else None, escalation_config=row[7] if len(row) > 7 else None)
        finally:
            con.close()

    def get_bot(self, bot_id: str) -> Optional[Bot]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key, widget_config, agent_config, escalation_config FROM bots WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4], widget_config=row[5] if len(row) > 5 else None, agent_config=row[6] if len(row) > 6 else None, escalation_config=row[7] if len(row) > 7 else None)
        finally:
            con.close()

    def get_bot_record(self, bot_id: str) -> Optional[BotRecord]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at, widget_config, agent_config, escalation_config
                FROM bots
                WHERE bot_id = %s
                """,
                (bid,),
            ).fetchone()
            if not row:
                return None
            return BotRecord(
                bot_id=row[0],
                org_id=row[1],
                display_name=row[2],
                publishable_key=row[3],
                secret_key=row[4],
                created_at=row[5],
                updated_at=row[6],
                widget_config=row[7] if len(row) > 7 else None,
                agent_config=row[8] if len(row) > 8 else None,
                escalation_config=row[9] if len(row) > 9 else None,
            )
        finally:
            con.close()

    def list_bots(self, org_id: Optional[str] = None) -> List[BotRecord]:
        con = _connect()
        try:
            if org_id:
                rows = con.execute(
                    """
                    SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at, widget_config, agent_config, escalation_config
                    FROM bots
                    WHERE org_id = %s
                    ORDER BY created_at DESC
                    """,
                    (org_id,),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at, widget_config, agent_config, escalation_config
                    FROM bots
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            return [
                BotRecord(
                    bot_id=row[0],
                    org_id=row[1],
                    display_name=row[2],
                    publishable_key=row[3],
                    secret_key=row[4],
                    created_at=row[5],
                    updated_at=row[6],
                    widget_config=row[7] if len(row) > 7 else None,
                    agent_config=row[8] if len(row) > 8 else None,
                    escalation_config=row[9] if len(row) > 9 else None,
                )
                for row in (rows or [])
            ]
        finally:
            con.close()

    def update_display_name(self, bot_id: str, display_name: str) -> None:
        bid = (bot_id or "").strip()
        name = (display_name or "").strip()
        if not bid:
            raise ValueError("bot_id is required")
        if not name:
            raise ValueError("display_name is required")
        con = _connect()
        try:
            now = _utc_now()
            con.execute(
                "UPDATE bots SET display_name = %s, updated_at = %s WHERE bot_id = %s",
                (name, now, bid),
            )
            con.commit()
        finally:
            con.close()

    def update_widget_config(self, bot_id: str, config_json: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            raise ValueError("bot_id is required")
        con = _connect()
        try:
            now = _utc_now()
            con.execute(
                "UPDATE bots SET widget_config = %s, updated_at = %s WHERE bot_id = %s",
                (config_json, now, bid),
            )
            con.commit()
        finally:
            con.close()

    def update_agent_config(self, bot_id: str, config_json: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            raise ValueError("bot_id is required")
        con = _connect()
        try:
            now = _utc_now()
            con.execute(
                "UPDATE bots SET agent_config = %s, updated_at = %s WHERE bot_id = %s",
                (config_json, now, bid),
            )
            con.commit()
        finally:
            con.close()

    def update_escalation_config(self, bot_id: str, config_json: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            raise ValueError("bot_id is required")
        con = _connect()
        try:
            now = _utc_now()
            con.execute(
                "UPDATE bots SET escalation_config = %s, updated_at = %s WHERE bot_id = %s",
                (config_json, now, bid),
            )
            con.commit()
        finally:
            con.close()

    def delete_bot(self, bot_id: str) -> None:
        """Delete a bot and all related data (channels, sessions, jobs, conversations, assets, etc.)."""
        bid = (bot_id or "").strip()
        if not bid:
            raise ValueError("bot_id is required")
        con = _connect()
        try:
            # Delete in dependency order (children before parents)
            con.execute("DELETE FROM instagram_user_sessions WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM line_user_sessions WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM instagram_channels WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM line_channels WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM availability_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM booking_link_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_suggested_message_packs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM topic_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM index_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM discovery_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM asset_extraction_jobs WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM conversation_messages WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM conversation_escalations WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM conversation_feedback WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM conversation_sessions WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM topic_question_mappings WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_extracted_topics WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_usage_daily WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_sources_daily WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_topics_daily WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_assets WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_sources WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_domains WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bot_corpora WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM rollup_watermarks WHERE bot_id = %s", (bid,))
            con.execute("DELETE FROM bots WHERE bot_id = %s", (bid,))
            con.commit()
        finally:
            con.close()


class PostgresBotDomainRepository:
    def add_domain(self, bot_id: str, hostname: str) -> Tuple[str, str]:
        bid = (bot_id or "").strip()
        host = _normalize_hostname(hostname)
        if not bid or not host:
            raise ValueError("bot_id and hostname are required")
        con = _connect()
        now = _utc_now()
        token = "verify_" + secrets.token_urlsafe(24)
        try:
            existing = con.execute(
                "SELECT status, verification_token FROM bot_domains WHERE bot_id = %s AND hostname = %s",
                (bid, host),
            ).fetchone()
            if existing:
                return (existing[0], existing[1])
            bot = PostgresBotRepository().get_bot(bid)
            if not bot:
                raise ValueError("Unknown bot_id")
            con.execute(
                """
                INSERT INTO bot_domains(org_id, bot_id, hostname, status, verification_token, verified_at, created_at, updated_at)
                VALUES (%s, %s, %s, 'pending', %s, NULL, %s, %s)
                """,
                (bot.org_id, bid, host, token, now, now),
            )
            con.commit()
            return ("pending", token)
        finally:
            con.close()

    def list_verified_hosts(self, bot_id: str) -> List[str]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                "SELECT hostname FROM bot_domains WHERE bot_id = %s AND status = 'verified'",
                (bid,),
            ).fetchall()
            return [r[0] for r in rows] if rows else []
        finally:
            con.close()

    def mark_domain_verified(self, bot_id: str, hostname: str) -> None:
        bid = (bot_id or "").strip()
        host = _normalize_hostname(hostname)
        if not bid or not host:
            raise ValueError("bot_id and hostname required")
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE bot_domains
                SET status='verified', verified_at=%s, updated_at=%s
                WHERE bot_id=%s AND hostname=%s
                """,
                (now, now, bid, host),
            )
            con.commit()
        finally:
            con.close()

    def list_domains(self, bot_id: str) -> List[BotDomainRecord]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT org_id, bot_id, hostname, status, verification_token, verified_at, created_at, updated_at
                FROM bot_domains
                WHERE bot_id = %s
                ORDER BY created_at DESC
                """,
                (bid,),
            ).fetchall()
            return [
                BotDomainRecord(
                    org_id=row[0],
                    bot_id=row[1],
                    hostname=row[2],
                    status=row[3],
                    verification_token=row[4],
                    verified_at=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                )
                for row in (rows or [])
            ]
        finally:
            con.close()


class PostgresBotCorpusRepository:
    def upsert_bot_corpus(self, bot_id: str, corpus_resource: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            raise ValueError("bot_id required")
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO bot_corpora(bot_id, corpus_resource, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(bot_id) DO UPDATE SET
                  corpus_resource=excluded.corpus_resource,
                  updated_at=excluded.updated_at
                """,
                (bid, corpus_resource, now, now),
            )
            con.commit()
        finally:
            con.close()

    def get_bot_corpus(self, bot_id: str) -> Optional[str]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT corpus_resource FROM bot_corpora WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            return row[0] if row else None
        finally:
            con.close()


class PostgresOrgRepository:
    def create_org(self, name: str) -> str:
        oid = "org_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
        now = _utc_now()
        con = _connect()
        try:
            try:
                con.execute(
                    """
                    INSERT INTO organizations(org_id, name, status, created_at, updated_at)
                    VALUES (%s, %s, 'active', %s, %s)
                    """,
                    (oid, name, now, now),
                )
                con.commit()
                return oid
            except pg_errors.UniqueViolation:
                con.rollback()
                existing = self.get_org_by_name(name)
                if existing:
                    return existing.org_id
                raise
        finally:
            con.close()

    def list_orgs(self) -> List[OrgRecord]:
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT org_id, name, status, plan, stripe_customer_id, stripe_subscription_id, created_at, updated_at
                FROM organizations
                ORDER BY created_at DESC
                """
            ).fetchall()
            return [
                OrgRecord(
                    org_id=r[0],
                    name=r[1],
                    status=r[2],
                    plan=r[3],
                    stripe_customer_id=r[4],
                    stripe_subscription_id=r[5],
                    created_at=r[6],
                    updated_at=r[7],
                )
                for r in (rows or [])
            ]
        finally:
            con.close()

    def get_org_by_name(self, name: str) -> Optional[OrgRecord]:
        nm = (name or "").strip()
        if not nm:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT org_id, name, status, plan, stripe_customer_id, stripe_subscription_id, created_at, updated_at
                FROM organizations
                WHERE lower(name) = lower(%s)
                LIMIT 1
                """,
                (nm,),
            ).fetchone()
            if not row:
                return None
            return OrgRecord(
                org_id=row[0],
                name=row[1],
                status=row[2],
                plan=row[3],
                stripe_customer_id=row[4],
                stripe_subscription_id=row[5],
                created_at=row[6],
                updated_at=row[7],
            )
        finally:
            con.close()

    def get_org(self, org_id: str) -> Optional[OrgRecord]:
        oid = (org_id or "").strip()
        if not oid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT org_id, name, status, plan, stripe_customer_id, stripe_subscription_id, created_at, updated_at
                FROM organizations
                WHERE org_id = %s
                """,
                (oid,),
            ).fetchone()
            if not row:
                return None
            return OrgRecord(
                org_id=row[0],
                name=row[1],
                status=row[2],
                plan=row[3],
                stripe_customer_id=row[4],
                stripe_subscription_id=row[5],
                created_at=row[6],
                updated_at=row[7],
            )
        finally:
            con.close()

    def update_org_name(self, org_id: str, name: str) -> None:
        oid = (org_id or "").strip()
        nm = (name or "").strip()
        if not oid or not nm:
            raise ValueError("org_id and name required")
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                "UPDATE organizations SET name = %s, updated_at = %s WHERE org_id = %s",
                (nm, now, oid),
            )
            con.commit()
        finally:
            con.close()

    def set_org_status(self, org_id: str, status: str) -> None:
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                "UPDATE organizations SET status = %s, updated_at = %s WHERE org_id = %s",
                (status, now, org_id),
            )
            con.commit()
        finally:
            con.close()


class PostgresUserRepository:
    def upsert_user_from_claims(
        self,
        *,
        subject: str,
        email: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> UserRecord:
        now = _utc_now()
        con = _connect()
        try:
            existing = con.execute(
                "SELECT user_id, idp_subject, email, first_name, last_name FROM users WHERE idp_subject = %s OR email = %s",
                (subject, email),
            ).fetchone()
            if existing:
                user_id = existing[0]
                existing_first = existing[3]
                existing_last = existing[4]
                resolved_first = first_name or existing_first
                resolved_last = last_name or existing_last
                con.execute(
                    """
                    UPDATE users SET idp_subject = %s, email = %s, first_name = %s, last_name = %s, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (subject, email, resolved_first, resolved_last, now, user_id),
                )
                con.commit()
                return UserRecord(
                    user_id=user_id,
                    idp_subject=subject,
                    email=email,
                    first_name=resolved_first,
                    last_name=resolved_last,
                )
            user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
            con.execute(
                """
                INSERT INTO users(user_id, idp_subject, email, first_name, last_name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, subject, email, first_name, last_name, now, now),
            )
            con.commit()
            return UserRecord(user_id=user_id, idp_subject=subject, email=email, first_name=first_name, last_name=last_name)
        finally:
            con.close()

    def create_user_placeholder(self, email: str) -> UserRecord:
        em = (email or "").strip().lower()
        if not em:
            raise ValueError("email required")
        now = _utc_now()
        con = _connect()
        try:
            existing = con.execute(
                "SELECT user_id, idp_subject, email, first_name, last_name FROM users WHERE email = %s",
                (em,),
            ).fetchone()
            if existing:
                return UserRecord(
                    user_id=existing[0],
                    idp_subject=existing[1],
                    email=existing[2],
                    first_name=existing[3],
                    last_name=existing[4],
                )
            user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
            con.execute(
                """
                INSERT INTO users(user_id, idp_subject, email, first_name, last_name, created_at, updated_at)
                VALUES (%s, NULL, %s, NULL, NULL, %s, %s)
                """,
                (user_id, em, now, now),
            )
            con.commit()
            return UserRecord(user_id=user_id, idp_subject=None, email=em, first_name=None, last_name=None)
        finally:
            con.close()

    def get_user_by_subject(self, subject: str) -> Optional[UserRecord]:
        sub = (subject or "").strip()
        if not sub:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT user_id, idp_subject, email, first_name, last_name FROM users WHERE idp_subject = %s",
                (sub,),
            ).fetchone()
            if not row:
                return None
            return UserRecord(user_id=row[0], idp_subject=row[1], email=row[2], first_name=row[3], last_name=row[4])
        finally:
            con.close()


class PostgresOrgMembershipRepository:
    def add_membership(self, org_id: str, user_id: str, role: str) -> None:
        oid = (org_id or "").strip()
        uid = (user_id or "").strip()
        if not oid or not uid:
            raise ValueError("org_id and user_id required")
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO org_memberships(user_id, org_id, role, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(user_id, org_id) DO UPDATE SET
                  role=excluded.role,
                  updated_at=excluded.updated_at
                """,
                (uid, oid, role, now, now),
            )
            con.commit()
        finally:
            con.close()

    def get_org_memberships(self, user_id: str) -> List[Dict[str, str]]:
        uid = (user_id or "").strip()
        if not uid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                "SELECT org_id, role FROM org_memberships WHERE user_id = %s",
                (uid,),
            ).fetchall()
            return [{"org_id": r[0], "role": r[1]} for r in (rows or [])]
        finally:
            con.close()

    def list_org_members(self, org_id: str) -> List[OrgMemberRecord]:
        oid = (org_id or "").strip()
        if not oid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT u.user_id, u.email, u.first_name, u.last_name, m.role, m.created_at, m.updated_at
                FROM org_memberships m
                JOIN users u ON u.user_id = m.user_id
                WHERE m.org_id = %s
                ORDER BY m.created_at DESC
                """,
                (oid,),
            ).fetchall()
            return [
                OrgMemberRecord(
                    user_id=r[0],
                    email=r[1],
                    first_name=r[2],
                    last_name=r[3],
                    role=r[4],
                    created_at=r[5],
                    updated_at=r[6],
                )
                for r in (rows or [])
            ]
        finally:
            con.close()


class PostgresDomainCorpusRepository:
    def get_corpus_for_host(self, hostname: str) -> Optional[str]:
        h = (hostname or "").strip().lower()
        if not h:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT corpus_resource FROM domain_corpora WHERE hostname = %s",
                (h,),
            ).fetchone()
            return row[0] if row else None
        finally:
            con.close()

    def upsert_corpus_for_host(self, hostname: str, corpus_resource: str) -> None:
        h = (hostname or "").strip().lower()
        if not h:
            raise ValueError("hostname required")
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO domain_corpora(hostname, corpus_resource, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(hostname) DO UPDATE SET
                  corpus_resource=excluded.corpus_resource,
                  updated_at=excluded.updated_at
                """,
                (h, corpus_resource, now, now),
            )
            con.commit()
        finally:
            con.close()

    def delete_corpus_mapping_for_host(self, hostname: str) -> None:
        h = (hostname or "").strip().lower()
        if not h:
            return
        con = _connect()
        try:
            con.execute("DELETE FROM domain_corpora WHERE hostname = %s", (h,))
            con.commit()
        finally:
            con.close()


class PostgresBotSourceRepository(BotSourceRepository):
    _sync_columns_ensured = False

    @classmethod
    def _ensure_sync_columns(cls) -> None:
        if cls._sync_columns_ensured:
            return
        con = _connect()
        try:
            for stmt in [
                "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS sync_enabled BOOLEAN DEFAULT FALSE",
                "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS sync_frequency VARCHAR(20) DEFAULT 'daily'",
                "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS sync_time_utc VARCHAR(5) DEFAULT '00:00'",
                "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS sync_timezone VARCHAR(50) DEFAULT 'UTC'",
                "ALTER TABLE bot_sources ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ",
            ]:
                con.execute(stmt)
            con.commit()
            cls._sync_columns_ensured = True
        except Exception:
            cls._sync_columns_ensured = True  # Don't retry on error; columns may already exist
        finally:
            con.close()

    @staticmethod
    def _row_to_source(row) -> BotSource:
        try:
            config = json.loads(row[3]) if isinstance(row[3], str) else (row[3] or {})
        except (TypeError, ValueError):
            config = {}
        return BotSource(
            source_id=row[0],
            bot_id=row[1],
            type=row[2],
            config=config if isinstance(config, dict) else {},
            display_name=row[4],
            created_at=row[5] or "",
            updated_at=row[6] or "",
            sync_enabled=bool(row[7]) if len(row) > 7 and row[7] is not None else False,
            sync_frequency=row[8] or "daily" if len(row) > 8 else "daily",
            sync_time_utc=row[9] or "00:00" if len(row) > 9 else "00:00",
            sync_timezone=row[10] or "UTC" if len(row) > 10 else "UTC",
            last_synced_at=row[11] if len(row) > 11 else None,
        )

    _SELECT_COLS = "source_id, bot_id, type, config, display_name, created_at, updated_at, sync_enabled, sync_frequency, sync_time_utc, sync_timezone, last_synced_at"

    def create_source(self, source: BotSource) -> None:
        self._ensure_sync_columns()
        con = _connect()
        try:
            config_json = json.dumps(source.config if isinstance(source.config, dict) else {})
            con.execute(
                """
                INSERT INTO bot_sources (source_id, bot_id, type, config, display_name, created_at, updated_at,
                                         sync_enabled, sync_frequency, sync_time_utc, sync_timezone, last_synced_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    source.source_id,
                    source.bot_id,
                    source.type,
                    config_json,
                    source.display_name,
                    source.created_at,
                    source.updated_at,
                    source.sync_enabled,
                    source.sync_frequency,
                    source.sync_time_utc,
                    source.sync_timezone,
                    source.last_synced_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_source(self, source: BotSource) -> None:
        bid = (source.bot_id or "").strip()
        sid = (source.source_id or "").strip()
        if not bid or not sid:
            return
        self._ensure_sync_columns()
        con = _connect()
        try:
            config_json = json.dumps(source.config if isinstance(source.config, dict) else {}, ensure_ascii=False)
            now = source.updated_at or datetime.now(timezone.utc).isoformat()
            con.execute(
                """
                UPDATE bot_sources
                SET type = %s, config = %s, display_name = %s, updated_at = %s
                WHERE bot_id = %s AND source_id = %s
                """,
                (source.type, config_json, source.display_name, now, bid, sid),
            )
            con.commit()
        finally:
            con.close()

    def get_source(self, bot_id: str, source_id: str) -> Optional[BotSource]:
        bid = (bot_id or "").strip()
        sid = (source_id or "").strip()
        if not bid or not sid:
            return None
        self._ensure_sync_columns()
        con = _connect()
        try:
            row = con.execute(
                f"SELECT {self._SELECT_COLS} FROM bot_sources WHERE bot_id = %s AND source_id = %s",
                (bid, sid),
            ).fetchone()
            if not row:
                return None
            return self._row_to_source(row)
        finally:
            con.close()

    def list_sources_for_bot(self, bot_id: str) -> List[BotSource]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        self._ensure_sync_columns()
        con = _connect()
        try:
            rows = con.execute(
                f"SELECT {self._SELECT_COLS} FROM bot_sources WHERE bot_id = %s ORDER BY updated_at DESC",
                (bid,),
            ).fetchall()
            return [self._row_to_source(row) for row in rows]
        finally:
            con.close()

    def delete_source(self, bot_id: str, source_id: str) -> None:
        bid = (bot_id or "").strip()
        sid = (source_id or "").strip()
        if not bid or not sid:
            return
        con = _connect()
        try:
            con.execute("DELETE FROM bot_sources WHERE bot_id = %s AND source_id = %s", (bid, sid))
            con.commit()
        finally:
            con.close()

    def update_sync_settings(
        self, bot_id: str, source_id: str, sync_enabled: bool, sync_frequency: str, sync_time_utc: str, sync_timezone: str
    ) -> None:
        bid = (bot_id or "").strip()
        sid = (source_id or "").strip()
        if not bid or not sid:
            return
        self._ensure_sync_columns()
        con = _connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                """
                UPDATE bot_sources
                SET sync_enabled = %s, sync_frequency = %s, sync_time_utc = %s, sync_timezone = %s, updated_at = %s
                WHERE bot_id = %s AND source_id = %s
                """,
                (sync_enabled, sync_frequency, sync_time_utc, sync_timezone, now, bid, sid),
            )
            con.commit()
        finally:
            con.close()

    def update_last_synced(self, bot_id: str, source_id: str) -> None:
        bid = (bot_id or "").strip()
        sid = (source_id or "").strip()
        if not bid or not sid:
            return
        self._ensure_sync_columns()
        con = _connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                "UPDATE bot_sources SET last_synced_at = %s, updated_at = %s WHERE bot_id = %s AND source_id = %s",
                (now, now, bid, sid),
            )
            con.commit()
        finally:
            con.close()

    def list_sources_due_for_sync(self) -> List[BotSource]:
        """Return URL sources with sync_enabled=True whose schedule matches the current 15-min window."""
        self._ensure_sync_columns()
        now_utc = datetime.now(timezone.utc)
        current_hhmm = now_utc.strftime("%H:%M")
        # Build a 15-minute window around the scheduled time
        window_start = (now_utc - timedelta(minutes=7)).strftime("%H:%M")
        window_end = (now_utc + timedelta(minutes=7)).strftime("%H:%M")

        con = _connect()
        try:
            if window_start <= window_end:
                rows = con.execute(
                    f"""
                    SELECT {self._SELECT_COLS} FROM bot_sources
                    WHERE sync_enabled = TRUE AND type = 'url'
                      AND sync_time_utc >= %s AND sync_time_utc <= %s
                    ORDER BY updated_at DESC
                    """,
                    (window_start, window_end),
                ).fetchall()
            else:
                # Window wraps around midnight (e.g. 23:55 - 00:05)
                rows = con.execute(
                    f"""
                    SELECT {self._SELECT_COLS} FROM bot_sources
                    WHERE sync_enabled = TRUE AND type = 'url'
                      AND (sync_time_utc >= %s OR sync_time_utc <= %s)
                    ORDER BY updated_at DESC
                    """,
                    (window_start, window_end),
                ).fetchall()

            sources = [self._row_to_source(row) for row in rows]

            # Filter by frequency: skip if already synced within the period
            result = []
            for s in sources:
                if s.last_synced_at:
                    try:
                        last = datetime.fromisoformat(s.last_synced_at.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        last = None
                    if last:
                        if s.sync_frequency == "daily" and (now_utc - last).total_seconds() < 23 * 3600:
                            continue
                        elif s.sync_frequency == "weekly" and (now_utc - last).total_seconds() < 6.5 * 24 * 3600:
                            continue
                        elif s.sync_frequency == "monthly" and (now_utc - last).total_seconds() < 29 * 24 * 3600:
                            continue
                result.append(s)
            return result
        finally:
            con.close()


class PostgresIndexJobRepository(IndexJobRepository):
    def create_job(self, job: IndexJob) -> None:
        con = _connect()
        try:
            crawled_urls_json = json.dumps(getattr(job, "crawled_urls", None) or [])
            source_id = getattr(job, "source_id", None)
            con.execute(
                """
                INSERT INTO index_jobs(
                  job_id, bot_id, source_id, url, hostname, celery_task_id, stage,
                  pages_crawled, docs_count, last_crawled_url, last_depth,
                  gcs_prefix, last_error, crawled_urls, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id,
                    job.bot_id,
                    source_id,
                    job.url,
                    job.hostname,
                    job.celery_task_id,
                    job.stage,
                    job.pages_crawled,
                    job.docs_count,
                    job.last_crawled_url,
                    job.last_depth,
                    job.gcs_prefix,
                    job.last_error,
                    crawled_urls_json,
                    job.created_at,
                    job.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get_job(self, bot_id: str, job_key: str) -> Optional[IndexJob]:
        bid = (bot_id or "").strip()
        key = (job_key or "").strip()
        if not bid or not key:
            return None
        con = _connect()
        try:
            # Try by job_id first
            row = con.execute(
                """
                SELECT job_id, bot_id, source_id, url, hostname, celery_task_id, stage,
                       pages_crawled, docs_count, last_crawled_url, last_depth,
                       gcs_prefix, last_error, crawled_urls, created_at, updated_at
                FROM index_jobs
                WHERE bot_id = %s AND job_id = %s
                """,
                (bid, key),
            ).fetchone()
            if row:
                crawled = row[13] if len(row) > 13 else "[]"
                try:
                    crawled_list = json.loads(crawled) if isinstance(crawled, str) else (crawled or [])
                except (TypeError, ValueError):
                    crawled_list = []
                return IndexJob(
                    job_id=row[0],
                    bot_id=row[1],
                    url=row[3],
                    hostname=row[4],
                    celery_task_id=row[5],
                    stage=row[6],
                    pages_crawled=row[7] or 0,
                    docs_count=row[8] or 0,
                    last_crawled_url=row[9] or "",
                    last_depth=row[10] or -1,
                    gcs_prefix=row[11] or "",
                    last_error=row[12] or "",
                    crawled_urls=crawled_list if isinstance(crawled_list, list) else [],
                    created_at=row[14],
                    updated_at=row[15],
                    source_id=row[2],
                )
            # Try by hostname (for backward compatibility)
            row = con.execute(
                """
                SELECT job_id, bot_id, source_id, url, hostname, celery_task_id, stage,
                       pages_crawled, docs_count, last_crawled_url, last_depth,
                       gcs_prefix, last_error, crawled_urls, created_at, updated_at
                FROM index_jobs
                WHERE bot_id = %s AND hostname = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (bid, key),
            ).fetchone()
            if row:
                crawled = row[13] if len(row) > 13 else "[]"
                try:
                    crawled_list = json.loads(crawled) if isinstance(crawled, str) else (crawled or [])
                except (TypeError, ValueError):
                    crawled_list = []
                return IndexJob(
                    job_id=row[0],
                    bot_id=row[1],
                    url=row[3],
                    hostname=row[4],
                    celery_task_id=row[5],
                    stage=row[6],
                    pages_crawled=row[7] or 0,
                    docs_count=row[8] or 0,
                    last_crawled_url=row[9] or "",
                    last_depth=row[10] or -1,
                    gcs_prefix=row[11] or "",
                    last_error=row[12] or "",
                    crawled_urls=crawled_list if isinstance(crawled_list, list) else [],
                    created_at=row[14],
                    updated_at=row[15],
                    source_id=row[2],
                )
            return None
        finally:
            con.close()

    def get_job_by_hostname(self, bot_id: str, hostname: str) -> Optional[IndexJob]:
        bid = (bot_id or "").strip()
        host = (hostname or "").strip().lower()
        if not bid or not host:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, bot_id, source_id, url, hostname, celery_task_id, stage,
                       pages_crawled, docs_count, last_crawled_url, last_depth,
                       gcs_prefix, last_error, crawled_urls, created_at, updated_at
                FROM index_jobs
                WHERE bot_id = %s AND hostname = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (bid, host),
            ).fetchone()
            if not row:
                return None
            crawled = row[13] if len(row) > 13 else "[]"
            try:
                crawled_list = json.loads(crawled) if isinstance(crawled, str) else (crawled or [])
            except (TypeError, ValueError):
                crawled_list = []
            return IndexJob(
                job_id=row[0],
                bot_id=row[1],
                url=row[3],
                hostname=row[4],
                stage=row[6],
                pages_crawled=row[7] or 0,
                docs_count=row[8] or 0,
                last_crawled_url=row[9] or "",
                last_depth=row[10] or -1,
                gcs_prefix=row[11] or "",
                last_error=row[12] or "",
                crawled_urls=crawled_list if isinstance(crawled_list, list) else [],
                created_at=row[14],
                updated_at=row[15],
                celery_task_id=row[5],
                source_id=row[2],
            )
        finally:
            con.close()

    def update_job(self, job: IndexJob) -> None:
        now = _utc_now()
        crawled_urls_json = json.dumps(getattr(job, "crawled_urls", None) or [])
        source_id = getattr(job, "source_id", None)
        con = _connect()
        try:
            con.execute(
                """
                UPDATE index_jobs
                SET stage = %s, pages_crawled = %s, docs_count = %s,
                    last_crawled_url = %s, last_depth = %s, gcs_prefix = %s,
                    last_error = %s, crawled_urls = %s, source_id = COALESCE(%s, source_id),
                    updated_at = %s, celery_task_id = COALESCE(%s, celery_task_id)
                WHERE job_id = %s
                """,
                (
                    job.stage,
                    job.pages_crawled,
                    job.docs_count,
                    job.last_crawled_url,
                    job.last_depth,
                    job.gcs_prefix,
                    job.last_error,
                    crawled_urls_json,
                    source_id,
                    now,
                    job.celery_task_id,
                    job.job_id,
                ),
            )
            con.commit()
        finally:
            con.close()

    def list_jobs_for_bot(self, bot_id: str) -> List[IndexJob]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT job_id, bot_id, source_id, url, hostname, celery_task_id, stage,
                       pages_crawled, docs_count, last_crawled_url, last_depth,
                       gcs_prefix, last_error, crawled_urls, created_at, updated_at
                FROM index_jobs
                WHERE bot_id = %s
                ORDER BY updated_at DESC
                """,
                (bid,),
            ).fetchall()
            result = []
            for row in rows:
                crawled = row[13] if len(row) > 13 else "[]"
                try:
                    crawled_list = json.loads(crawled) if isinstance(crawled, str) else (crawled or [])
                except (TypeError, ValueError):
                    crawled_list = []
                result.append(
                    IndexJob(
                        job_id=row[0],
                        bot_id=row[1],
                        url=row[3],
                        hostname=row[4],
                        stage=row[6],
                        pages_crawled=row[7] or 0,
                        docs_count=row[8] or 0,
                        last_crawled_url=row[9] or "",
                        last_depth=row[10] or -1,
                        gcs_prefix=row[11] or "",
                        last_error=row[12] or "",
                        crawled_urls=crawled_list if isinstance(crawled_list, list) else [],
                        created_at=row[14],
                        updated_at=row[15],
                        source_id=row[2],
                    )
                )
            return result
        finally:
            con.close()


class PostgresDiscoveryJobRepository(DiscoveryJobRepository):
    def create(self, job: DiscoveryJob) -> None:
        con = _connect()
        try:
            urls_json = json.dumps(getattr(job, "discovered_urls", None) or [])
            con.execute(
                """
                INSERT INTO discovery_jobs(
                  job_id, bot_id, root_url, method, status,
                  discovered_urls, error, celery_task_id, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id,
                    job.bot_id,
                    job.root_url,
                    job.method,
                    job.status,
                    urls_json,
                    job.error,
                    job.celery_task_id,
                    job.created_at,
                    job.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get(self, bot_id: str, job_id: str) -> Optional[DiscoveryJob]:
        bid = (bot_id or "").strip()
        jid = (job_id or "").strip()
        if not bid or not jid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, bot_id, root_url, method, status,
                       discovered_urls, error, celery_task_id, created_at, updated_at
                FROM discovery_jobs
                WHERE bot_id = %s AND job_id = %s
                """,
                (bid, jid),
            ).fetchone()
            if not row:
                return None
            urls_raw = row[5] if len(row) > 5 else "[]"
            try:
                urls_list = json.loads(urls_raw) if isinstance(urls_raw, str) else (urls_raw or [])
            except (TypeError, ValueError):
                urls_list = []
            return DiscoveryJob(
                job_id=row[0],
                bot_id=row[1],
                root_url=row[2],
                method=row[3],
                status=row[4],
                discovered_urls=urls_list if isinstance(urls_list, list) else [],
                error=row[6],
                celery_task_id=row[7],
                created_at=row[8],
                updated_at=row[9],
            )
        finally:
            con.close()

    def list_by_bot(self, bot_id: str) -> List[DiscoveryJob]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT job_id, bot_id, root_url, method, status,
                       discovered_urls, error, celery_task_id, created_at, updated_at
                FROM discovery_jobs
                WHERE bot_id = %s
                ORDER BY created_at DESC
                """,
                (bid,),
            ).fetchall()
            result = []
            for row in rows:
                urls_raw = row[5] if len(row) > 5 else "[]"
                try:
                    urls_list = json.loads(urls_raw) if isinstance(urls_raw, str) else (urls_raw or [])
                except (TypeError, ValueError):
                    urls_list = []
                result.append(
                    DiscoveryJob(
                        job_id=row[0],
                        bot_id=row[1],
                        root_url=row[2],
                        method=row[3],
                        status=row[4],
                        discovered_urls=urls_list if isinstance(urls_list, list) else [],
                        error=row[6],
                        celery_task_id=row[7],
                        created_at=row[8],
                        updated_at=row[9],
                    )
                )
            return result
        finally:
            con.close()

    def update(self, job: DiscoveryJob) -> None:
        now = _utc_now()
        urls_json = json.dumps(getattr(job, "discovered_urls", None) or [])
        con = _connect()
        try:
            con.execute(
                """
                UPDATE discovery_jobs
                SET status = %s, discovered_urls = %s, error = %s,
                    celery_task_id = COALESCE(%s, celery_task_id), updated_at = %s
                WHERE job_id = %s
                """,
                (job.status, urls_json, job.error, job.celery_task_id, now, job.job_id),
            )
            con.commit()
        finally:
            con.close()


class PostgresBookingLinkJobRepository(BookingLinkJobRepository):
    def create(self, job: BookingLinkJob) -> None:
        con = _connect()
        try:
            links_json = json.dumps(getattr(job, "links", None) or [])
            con.execute(
                """
                INSERT INTO booking_link_jobs(
                  job_id, bot_id, index_job_id, root_url, status,
                  links, error, celery_task_id, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id,
                    job.bot_id,
                    job.index_job_id,
                    job.root_url,
                    job.status,
                    links_json,
                    job.error,
                    job.celery_task_id,
                    job.created_at,
                    job.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get(self, bot_id: str, job_id: str) -> Optional[BookingLinkJob]:
        bid = (bot_id or "").strip()
        jid = (job_id or "").strip()
        if not bid or not jid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, bot_id, index_job_id, root_url, status,
                       links, error, celery_task_id, created_at, updated_at
                FROM booking_link_jobs
                WHERE bot_id = %s AND job_id = %s
                """,
                (bid, jid),
            ).fetchone()
            if not row:
                return None
            links_raw = row[5] if len(row) > 5 else "[]"
            try:
                links_list = json.loads(links_raw) if isinstance(links_raw, str) else (links_raw or [])
            except (TypeError, ValueError):
                links_list = []
            return BookingLinkJob(
                job_id=row[0],
                bot_id=row[1],
                index_job_id=row[2],
                root_url=row[3],
                status=row[4],
                links=links_list if isinstance(links_list, list) else [],
                error=row[6],
                celery_task_id=row[7],
                created_at=row[8],
                updated_at=row[9],
            )
        finally:
            con.close()

    def list_by_bot(self, bot_id: str) -> List[BookingLinkJob]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT job_id, bot_id, index_job_id, root_url, status,
                       links, error, celery_task_id, created_at, updated_at
                FROM booking_link_jobs
                WHERE bot_id = %s
                ORDER BY created_at DESC
                """,
                (bid,),
            ).fetchall()
            result = []
            for row in rows:
                links_raw = row[5] if len(row) > 5 else "[]"
                try:
                    links_list = json.loads(links_raw) if isinstance(links_raw, str) else (links_raw or [])
                except (TypeError, ValueError):
                    links_list = []
                result.append(
                    BookingLinkJob(
                        job_id=row[0],
                        bot_id=row[1],
                        index_job_id=row[2],
                        root_url=row[3],
                        status=row[4],
                        links=links_list if isinstance(links_list, list) else [],
                        error=row[6],
                        celery_task_id=row[7],
                        created_at=row[8],
                        updated_at=row[9],
                    )
                )
            return result
        finally:
            con.close()

    def update(self, job: BookingLinkJob) -> None:
        now = _utc_now()
        links_json = json.dumps(getattr(job, "links", None) or [])
        con = _connect()
        try:
            con.execute(
                """
                UPDATE booking_link_jobs
                SET status = %s, links = %s, error = %s,
                    celery_task_id = COALESCE(%s, celery_task_id), updated_at = %s
                WHERE job_id = %s
                """,
                (
                    job.status,
                    links_json,
                    job.error,
                    job.celery_task_id,
                    now,
                    job.job_id,
                ),
            )
            con.commit()
        finally:
            con.close()


class PostgresSuggestedMessagePackRepository(SuggestedMessagePackRepository):
    @staticmethod
    def _parse_json_list(raw: Any) -> list:
        try:
            value = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (TypeError, ValueError):
            value = []
        return value if isinstance(value, list) else []

    @classmethod
    def _row_to_entity(cls, row) -> SuggestedMessagePack:
        return SuggestedMessagePack(
            pack_id=row[0],
            bot_id=row[1],
            org_id=row[2],
            lang=row[3],
            suggested_message_id=row[4],
            label=row[5],
            prompt=row[6],
            pack_mode=row[7],
            status=row[8],
            version_hash=row[9],
            source_urls=[str(item).strip() for item in cls._parse_json_list(row[10]) if str(item).strip()],
            evidence_snippets=cls._parse_json_list(row[11]),
            link_targets=cls._parse_json_list(row[12]),
            instruction=row[13] or "",
            citations=cls._parse_json_list(row[14]),
            error=row[15],
            created_at=row[16],
            updated_at=row[17],
        )

    def upsert(self, pack: SuggestedMessagePack) -> None:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO bot_suggested_message_packs(
                  pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                  pack_mode, status, version_hash, source_urls, evidence_snippets,
                  link_targets, instruction, citations, error, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (bot_id, lang, suggested_message_id, version_hash)
                DO UPDATE SET
                  label = EXCLUDED.label,
                  prompt = EXCLUDED.prompt,
                  pack_mode = EXCLUDED.pack_mode,
                  status = EXCLUDED.status,
                  source_urls = EXCLUDED.source_urls,
                  evidence_snippets = EXCLUDED.evidence_snippets,
                  link_targets = EXCLUDED.link_targets,
                  instruction = EXCLUDED.instruction,
                  citations = EXCLUDED.citations,
                  error = EXCLUDED.error,
                  updated_at = EXCLUDED.updated_at
                """,
                (
                    pack.pack_id,
                    pack.bot_id,
                    pack.org_id,
                    pack.lang,
                    pack.suggested_message_id,
                    pack.label,
                    pack.prompt,
                    pack.pack_mode,
                    pack.status,
                    pack.version_hash,
                    json.dumps(pack.source_urls or [], ensure_ascii=False),
                    json.dumps(pack.evidence_snippets or [], ensure_ascii=False),
                    json.dumps(pack.link_targets or [], ensure_ascii=False),
                    pack.instruction or "",
                    json.dumps(pack.citations or [], ensure_ascii=False),
                    pack.error,
                    pack.created_at,
                    pack.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get_latest(
        self,
        *,
        bot_id: str,
        lang: str,
        suggested_message_id: str,
    ) -> Optional[SuggestedMessagePack]:
        bid = (bot_id or "").strip()
        normalized_lang = (lang or "").strip().lower()
        sid = (suggested_message_id or "").strip()
        if not bid or not normalized_lang or not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                       pack_mode, status, version_hash, source_urls, evidence_snippets,
                       link_targets, instruction, citations, error, created_at, updated_at
                FROM bot_suggested_message_packs
                WHERE bot_id = %s AND lang = %s AND suggested_message_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (bid, normalized_lang, sid),
            ).fetchone()
            return self._row_to_entity(row) if row else None
        finally:
            con.close()

    def get_by_version(
        self,
        *,
        bot_id: str,
        lang: str,
        suggested_message_id: str,
        version_hash: str,
    ) -> Optional[SuggestedMessagePack]:
        bid = (bot_id or "").strip()
        normalized_lang = (lang or "").strip().lower()
        sid = (suggested_message_id or "").strip()
        version = (version_hash or "").strip()
        if not bid or not normalized_lang or not sid or not version:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                       pack_mode, status, version_hash, source_urls, evidence_snippets,
                       link_targets, instruction, citations, error, created_at, updated_at
                FROM bot_suggested_message_packs
                WHERE bot_id = %s AND lang = %s AND suggested_message_id = %s AND version_hash = %s
                LIMIT 1
                """,
                (bid, normalized_lang, sid, version),
            ).fetchone()
            return self._row_to_entity(row) if row else None
        finally:
            con.close()

    def list_by_bot(self, bot_id: str, *, lang: Optional[str] = None) -> List[SuggestedMessagePack]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            if lang:
                rows = con.execute(
                    """
                    SELECT pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                           pack_mode, status, version_hash, source_urls, evidence_snippets,
                           link_targets, instruction, citations, error, created_at, updated_at
                    FROM bot_suggested_message_packs
                    WHERE bot_id = %s AND lang = %s
                    ORDER BY updated_at DESC
                    """,
                    (bid, (lang or "").strip().lower()),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                           pack_mode, status, version_hash, source_urls, evidence_snippets,
                           link_targets, instruction, citations, error, created_at, updated_at
                    FROM bot_suggested_message_packs
                    WHERE bot_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (bid,),
                ).fetchall()
            return [self._row_to_entity(row) for row in rows or []]
        finally:
            con.close()

    def replace_for_bot_lang(self, *, bot_id: str, lang: str, packs: List[SuggestedMessagePack]) -> None:
        bid = (bot_id or "").strip()
        normalized_lang = (lang or "").strip().lower()
        if not bid or not normalized_lang:
            raise ValueError("bot_id and lang are required")
        con = _connect()
        try:
            con.execute(
                "DELETE FROM bot_suggested_message_packs WHERE bot_id = %s AND lang = %s",
                (bid, normalized_lang),
            )
            for pack in packs or []:
                con.execute(
                    """
                    INSERT INTO bot_suggested_message_packs(
                      pack_id, bot_id, org_id, lang, suggested_message_id, label, prompt,
                      pack_mode, status, version_hash, source_urls, evidence_snippets,
                      link_targets, instruction, citations, error, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        pack.pack_id,
                        pack.bot_id,
                        pack.org_id,
                        pack.lang,
                        pack.suggested_message_id,
                        pack.label,
                        pack.prompt,
                        pack.pack_mode,
                        pack.status,
                        pack.version_hash,
                        json.dumps(pack.source_urls or [], ensure_ascii=False),
                        json.dumps(pack.evidence_snippets or [], ensure_ascii=False),
                        json.dumps(pack.link_targets or [], ensure_ascii=False),
                        pack.instruction or "",
                        json.dumps(pack.citations or [], ensure_ascii=False),
                        pack.error,
                        pack.created_at,
                        pack.updated_at,
                    ),
                )
            con.commit()
        finally:
            con.close()

    def delete_for_bot(self, bot_id: str) -> None:
        bid = (bot_id or "").strip()
        if not bid:
            return
        con = _connect()
        try:
            con.execute("DELETE FROM bot_suggested_message_packs WHERE bot_id = %s", (bid,))
            con.commit()
        finally:
            con.close()


class PostgresTopicJobRepository(TopicJobRepository):
    def create(self, job: TopicJob) -> None:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO topic_jobs(
                  job_id, org_id, bot_id, status, stage, gcs_prefix,
                  docs_count, topics_count, last_error, celery_task_id,
                  created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id,
                    job.org_id,
                    job.bot_id,
                    job.status,
                    job.stage,
                    job.gcs_prefix,
                    job.docs_count,
                    job.topics_count,
                    job.last_error,
                    job.celery_task_id,
                    job.created_at,
                    job.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get(self, bot_id: str, job_id: str) -> Optional[TopicJob]:
        bid = (bot_id or "").strip()
        jid = (job_id or "").strip()
        if not bid or not jid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, org_id, bot_id, status, stage, gcs_prefix,
                       docs_count, topics_count, last_error, celery_task_id,
                       created_at, updated_at
                FROM topic_jobs
                WHERE bot_id = %s AND job_id = %s
                """,
                (bid, jid),
            ).fetchone()
            if not row:
                return None
            return TopicJob(
                job_id=row[0],
                org_id=row[1],
                bot_id=row[2],
                status=row[3],
                stage=row[4],
                gcs_prefix=row[5],
                docs_count=row[6] or 0,
                topics_count=row[7] or 0,
                last_error=row[8],
                celery_task_id=row[9],
                created_at=row[10],
                updated_at=row[11],
            )
        finally:
            con.close()

    def list_by_bot(self, bot_id: str) -> List[TopicJob]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT job_id, org_id, bot_id, status, stage, gcs_prefix,
                       docs_count, topics_count, last_error, celery_task_id,
                       created_at, updated_at
                FROM topic_jobs
                WHERE bot_id = %s
                ORDER BY updated_at DESC
                """,
                (bid,),
            ).fetchall()
            return [
                TopicJob(
                    job_id=row[0],
                    org_id=row[1],
                    bot_id=row[2],
                    status=row[3],
                    stage=row[4],
                    gcs_prefix=row[5],
                    docs_count=row[6] or 0,
                    topics_count=row[7] or 0,
                    last_error=row[8],
                    celery_task_id=row[9],
                    created_at=row[10],
                    updated_at=row[11],
                )
                for row in rows or []
            ]
        finally:
            con.close()

    def update(self, job: TopicJob) -> None:
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE topic_jobs
                SET status = %s,
                    stage = %s,
                    gcs_prefix = COALESCE(%s, gcs_prefix),
                    docs_count = %s,
                    topics_count = %s,
                    last_error = %s,
                    celery_task_id = COALESCE(%s, celery_task_id),
                    updated_at = %s
                WHERE job_id = %s
                """,
                (
                    job.status,
                    job.stage,
                    job.gcs_prefix,
                    job.docs_count,
                    job.topics_count,
                    job.last_error,
                    job.celery_task_id,
                    now,
                    job.job_id,
                ),
            )
            con.commit()
        finally:
            con.close()


class PostgresAvailabilityJobRepository(AvailabilityJobRepository):
    def create(self, job: AvailabilityJob) -> None:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO availability_jobs(
                  job_id, org_id, bot_id, url, status, question, summary, raw_text_path, raw_html_path, last_error,
                  max_seconds, steps_count, screenshots_dir, celery_task_id,
                  created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id,
                    job.org_id,
                    job.bot_id,
                    job.url,
                    job.status,
                    job.question,
                    job.summary,
                    job.raw_text_path,
                    job.raw_html_path,
                    job.last_error,
                    job.max_seconds,
                    job.steps_count,
                    job.screenshots_dir,
                    job.celery_task_id,
                    job.created_at,
                    job.updated_at,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get(self, bot_id: str, job_id: str) -> Optional[AvailabilityJob]:
        bid = (bot_id or "").strip()
        jid = (job_id or "").strip()
        if not bid or not jid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, org_id, bot_id, url, status, question, summary, raw_text_path, raw_html_path, last_error,
                       max_seconds, steps_count, screenshots_dir, celery_task_id,
                       created_at, updated_at
                FROM availability_jobs
                WHERE bot_id = %s AND job_id = %s
                """,
                (bid, jid),
            ).fetchone()
            if not row:
                return None
            return AvailabilityJob(
                job_id=row[0],
                org_id=row[1],
                bot_id=row[2],
                url=row[3],
                status=row[4],
                question=row[5],
                summary=row[6],
                raw_text_path=row[7],
                raw_html_path=row[8],
                last_error=row[9],
                max_seconds=row[10] or 60,
                steps_count=row[11] or 0,
                screenshots_dir=row[12],
                celery_task_id=row[13],
                created_at=row[14],
                updated_at=row[15],
            )
        finally:
            con.close()

    def list_by_bot(self, bot_id: str) -> List[AvailabilityJob]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT job_id, org_id, bot_id, url, status, question, summary, raw_text_path, raw_html_path, last_error,
                       max_seconds, steps_count, screenshots_dir, celery_task_id,
                       created_at, updated_at
                FROM availability_jobs
                WHERE bot_id = %s
                ORDER BY updated_at DESC
                """,
                (bid,),
            ).fetchall()
            return [
                AvailabilityJob(
                    job_id=row[0],
                    org_id=row[1],
                    bot_id=row[2],
                    url=row[3],
                    status=row[4],
                    question=row[5],
                    summary=row[6],
                    raw_text_path=row[7],
                    raw_html_path=row[8],
                    last_error=row[9],
                    max_seconds=row[10] or 60,
                    steps_count=row[11] or 0,
                    screenshots_dir=row[12],
                    celery_task_id=row[13],
                    created_at=row[14],
                    updated_at=row[15],
                )
                for row in rows or []
            ]
        finally:
            con.close()

    def update(self, job: AvailabilityJob) -> None:
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE availability_jobs
                SET status = %s,
                    question = COALESCE(%s, question),
                    summary = %s,
                    raw_text_path = COALESCE(%s, raw_text_path),
                    raw_html_path = COALESCE(%s, raw_html_path),
                    last_error = %s,
                    max_seconds = %s,
                    steps_count = %s,
                    screenshots_dir = %s,
                    celery_task_id = COALESCE(%s, celery_task_id),
                    updated_at = %s
                WHERE job_id = %s
                """,
                (
                    job.status,
                    job.question,
                    job.summary,
                    job.raw_text_path,
                    job.raw_html_path,
                    job.last_error,
                    job.max_seconds,
                    job.steps_count,
                    job.screenshots_dir,
                    job.celery_task_id,
                    now,
                    job.job_id,
                ),
            )
            con.commit()
        finally:
            con.close()


class PostgresJobPipelineRepository(JobPipelineRepository):
    def create_run(self, run: JobPipelineRun, steps: List[JobPipelineStep]) -> None:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO job_pipeline_runs(
                  run_id, org_id, bot_id, workflow_id, trigger, status,
                  current_step_index, progress_pct, current_step_id, current_stage_key, current_message,
                  context_json, last_error, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run.run_id,
                    run.org_id,
                    run.bot_id,
                    run.workflow_id,
                    run.trigger,
                    run.status,
                    int(run.current_step_index or 0),
                    int(run.progress_pct or 0),
                    run.current_step_id,
                    run.current_stage_key,
                    run.current_message,
                    json.dumps(run.context if isinstance(run.context, dict) else {}),
                    run.last_error,
                    run.created_at,
                    run.updated_at,
                ),
            )
            for step in steps or []:
                con.execute(
                    """
                    INSERT INTO job_pipeline_steps(
                      run_id, step_index, job_id, runner_ref, on_failure, status,
                      progress_pct, current_stage_key, current_message, attempt, celery_task_id,
                      linked_job_type, linked_job_id, output_json, last_error,
                      started_at, completed_at, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        step.run_id,
                        int(step.step_index),
                        step.job_id,
                        step.runner_ref,
                        step.on_failure,
                        step.status,
                        int(step.progress_pct or 0),
                        step.current_stage_key,
                        step.current_message,
                        int(step.attempt or 0),
                        step.celery_task_id,
                        step.linked_job_type,
                        step.linked_job_id,
                        json.dumps(step.output if isinstance(step.output, dict) else {}),
                        step.last_error,
                        step.started_at,
                        step.completed_at,
                        step.created_at,
                        step.updated_at,
                    ),
                )
            con.commit()
        finally:
            con.close()

    def get_run(self, run_id: str) -> Optional[JobPipelineRun]:
        rid = (run_id or "").strip()
        if not rid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT run_id, org_id, bot_id, workflow_id, trigger, status,
                       current_step_index, progress_pct, current_step_id, current_stage_key, current_message,
                       context_json, last_error, created_at, updated_at
                FROM job_pipeline_runs
                WHERE run_id = %s
                """,
                (rid,),
            ).fetchone()
            if not row:
                return None
            try:
                context_json = json.loads(row[11]) if row[11] else {}
            except Exception:
                context_json = {}
            return JobPipelineRun(
                run_id=row[0],
                org_id=row[1],
                bot_id=row[2],
                workflow_id=row[3],
                trigger=row[4],
                status=row[5],
                current_step_index=row[6] or 0,
                progress_pct=row[7] or 0,
                current_step_id=row[8],
                current_stage_key=row[9],
                current_message=row[10],
                context=context_json if isinstance(context_json, dict) else {},
                last_error=row[12],
                created_at=row[13],
                updated_at=row[14],
            )
        finally:
            con.close()

    def get_latest_run_for_bot(self, bot_id: str) -> Optional[JobPipelineRun]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT run_id, org_id, bot_id, workflow_id, trigger, status,
                       current_step_index, progress_pct, current_step_id, current_stage_key, current_message,
                       context_json, last_error, created_at, updated_at
                FROM job_pipeline_runs
                WHERE bot_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (bid,),
            ).fetchone()
            if not row:
                return None
            try:
                context_json = json.loads(row[11]) if row[11] else {}
            except Exception:
                context_json = {}
            return JobPipelineRun(
                run_id=row[0],
                org_id=row[1],
                bot_id=row[2],
                workflow_id=row[3],
                trigger=row[4],
                status=row[5],
                current_step_index=row[6] or 0,
                progress_pct=row[7] or 0,
                current_step_id=row[8],
                current_stage_key=row[9],
                current_message=row[10],
                context=context_json if isinstance(context_json, dict) else {},
                last_error=row[12],
                created_at=row[13],
                updated_at=row[14],
            )
        finally:
            con.close()

    def list_steps(self, run_id: str) -> List[JobPipelineStep]:
        rid = (run_id or "").strip()
        if not rid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT run_id, step_index, job_id, runner_ref, on_failure, status,
                       progress_pct, current_stage_key, current_message, attempt, celery_task_id,
                       linked_job_type, linked_job_id, output_json, last_error,
                       started_at, completed_at, created_at, updated_at
                FROM job_pipeline_steps
                WHERE run_id = %s
                ORDER BY step_index ASC
                """,
                (rid,),
            ).fetchall()
            out: List[JobPipelineStep] = []
            for row in rows or []:
                try:
                    output_json = json.loads(row[13]) if row[13] else {}
                except Exception:
                    output_json = {}
                out.append(
                    JobPipelineStep(
                        run_id=row[0],
                        step_index=row[1] or 0,
                        job_id=row[2],
                        runner_ref=row[3],
                        on_failure=row[4],
                        status=row[5],
                        progress_pct=row[6] or 0,
                        current_stage_key=row[7],
                        current_message=row[8],
                        attempt=row[9] or 0,
                        celery_task_id=row[10],
                        linked_job_type=row[11],
                        linked_job_id=row[12],
                        output=output_json if isinstance(output_json, dict) else {},
                        last_error=row[14],
                        started_at=row[15],
                        completed_at=row[16],
                        created_at=row[17],
                        updated_at=row[18],
                    )
                )
            return out
        finally:
            con.close()

    def list_events(self, run_id: str) -> List[JobPipelineStepEvent]:
        rid = (run_id or "").strip()
        if not rid:
            return []
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT event_id, run_id, step_index, event_type, stage_key, message,
                       progress_pct, details_json, created_at
                FROM job_pipeline_step_events
                WHERE run_id = %s
                ORDER BY created_at ASC
                """,
                (rid,),
            ).fetchall()
            out: List[JobPipelineStepEvent] = []
            for row in rows or []:
                try:
                    details_json = json.loads(row[7]) if row[7] else {}
                except Exception:
                    details_json = {}
                out.append(
                    JobPipelineStepEvent(
                        event_id=row[0],
                        run_id=row[1],
                        step_index=row[2],
                        event_type=row[3],
                        stage_key=row[4],
                        message=row[5],
                        progress_pct=row[6],
                        details=details_json if isinstance(details_json, dict) else {},
                        created_at=row[8],
                    )
                )
            return out
        finally:
            con.close()

    def update_run(self, run: JobPipelineRun) -> None:
        now = _utc_now()
        run.updated_at = now
        con = _connect()
        try:
            con.execute(
                """
                UPDATE job_pipeline_runs
                SET status = %s,
                    current_step_index = %s,
                    progress_pct = %s,
                    current_step_id = %s,
                    current_stage_key = %s,
                    current_message = %s,
                    context_json = %s,
                    last_error = %s,
                    updated_at = %s
                WHERE run_id = %s
                """,
                (
                    run.status,
                    int(run.current_step_index or 0),
                    int(run.progress_pct or 0),
                    run.current_step_id,
                    run.current_stage_key,
                    run.current_message,
                    json.dumps(run.context if isinstance(run.context, dict) else {}),
                    run.last_error,
                    now,
                    run.run_id,
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_step(self, step: JobPipelineStep) -> None:
        now = _utc_now()
        step.updated_at = now
        con = _connect()
        try:
            con.execute(
                """
                UPDATE job_pipeline_steps
                SET status = %s,
                    progress_pct = %s,
                    current_stage_key = %s,
                    current_message = %s,
                    attempt = %s,
                    celery_task_id = %s,
                    linked_job_type = %s,
                    linked_job_id = %s,
                    output_json = %s,
                    last_error = %s,
                    started_at = %s,
                    completed_at = %s,
                    updated_at = %s
                WHERE run_id = %s AND step_index = %s
                """,
                (
                    step.status,
                    int(step.progress_pct or 0),
                    step.current_stage_key,
                    step.current_message,
                    int(step.attempt or 0),
                    step.celery_task_id,
                    step.linked_job_type,
                    step.linked_job_id,
                    json.dumps(step.output if isinstance(step.output, dict) else {}),
                    step.last_error,
                    step.started_at,
                    step.completed_at,
                    now,
                    step.run_id,
                    int(step.step_index),
                ),
            )
            con.commit()
        finally:
            con.close()

    def append_event(self, event: JobPipelineStepEvent) -> None:
        eid = (event.event_id or "").strip() or _new_pipeline_event_id()
        created = (event.created_at or "").strip() or _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO job_pipeline_step_events(
                  event_id, run_id, step_index, event_type, stage_key, message,
                  progress_pct, details_json, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    eid,
                    event.run_id,
                    event.step_index,
                    event.event_type,
                    event.stage_key,
                    event.message,
                    event.progress_pct,
                    json.dumps(event.details if isinstance(event.details, dict) else {}),
                    created,
                ),
            )
            con.commit()
        finally:
            con.close()


class PostgresConversationRepository:
    _SESSION_SELECT = """
        s.session_id, s.bot_id, s.org_id, s.channel, s.status, s.title, s.site_url, s.site_title,
        s.message_count, s.started_at, s.last_active_at, s.ended_at, s.user_agent, s.ip,
        COALESCE(h.assistant_state, 'bot') AS assistant_state,
        CASE
          WHEN COALESCE(h.assistant_state, 'bot') IN ('awaiting_support_details', 'human_handoff') THEN TRUE
          ELSE FALSE
        END AS handoff_active,
        esc.escalation_id AS support_request_id,
        esc.status AS support_request_status
    """
    _SESSION_SUPPORT_JOIN = """
        LEFT JOIN conversation_session_handoffs h ON h.session_id = s.session_id
        LEFT JOIN LATERAL (
          SELECT e.escalation_id, e.status
          FROM conversation_escalations e
          WHERE e.session_id = s.session_id
          ORDER BY e.created_at DESC, e.escalation_id DESC
          LIMIT 1
        ) esc ON TRUE
    """

    @staticmethod
    def _conversation_session_from_row(row) -> ConversationSession:
        return ConversationSession(
            session_id=row[0],
            bot_id=row[1],
            org_id=row[2],
            channel=row[3],
            status=row[4],
            title=row[5],
            site_url=row[6],
            site_title=row[7],
            message_count=row[8] or 0,
            started_at=row[9],
            last_active_at=row[10],
            ended_at=row[11],
            user_agent=row[12],
            ip=row[13],
            assistant_state=(row[14] or "bot") if len(row) > 14 else "bot",
            handoff_active=bool(row[15]) if len(row) > 15 else False,
            support_request_id=row[16] if len(row) > 16 else None,
            support_request_status=row[17] if len(row) > 17 else None,
        )

    @staticmethod
    def _contact_from_row(row) -> ConversationChannelContact:
        metadata_raw = row[5] if len(row) > 5 else "{}"
        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return ConversationChannelContact(
            contact_id=row[0],
            bot_id=row[1],
            channel=row[2],
            external_user_id=row[3],
            display_name=row[4],
            metadata=metadata,
            current_session_id=row[6] if len(row) > 6 else None,
            created_at=row[7],
            updated_at=row[8],
        )

    @staticmethod
    def _handoff_from_row(row) -> ConversationSessionHandoff:
        return ConversationSessionHandoff(
            handoff_id=row[0],
            bot_id=row[1],
            session_id=row[2],
            contact_id=row[3],
            source_channel=row[4],
            assistant_state=row[5],
            support_request_id=row[6],
            started_by=row[7],
            ended_reason=row[8],
            created_at=row[9],
            updated_at=row[10],
            ended_at=row[11],
        )

    def create_session(
        self,
        *,
        bot_id: str,
        org_id: str,
        channel: str,
        site_url: Optional[str],
        site_title: Optional[str],
        user_agent: Optional[str],
        ip: Optional[str],
    ) -> ConversationSession:
        bid = (bot_id or "").strip()
        oid = (org_id or "").strip()
        if not bid or not oid:
            raise ValueError("bot_id and org_id are required")
        now = _utc_now()
        session_id = _new_conversation_id()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO conversation_sessions(
                  session_id, bot_id, org_id, channel, status, title, site_url, site_title,
                  message_count, started_at, last_active_at, ended_at, user_agent, ip
                )
                VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, 0, %s, %s, NULL, %s, %s)
                """,
                (
                    session_id,
                    bid,
                    oid,
                    channel,
                    "active",
                    site_url,
                    site_title,
                    now,
                    now,
                    user_agent,
                    ip,
                ),
            )
            con.commit()
            return ConversationSession(
                session_id=session_id,
                bot_id=bid,
                org_id=oid,
                channel=channel,
                status="active",
                title=None,
                site_url=site_url,
                site_title=site_title,
                message_count=0,
                started_at=now,
                last_active_at=now,
                ended_at=None,
                user_agent=user_agent,
                ip=ip,
            )
        finally:
            con.close()

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT
                  """
                + self._SESSION_SELECT
                + """
                FROM conversation_sessions s
                """
                + self._SESSION_SUPPORT_JOIN
                + """
                WHERE s.session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return self._conversation_session_from_row(row)
        finally:
            con.close()

    def list_sessions_for_bot(self, bot_id: str, *, limit: int = 50, before: Optional[str] = None) -> List[ConversationSession]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        lim = max(1, min(int(limit or 50), 200))
        before_ts = None
        before_id = None
        if before:
            if "|" in before:
                before_ts, before_id = before.split("|", 1)
            else:
                before_ts = before
        con = _connect()
        try:
            if before_ts:
                rows = con.execute(
                    """
                    SELECT
                      """
                    + self._SESSION_SELECT
                    + """
                    FROM conversation_sessions s
                    """
                    + self._SESSION_SUPPORT_JOIN
                    + """
                    WHERE s.bot_id = %s AND (s.last_active_at, s.session_id) < (%s, %s)
                    ORDER BY s.last_active_at DESC, s.session_id DESC
                    LIMIT %s
                    """,
                    (bid, before_ts, before_id or "", lim),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT
                      """
                    + self._SESSION_SELECT
                    + """
                    FROM conversation_sessions s
                    """
                    + self._SESSION_SUPPORT_JOIN
                    + """
                    WHERE s.bot_id = %s
                    ORDER BY s.last_active_at DESC, s.session_id DESC
                    LIMIT %s
                    """,
                    (bid, lim),
                ).fetchall()
            return [self._conversation_session_from_row(row) for row in rows or []]
        finally:
            con.close()

    def count_sessions_for_bot(self, bot_id: str) -> int:
        bid = (bot_id or "").strip()
        if not bid:
            return 0
        con = _connect()
        try:
            row = con.execute(
                "SELECT COUNT(1) FROM conversation_sessions WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def touch_session(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE conversation_sessions
                SET last_active_at = %s
                WHERE session_id = %s
                """,
                (now, sid),
            )
            con.commit()
        finally:
            con.close()

    def update_session_title(self, session_id: str, title: Optional[str]) -> bool:
        sid = (session_id or "").strip()
        next_title = (title or "").strip()
        if not sid or not next_title:
            return False
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE conversation_sessions
                SET title = %s
                WHERE session_id = %s
                """,
                (next_title, sid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def end_session(self, session_id: str, status: str = "ended") -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE conversation_sessions
                SET status = %s, ended_at = %s, last_active_at = %s
                WHERE session_id = %s
                """,
                (status, now, now, sid),
            )
            con.commit()
        finally:
            con.close()

    def add_message(
        self,
        *,
        session_id: str,
        bot_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        sender_name: Optional[str] = None,
    ) -> ConversationMessage:
        sid = (session_id or "").strip()
        bid = (bot_id or "").strip()
        if not sid or not bid:
            raise ValueError("session_id and bot_id are required")
        now = _utc_now()
        msg_id = _new_message_id()
        payload = json.dumps(citations or [])
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO conversation_messages(
                  message_id, session_id, bot_id, role, sender_name, content, citations, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (msg_id, sid, bid, role, sender_name, content or "", payload, now),
            )
            con.execute(
                """
                UPDATE conversation_sessions
                SET message_count = message_count + 1,
                    last_active_at = %s,
                    title = CASE
                      WHEN (title IS NULL OR title = '') AND %s = 'user' AND %s <> '' THEN %s
                      ELSE title
                    END
                WHERE session_id = %s
                """,
                (now, role, content or "", content or "", sid),
            )
            con.commit()
            return ConversationMessage(
                message_id=msg_id,
                session_id=sid,
                bot_id=bid,
                role=role,
                sender_name=sender_name,
                content=content or "",
                citations=citations or [],
                created_at=now,
            )
        finally:
            con.close()

    def list_messages(self, session_id: str, *, limit: int = 200) -> List[ConversationMessage]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        lim = max(1, min(int(limit or 200), 500))
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT message_id, session_id, bot_id, role, sender_name, content, citations, created_at
                FROM conversation_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (sid, lim),
            ).fetchall()
            result = []
            for row in rows or []:
                citations_raw = row[6] if len(row) > 6 else "[]"
                try:
                    citations = json.loads(citations_raw) if isinstance(citations_raw, str) else (citations_raw or [])
                except (TypeError, ValueError):
                    citations = []
                result.append(
                    ConversationMessage(
                        message_id=row[0],
                        session_id=row[1],
                        bot_id=row[2],
                        role=row[3],
                        sender_name=row[4],
                        content=row[5],
                        citations=citations if isinstance(citations, list) else [],
                        created_at=row[7],
                    )
                )
            return result
        finally:
            con.close()

    def list_messages_recent(self, session_id: str, *, limit: int = 20) -> List[ConversationMessage]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        lim = max(1, min(int(limit or 20), 200))
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT message_id, session_id, bot_id, role, sender_name, content, citations, created_at
                FROM conversation_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (sid, lim),
            ).fetchall()
            result = []
            for row in rows or []:
                citations_raw = row[6] if len(row) > 6 else "[]"
                try:
                    citations = json.loads(citations_raw) if isinstance(citations_raw, str) else (citations_raw or [])
                except (TypeError, ValueError):
                    citations = []
                result.append(
                    ConversationMessage(
                        message_id=row[0],
                        session_id=row[1],
                        bot_id=row[2],
                        role=row[3],
                        sender_name=row[4],
                        content=row[5],
                        citations=citations if isinstance(citations, list) else [],
                        created_at=row[7],
                    )
                )
            result.reverse()
            return result
        finally:
            con.close()

    def get_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
    ) -> Optional[ConversationChannelContact]:
        bid = (bot_id or "").strip()
        ch = (channel or "").strip().lower()
        external_id = (external_user_id or "").strip()
        if not bid or not ch or not external_id:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT contact_id, bot_id, channel, external_user_id, display_name, metadata_json,
                       current_session_id, created_at, updated_at
                FROM conversation_channel_contacts
                WHERE bot_id = %s AND channel = %s AND external_user_id = %s
                """,
                (bid, ch, external_id),
            ).fetchone()
            if not row:
                return None
            return self._contact_from_row(row)
        finally:
            con.close()

    def upsert_channel_contact(
        self,
        *,
        bot_id: str,
        channel: str,
        external_user_id: str,
        current_session_id: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationChannelContact:
        bid = (bot_id or "").strip()
        ch = (channel or "").strip().lower()
        external_id = (external_user_id or "").strip()
        name = (display_name or "").strip() or None
        if not bid or not ch or not external_id:
            raise ValueError("bot_id, channel, and external_user_id are required")
        now = _utc_now()
        con = _connect()
        try:
            existing_row = con.execute(
                """
                SELECT contact_id, bot_id, channel, external_user_id, display_name, metadata_json,
                       current_session_id, created_at, updated_at
                FROM conversation_channel_contacts
                WHERE bot_id = %s AND channel = %s AND external_user_id = %s
                """,
                (bid, ch, external_id),
            ).fetchone()
            if existing_row:
                existing = self._contact_from_row(existing_row)
                next_metadata = dict(existing.metadata or {})
                if isinstance(metadata, dict):
                    next_metadata.update(metadata)
                next_name = name or existing.display_name
                next_session_id = (current_session_id or "").strip() or existing.current_session_id
                con.execute(
                    """
                    UPDATE conversation_channel_contacts
                    SET display_name = %s,
                        metadata_json = %s,
                        current_session_id = %s,
                        updated_at = %s
                    WHERE contact_id = %s
                    """,
                    (
                        next_name,
                        json.dumps(next_metadata),
                        next_session_id,
                        now,
                        existing.contact_id,
                    ),
                )
                con.commit()
                return ConversationChannelContact(
                    contact_id=existing.contact_id,
                    bot_id=existing.bot_id,
                    channel=existing.channel,
                    external_user_id=existing.external_user_id,
                    display_name=next_name,
                    metadata=next_metadata,
                    current_session_id=next_session_id,
                    created_at=existing.created_at,
                    updated_at=now,
                )

            contact_id = _new_contact_id()
            metadata_payload = metadata if isinstance(metadata, dict) else {}
            next_session_id = (current_session_id or "").strip() or None
            con.execute(
                """
                INSERT INTO conversation_channel_contacts(
                  contact_id, bot_id, channel, external_user_id, display_name, metadata_json,
                  current_session_id, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    contact_id,
                    bid,
                    ch,
                    external_id,
                    name,
                    json.dumps(metadata_payload),
                    next_session_id,
                    now,
                    now,
                ),
            )
            con.commit()
            return ConversationChannelContact(
                contact_id=contact_id,
                bot_id=bid,
                channel=ch,
                external_user_id=external_id,
                display_name=name,
                metadata=metadata_payload,
                current_session_id=next_session_id,
                created_at=now,
                updated_at=now,
            )
        finally:
            con.close()

    def get_session_handoff(self, session_id: str) -> Optional[ConversationSessionHandoff]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state,
                       support_request_id, started_by, ended_reason, created_at, updated_at, ended_at
                FROM conversation_session_handoffs
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return self._handoff_from_row(row)
        finally:
            con.close()

    def upsert_session_handoff(
        self,
        *,
        bot_id: str,
        session_id: str,
        assistant_state: str,
        contact_id: Optional[str] = None,
        source_channel: Optional[str] = None,
        support_request_id: Optional[str] = None,
        started_by: Optional[str] = None,
        ended_reason: Optional[str] = None,
        ended_at: Optional[str] = None,
    ) -> ConversationSessionHandoff:
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        state = (assistant_state or "").strip() or "bot"
        if not bid or not sid:
            raise ValueError("bot_id and session_id are required")
        now = _utc_now()
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state,
                       support_request_id, started_by, ended_reason, created_at, updated_at, ended_at
                FROM conversation_session_handoffs
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if row:
                existing = self._handoff_from_row(row)
                next_contact_id = contact_id if contact_id is not None else existing.contact_id
                next_source_channel = source_channel if source_channel is not None else existing.source_channel
                next_support_request_id = support_request_id if support_request_id is not None else existing.support_request_id
                next_started_by = started_by if started_by is not None else existing.started_by
                con.execute(
                    """
                    UPDATE conversation_session_handoffs
                    SET bot_id = %s,
                        contact_id = %s,
                        source_channel = %s,
                        assistant_state = %s,
                        support_request_id = %s,
                        started_by = %s,
                        ended_reason = %s,
                        updated_at = %s,
                        ended_at = %s
                    WHERE session_id = %s
                    """,
                    (
                        bid,
                        next_contact_id,
                        next_source_channel,
                        state,
                        next_support_request_id,
                        next_started_by,
                        ended_reason,
                        now,
                        ended_at,
                        sid,
                    ),
                )
                con.commit()
                return ConversationSessionHandoff(
                    handoff_id=existing.handoff_id,
                    bot_id=bid,
                    session_id=sid,
                    contact_id=next_contact_id,
                    source_channel=next_source_channel,
                    assistant_state=state,
                    support_request_id=next_support_request_id,
                    started_by=next_started_by,
                    ended_reason=ended_reason,
                    created_at=existing.created_at,
                    updated_at=now,
                    ended_at=ended_at,
                )

            handoff_id = _new_handoff_id()
            con.execute(
                """
                INSERT INTO conversation_session_handoffs(
                  handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state,
                  support_request_id, started_by, ended_reason, created_at, updated_at, ended_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    handoff_id,
                    bid,
                    sid,
                    contact_id,
                    source_channel,
                    state,
                    support_request_id,
                    started_by,
                    ended_reason,
                    now,
                    now,
                    ended_at,
                ),
            )
            con.commit()
            return ConversationSessionHandoff(
                handoff_id=handoff_id,
                bot_id=bid,
                session_id=sid,
                contact_id=contact_id,
                source_channel=source_channel,
                assistant_state=state,
                support_request_id=support_request_id,
                started_by=started_by,
                ended_reason=ended_reason,
                created_at=now,
                updated_at=now,
                ended_at=ended_at,
            )
        finally:
            con.close()

    @staticmethod
    def _escalation_record_from_row(row) -> EscalationRecord:
        return EscalationRecord(
            escalation_id=row[0],
            bot_id=row[1],
            session_id=row[2],
            visitor_email=row[3],
            status=row[5],
            created_at=row[6],
            details=row[4],
            session_title=row[7],
            site_url=row[8],
            site_title=row[9],
            last_active_at=row[10],
            session_status=row[11],
            notification_read_at=row[12] if len(row) > 12 else None,
            notification_is_unread=bool(row[13]) if len(row) > 13 else False,
        )

    def get_open_escalation_for_session(self, bot_id: str, session_id: str) -> Optional[EscalationRecord]:
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        if not bid or not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                       s.title, s.site_url, s.site_title, s.last_active_at, s.status
                FROM conversation_escalations e
                LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                WHERE e.bot_id = %s AND e.session_id = %s AND e.status = 'open'
                ORDER BY e.created_at DESC, e.escalation_id DESC
                LIMIT 1
                """,
                (bid, sid),
            ).fetchone()
            if not row:
                return None
            return self._escalation_record_from_row(row)
        finally:
            con.close()

    def create_escalation(self, *, bot_id: str, session_id: str, visitor_email: str, details: Optional[str] = None) -> EscalationRecord:
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        email = (visitor_email or "").strip().lower()
        if not bid or not sid or not email:
            raise ValueError("bot_id, session_id, and visitor_email are required")
        eid = _new_escalation_id()
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO conversation_escalations(
                  escalation_id, bot_id, session_id, visitor_email, details, status, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (eid, bid, sid, email, details, "open", now),
            )
            con.commit()
            return EscalationRecord(
                escalation_id=eid,
                bot_id=bid,
                session_id=sid,
                visitor_email=email,
                created_at=now,
                status="open",
            )
        finally:
            con.close()

    def count_escalations_for_bot(self, bot_id: str) -> int:
        bid = (bot_id or "").strip()
        if not bid:
            return 0
        con = _connect()
        try:
            row = con.execute(
                "SELECT COUNT(1) FROM conversation_escalations WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def count_open_escalations_for_bot(self, bot_id: str) -> int:
        bid = (bot_id or "").strip()
        if not bid:
            return 0
        con = _connect()
        try:
            row = con.execute(
                "SELECT COUNT(1) FROM conversation_escalations WHERE bot_id = %s AND status = %s",
                (bid, "open"),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def count_unread_escalations_for_bot(self, bot_id: str, user_id: str) -> int:
        bid = (bot_id or "").strip()
        uid = (user_id or "").strip()
        if not bid or not uid:
            return 0
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT COUNT(1)
                FROM conversation_escalations e
                LEFT JOIN conversation_escalation_reads r
                  ON r.escalation_id = e.escalation_id AND r.user_id = %s
                WHERE e.bot_id = %s AND r.read_at IS NULL
                """,
                (uid, bid),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()

    def list_escalations_for_bot(
        self, bot_id: str, *, limit: int = 10, before: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[EscalationRecord]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        lim = max(1, min(int(limit or 10), 200))
        before_ts = None
        before_id = None
        if before:
            if "|" in before:
                before_ts, before_id = before.split("|", 1)
            else:
                before_ts = before
        uid = (user_id or "").strip()
        con = _connect()
        try:
            if uid:
                if before_ts:
                    rows = con.execute(
                        """
                        SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                               s.title, s.site_url, s.site_title, s.last_active_at, s.status,
                               r.read_at,
                               CASE WHEN r.read_at IS NULL THEN TRUE ELSE FALSE END AS is_unread
                        FROM conversation_escalations e
                        LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                        LEFT JOIN conversation_escalation_reads r
                          ON r.escalation_id = e.escalation_id AND r.user_id = %s
                        WHERE e.bot_id = %s AND (e.created_at, e.escalation_id) < (%s, %s)
                        ORDER BY e.created_at DESC, e.escalation_id DESC
                        LIMIT %s
                        """,
                        (uid, bid, before_ts, before_id or "", lim),
                    ).fetchall()
                else:
                    rows = con.execute(
                        """
                        SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                               s.title, s.site_url, s.site_title, s.last_active_at, s.status,
                               r.read_at,
                               CASE WHEN r.read_at IS NULL THEN TRUE ELSE FALSE END AS is_unread
                        FROM conversation_escalations e
                        LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                        LEFT JOIN conversation_escalation_reads r
                          ON r.escalation_id = e.escalation_id AND r.user_id = %s
                        WHERE e.bot_id = %s
                        ORDER BY e.created_at DESC, e.escalation_id DESC
                        LIMIT %s
                        """,
                        (uid, bid, lim),
                    ).fetchall()
            else:
                if before_ts:
                    rows = con.execute(
                        """
                        SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                               s.title, s.site_url, s.site_title, s.last_active_at, s.status
                        FROM conversation_escalations e
                        LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                        WHERE e.bot_id = %s AND (e.created_at, e.escalation_id) < (%s, %s)
                        ORDER BY e.created_at DESC, e.escalation_id DESC
                        LIMIT %s
                        """,
                        (bid, before_ts, before_id or "", lim),
                    ).fetchall()
                else:
                    rows = con.execute(
                        """
                        SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                               s.title, s.site_url, s.site_title, s.last_active_at, s.status
                        FROM conversation_escalations e
                        LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                        WHERE e.bot_id = %s
                        ORDER BY e.created_at DESC, e.escalation_id DESC
                        LIMIT %s
                        """,
                        (bid, lim),
                    ).fetchall()
            return [self._escalation_record_from_row(row) for row in rows or []]
        finally:
            con.close()

    def get_escalation_for_session(self, bot_id: str, session_id: str, *, user_id: Optional[str] = None) -> Optional[EscalationRecord]:
        bid = (bot_id or "").strip()
        sid = (session_id or "").strip()
        if not bid or not sid:
            return None
        uid = (user_id or "").strip()
        con = _connect()
        try:
            if uid:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status,
                           r.read_at,
                           CASE WHEN r.read_at IS NULL THEN TRUE ELSE FALSE END AS is_unread
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    LEFT JOIN conversation_escalation_reads r
                      ON r.escalation_id = e.escalation_id AND r.user_id = %s
                    WHERE e.bot_id = %s AND e.session_id = %s
                    ORDER BY e.created_at DESC, e.escalation_id DESC
                    LIMIT 1
                    """,
                    (uid, bid, sid),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    WHERE e.bot_id = %s AND e.session_id = %s
                    ORDER BY e.created_at DESC, e.escalation_id DESC
                    LIMIT 1
                    """,
                    (bid, sid),
                ).fetchone()
            if not row:
                return None
            return self._escalation_record_from_row(row)
        finally:
            con.close()

    def get_latest_escalation_for_visitor(self, bot_id: str, visitor_email: str, *, user_id: Optional[str] = None) -> Optional[EscalationRecord]:
        bid = (bot_id or "").strip()
        email = (visitor_email or "").strip().lower()
        if not bid or not email:
            return None
        uid = (user_id or "").strip()
        con = _connect()
        try:
            if uid:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status,
                           r.read_at,
                           CASE WHEN r.read_at IS NULL THEN TRUE ELSE FALSE END AS is_unread
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    LEFT JOIN conversation_escalation_reads r
                      ON r.escalation_id = e.escalation_id AND r.user_id = %s
                    WHERE e.bot_id = %s AND e.visitor_email = %s
                    ORDER BY e.created_at DESC, e.escalation_id DESC
                    LIMIT 1
                    """,
                    (uid, bid, email),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    WHERE e.bot_id = %s AND e.visitor_email = %s
                    ORDER BY e.created_at DESC, e.escalation_id DESC
                    LIMIT 1
                    """,
                    (bid, email),
                ).fetchone()
            if not row:
                return None
            return self._escalation_record_from_row(row)
        finally:
            con.close()

    def get_escalation_by_id(self, bot_id: str, escalation_id: str, *, user_id: Optional[str] = None) -> Optional[EscalationRecord]:
        bid = (bot_id or "").strip()
        eid = (escalation_id or "").strip()
        if not bid or not eid:
            return None
        uid = (user_id or "").strip()
        con = _connect()
        try:
            if uid:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status,
                           r.read_at,
                           CASE WHEN r.read_at IS NULL THEN TRUE ELSE FALSE END AS is_unread
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    LEFT JOIN conversation_escalation_reads r
                      ON r.escalation_id = e.escalation_id AND r.user_id = %s
                    WHERE e.bot_id = %s AND e.escalation_id = %s
                    LIMIT 1
                    """,
                    (uid, bid, eid),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT e.escalation_id, e.bot_id, e.session_id, e.visitor_email, e.details, e.status, e.created_at,
                           s.title, s.site_url, s.site_title, s.last_active_at, s.status
                    FROM conversation_escalations e
                    LEFT JOIN conversation_sessions s ON s.session_id = e.session_id
                    WHERE e.bot_id = %s AND e.escalation_id = %s
                    LIMIT 1
                    """,
                    (bid, eid),
                ).fetchone()
            if not row:
                return None
            return self._escalation_record_from_row(row)
        finally:
            con.close()

    def update_escalation_status(self, bot_id: str, escalation_id: str, status: str) -> bool:
        bid = (bot_id or "").strip()
        eid = (escalation_id or "").strip()
        if not bid or not eid:
            return False
        con = _connect()
        try:
            result = con.execute(
                "UPDATE conversation_escalations SET status = %s WHERE bot_id = %s AND escalation_id = %s",
                (status, bid, eid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def mark_escalation_read(self, bot_id: str, escalation_id: str, user_id: str) -> bool:
        bid = (bot_id or "").strip()
        eid = (escalation_id or "").strip()
        uid = (user_id or "").strip()
        if not bid or not eid or not uid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            exists = con.execute(
                "SELECT 1 FROM conversation_escalations WHERE bot_id = %s AND escalation_id = %s",
                (bid, eid),
            ).fetchone()
            if not exists:
                return False
            con.execute(
                """
                INSERT INTO conversation_escalation_reads(escalation_id, user_id, read_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (escalation_id, user_id)
                DO UPDATE SET read_at = EXCLUDED.read_at
                """,
                (eid, uid, now),
            )
            con.commit()
            return True
        finally:
            con.close()


class PostgresAnalyticsRepository:
    """Rollups + analytics queries for the dashboard.

    Phase 1 uses a recompute/backfill endpoint; rollups are stored in *_daily tables.
    """

    def recompute_bot_rollups(self, *, org_id: str, bot_id: str, start_day: date, end_day: date) -> None:
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        if not oid or not bid:
            raise ValueError("org_id and bot_id are required")
        if end_day < start_day:
            raise ValueError("end_day must be >= start_day")

        # inclusive date range
        days: List[date] = []
        d = start_day
        while d <= end_day:
            days.append(d)
            d = d + timedelta(days=1)

        # We'll compute over UTC day boundaries.
        start_iso = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc).isoformat()
        end_exclusive = end_day + timedelta(days=1)
        end_iso = datetime(end_exclusive.year, end_exclusive.month, end_exclusive.day, tzinfo=timezone.utc).isoformat()

        con = _connect()
        try:
            # Clear existing rollups for days in range (idempotent).
            day_keys = [dd.isoformat() for dd in days]
            con.execute(
                "DELETE FROM bot_usage_daily WHERE org_id=%s AND bot_id=%s AND day = ANY(%s)",
                (oid, bid, day_keys),
            )
            con.execute(
                "DELETE FROM bot_sources_daily WHERE org_id=%s AND bot_id=%s AND day = ANY(%s)",
                (oid, bid, day_keys),
            )
            con.execute(
                "DELETE FROM bot_topics_daily WHERE org_id=%s AND bot_id=%s AND day = ANY(%s)",
                (oid, bid, day_keys),
            )

            # Usage: conversations/day from sessions.started_at
            rows = con.execute(
                """
                SELECT substring(started_at, 1, 10) AS day, COUNT(1)
                FROM conversation_sessions
                WHERE org_id=%s AND bot_id=%s AND started_at >= %s AND started_at < %s
                GROUP BY 1
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchall()
            conv_by_day = {r[0]: int(r[1] or 0) for r in (rows or [])}

            # Usage: messages/day split by role from messages.created_at joined to sessions for org_id.
            rows = con.execute(
                """
                SELECT substring(m.created_at, 1, 10) AS day,
                       SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS user_msgs,
                       SUM(CASE WHEN m.role='bot' THEN 1 ELSE 0 END)  AS bot_msgs
                FROM conversation_messages m
                JOIN conversation_sessions s ON s.session_id = m.session_id
                WHERE s.org_id=%s AND m.bot_id=%s AND m.created_at >= %s AND m.created_at < %s
                GROUP BY 1
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchall()
            msgs_by_day = {r[0]: (int(r[1] or 0), int(r[2] or 0)) for r in (rows or [])}

            # Escalations/day from escalations.created_at joined to sessions for org_id.
            rows = con.execute(
                """
                SELECT substring(e.created_at, 1, 10) AS day, COUNT(1)
                FROM conversation_escalations e
                JOIN conversation_sessions s ON s.session_id = e.session_id
                WHERE s.org_id=%s AND e.bot_id=%s AND e.created_at >= %s AND e.created_at < %s
                GROUP BY 1
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchall()
            esc_by_day = {r[0]: int(r[1] or 0) for r in (rows or [])}

            # Approx unique visitors: distinct ip per day (may be null/empty).
            rows = con.execute(
                """
                SELECT substring(started_at, 1, 10) AS day, COUNT(DISTINCT ip)
                FROM conversation_sessions
                WHERE org_id=%s AND bot_id=%s AND started_at >= %s AND started_at < %s AND ip IS NOT NULL AND ip <> ''
                GROUP BY 1
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchall()
            uniq_by_day = {r[0]: int(r[1] or 0) for r in (rows or [])}

            for dk in day_keys:
                user_msgs, bot_msgs = msgs_by_day.get(dk, (0, 0))
                con.execute(
                    """
                    INSERT INTO bot_usage_daily(org_id, bot_id, day, conversations, messages_user, messages_bot, escalations, unique_visitors_est)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        oid,
                        bid,
                        dk,
                        conv_by_day.get(dk, 0),
                        user_msgs,
                        bot_msgs,
                        esc_by_day.get(dk, 0),
                        uniq_by_day.get(dk, 0),
                    ),
                )

            # Top sources/day from bot message citations JSON (simple URL counts).
            rows = con.execute(
                """
                SELECT substring(m.created_at, 1, 10) AS day, m.citations
                FROM conversation_messages m
                JOIN conversation_sessions s ON s.session_id = m.session_id
                WHERE s.org_id=%s AND m.bot_id=%s AND m.role='bot' AND m.created_at >= %s AND m.created_at < %s
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchall()
            src_counts: Dict[Tuple[str, str], int] = {}
            for day_s, citations_raw in rows or []:
                try:
                    citations = json.loads(citations_raw) if isinstance(citations_raw, str) else (citations_raw or [])
                except (TypeError, ValueError):
                    citations = []
                if not isinstance(citations, list):
                    continue
                for c in citations:
                    if not isinstance(c, dict):
                        continue
                    u = str(c.get("url") or "").strip()
                    if not u:
                        continue
                    key = (day_s, u)
                    src_counts[key] = src_counts.get(key, 0) + 1
            for (day_s, u), ct in src_counts.items():
                con.execute(
                    """
                    INSERT INTO bot_sources_daily(org_id, bot_id, day, source_url, count)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (oid, bid, day_s, u, int(ct)),
                )

            # Topics/day: LLM-based clustering.
            # DISABLED for now per user request.
            # rows = con.execute(
            #     """
            #     SELECT substring(started_at, 1, 10) AS day, COALESCE(NULLIF(title,''), '')
            #     FROM conversation_sessions
            #     WHERE org_id=%s AND bot_id=%s AND started_at >= %s AND started_at < %s
            #     """,
            #     (oid, bid, start_iso, end_iso),
            # ).fetchall()
            #
            # day_titles: Dict[str, List[str]] = {}
            # for day_s, title in rows or []:
            #     t = (title or "").strip()
            #     if not t:
            #         continue
            #     if day_s not in day_titles:
            #         day_titles[day_s] = []
            #     day_titles[day_s].append(t)
            #
            # for day_s, titles in day_titles.items():
            #     if not titles:
            #         continue
            #     try:
            #         daily_topics = extract_topics_from_titles(titles)
            #         for topic, ct in daily_topics.items():
            #             con.execute(
            #                 """
            #                 INSERT INTO bot_topics_daily(org_id, bot_id, day, topic, count)
            #                 VALUES (%s,%s,%s,%s,%s)
            #                 """,
            #                 (oid, bid, day_s, topic, int(ct)),
            #             )
            #     except Exception as e:
            #         print(f"[Analytics] Error extracting topics for {day_s}: {e}")

            # Watermark: set to end_iso (exclusive).
            con.execute(
                """
                INSERT INTO rollup_watermarks(bot_id, last_processed_at)
                VALUES (%s, %s)
                ON CONFLICT (bot_id) DO UPDATE SET last_processed_at = EXCLUDED.last_processed_at
                """,
                (bid, end_iso),
            )
            con.commit()
        finally:
            con.close()

    def get_usage_timeseries(self, *, org_id: str, bot_id: str, start_day: date, end_day: date) -> List[Dict[str, Any]]:
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT day, conversations, messages_user, messages_bot, escalations, unique_visitors_est
                FROM bot_usage_daily
                WHERE org_id=%s AND bot_id=%s AND day >= %s AND day <= %s
                ORDER BY day ASC
                """,
                (oid, bid, start_day.isoformat(), end_day.isoformat()),
            ).fetchall()
            return [
                {
                    "day": r[0],
                    "conversations": int(r[1] or 0),
                    "messages_user": int(r[2] or 0),
                    "messages_bot": int(r[3] or 0),
                    "escalations": int(r[4] or 0),
                    "unique_visitors_est": int(r[5] or 0),
                }
                for r in (rows or [])
            ]
        finally:
            con.close()

    def get_top_sources(self, *, org_id: str, bot_id: str, start_day: date, end_day: date, limit: int = 10) -> List[Dict[str, Any]]:
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        lim = max(1, min(int(limit or 10), 50))
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT source_url, SUM(count) AS total
                FROM bot_sources_daily
                WHERE org_id=%s AND bot_id=%s AND day >= %s AND day <= %s
                GROUP BY source_url
                ORDER BY total DESC
                LIMIT %s
                """,
                (oid, bid, start_day.isoformat(), end_day.isoformat(), lim),
            ).fetchall()
            return [{"source_url": r[0], "count": int(r[1] or 0)} for r in (rows or [])]
        finally:
            con.close()

    def get_top_topics(self, *, org_id: str, bot_id: str, start_day: date, end_day: date, limit: int = 20) -> List[Dict[str, Any]]:
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        lim = max(1, min(int(limit or 20), 100))
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT topic, SUM(count) AS total
                FROM bot_topics_daily
                WHERE org_id=%s AND bot_id=%s AND day >= %s AND day <= %s
                GROUP BY topic
                ORDER BY total DESC
                LIMIT %s
                """,
                (oid, bid, start_day.isoformat(), end_day.isoformat(), lim),
            ).fetchall()
            return [{"topic": r[0], "count": int(r[1] or 0)} for r in (rows or [])]
        finally:
            con.close()

    def get_feedback_counts(self, *, org_id: str, bot_id: str, start_iso: str, end_iso: str) -> Dict[str, int]:
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT
                  SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS pos,
                  SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS neg
                FROM conversation_feedback
                WHERE org_id=%s AND bot_id=%s AND created_at >= %s AND created_at < %s
                """,
                (oid, bid, start_iso, end_iso),
            ).fetchone()
            return {"positive": int((row[0] if row else 0) or 0), "negative": int((row[1] if row else 0) or 0)}
        finally:
            con.close()

    def insert_feedback(
        self,
        *,
        feedback_id: str,
        org_id: str,
        bot_id: str,
        session_id: str,
        message_id: Optional[str],
        rating: int,
        comment: Optional[str],
        created_at: str,
    ) -> None:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO conversation_feedback(feedback_id, org_id, bot_id, session_id, message_id, rating, comment, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (feedback_id, org_id, bot_id, session_id, message_id, int(rating), comment, created_at),
            )
            con.commit()
        finally:
            con.close()


def _new_topic_id() -> str:
    return "topic_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


@dataclass
class ExtractedTopic:
    topic_id: str
    org_id: str
    bot_id: str
    topic: str
    category: Optional[str]
    confidence: float
    source_urls: List[str]
    occurrence_count: int
    is_active: bool
    extracted_at: str
    updated_at: str
    source_url: Optional[str] = None
    origin: str = "extracted"


class PostgresExtractedTopicRepository:
    """Repository for managing extracted topics from website content."""

    def save_extracted_topics(
        self,
        *,
        org_id: str,
        bot_id: str,
        topics: List[Dict[str, Any]],
    ) -> List[ExtractedTopic]:
        """
        Upsert extracted topics for a bot.
        Each topic dict should have: topic, category (optional), confidence (optional), source_urls (optional)
        """
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        if not oid or not bid:
            raise ValueError("org_id and bot_id are required")

        now = _utc_now()
        results: List[ExtractedTopic] = []
        con = _connect()
        try:
            for t in topics:
                topic_text = (t.get("topic") or "").strip().lower()
                if not topic_text:
                    continue

                category = (t.get("category") or "").strip() or None
                confidence = float(t.get("confidence", 1.0))
                source_urls = t.get("source_urls", [])
                if isinstance(source_urls, str):
                    try:
                        source_urls = json.loads(source_urls)
                    except json.JSONDecodeError:
                        source_urls = []
                source_url = (t.get("source_url") or "").strip() or None
                origin = (t.get("origin") or "extracted").strip() or "extracted"

                # Check if topic already exists for this bot
                existing = con.execute(
                    """
                    SELECT topic_id, source_urls, occurrence_count, source_url, origin
                    FROM bot_extracted_topics
                    WHERE org_id = %s AND bot_id = %s AND lower(topic) = %s
                    """,
                    (oid, bid, topic_text),
                ).fetchone()

                if existing:
                    # Update existing topic
                    topic_id = existing[0]
                    existing_urls = json.loads(existing[1] or "[]")
                    merged_urls = list(set(existing_urls + source_urls))
                    new_count = (existing[2] or 1) + 1
                    # Keep existing source_url unless a new one is provided
                    resolved_source_url = source_url or existing[3]
                    # url_bank origin takes precedence (don't downgrade to extracted)
                    resolved_origin = existing[4] if existing[4] == "url_bank" else origin

                    con.execute(
                        """
                        UPDATE bot_extracted_topics
                        SET category = COALESCE(%s, category),
                            confidence = %s,
                            source_urls = %s,
                            occurrence_count = %s,
                            source_url = COALESCE(%s, source_url),
                            origin = %s,
                            updated_at = %s
                        WHERE topic_id = %s
                        """,
                        (category, confidence, json.dumps(merged_urls), new_count,
                         resolved_source_url, resolved_origin, now, topic_id),
                    )
                    results.append(ExtractedTopic(
                        topic_id=topic_id,
                        org_id=oid,
                        bot_id=bid,
                        topic=topic_text,
                        category=category,
                        confidence=confidence,
                        source_urls=merged_urls,
                        occurrence_count=new_count,
                        is_active=True,
                        extracted_at=now,
                        updated_at=now,
                        source_url=resolved_source_url,
                        origin=resolved_origin,
                    ))
                else:
                    # Insert new topic
                    topic_id = _new_topic_id()
                    con.execute(
                        """
                        INSERT INTO bot_extracted_topics
                        (topic_id, org_id, bot_id, topic, category, confidence, source_urls, occurrence_count, is_active, extracted_at, updated_at, source_url, origin)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (topic_id, oid, bid, topic_text, category, confidence, json.dumps(source_urls), 1, True, now, now, source_url, origin),
                    )
                    results.append(ExtractedTopic(
                        topic_id=topic_id,
                        org_id=oid,
                        bot_id=bid,
                        topic=topic_text,
                        category=category,
                        confidence=confidence,
                        source_urls=source_urls,
                        occurrence_count=1,
                        is_active=True,
                        extracted_at=now,
                        updated_at=now,
                        source_url=source_url,
                        origin=origin,
                    ))

            con.commit()
            return results
        finally:
            con.close()

    def get_extracted_topics(
        self,
        *,
        org_id: str,
        bot_id: str,
        active_only: bool = False,
        limit: int = 100,
    ) -> List[ExtractedTopic]:
        """Get extracted topics for a bot."""
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        lim = max(1, min(int(limit or 100), 500))

        con = _connect()
        try:
            query = """
                SELECT topic_id, org_id, bot_id, topic, category, confidence,
                       source_urls, occurrence_count, is_active, extracted_at, updated_at,
                       source_url, origin
                FROM bot_extracted_topics
                WHERE org_id = %s AND bot_id = %s
            """
            params: List[Any] = [oid, bid]

            if active_only:
                query += " AND is_active = TRUE"

            query += " ORDER BY occurrence_count DESC, extracted_at DESC LIMIT %s"
            params.append(lim)

            rows = con.execute(query, params).fetchall()
            results: List[ExtractedTopic] = []
            for r in rows or []:
                source_urls = []
                try:
                    source_urls = json.loads(r[6] or "[]")
                except json.JSONDecodeError:
                    pass
                results.append(ExtractedTopic(
                    topic_id=r[0],
                    org_id=r[1],
                    bot_id=r[2],
                    topic=r[3],
                    category=r[4],
                    confidence=float(r[5] or 1.0),
                    source_urls=source_urls,
                    occurrence_count=int(r[7] or 1),
                    is_active=bool(r[8]),
                    extracted_at=r[9],
                    updated_at=r[10],
                    source_url=r[11],
                    origin=(r[12] or "extracted"),
                ))
            return results
        finally:
            con.close()

    def update_topic(
        self,
        *,
        topic_id: str,
        is_active: Optional[bool] = None,
        category: Optional[str] = None,
    ) -> Optional[ExtractedTopic]:
        """Update a topic's active status or category."""
        tid = (topic_id or "").strip()
        if not tid:
            return None

        now = _utc_now()
        con = _connect()
        try:
            updates: List[str] = ["updated_at = %s"]
            params: List[Any] = [now]

            if is_active is not None:
                updates.append("is_active = %s")
                params.append(is_active)

            if category is not None:
                updates.append("category = %s")
                params.append(category if category else None)

            params.append(tid)

            con.execute(
                f"""
                UPDATE bot_extracted_topics
                SET {', '.join(updates)}
                WHERE topic_id = %s
                """,
                params,
            )
            con.commit()

            # Fetch and return the updated topic
            row = con.execute(
                """
                SELECT topic_id, org_id, bot_id, topic, category, confidence,
                       source_urls, occurrence_count, is_active, extracted_at, updated_at,
                       source_url, origin
                FROM bot_extracted_topics
                WHERE topic_id = %s
                """,
                (tid,),
            ).fetchone()

            if not row:
                return None

            source_urls = []
            try:
                source_urls = json.loads(row[6] or "[]")
            except json.JSONDecodeError:
                pass

            return ExtractedTopic(
                topic_id=row[0],
                org_id=row[1],
                bot_id=row[2],
                topic=row[3],
                category=row[4],
                confidence=float(row[5] or 1.0),
                source_urls=source_urls,
                occurrence_count=int(row[7] or 1),
                is_active=bool(row[8]),
                extracted_at=row[9],
                updated_at=row[10],
                source_url=row[11],
                origin=(row[12] or "extracted"),
            )
        finally:
            con.close()

    def delete_topic(self, *, topic_id: str) -> bool:
        """Delete a topic by ID."""
        tid = (topic_id or "").strip()
        if not tid:
            return False

        con = _connect()
        try:
            result = con.execute(
                "DELETE FROM bot_extracted_topics WHERE topic_id = %s",
                (tid,),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def delete_all_topics_for_bot(self, *, org_id: str, bot_id: str) -> int:
        """Delete all extracted topics for a bot. Returns count of deleted topics."""
        oid = (org_id or "").strip()
        bid = (bot_id or "").strip()
        if not oid or not bid:
            return 0

        con = _connect()
        try:
            result = con.execute(
                "DELETE FROM bot_extracted_topics WHERE org_id = %s AND bot_id = %s",
                (oid, bid),
            )
            con.commit()
            return result.rowcount
        finally:
            con.close()


# ═══════════════════════════════════════════════════════════════════════
# Topic-question mapping repository
# ═══════════════════════════════════════════════════════════════════════


def _new_tqm_id() -> str:
    return "tqm_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


@dataclass
class TopicQuestion:
    id: str
    topic_id: str
    org_id: str
    bot_id: str
    session_id: str
    message_id: Optional[str]
    question_text: Optional[str]
    matched_at: str
    match_method: str


class PostgresTopicQuestionRepository:
    """Repository for topic-question mapping."""

    def upsert_mapping(
        self,
        *,
        topic_id: str,
        org_id: str,
        bot_id: str,
        session_id: str,
        message_id: Optional[str],
        question_text: Optional[str],
        match_method: str = "keyword",
    ) -> bool:
        """Insert a mapping if it doesn't exist yet. Returns True if inserted."""
        con = _connect()
        try:
            existing = con.execute(
                "SELECT id FROM topic_question_mappings WHERE topic_id = %s AND session_id = %s AND message_id = %s",
                (topic_id, session_id, message_id),
            ).fetchone()
            if existing:
                return False
            con.execute(
                """
                INSERT INTO topic_question_mappings
                (id, topic_id, org_id, bot_id, session_id, message_id, question_text, matched_at, match_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (_new_tqm_id(), topic_id, org_id, bot_id, session_id, message_id, question_text, _utc_now(), match_method),
            )
            con.commit()
            return True
        finally:
            con.close()

    def get_questions_for_topic(self, *, topic_id: str, limit: int = 50) -> List[TopicQuestion]:
        """Return questions linked to a topic."""
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT id, topic_id, org_id, bot_id, session_id, message_id, question_text, matched_at, match_method
                FROM topic_question_mappings
                WHERE topic_id = %s
                ORDER BY matched_at DESC
                LIMIT %s
                """,
                (topic_id, max(1, min(int(limit), 200))),
            ).fetchall()
            return [
                TopicQuestion(
                    id=r[0], topic_id=r[1], org_id=r[2], bot_id=r[3],
                    session_id=r[4], message_id=r[5], question_text=r[6],
                    matched_at=r[7], match_method=(r[8] or "keyword"),
                )
                for r in (rows or [])
            ]
        finally:
            con.close()

    def get_usage_counts(self, *, org_id: str, bot_id: str) -> Dict[str, int]:
        """Return {topic_id: question_count} for all topics of a bot."""
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT topic_id, COUNT(*) as cnt
                FROM topic_question_mappings
                WHERE org_id = %s AND bot_id = %s
                GROUP BY topic_id
                """,
                (org_id, bot_id),
            ).fetchall()
            return {r[0]: int(r[1]) for r in (rows or [])}
        finally:
            con.close()

    def delete_for_bot(self, *, org_id: str, bot_id: str) -> int:
        """Delete all mappings for a bot. Returns count deleted."""
        con = _connect()
        try:
            result = con.execute(
                "DELETE FROM topic_question_mappings WHERE org_id = %s AND bot_id = %s",
                (org_id, bot_id),
            )
            con.commit()
            return result.rowcount
        finally:
            con.close()


# ═══════════════════════════════════════════════════════════════════════
# LINE integration repositories
# ═══════════════════════════════════════════════════════════════════════


def _new_line_channel_id() -> str:
    return "lch_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_line_rich_menu_state_id() -> str:
    return "lrm_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


class PostgresLineChannelRepository:
    """CRUD for LINE channel credentials linked to a bot."""

    def upsert(
        self,
        *,
        bot_id: str,
        org_id: str,
        line_channel_id: str,
        line_channel_secret: str,
        line_channel_access_token: str,
        is_active: bool = True,
    ) -> LineChannel:
        bid = (bot_id or "").strip()
        oid = (org_id or "").strip()
        if not bid or not oid:
            raise ValueError("bot_id and org_id are required")
        now = _utc_now()
        cid = _new_line_channel_id()
        con = _connect()
        try:
            # Try update first
            row = con.execute(
                "SELECT channel_id FROM line_channels WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if row:
                cid = row[0]
                con.execute(
                    """
                    UPDATE line_channels
                    SET line_channel_id = %s,
                        line_channel_secret = %s,
                        line_channel_access_token = %s,
                        is_active = %s,
                        updated_at = %s
                    WHERE bot_id = %s
                    """,
                    (line_channel_id, line_channel_secret, line_channel_access_token, is_active, now, bid),
                )
            else:
                con.execute(
                    """
                    INSERT INTO line_channels(
                      channel_id, bot_id, org_id, line_channel_id,
                      line_channel_secret, line_channel_access_token,
                      is_active, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (cid, bid, oid, line_channel_id, line_channel_secret, line_channel_access_token, is_active, now, now),
                )
            con.commit()
            return LineChannel(
                channel_id=cid,
                bot_id=bid,
                org_id=oid,
                line_channel_id=line_channel_id,
                line_channel_secret=line_channel_secret,
                line_channel_access_token=line_channel_access_token,
                is_active=is_active,
                created_at=now,
                updated_at=now,
            )
        finally:
            con.close()

    def get_by_bot_id(self, bot_id: str) -> Optional[LineChannel]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT channel_id, bot_id, org_id, line_channel_id,
                       line_channel_secret, line_channel_access_token,
                       is_active, created_at, updated_at
                FROM line_channels
                WHERE bot_id = %s
                """,
                (bid,),
            ).fetchone()
            if not row:
                return None
            return LineChannel(
                channel_id=row[0],
                bot_id=row[1],
                org_id=row[2],
                line_channel_id=row[3],
                line_channel_secret=row[4],
                line_channel_access_token=row[5],
                is_active=bool(row[6]),
                created_at=row[7],
                updated_at=row[8],
            )
        finally:
            con.close()

    def delete_by_bot_id(self, bot_id: str) -> bool:
        bid = (bot_id or "").strip()
        if not bid:
            return False
        con = _connect()
        try:
            result = con.execute(
                "DELETE FROM line_channels WHERE bot_id = %s",
                (bid,),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()


class PostgresLineRichMenuStateRepository:
    """Persistence for managed LINE rich-menu sync state per bot/channel."""

    @staticmethod
    def _state_from_row(row) -> LineRichMenuState:
        variants_raw = row[5] if len(row) > 5 else "{}"
        try:
            variants = json.loads(variants_raw) if isinstance(variants_raw, str) else (variants_raw or {})
        except (TypeError, ValueError):
            variants = {}
        return LineRichMenuState(
            state_id=row[0],
            channel_id=row[1],
            bot_id=row[2],
            config_hash=row[3],
            default_variant=row[4],
            rich_menu_variants=variants if isinstance(variants, dict) else {},
            sync_status=row[6] or "pending",
            last_synced_at=row[7],
            last_error=row[8],
            created_at=row[9],
            updated_at=row[10],
        )

    def get_by_bot_id(self, bot_id: str) -> Optional[LineRichMenuState]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT state_id, channel_id, bot_id, config_hash, default_variant, rich_menu_variants_json,
                       sync_status, last_synced_at, last_error, created_at, updated_at
                FROM line_rich_menu_states
                WHERE bot_id = %s
                """,
                (bid,),
            ).fetchone()
            return self._state_from_row(row) if row else None
        finally:
            con.close()

    def upsert(
        self,
        *,
        channel_id: str,
        bot_id: str,
        config_hash: Optional[str],
        default_variant: Optional[str],
        rich_menu_variants: Optional[Dict[str, str]],
        sync_status: str,
        last_synced_at: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> LineRichMenuState:
        cid = (channel_id or "").strip()
        bid = (bot_id or "").strip()
        if not cid or not bid:
            raise ValueError("channel_id and bot_id are required")
        now = _utc_now()
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT state_id, channel_id, bot_id, config_hash, default_variant, rich_menu_variants_json,
                       sync_status, last_synced_at, last_error, created_at, updated_at
                FROM line_rich_menu_states
                WHERE bot_id = %s
                """,
                (bid,),
            ).fetchone()
            payload = json.dumps(rich_menu_variants or {})
            synced_at = (last_synced_at or "").strip() or None
            error_text = (last_error or "").strip() or None
            if row:
                existing = self._state_from_row(row)
                con.execute(
                    """
                    UPDATE line_rich_menu_states
                    SET channel_id = %s,
                        config_hash = %s,
                        default_variant = %s,
                        rich_menu_variants_json = %s,
                        sync_status = %s,
                        last_synced_at = %s,
                        last_error = %s,
                        updated_at = %s
                    WHERE state_id = %s
                    """,
                    (
                        cid,
                        config_hash,
                        default_variant,
                        payload,
                        sync_status,
                        synced_at,
                        error_text,
                        now,
                        existing.state_id,
                    ),
                )
                con.commit()
                return LineRichMenuState(
                    state_id=existing.state_id,
                    channel_id=cid,
                    bot_id=bid,
                    config_hash=config_hash,
                    default_variant=default_variant,
                    rich_menu_variants=rich_menu_variants or {},
                    sync_status=sync_status,
                    last_synced_at=synced_at,
                    last_error=error_text,
                    created_at=existing.created_at,
                    updated_at=now,
                )

            state_id = _new_line_rich_menu_state_id()
            con.execute(
                """
                INSERT INTO line_rich_menu_states(
                  state_id, channel_id, bot_id, config_hash, default_variant, rich_menu_variants_json,
                  sync_status, last_synced_at, last_error, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state_id,
                    cid,
                    bid,
                    config_hash,
                    default_variant,
                    payload,
                    sync_status,
                    synced_at,
                    error_text,
                    now,
                    now,
                ),
            )
            con.commit()
            return LineRichMenuState(
                state_id=state_id,
                channel_id=cid,
                bot_id=bid,
                config_hash=config_hash,
                default_variant=default_variant,
                rich_menu_variants=rich_menu_variants or {},
                sync_status=sync_status,
                last_synced_at=synced_at,
                last_error=error_text,
                created_at=now,
                updated_at=now,
            )
        finally:
            con.close()

    def delete_by_bot_id(self, bot_id: str) -> bool:
        bid = (bot_id or "").strip()
        if not bid:
            return False
        con = _connect()
        try:
            result = con.execute("DELETE FROM line_rich_menu_states WHERE bot_id = %s", (bid,))
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()


class PostgresLineUserSessionRepository:
    """Maps LINE user IDs to conversation sessions per bot."""

    def get_or_create(
        self,
        *,
        line_user_id: str,
        bot_id: str,
        session_id: str,
        display_name: Optional[str] = None,
    ) -> LineUserSession:
        """Get existing mapping or create one. Returns the mapping."""
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        name = (display_name or "").strip() or None
        if not uid or not bid:
            raise ValueError("line_user_id and bot_id are required")
        now = _utc_now()
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT line_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg, display_name
                FROM line_user_sessions
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (uid, bid),
            ).fetchone()
            if row:
                return LineUserSession(
                    line_user_id=row[0],
                    bot_id=row[1],
                    session_id=row[2],
                    is_escalated=bool(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                    awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                    display_name=row[7] if len(row) > 7 else None,
                )
            con.execute(
                """
                INSERT INTO line_user_sessions(line_user_id, bot_id, session_id, is_escalated, awaiting_escalation_msg, display_name, created_at, updated_at)
                VALUES (%s, %s, %s, FALSE, FALSE, %s, %s, %s)
                """,
                (uid, bid, session_id, name, now, now),
            )
            con.commit()
            return LineUserSession(
                line_user_id=uid,
                bot_id=bid,
                session_id=session_id,
                is_escalated=False,
                awaiting_escalation_msg=False,
                created_at=now,
                updated_at=now,
                display_name=name,
            )
        finally:
            con.close()

    def get(self, *, line_user_id: str, bot_id: str) -> Optional[LineUserSession]:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT line_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg, display_name
                FROM line_user_sessions
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (uid, bid),
            ).fetchone()
            if not row:
                return None
            return LineUserSession(
                line_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                display_name=row[7] if len(row) > 7 else None,
            )
        finally:
            con.close()

    def get_by_session_id(self, session_id: str) -> Optional[LineUserSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT line_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg, display_name
                FROM line_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return LineUserSession(
                line_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                display_name=row[7] if len(row) > 7 else None,
            )
        finally:
            con.close()

    def set_display_name(self, *, line_user_id: str, bot_id: str, display_name: Optional[str]) -> bool:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        name = (display_name or "").strip() or None
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE line_user_sessions
                SET display_name = %s, updated_at = %s
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (name, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def set_escalated(self, *, line_user_id: str, bot_id: str, escalated: bool) -> bool:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE line_user_sessions
                SET is_escalated = %s, updated_at = %s
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (escalated, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def set_awaiting_escalation_msg(self, *, line_user_id: str, bot_id: str, awaiting: bool) -> bool:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE line_user_sessions
                SET awaiting_escalation_msg = %s, updated_at = %s
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (awaiting, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def update_session_id(self, *, line_user_id: str, bot_id: str, session_id: str) -> bool:
        uid = (line_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE line_user_sessions
                SET session_id = %s, updated_at = %s
                WHERE line_user_id = %s AND bot_id = %s
                """,
                (session_id, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def escalate_by_session_id(self, session_id: str) -> Optional[LineUserSession]:
        """Set is_escalated=True for the mapping matching this session_id (client takeover)."""
        sid = (session_id or "").strip()
        if not sid:
            return None
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE line_user_sessions
                SET is_escalated = TRUE, awaiting_escalation_msg = FALSE, updated_at = %s
                WHERE session_id = %s
                """,
                (now, sid),
            )
            con.commit()
            row = con.execute(
                """
                SELECT line_user_id, bot_id, session_id, is_escalated, created_at, updated_at, display_name
                FROM line_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return LineUserSession(
                line_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                display_name=row[6] if len(row) > 6 else None,
            )
        finally:
            con.close()

    def de_escalate_by_session_id(self, session_id: str) -> Optional[LineUserSession]:
        """Reset is_escalated=False for the mapping matching this session_id. Returns updated mapping or None."""
        sid = (session_id or "").strip()
        if not sid:
            return None
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE line_user_sessions
                SET is_escalated = FALSE, updated_at = %s
                WHERE session_id = %s
                """,
                (now, sid),
            )
            con.commit()
            row = con.execute(
                """
                SELECT line_user_id, bot_id, session_id, is_escalated, created_at, updated_at, display_name
                FROM line_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return LineUserSession(
                line_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                display_name=row[6] if len(row) > 6 else None,
            )
        finally:
            con.close()


# ══════════════════════════════════════════════════════════════════════
# Instagram integration
# ══════════════════════════════════════════════════════════════════════


def _new_ig_channel_id() -> str:
    return "igch_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_ig_verify_token() -> str:
    return secrets.token_urlsafe(32)


class PostgresInstagramChannelRepository:
    """CRUD for Instagram channel credentials linked to a bot."""

    _SELECT_COLS = """channel_id, bot_id, org_id, ig_page_id,
                      app_secret, page_access_token, verify_token,
                      is_active, created_at, updated_at,
                      ig_user_id, ig_username, token_expires_at, connection_method"""

    @staticmethod
    def _row_to_entity(row) -> InstagramChannel:
        return InstagramChannel(
            channel_id=row[0],
            bot_id=row[1],
            org_id=row[2],
            ig_page_id=row[3],
            app_secret=row[4],
            page_access_token=row[5],
            verify_token=row[6],
            is_active=bool(row[7]),
            created_at=row[8],
            updated_at=row[9],
            ig_user_id=row[10] if len(row) > 10 else None,
            ig_username=row[11] if len(row) > 11 else None,
            token_expires_at=row[12] if len(row) > 12 else None,
            connection_method=row[13] if len(row) > 13 else "manual",
        )

    def upsert(
        self,
        *,
        bot_id: str,
        org_id: str,
        ig_page_id: str,
        app_secret: str,
        page_access_token: str,
        is_active: bool = True,
    ) -> InstagramChannel:
        bid = (bot_id or "").strip()
        oid = (org_id or "").strip()
        if not bid or not oid:
            raise ValueError("bot_id and org_id are required")
        now = _utc_now()
        cid = _new_ig_channel_id()
        verify_token = _new_ig_verify_token()
        con = _connect()
        try:
            row = con.execute(
                "SELECT channel_id, verify_token FROM instagram_channels WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if row:
                cid = row[0]
                verify_token = row[1]  # Keep existing verify_token on update
                con.execute(
                    """
                    UPDATE instagram_channels
                    SET ig_page_id = %s,
                        app_secret = %s,
                        page_access_token = %s,
                        is_active = %s,
                        updated_at = %s
                    WHERE bot_id = %s
                    """,
                    (ig_page_id, app_secret, page_access_token, is_active, now, bid),
                )
            else:
                con.execute(
                    """
                    INSERT INTO instagram_channels(
                      channel_id, bot_id, org_id, ig_page_id,
                      app_secret, page_access_token, verify_token,
                      is_active, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (cid, bid, oid, ig_page_id, app_secret, page_access_token, verify_token, is_active, now, now),
                )
            con.commit()
            return InstagramChannel(
                channel_id=cid,
                bot_id=bid,
                org_id=oid,
                ig_page_id=ig_page_id,
                app_secret=app_secret,
                page_access_token=page_access_token,
                verify_token=verify_token,
                is_active=is_active,
                created_at=now,
                updated_at=now,
            )
        finally:
            con.close()

    def upsert_oauth(
        self,
        *,
        bot_id: str,
        org_id: str,
        ig_user_id: str,
        ig_username: str,
        access_token: str,
        token_expires_at: str,
        ig_webhook_id: str | None = None,
    ) -> InstagramChannel:
        """Create or update a channel via OAuth flow (no manual credentials).

        ig_webhook_id: The IGSID used in webhook recipient.id, which may
        differ from the app-scoped ig_user_id.  Stored in ig_page_id for
        webhook routing lookups.
        """
        bid = (bot_id or "").strip()
        oid = (org_id or "").strip()
        if not bid or not oid:
            raise ValueError("bot_id and org_id are required")
        now = _utc_now()
        cid = _new_ig_channel_id()
        verify_token = _new_ig_verify_token()
        # For OAuth channels the app_secret is global (from env), stored as placeholder
        app_secret_placeholder = "__OAUTH__"
        # ig_page_id stores the webhook IGSID if available, else the app-scoped ID
        page_id_value = ig_webhook_id or ig_user_id
        con = _connect()
        try:
            row = con.execute(
                "SELECT channel_id, verify_token FROM instagram_channels WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if row:
                cid = row[0]
                verify_token = row[1]
                con.execute(
                    """
                    UPDATE instagram_channels
                    SET ig_page_id = %s,
                        app_secret = %s,
                        page_access_token = %s,
                        is_active = TRUE,
                        updated_at = %s,
                        ig_user_id = %s,
                        ig_username = %s,
                        token_expires_at = %s,
                        connection_method = 'oauth'
                    WHERE bot_id = %s
                    """,
                    (page_id_value, app_secret_placeholder, access_token, now,
                     ig_user_id, ig_username, token_expires_at, bid),
                )
            else:
                con.execute(
                    """
                    INSERT INTO instagram_channels(
                      channel_id, bot_id, org_id, ig_page_id,
                      app_secret, page_access_token, verify_token,
                      is_active, created_at, updated_at,
                      ig_user_id, ig_username, token_expires_at, connection_method
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s, %s, %s, %s, 'oauth')
                    """,
                    (cid, bid, oid, page_id_value,
                     app_secret_placeholder, access_token, verify_token,
                     now, now,
                     ig_user_id, ig_username, token_expires_at),
                )
            con.commit()
            return InstagramChannel(
                channel_id=cid,
                bot_id=bid,
                org_id=oid,
                ig_page_id=ig_user_id,
                app_secret=app_secret_placeholder,
                page_access_token=access_token,
                verify_token=verify_token,
                is_active=True,
                created_at=now,
                updated_at=now,
                ig_user_id=ig_user_id,
                ig_username=ig_username,
                token_expires_at=token_expires_at,
                connection_method="oauth",
            )
        finally:
            con.close()

    def get_by_bot_id(self, bot_id: str) -> Optional[InstagramChannel]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                f"SELECT {self._SELECT_COLS} FROM instagram_channels WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)
        finally:
            con.close()

    def get_by_page_id(self, ig_page_id: str) -> Optional[InstagramChannel]:
        pid = (ig_page_id or "").strip()
        if not pid:
            return None
        con = _connect()
        try:
            row = con.execute(
                f"SELECT {self._SELECT_COLS} FROM instagram_channels WHERE ig_page_id = %s",
                (pid,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)
        finally:
            con.close()

    def get_by_ig_user_id(self, ig_user_id: str) -> Optional[InstagramChannel]:
        """Look up channel by Instagram-scoped user ID (for OAuth webhook routing)."""
        uid = (ig_user_id or "").strip()
        if not uid:
            return None
        con = _connect()
        try:
            row = con.execute(
                f"SELECT {self._SELECT_COLS} FROM instagram_channels WHERE ig_user_id = %s",
                (uid,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)
        finally:
            con.close()

    def resolve_by_webhook_id(self, webhook_recipient_id: str) -> Optional[InstagramChannel]:
        """Try to match a webhook recipient ID to an OAuth channel.

        Instagram webhooks may send an IGSID that differs from the app-scoped
        user ID stored during OAuth.  This uses each channel's IGAA token to
        query the webhook recipient ID on graph.instagram.com.  If the token
        can read that ID, it means the channel owns it.  When a match is found,
        ``ig_page_id`` is updated so future lookups are instant.
        """
        import httpx
        import logging as _log
        _logger = _log.getLogger(__name__)
        wid = (webhook_recipient_id or "").strip()
        if not wid:
            return None
        con = _connect()
        try:
            rows = con.execute(
                f"SELECT {self._SELECT_COLS} FROM instagram_channels "
                "WHERE connection_method = 'oauth' AND is_active = TRUE",
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return None

        for row in rows:
            ch = self._row_to_entity(row)
            token = ch.page_access_token
            if not token:
                continue
            try:
                # Use the IGAA token to query the webhook recipient ID
                # on graph.instagram.com.  If this token owns that account,
                # the API will return data successfully.
                with httpx.Client(timeout=8) as client:
                    resp = client.get(
                        f"https://graph.instagram.com/v21.0/{wid}",
                        params={"fields": "username", "access_token": token},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    _logger.info(
                        "resolve_by_webhook_id: matched IGSID %s -> @%s (bot %s)",
                        wid, data.get("username", "?"), ch.bot_id,
                    )
                    self._update_page_id(ch.bot_id, wid)
                    ch = InstagramChannel(
                        channel_id=ch.channel_id, bot_id=ch.bot_id,
                        org_id=ch.org_id, ig_page_id=wid,
                        app_secret=ch.app_secret,
                        page_access_token=ch.page_access_token,
                        verify_token=ch.verify_token, is_active=ch.is_active,
                        created_at=ch.created_at, updated_at=ch.updated_at,
                        ig_user_id=ch.ig_user_id, ig_username=ch.ig_username,
                        token_expires_at=ch.token_expires_at,
                        connection_method=ch.connection_method,
                    )
                    return ch
                else:
                    _logger.debug("resolve_by_webhook_id: %s returned %s for bot %s", wid, resp.status_code, ch.bot_id)
            except Exception as exc:
                _logger.warning("resolve_by_webhook_id: error checking bot %s: %s", ch.bot_id, exc)
                continue
        return None

    def _update_page_id(self, bot_id: str, new_page_id: str) -> None:
        """Update ig_page_id for a channel (self-healing webhook ID mapping)."""
        con = _connect()
        try:
            con.execute(
                "UPDATE instagram_channels SET ig_page_id = %s, updated_at = %s WHERE bot_id = %s",
                (new_page_id, _utc_now(), bot_id),
            )
            con.commit()
        finally:
            con.close()

    def get_channels_expiring_soon(self, days: int = 7) -> list:
        """Return OAuth channels whose token expires within `days` days."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        con = _connect()
        try:
            rows = con.execute(
                f"""SELECT {self._SELECT_COLS} FROM instagram_channels
                    WHERE connection_method = 'oauth'
                      AND token_expires_at IS NOT NULL
                      AND token_expires_at <= %s
                      AND is_active = TRUE""",
                (cutoff,),
            ).fetchall()
            return [self._row_to_entity(r) for r in rows]
        finally:
            con.close()

    def update_token(self, channel_id: str, *, access_token: str, token_expires_at: str) -> None:
        """Update the access token and expiration (used by token refresh job)."""
        con = _connect()
        try:
            con.execute(
                """UPDATE instagram_channels
                   SET page_access_token = %s, token_expires_at = %s, updated_at = %s
                   WHERE channel_id = %s""",
                (access_token, token_expires_at, _utc_now(), channel_id),
            )
            con.commit()
        finally:
            con.close()

    def delete_by_bot_id(self, bot_id: str) -> bool:
        bid = (bot_id or "").strip()
        if not bid:
            return False
        con = _connect()
        try:
            result = con.execute(
                "DELETE FROM instagram_channels WHERE bot_id = %s",
                (bid,),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()


class PostgresInstagramUserSessionRepository:
    """Maps Instagram user IDs (IGSID) to conversation sessions per bot."""

    def get_or_create(
        self,
        *,
        ig_user_id: str,
        bot_id: str,
        session_id: str,
    ) -> InstagramUserSession:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            raise ValueError("ig_user_id and bot_id are required")
        now = _utc_now()
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT ig_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg,
                       COALESCE(awaiting_staff_takeover, FALSE)
                FROM instagram_user_sessions
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (uid, bid),
            ).fetchone()
            if row:
                return InstagramUserSession(
                    ig_user_id=row[0],
                    bot_id=row[1],
                    session_id=row[2],
                    is_escalated=bool(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                    awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                    awaiting_staff_takeover=bool(row[7]) if len(row) > 7 and row[7] is not None else False,
                )
            con.execute(
                """
                INSERT INTO instagram_user_sessions(ig_user_id, bot_id, session_id, is_escalated, awaiting_escalation_msg, awaiting_staff_takeover, created_at, updated_at)
                VALUES (%s, %s, %s, FALSE, FALSE, FALSE, %s, %s)
                """,
                (uid, bid, session_id, now, now),
            )
            con.commit()
            return InstagramUserSession(
                ig_user_id=uid,
                bot_id=bid,
                session_id=session_id,
                is_escalated=False,
                awaiting_escalation_msg=False,
                awaiting_staff_takeover=False,
                created_at=now,
                updated_at=now,
            )
        finally:
            con.close()

    def get(self, *, ig_user_id: str, bot_id: str) -> Optional[InstagramUserSession]:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT ig_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg,
                       COALESCE(awaiting_staff_takeover, FALSE)
                FROM instagram_user_sessions
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (uid, bid),
            ).fetchone()
            if not row:
                return None
            return InstagramUserSession(
                ig_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                awaiting_staff_takeover=bool(row[7]) if len(row) > 7 and row[7] is not None else False,
            )
        finally:
            con.close()

    def get_by_session_id(self, session_id: str) -> Optional[InstagramUserSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT ig_user_id, bot_id, session_id, is_escalated, created_at, updated_at, awaiting_escalation_msg,
                       COALESCE(awaiting_staff_takeover, FALSE)
                FROM instagram_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return InstagramUserSession(
                ig_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
                awaiting_escalation_msg=bool(row[6]) if row[6] is not None else False,
                awaiting_staff_takeover=bool(row[7]) if len(row) > 7 and row[7] is not None else False,
            )
        finally:
            con.close()

    def set_escalated(self, *, ig_user_id: str, bot_id: str, escalated: bool) -> bool:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE instagram_user_sessions
                SET is_escalated = %s, updated_at = %s
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (escalated, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def set_awaiting_escalation_msg(self, *, ig_user_id: str, bot_id: str, awaiting: bool) -> bool:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE instagram_user_sessions
                SET awaiting_escalation_msg = %s, updated_at = %s
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (awaiting, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def set_awaiting_staff_takeover(self, *, ig_user_id: str, bot_id: str, awaiting: bool) -> bool:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE instagram_user_sessions
                SET awaiting_staff_takeover = %s, updated_at = %s
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (awaiting, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def set_escalated_and_clear_awaiting_staff(self, *, ig_user_id: str, bot_id: str) -> bool:
        """Set is_escalated=True and awaiting_staff_takeover=False (client tapped Take over or replied)."""
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE instagram_user_sessions
                SET is_escalated = TRUE, awaiting_staff_takeover = FALSE, updated_at = %s
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def update_session_id(self, *, ig_user_id: str, bot_id: str, session_id: str) -> bool:
        uid = (ig_user_id or "").strip()
        bid = (bot_id or "").strip()
        if not uid or not bid:
            return False
        now = _utc_now()
        con = _connect()
        try:
            result = con.execute(
                """
                UPDATE instagram_user_sessions
                SET session_id = %s, updated_at = %s
                WHERE ig_user_id = %s AND bot_id = %s
                """,
                (session_id, now, uid, bid),
            )
            con.commit()
            return result.rowcount > 0
        finally:
            con.close()

    def escalate_by_session_id(self, session_id: str) -> Optional[InstagramUserSession]:
        """Set is_escalated=True for the mapping matching this session_id (client takeover)."""
        sid = (session_id or "").strip()
        if not sid:
            return None
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE instagram_user_sessions
                SET is_escalated = TRUE, awaiting_escalation_msg = FALSE, updated_at = %s
                WHERE session_id = %s
                """,
                (now, sid),
            )
            con.commit()
            row = con.execute(
                """
                SELECT ig_user_id, bot_id, session_id, is_escalated, created_at, updated_at
                FROM instagram_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return InstagramUserSession(
                ig_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
            )
        finally:
            con.close()

    def de_escalate_by_session_id(self, session_id: str) -> Optional[InstagramUserSession]:
        """Reset is_escalated=False for the mapping matching this session_id."""
        sid = (session_id or "").strip()
        if not sid:
            return None
        now = _utc_now()
        con = _connect()
        try:
            con.execute(
                """
                UPDATE instagram_user_sessions
                SET is_escalated = FALSE, updated_at = %s
                WHERE session_id = %s
                """,
                (now, sid),
            )
            con.commit()
            row = con.execute(
                """
                SELECT ig_user_id, bot_id, session_id, is_escalated, created_at, updated_at
                FROM instagram_user_sessions
                WHERE session_id = %s
                """,
                (sid,),
            ).fetchone()
            if not row:
                return None
            return InstagramUserSession(
                ig_user_id=row[0],
                bot_id=row[1],
                session_id=row[2],
                is_escalated=bool(row[3]),
                created_at=row[4],
                updated_at=row[5],
            )
        finally:
            con.close()


# ── Bot Assets ──────────────────────────────────────────────────────────────



@dataclass
class AssetExtractionJob:
    job_id: str
    bot_id: str
    org_id: str
    status: str
    created_at: str
    updated_at: str
    gcs_prefix: Optional[str] = None
    page_urls: Optional[List[str]] = None
    assets_discovered: int = 0
    assets_downloaded: int = 0
    assets_created: int = 0
    error: Optional[str] = None
    celery_task_id: Optional[str] = None


class PostgresAssetExtractionJobRepository:
    def create_job(self, job: AssetExtractionJob) -> None:
        if not job.job_id or not job.bot_id or not job.org_id:
            raise ValueError("job_id, bot_id, and org_id are required")
        
        con = _connect()
        try:
            urls_json = json.dumps(job.page_urls or [])
            con.execute(
                """
                INSERT INTO asset_extraction_jobs (
                    job_id, bot_id, org_id, status, gcs_prefix, page_urls,
                    assets_discovered, assets_downloaded, assets_created,
                    error, celery_task_id, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.job_id, job.bot_id, job.org_id, job.status, job.gcs_prefix, urls_json,
                    job.assets_discovered, job.assets_downloaded, job.assets_created,
                    job.error, job.celery_task_id, job.created_at, job.updated_at
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_job(self, job: AssetExtractionJob) -> None:
        job.updated_at = _utc_now()
        con = _connect()
        try:
            urls_json = json.dumps(job.page_urls or [])
            # Keep "cancelled" sticky: once cancelled, background progress updates must not
            # flip the row back to running/done.
            where_clause = "WHERE job_id=%s" if job.status == "cancelled" else "WHERE job_id=%s AND status <> 'cancelled'"
            con.execute(
                f"""
                UPDATE asset_extraction_jobs
                SET status=%s, gcs_prefix=%s, page_urls=%s,
                    assets_discovered=%s, assets_downloaded=%s, assets_created=%s,
                    error=%s, celery_task_id=%s, updated_at=%s
                {where_clause}
                """,
                (
                    job.status, job.gcs_prefix, urls_json,
                    job.assets_discovered, job.assets_downloaded, job.assets_created,
                    job.error, job.celery_task_id, job.updated_at,
                    job.job_id
                ),
            )
            con.commit()
        finally:
            con.close()

    def get_job(self, job_id: str) -> Optional[AssetExtractionJob]:
        if not job_id:
            return None
        con = _connect()
        try:
            row = con.execute(
                """
                SELECT job_id, bot_id, org_id, status, gcs_prefix, page_urls,
                       assets_discovered, assets_downloaded, assets_created,
                       error, celery_task_id, created_at, updated_at
                FROM asset_extraction_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
            if not row:
                return None
            
            try:
                page_urls = json.loads(row[5]) if row[5] else []
            except Exception:
                page_urls = []

            return AssetExtractionJob(
                job_id=row[0],
                bot_id=row[1],
                org_id=row[2],
                status=row[3],
                gcs_prefix=row[4],
                page_urls=page_urls,
                assets_discovered=row[6],
                assets_downloaded=row[7],
                assets_created=row[8],
                error=row[9],
                celery_task_id=row[10],
                created_at=row[11],
                updated_at=row[12],
            )
        finally:
            con.close()

    def get_latest_job_for_bot(self, bot_id: str, prefix: Optional[str] = None) -> Optional[AssetExtractionJob]:
        if not bot_id:
            return None
        con = _connect()
        try:
            if prefix:
                row = con.execute(
                    """
                    SELECT job_id, bot_id, org_id, status, gcs_prefix, page_urls,
                           assets_discovered, assets_downloaded, assets_created,
                           error, celery_task_id, created_at, updated_at
                    FROM asset_extraction_jobs
                    WHERE bot_id = %s AND job_id LIKE %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (bot_id, prefix + "%"),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT job_id, bot_id, org_id, status, gcs_prefix, page_urls,
                           assets_discovered, assets_downloaded, assets_created,
                           error, celery_task_id, created_at, updated_at
                    FROM asset_extraction_jobs
                    WHERE bot_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (bot_id,),
                ).fetchone()
            if not row:
                return None
            
            try:
                page_urls = json.loads(row[5]) if row[5] else []
            except Exception:
                page_urls = []

            return AssetExtractionJob(
                job_id=row[0],
                bot_id=row[1],
                org_id=row[2],
                status=row[3],
                gcs_prefix=row[4],
                page_urls=page_urls,
                assets_discovered=row[6],
                assets_downloaded=row[7],
                assets_created=row[8],
                error=row[9],
                celery_task_id=row[10],
                created_at=row[11],
                updated_at=row[12],
            )
        finally:
            con.close()


class PostgresBotAssetRepository:
    """CRUD for bot_assets (business image cards and menu items)."""

    _SELECT_COLS = """asset_id, bot_id, org_id, name, description,
                      image_gcs_uri, image_public_url, link_url,
                      keywords, metadata_json, is_active, created_at, updated_at, asset_type"""

    def create_asset(self, asset: BotAsset) -> None:
        con = _connect()
        try:
            kw_json = json.dumps(asset.keywords if isinstance(asset.keywords, list) else [])
            metadata_json = json.dumps(asset.metadata if isinstance(asset.metadata, dict) else {})
            con.execute(
                """
                INSERT INTO bot_assets
                  (asset_id, bot_id, org_id, name, description,
                   image_gcs_uri, image_public_url, link_url,
                   keywords, metadata_json, is_active, created_at, updated_at, asset_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    asset.asset_id, asset.bot_id, asset.org_id,
                    asset.name, asset.description,
                    asset.image_gcs_uri, asset.image_public_url, asset.link_url,
                    kw_json, metadata_json, asset.is_active,
                    asset.created_at, asset.updated_at, asset.asset_type,
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_asset(self, asset: BotAsset) -> None:
        aid = (asset.asset_id or "").strip()
        if not aid:
            return
        con = _connect()
        try:
            kw_json = json.dumps(asset.keywords if isinstance(asset.keywords, list) else [])
            metadata_json = json.dumps(asset.metadata if isinstance(asset.metadata, dict) else {})
            now = asset.updated_at or _utc_now()
            con.execute(
                """
                UPDATE bot_assets
                SET name=%s, description=%s, image_gcs_uri=%s,
                    image_public_url=%s, link_url=%s, keywords=%s,
                    metadata_json=%s, is_active=%s, updated_at=%s
                WHERE asset_id=%s
                """,
                (
                    asset.name, asset.description,
                    asset.image_gcs_uri, asset.image_public_url, asset.link_url,
                    kw_json, metadata_json, asset.is_active, now, aid,
                ),
            )
            con.commit()
        finally:
            con.close()

    def get_asset(self, asset_id: str) -> Optional[BotAsset]:
        aid = (asset_id or "").strip()
        if not aid:
            return None
        con = _connect()
        try:
            row = con.execute(
                f"SELECT {self._SELECT_COLS} FROM bot_assets WHERE asset_id=%s",
                (aid,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_asset(row)
        finally:
            con.close()

    def list_assets_for_bot(
        self, bot_id: str, *, active_only: bool = False, asset_type: Optional[str] = None,
    ) -> List[BotAsset]:
        bid = (bot_id or "").strip()
        if not bid:
            return []
        con = _connect()
        try:
            where = "bot_id=%s"
            params: list = [bid]
            if active_only:
                where += " AND is_active=TRUE"
            if asset_type:
                where += " AND asset_type=%s"
                params.append(asset_type)
            rows = con.execute(
                f"SELECT {self._SELECT_COLS} FROM bot_assets WHERE {where} ORDER BY created_at ASC",
                tuple(params),
            ).fetchall()
            return [self._row_to_asset(r) for r in rows]
        finally:
            con.close()

    def list_assets_for_bot_paginated(
        self,
        bot_id: str,
        *,
        active_only: bool = False,
        asset_type: Optional[str] = None,
        page_size: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BotAsset], int, Optional[str]]:
        bid = (bot_id or "").strip()
        if not bid:
            return [], 0, None
        lim = max(1, min(int(page_size or 50), 200))
        off = max(0, int(offset or 0))
        con = _connect()
        try:
            where = "bot_id=%s"
            params: list = [bid]
            if active_only:
                where += " AND is_active=TRUE"
            if asset_type:
                where += " AND asset_type=%s"
                params.append(asset_type)

            stats_row = con.execute(
                f"SELECT COUNT(*), MAX(updated_at) FROM bot_assets WHERE {where}",
                tuple(params),
            ).fetchone()
            total_count = int(stats_row[0] or 0) if stats_row else 0
            latest_updated = str(stats_row[1]) if stats_row and stats_row[1] is not None else None

            page_params = [*params, lim, off]
            rows = con.execute(
                f"SELECT {self._SELECT_COLS} FROM bot_assets WHERE {where} ORDER BY created_at ASC LIMIT %s OFFSET %s",
                tuple(page_params),
            ).fetchall()
            return [self._row_to_asset(r) for r in rows], total_count, latest_updated
        finally:
            con.close()

    def delete_asset(self, bot_id: str, asset_id: str) -> None:
        bid = (bot_id or "").strip()
        aid = (asset_id or "").strip()
        if not bid or not aid:
            return
        con = _connect()
        try:
            con.execute(
                "DELETE FROM bot_assets WHERE bot_id=%s AND asset_id=%s",
                (bid, aid),
            )
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _row_to_asset(row) -> BotAsset:
        try:
            kw = json.loads(row[8]) if isinstance(row[8], str) else (row[8] or [])
        except (TypeError, ValueError):
            kw = []
        try:
            metadata = json.loads(row[9]) if isinstance(row[9], str) else (row[9] or {})
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return BotAsset(
            asset_id=row[0],
            bot_id=row[1],
            org_id=row[2],
            name=row[3],
            description=row[4] or "",
            image_gcs_uri=row[5],
            image_public_url=row[6] or "",
            link_url=row[7],
            keywords=kw if isinstance(kw, list) else [],
            metadata=metadata,
            is_active=bool(row[10]),
            asset_type=row[13] if len(row) > 13 and row[13] else "image",
            created_at=row[11],
            updated_at=row[12],
        )
