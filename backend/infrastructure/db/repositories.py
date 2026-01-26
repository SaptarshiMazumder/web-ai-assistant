import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg import errors as pg_errors

from domain.entities import Bot, BotDomainRecord, BotRecord, OrgMemberRecord, OrgRecord, UserRecord
from infrastructure.db.connection import ensure_default_org, get_connection


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    con = get_connection()
    ensure_default_org(con, _utc_now())
    return con


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


class PostgresBotRepository:
    def create_bot(self, display_name: str, org_id: str) -> Bot:
        bot_id = _new_bot_id()
        pk = _new_publishable_key()
        sk = _new_secret_key()
        now = _utc_now()
        oid = (org_id or "").strip() or "org_default"
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO bots(bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (bot_id, oid, display_name, pk, sk, now, now),
            )
            con.commit()
            return Bot(bot_id=bot_id, org_id=oid, display_name=display_name, publishable_key=pk, secret_key=sk)
        finally:
            con.close()

    def get_bot_by_publishable_key(self, publishable_key: str) -> Optional[Bot]:
        pk = (publishable_key or "").strip()
        if not pk:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE publishable_key = %s",
                (pk,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
        finally:
            con.close()

    def get_bot_by_secret_key(self, secret_key: str) -> Optional[Bot]:
        sk = (secret_key or "").strip()
        if not sk:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE secret_key = %s",
                (sk,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
        finally:
            con.close()

    def get_bot(self, bot_id: str) -> Optional[Bot]:
        bid = (bot_id or "").strip()
        if not bid:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE bot_id = %s",
                (bid,),
            ).fetchone()
            if not row:
                return None
            return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
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
                SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at
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
            )
        finally:
            con.close()

    def list_bots(self, org_id: Optional[str] = None) -> List[BotRecord]:
        con = _connect()
        try:
            if org_id:
                rows = con.execute(
                    """
                    SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at
                    FROM bots
                    WHERE org_id = %s
                    ORDER BY created_at DESC
                    """,
                    (org_id,),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at
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
                )
                for row in (rows or [])
            ]
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
    def upsert_user_from_claims(self, *, subject: str, email: str) -> UserRecord:
        now = _utc_now()
        con = _connect()
        try:
            existing = con.execute(
                "SELECT user_id, idp_subject, email FROM users WHERE idp_subject = %s OR email = %s",
                (subject, email),
            ).fetchone()
            if existing:
                user_id = existing[0]
                con.execute(
                    """
                    UPDATE users SET idp_subject = %s, email = %s, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (subject, email, now, user_id),
                )
                con.commit()
                return UserRecord(user_id=user_id, idp_subject=subject, email=email)
            user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
            con.execute(
                """
                INSERT INTO users(user_id, idp_subject, email, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, subject, email, now, now),
            )
            con.commit()
            return UserRecord(user_id=user_id, idp_subject=subject, email=email)
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
                "SELECT user_id, idp_subject, email FROM users WHERE email = %s",
                (em,),
            ).fetchone()
            if existing:
                return UserRecord(user_id=existing[0], idp_subject=existing[1], email=existing[2])
            user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
            con.execute(
                """
                INSERT INTO users(user_id, idp_subject, email, created_at, updated_at)
                VALUES (%s, NULL, %s, %s, %s)
                """,
                (user_id, em, now, now),
            )
            con.commit()
            return UserRecord(user_id=user_id, idp_subject=None, email=em)
        finally:
            con.close()

    def get_user_by_subject(self, subject: str) -> Optional[UserRecord]:
        sub = (subject or "").strip()
        if not sub:
            return None
        con = _connect()
        try:
            row = con.execute(
                "SELECT user_id, idp_subject, email FROM users WHERE idp_subject = %s",
                (sub,),
            ).fetchone()
            if not row:
                return None
            return UserRecord(user_id=row[0], idp_subject=row[1], email=row[2])
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
                SELECT u.user_id, u.email, m.role, m.created_at, m.updated_at
                FROM org_memberships m
                JOIN users u ON u.user_id = m.user_id
                WHERE m.org_id = %s
                ORDER BY m.created_at DESC
                """,
                (oid,),
            ).fetchall()
            return [
                OrgMemberRecord(user_id=r[0], email=r[1], role=r[2], created_at=r[3], updated_at=r[4])
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
