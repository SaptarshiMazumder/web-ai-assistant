import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> str:
    # Local persistent registry. For production, swap to Postgres.
    root = os.path.join(os.path.dirname(__file__), "_data")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "bot_registry.sqlite3")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path(), timeout=30)
    con.execute("PRAGMA busy_timeout=5000;")
    try:
        con.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        # Some mounted volumes (esp. on Windows) don't support WAL.
        con.execute("PRAGMA journal_mode=DELETE;")
    try:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS organizations_name_unique ON organizations (lower(name))")
    except (sqlite3.OperationalError, sqlite3.IntegrityError):
        # If duplicates already exist, index creation can fail; skip to keep app running.
        pass
    con.execute(
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
        """
    )
    con.execute(
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
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_corpora (
          bot_id TEXT PRIMARY KEY,
          corpus_resource TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
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
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          user_id TEXT PRIMARY KEY,
          idp_subject TEXT UNIQUE,
          email TEXT UNIQUE,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS org_memberships (
          user_id TEXT NOT NULL,
          org_id TEXT NOT NULL,
          role TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (user_id, org_id)
        )
        """
    )
    _ensure_columns(con, "bots", {"org_id": "TEXT"})
    _ensure_columns(con, "bot_domains", {"org_id": "TEXT"})
    _ensure_columns(
        con,
        "organizations",
        {
            "plan": "TEXT",
            "stripe_customer_id": "TEXT",
            "stripe_subscription_id": "TEXT",
        },
    )
    _ensure_default_org(con)
    con.execute("UPDATE bots SET org_id = 'org_default' WHERE org_id IS NULL OR org_id = ''")
    con.execute("UPDATE bot_domains SET org_id = 'org_default' WHERE org_id IS NULL OR org_id = ''")
    con.commit()
    return con


def _ensure_columns(con: sqlite3.Connection, table: str, cols: Dict[str, str]) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, col_type in cols.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _ensure_default_org(con: sqlite3.Connection) -> None:
    now = _utc_now()
    try:
        con.execute(
            """
            INSERT OR IGNORE INTO organizations(org_id, name, status, created_at, updated_at)
            VALUES ('org_default', 'Default Org', 'active', ?, ?)
            """,
            (now, now),
        )
    except sqlite3.OperationalError:
        # If the DB is locked during startup, retry on next request.
        return


def _normalize_hostname(hostname: str) -> str:
    h = (hostname or "").strip().lower()
    h = re.sub(r"^https?://", "", h)
    h = h.split("/")[0]
    h = h.split(":")[0]
    # very light validation; full IDNA/punycode handling can be added later
    return h


def _new_bot_id() -> str:
    return "bot_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_publishable_key() -> str:
    return "pk_" + secrets.token_urlsafe(24)


def _new_secret_key() -> str:
    return "sk_" + secrets.token_urlsafe(32)


@dataclass
class Bot:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str


@dataclass
class BotRecord:
    bot_id: str
    org_id: str
    display_name: str
    publishable_key: str
    secret_key: str
    created_at: str
    updated_at: str


@dataclass
class BotDomainRecord:
    org_id: str
    bot_id: str
    hostname: str
    status: str
    verification_token: str
    verified_at: Optional[str]
    created_at: str
    updated_at: str


def create_bot(display_name: str, org_id: str) -> Bot:
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
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (bot_id, oid, display_name, pk, sk, now, now),
        )
        con.commit()
        return Bot(bot_id=bot_id, org_id=oid, display_name=display_name, publishable_key=pk, secret_key=sk)
    finally:
        con.close()


def get_bot_by_publishable_key(publishable_key: str) -> Optional[Bot]:
    pk = (publishable_key or "").strip()
    if not pk:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE publishable_key = ?",
            (pk,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
    finally:
        con.close()


def get_bot_by_secret_key(secret_key: str) -> Optional[Bot]:
    sk = (secret_key or "").strip()
    if not sk:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE secret_key = ?",
            (sk,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
    finally:
        con.close()


def get_bot(bot_id: str) -> Optional[Bot]:
    bid = (bot_id or "").strip()
    if not bid:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, org_id, display_name, publishable_key, secret_key FROM bots WHERE bot_id = ?",
            (bid,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], org_id=row[1], display_name=row[2], publishable_key=row[3], secret_key=row[4])
    finally:
        con.close()


def get_bot_record(bot_id: str) -> Optional[BotRecord]:
    bid = (bot_id or "").strip()
    if not bid:
        return None
    con = _connect()
    try:
        row = con.execute(
            """
            SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at
            FROM bots
            WHERE bot_id = ?
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


def list_bots(org_id: Optional[str] = None) -> List[BotRecord]:
    con = _connect()
    try:
        if org_id:
            rows = con.execute(
                """
                SELECT bot_id, org_id, display_name, publishable_key, secret_key, created_at, updated_at
                FROM bots
                WHERE org_id = ?
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


def add_domain(bot_id: str, hostname: str) -> Tuple[str, str]:
    """
    Returns (status, verification_token). If already exists, returns existing.
    """
    bid = (bot_id or "").strip()
    host = _normalize_hostname(hostname)
    if not bid or not host:
        raise ValueError("bot_id and hostname are required")
    con = _connect()
    now = _utc_now()
    token = "verify_" + secrets.token_urlsafe(24)
    try:
        existing = con.execute(
            "SELECT status, verification_token FROM bot_domains WHERE bot_id = ? AND hostname = ?",
            (bid, host),
        ).fetchone()
        if existing:
            return (existing[0], existing[1])
        bot = get_bot(bid)
        if not bot:
            raise ValueError("Unknown bot_id")
        con.execute(
            """
            INSERT INTO bot_domains(org_id, bot_id, hostname, status, verification_token, verified_at, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, NULL, ?, ?)
            """,
            (bot.org_id, bid, host, token, now, now),
        )
        con.commit()
        return ("pending", token)
    finally:
        con.close()


def list_verified_hosts(bot_id: str) -> List[str]:
    bid = (bot_id or "").strip()
    if not bid:
        return []
    con = _connect()
    try:
        rows = con.execute(
            "SELECT hostname FROM bot_domains WHERE bot_id = ? AND status = 'verified'",
            (bid,),
        ).fetchall()
        return [r[0] for r in rows] if rows else []
    finally:
        con.close()


def get_domain_status(bot_id: str, hostname: str) -> Optional[Tuple[str, str]]:
    bid = (bot_id or "").strip()
    host = _normalize_hostname(hostname)
    if not bid or not host:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT status, verification_token FROM bot_domains WHERE bot_id = ? AND hostname = ?",
            (bid, host),
        ).fetchone()
        if not row:
            return None
        return (row[0], row[1])
    finally:
        con.close()


def mark_domain_verified(bot_id: str, hostname: str) -> None:
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
            SET status='verified', verified_at=?, updated_at=?
            WHERE bot_id=? AND hostname=?
            """,
            (now, now, bid, host),
        )
        con.commit()
    finally:
        con.close()


def list_domains(bot_id: str) -> List[BotDomainRecord]:
    bid = (bot_id or "").strip()
    if not bid:
        return []
    con = _connect()
    try:
        rows = con.execute(
            """
            SELECT org_id, bot_id, hostname, status, verification_token, verified_at, created_at, updated_at
            FROM bot_domains
            WHERE bot_id = ?
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


def upsert_bot_corpus(bot_id: str, corpus_resource: str) -> None:
    bid = (bot_id or "").strip()
    if not bid:
        raise ValueError("bot_id required")
    now = _utc_now()
    con = _connect()
    try:
        con.execute(
            """
            INSERT INTO bot_corpora(bot_id, corpus_resource, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(bot_id) DO UPDATE SET
              corpus_resource=excluded.corpus_resource,
              updated_at=excluded.updated_at
            """,
            (bid, corpus_resource, now, now),
        )
        con.commit()
    finally:
        con.close()


def get_bot_corpus(bot_id: str) -> Optional[str]:
    bid = (bot_id or "").strip()
    if not bid:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT corpus_resource FROM bot_corpora WHERE bot_id = ?",
            (bid,),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def create_org(name: str) -> str:
    oid = "org_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
    now = _utc_now()
    con = _connect()
    try:
        try:
            con.execute(
                """
                INSERT INTO organizations(org_id, name, status, created_at, updated_at)
                VALUES (?, ?, 'active', ?, ?)
                """,
                (oid, name, now, now),
            )
            con.commit()
            return oid
        except sqlite3.IntegrityError:
            existing = get_org_by_name(name)
            if existing:
                return existing["org_id"]
            raise
    finally:
        con.close()


def list_orgs() -> List[Dict[str, Any]]:
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
            {
                "org_id": r[0],
                "name": r[1],
                "status": r[2],
                "plan": r[3],
                "stripe_customer_id": r[4],
                "stripe_subscription_id": r[5],
                "created_at": r[6],
                "updated_at": r[7],
            }
            for r in (rows or [])
        ]
    finally:
        con.close()


def get_org_by_name(name: str) -> Optional[Dict[str, Any]]:
    nm = (name or "").strip()
    if not nm:
        return None
    con = _connect()
    try:
        row = con.execute(
            """
            SELECT org_id, name, status, plan, stripe_customer_id, stripe_subscription_id, created_at, updated_at
            FROM organizations
            WHERE lower(name) = lower(?)
            LIMIT 1
            """,
            (nm,),
        ).fetchone()
        if not row:
            return None
        return {
            "org_id": row[0],
            "name": row[1],
            "status": row[2],
            "plan": row[3],
            "stripe_customer_id": row[4],
            "stripe_subscription_id": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }
    finally:
        con.close()


def get_org(org_id: str) -> Optional[Dict[str, Any]]:
    oid = (org_id or "").strip()
    if not oid:
        return None
    con = _connect()
    try:
        row = con.execute(
            """
            SELECT org_id, name, status, plan, stripe_customer_id, stripe_subscription_id, created_at, updated_at
            FROM organizations
            WHERE org_id = ?
            """,
            (oid,),
        ).fetchone()
        if not row:
            return None
        return {
            "org_id": row[0],
            "name": row[1],
            "status": row[2],
            "plan": row[3],
            "stripe_customer_id": row[4],
            "stripe_subscription_id": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }
    finally:
        con.close()


def update_org_name(org_id: str, name: str) -> None:
    oid = (org_id or "").strip()
    nm = (name or "").strip()
    if not oid or not nm:
        raise ValueError("org_id and name required")
    now = _utc_now()
    con = _connect()
    try:
        con.execute(
            "UPDATE organizations SET name = ?, updated_at = ? WHERE org_id = ?",
            (nm, now, oid),
        )
        con.commit()
    finally:
        con.close()


def set_org_status(org_id: str, status: str) -> None:
    now = _utc_now()
    con = _connect()
    try:
        con.execute(
            "UPDATE organizations SET status = ?, updated_at = ? WHERE org_id = ?",
            (status, now, org_id),
        )
        con.commit()
    finally:
        con.close()


def upsert_user_from_claims(*, subject: str, email: str) -> Dict[str, Any]:
    now = _utc_now()
    con = _connect()
    try:
        existing = con.execute(
            "SELECT user_id, idp_subject, email FROM users WHERE idp_subject = ? OR email = ?",
            (subject, email),
        ).fetchone()
        if existing:
            user_id = existing[0]
            con.execute(
                """
                UPDATE users SET idp_subject = ?, email = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (subject, email, now, user_id),
            )
            con.commit()
            return {"user_id": user_id, "idp_subject": subject, "email": email}
        user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
        con.execute(
            """
            INSERT INTO users(user_id, idp_subject, email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, subject, email, now, now),
        )
        con.commit()
        return {"user_id": user_id, "idp_subject": subject, "email": email}
    finally:
        con.close()


def create_user_placeholder(email: str) -> Dict[str, Any]:
    em = (email or "").strip().lower()
    if not em:
        raise ValueError("email required")
    now = _utc_now()
    con = _connect()
    try:
        existing = con.execute(
            "SELECT user_id, idp_subject, email FROM users WHERE email = ?",
            (em,),
        ).fetchone()
        if existing:
            return {"user_id": existing[0], "idp_subject": existing[1], "email": existing[2]}
        user_id = "user_" + secrets.token_urlsafe(10).replace("-", "_").replace(".", "_")
        con.execute(
            """
            INSERT INTO users(user_id, idp_subject, email, created_at, updated_at)
            VALUES (?, NULL, ?, ?, ?)
            """,
            (user_id, em, now, now),
        )
        con.commit()
        return {"user_id": user_id, "idp_subject": None, "email": em}
    finally:
        con.close()


def get_user_by_subject(subject: str) -> Optional[Dict[str, Any]]:
    sub = (subject or "").strip()
    if not sub:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT user_id, idp_subject, email FROM users WHERE idp_subject = ?",
            (sub,),
        ).fetchone()
        if not row:
            return None
        return {"user_id": row[0], "idp_subject": row[1], "email": row[2]}
    finally:
        con.close()


def add_membership(org_id: str, user_id: str, role: str) -> None:
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
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, org_id) DO UPDATE SET
              role=excluded.role,
              updated_at=excluded.updated_at
            """,
            (uid, oid, role, now, now),
        )
        con.commit()
    finally:
        con.close()


def get_org_memberships(user_id: str) -> List[Dict[str, Any]]:
    uid = (user_id or "").strip()
    if not uid:
        return []
    con = _connect()
    try:
        rows = con.execute(
            "SELECT org_id, role FROM org_memberships WHERE user_id = ?",
            (uid,),
        ).fetchall()
        return [{"org_id": r[0], "role": r[1]} for r in (rows or [])]
    finally:
        con.close()


def list_org_members(org_id: str) -> List[Dict[str, Any]]:
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
            WHERE m.org_id = ?
            ORDER BY m.created_at DESC
            """,
            (oid,),
        ).fetchall()
        return [
            {"user_id": r[0], "email": r[1], "role": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in (rows or [])
        ]
    finally:
        con.close()

