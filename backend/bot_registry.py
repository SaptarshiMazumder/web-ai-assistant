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
    con = sqlite3.connect(_db_path())
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
          bot_id TEXT PRIMARY KEY,
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
    return con


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
    display_name: str
    publishable_key: str
    secret_key: str


@dataclass
class BotRecord:
    bot_id: str
    display_name: str
    publishable_key: str
    secret_key: str
    created_at: str
    updated_at: str


@dataclass
class BotDomainRecord:
    bot_id: str
    hostname: str
    status: str
    verification_token: str
    verified_at: Optional[str]
    created_at: str
    updated_at: str


def create_bot(display_name: str) -> Bot:
    bot_id = _new_bot_id()
    pk = _new_publishable_key()
    sk = _new_secret_key()
    now = _utc_now()
    con = _connect()
    try:
        con.execute(
            """
            INSERT INTO bots(bot_id, display_name, publishable_key, secret_key, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bot_id, display_name, pk, sk, now, now),
        )
        con.commit()
        return Bot(bot_id=bot_id, display_name=display_name, publishable_key=pk, secret_key=sk)
    finally:
        con.close()


def get_bot_by_publishable_key(publishable_key: str) -> Optional[Bot]:
    pk = (publishable_key or "").strip()
    if not pk:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, display_name, publishable_key, secret_key FROM bots WHERE publishable_key = ?",
            (pk,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], display_name=row[1], publishable_key=row[2], secret_key=row[3])
    finally:
        con.close()


def get_bot_by_secret_key(secret_key: str) -> Optional[Bot]:
    sk = (secret_key or "").strip()
    if not sk:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, display_name, publishable_key, secret_key FROM bots WHERE secret_key = ?",
            (sk,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], display_name=row[1], publishable_key=row[2], secret_key=row[3])
    finally:
        con.close()


def get_bot(bot_id: str) -> Optional[Bot]:
    bid = (bot_id or "").strip()
    if not bid:
        return None
    con = _connect()
    try:
        row = con.execute(
            "SELECT bot_id, display_name, publishable_key, secret_key FROM bots WHERE bot_id = ?",
            (bid,),
        ).fetchone()
        if not row:
            return None
        return Bot(bot_id=row[0], display_name=row[1], publishable_key=row[2], secret_key=row[3])
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
            SELECT bot_id, display_name, publishable_key, secret_key, created_at, updated_at
            FROM bots
            WHERE bot_id = ?
            """,
            (bid,),
        ).fetchone()
        if not row:
            return None
        return BotRecord(
            bot_id=row[0],
            display_name=row[1],
            publishable_key=row[2],
            secret_key=row[3],
            created_at=row[4],
            updated_at=row[5],
        )
    finally:
        con.close()


def list_bots() -> List[BotRecord]:
    con = _connect()
    try:
        rows = con.execute(
            """
            SELECT bot_id, display_name, publishable_key, secret_key, created_at, updated_at
            FROM bots
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [
            BotRecord(
                bot_id=row[0],
                display_name=row[1],
                publishable_key=row[2],
                secret_key=row[3],
                created_at=row[4],
                updated_at=row[5],
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
        con.execute(
            """
            INSERT INTO bot_domains(bot_id, hostname, status, verification_token, verified_at, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, NULL, ?, ?)
            """,
            (bid, host, token, now, now),
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
            SELECT bot_id, hostname, status, verification_token, verified_at, created_at, updated_at
            FROM bot_domains
            WHERE bot_id = ?
            ORDER BY created_at DESC
            """,
            (bid,),
        ).fetchall()
        return [
            BotDomainRecord(
                bot_id=row[0],
                hostname=row[1],
                status=row[2],
                verification_token=row[3],
                verified_at=row[4],
                created_at=row[5],
                updated_at=row[6],
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

