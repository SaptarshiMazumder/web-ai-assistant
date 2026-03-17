from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from typing import Any, Dict, Optional

from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError

from common.config import config
from infrastructure.db.connection import get_connection


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _to_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str):
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


class RetryablePersistError(Exception):
    pass


class OutOfOrderPersistError(RetryablePersistError):
    pass


class RuntimePersistenceWorker:
    def __init__(self) -> None:
        self._logger = logging.getLogger("runtime_persistence_worker")
        self._trace_storage = bool(getattr(config, "CHAT_STORAGE_TRACE_LOGS", False))
        self._redis_url = (
            (getattr(config, "REDIS_URL", "") or "").strip()
            or (getattr(config, "CELERY_BROKER_URL", "") or "").strip()
        )
        self._stream = (
            getattr(config, "CHAT_RUNTIME_PERSIST_STREAM_KEY", "") or "webai:chatruntime:persist:v1"
        ).strip()
        self._stream_maxlen = int(getattr(config, "CHAT_RUNTIME_PERSIST_STREAM_MAXLEN", 200000))
        self._stream_maxlen = max(1000, min(self._stream_maxlen, 2_000_000))
        self._group = (
            getattr(config, "CHAT_RUNTIME_PERSIST_CONSUMER_GROUP", "") or "webai:chatruntime:persist:group:v1"
        ).strip()
        configured_consumer = (getattr(config, "CHAT_RUNTIME_PERSIST_CONSUMER_NAME", "") or "").strip()
        if configured_consumer:
            self._consumer = configured_consumer
        else:
            self._consumer = f"{socket.gethostname()}-{os.getpid()}"
        self._max_retries = _to_int(getattr(config, "CHAT_RUNTIME_PERSIST_MAX_RETRIES", 8), 8)
        self._max_retries = max(1, min(self._max_retries, 100))
        self._dlq_stream = (
            getattr(config, "CHAT_RUNTIME_PERSIST_DLQ_STREAM_KEY", "") or "webai:chatruntime:persist:dlq:v1"
        ).strip()
        self._redis: Optional[AsyncRedis] = None

    def _trace_log(self, message: str, *args: object) -> None:
        if self._trace_storage:
            self._logger.warning(message, *args)

    async def _ensure_redis(self) -> Optional[AsyncRedis]:
        if not self._redis_url:
            return None
        if self._redis is not None:
            return self._redis
        try:
            self._redis = AsyncRedis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            return self._redis
        except Exception as exc:
            self._logger.warning("Runtime persistence Redis unavailable: %s", exc)
            if self._redis is not None:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            self._redis = None
            return None

    async def _ensure_group(self) -> bool:
        redis = await self._ensure_redis()
        if redis is None:
            return False
        try:
            await redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
            return True
        except ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                return True
            self._logger.warning("Runtime persistence xgroup create failed: %s", exc)
            return False
        except Exception as exc:
            self._logger.warning("Runtime persistence xgroup create failed: %s", exc)
            return False

    async def run(self) -> None:
        while True:
            ok = await self._ensure_group()
            if not ok:
                await asyncio.sleep(2.0)
                continue
            redis = await self._ensure_redis()
            if redis is None:
                await asyncio.sleep(2.0)
                continue
            try:
                streams = await redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=100,
                    block=4000,
                )
                if not streams:
                    continue
                for _stream_name, entries in streams:
                    for entry_id, fields in entries:
                        await self._process_stream_entry(entry_id, fields)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception("Runtime persistence loop error: %s", exc)
                await asyncio.sleep(1.0)

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass
        self._redis = None

    async def _process_stream_entry(self, entry_id: str, fields: Dict[str, str]) -> None:
        redis = await self._ensure_redis()
        if redis is None:
            return
        try:
            await asyncio.to_thread(self._process_event_sync, fields)
        except OutOfOrderPersistError as exc:
            await self._retry_or_dlq(entry_id, fields, reason=f"out_of_order:{exc}")
            return
        except RetryablePersistError as exc:
            await self._retry_or_dlq(entry_id, fields, reason=f"retryable:{exc}")
            return
        except Exception as exc:
            await self._retry_or_dlq(entry_id, fields, reason=f"error:{type(exc).__name__}:{exc}")
            return
        await redis.xack(self._stream, self._group, entry_id)
        self._trace_log(
            "chat_storage db_event_acked entry_id=%s event_id=%s event_type=%s session_id=%s seq=%s",
            entry_id,
            str(fields.get("event_id") or ""),
            str(fields.get("event_type") or ""),
            str(fields.get("session_id") or ""),
            str(fields.get("seq") or ""),
        )

    async def _retry_or_dlq(self, entry_id: str, fields: Dict[str, str], *, reason: str) -> None:
        redis = await self._ensure_redis()
        if redis is None:
            return
        retry_count = _to_int(fields.get("retry_count"), 0)
        next_fields = dict(fields)
        next_fields["last_error"] = reason[:400]
        next_fields["updated_at"] = _utc_now()
        if retry_count >= self._max_retries:
            next_fields["failed_at"] = _utc_now()
            try:
                await redis.xadd(
                    self._dlq_stream,
                    next_fields,
                    maxlen=self._stream_maxlen,
                    approximate=True,
                )
            finally:
                await redis.xack(self._stream, self._group, entry_id)
            return
        next_fields["retry_count"] = str(retry_count + 1)
        await redis.xadd(
            self._stream,
            next_fields,
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        await redis.xack(self._stream, self._group, entry_id)

    def _process_event_sync(self, fields: Dict[str, str]) -> None:
        event_type = str(fields.get("event_type") or "").strip()
        session_id = str(fields.get("session_id") or "").strip()
        seq = _to_int(fields.get("seq"), 0)
        idempotency_key = str(fields.get("idempotency_key") or fields.get("event_id") or "").strip()
        event_id = str(fields.get("event_id") or "").strip()
        created_at = str(fields.get("created_at") or _utc_now())
        payload = _parse_json(fields.get("payload"), {})
        if not event_type:
            return

        con = get_connection()
        try:
            last_seq = 0
            if session_id and seq > 0:
                row = con.execute(
                    "SELECT last_seq FROM conversation_persist_session_seq WHERE session_id = %s FOR UPDATE",
                    (session_id,),
                ).fetchone()
                last_seq = int(row[0]) if row else 0
                if seq <= last_seq:
                    con.commit()
                    return
                if seq > (last_seq + 1):
                    raise OutOfOrderPersistError(f"expected {last_seq + 1}, got {seq}")

            if idempotency_key:
                result = con.execute(
                    """
                    INSERT INTO conversation_persist_events(
                      idempotency_key, event_id, session_id, event_type, created_at, processed_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    """,
                    (idempotency_key, event_id, session_id, event_type, created_at, _utc_now()),
                )
                if result.rowcount == 0:
                    if session_id and seq > 0:
                        con.execute(
                            """
                            INSERT INTO conversation_persist_session_seq(session_id, last_seq, updated_at)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET
                              last_seq = GREATEST(conversation_persist_session_seq.last_seq, EXCLUDED.last_seq),
                              updated_at = EXCLUDED.updated_at
                            """,
                            (session_id, seq, _utc_now()),
                        )
                    con.commit()
                    self._trace_log(
                        "chat_storage db_event_duplicate_skip event_id=%s event_type=%s session_id=%s seq=%s idempotency_key=%s",
                        event_id,
                        event_type,
                        session_id,
                        seq,
                        idempotency_key,
                    )
                    return

            self._apply_event(con, event_type=event_type, payload=payload)

            if session_id and seq > 0:
                con.execute(
                    """
                    INSERT INTO conversation_persist_session_seq(session_id, last_seq, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                      last_seq = GREATEST(conversation_persist_session_seq.last_seq, EXCLUDED.last_seq),
                      updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, seq, _utc_now()),
                )
            con.commit()
            self._trace_log(
                "chat_storage db_event_committed event_id=%s event_type=%s session_id=%s seq=%s idempotency_key=%s",
                event_id,
                event_type,
                session_id,
                seq,
                idempotency_key,
            )
        except OutOfOrderPersistError:
            con.rollback()
            raise
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _apply_event(self, con, *, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "session_upsert":
            self._apply_session_upsert(con, _parse_json(payload.get("session"), {}) if isinstance(payload, dict) else {})
            return
        if event_type == "message_upsert":
            msg = _parse_json(payload.get("message"), {}) if isinstance(payload, dict) else {}
            sess = _parse_json(payload.get("session"), {}) if isinstance(payload, dict) else {}
            self._apply_message_upsert(con, msg)
            if isinstance(sess, dict) and sess:
                self._apply_session_upsert(con, sess)
            return
        if event_type == "contact_upsert":
            self._apply_contact_upsert(con, _parse_json(payload.get("contact"), {}) if isinstance(payload, dict) else {})
            return
        if event_type == "line_user_session_upsert":
            self._apply_line_mapping_upsert(
                con,
                _parse_json(payload.get("line_user_session"), {}) if isinstance(payload, dict) else {},
            )
            return
        if event_type == "instagram_user_session_upsert":
            self._apply_instagram_mapping_upsert(
                con,
                _parse_json(payload.get("instagram_user_session"), {}) if isinstance(payload, dict) else {},
            )
            return
        if event_type == "handoff_upsert":
            self._apply_handoff_upsert(con, _parse_json(payload.get("handoff"), {}) if isinstance(payload, dict) else {})
            return
        if event_type == "escalation_upsert":
            self._apply_escalation_upsert(
                con,
                _parse_json(payload.get("escalation"), {}) if isinstance(payload, dict) else {},
            )
            return

    def _apply_session_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        session_id = str(payload.get("session_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        org_id = str(payload.get("org_id") or "").strip()
        channel = str(payload.get("channel") or "").strip()
        if not session_id or not bot_id or not org_id or not channel:
            return
        con.execute(
            """
            INSERT INTO conversation_sessions(
              session_id, bot_id, org_id, channel, status, title, site_url, site_title,
              message_count, started_at, last_active_at, ended_at, user_agent, ip
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
              bot_id = EXCLUDED.bot_id,
              org_id = EXCLUDED.org_id,
              channel = EXCLUDED.channel,
              status = EXCLUDED.status,
              title = EXCLUDED.title,
              site_url = EXCLUDED.site_url,
              site_title = EXCLUDED.site_title,
              message_count = GREATEST(conversation_sessions.message_count, EXCLUDED.message_count),
              started_at = LEAST(conversation_sessions.started_at, EXCLUDED.started_at),
              last_active_at = GREATEST(conversation_sessions.last_active_at, EXCLUDED.last_active_at),
              ended_at = COALESCE(EXCLUDED.ended_at, conversation_sessions.ended_at),
              user_agent = COALESCE(EXCLUDED.user_agent, conversation_sessions.user_agent),
              ip = COALESCE(EXCLUDED.ip, conversation_sessions.ip)
            """,
            (
                session_id,
                bot_id,
                org_id,
                channel,
                str(payload.get("status") or "active"),
                payload.get("title"),
                payload.get("site_url"),
                payload.get("site_title"),
                _to_int(payload.get("message_count"), 0),
                str(payload.get("started_at") or _utc_now()),
                str(payload.get("last_active_at") or _utc_now()),
                payload.get("ended_at"),
                payload.get("user_agent"),
                payload.get("ip"),
            ),
        )
        self._trace_log(
            "chat_storage db_session_upsert session_id=%s bot_id=%s org_id=%s channel=%s message_count=%s",
            session_id,
            bot_id,
            org_id,
            channel,
            _to_int(payload.get("message_count"), 0),
        )

    def _apply_message_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        message_id = str(payload.get("message_id") or "").strip()
        if not message_id:
            return
        con.execute(
            """
            INSERT INTO conversation_messages(
              message_id, session_id, bot_id, role, sender_name, content, citations, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (message_id) DO NOTHING
            """,
            (
                message_id,
                str(payload.get("session_id") or ""),
                str(payload.get("bot_id") or ""),
                str(payload.get("role") or ""),
                payload.get("sender_name"),
                str(payload.get("content") or ""),
                _to_json(_parse_json(payload.get("citations"), [])),
                str(payload.get("created_at") or _utc_now()),
            ),
        )
        self._trace_log(
            "chat_storage db_message_upsert message_id=%s session_id=%s bot_id=%s role=%s content_len=%s",
            message_id,
            str(payload.get("session_id") or ""),
            str(payload.get("bot_id") or ""),
            str(payload.get("role") or ""),
            len(str(payload.get("content") or "")),
        )

    def _apply_contact_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        contact_id = str(payload.get("contact_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        channel = str(payload.get("channel") or "").strip()
        external_user_id = str(payload.get("external_user_id") or "").strip()
        if not contact_id or not bot_id or not channel or not external_user_id:
            return
        con.execute(
            """
            INSERT INTO conversation_channel_contacts(
              contact_id, bot_id, channel, external_user_id, display_name, metadata_json,
              current_session_id, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bot_id, channel, external_user_id) DO UPDATE SET
              display_name = COALESCE(EXCLUDED.display_name, conversation_channel_contacts.display_name),
              metadata_json = EXCLUDED.metadata_json,
              current_session_id = COALESCE(EXCLUDED.current_session_id, conversation_channel_contacts.current_session_id),
              updated_at = EXCLUDED.updated_at
            """,
            (
                contact_id,
                bot_id,
                channel,
                external_user_id,
                payload.get("display_name"),
                _to_json(_parse_json(payload.get("metadata"), {})),
                payload.get("current_session_id"),
                str(payload.get("created_at") or _utc_now()),
                str(payload.get("updated_at") or _utc_now()),
            ),
        )

    def _apply_line_mapping_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        line_user_id = str(payload.get("line_user_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        if not line_user_id or not bot_id:
            return
        con.execute(
            """
            INSERT INTO line_user_sessions(
              line_user_id, bot_id, session_id, is_escalated, awaiting_escalation_msg, awaiting_staff_takeover,
              display_name, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (line_user_id, bot_id) DO UPDATE SET
              session_id = EXCLUDED.session_id,
              is_escalated = EXCLUDED.is_escalated,
              awaiting_escalation_msg = EXCLUDED.awaiting_escalation_msg,
              awaiting_staff_takeover = EXCLUDED.awaiting_staff_takeover,
              display_name = COALESCE(EXCLUDED.display_name, line_user_sessions.display_name),
              updated_at = EXCLUDED.updated_at
            """,
            (
                line_user_id,
                bot_id,
                str(payload.get("session_id") or ""),
                bool(payload.get("is_escalated")),
                bool(payload.get("awaiting_escalation_msg")),
                bool(payload.get("awaiting_staff_takeover")),
                payload.get("display_name"),
                str(payload.get("created_at") or _utc_now()),
                str(payload.get("updated_at") or _utc_now()),
            ),
        )

    def _apply_instagram_mapping_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        ig_user_id = str(payload.get("ig_user_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        if not ig_user_id or not bot_id:
            return
        con.execute(
            """
            INSERT INTO instagram_user_sessions(
              ig_user_id, bot_id, session_id, is_escalated, awaiting_escalation_msg, awaiting_staff_takeover,
              created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ig_user_id, bot_id) DO UPDATE SET
              session_id = EXCLUDED.session_id,
              is_escalated = EXCLUDED.is_escalated,
              awaiting_escalation_msg = EXCLUDED.awaiting_escalation_msg,
              awaiting_staff_takeover = EXCLUDED.awaiting_staff_takeover,
              updated_at = EXCLUDED.updated_at
            """,
            (
                ig_user_id,
                bot_id,
                str(payload.get("session_id") or ""),
                bool(payload.get("is_escalated")),
                bool(payload.get("awaiting_escalation_msg")),
                bool(payload.get("awaiting_staff_takeover")),
                str(payload.get("created_at") or _utc_now()),
                str(payload.get("updated_at") or _utc_now()),
            ),
        )

    def _apply_handoff_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        session_id = str(payload.get("session_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        if not session_id or not bot_id:
            return
        con.execute(
            """
            INSERT INTO conversation_session_handoffs(
              handoff_id, bot_id, session_id, contact_id, source_channel, assistant_state, support_request_id,
              started_by, ended_reason, created_at, updated_at, ended_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
              contact_id = EXCLUDED.contact_id,
              source_channel = EXCLUDED.source_channel,
              assistant_state = EXCLUDED.assistant_state,
              support_request_id = EXCLUDED.support_request_id,
              started_by = EXCLUDED.started_by,
              ended_reason = EXCLUDED.ended_reason,
              updated_at = EXCLUDED.updated_at,
              ended_at = EXCLUDED.ended_at
            """,
            (
                str(payload.get("handoff_id") or ""),
                bot_id,
                session_id,
                payload.get("contact_id"),
                payload.get("source_channel"),
                str(payload.get("assistant_state") or "bot"),
                payload.get("support_request_id"),
                payload.get("started_by"),
                payload.get("ended_reason"),
                str(payload.get("created_at") or _utc_now()),
                str(payload.get("updated_at") or _utc_now()),
                payload.get("ended_at"),
            ),
        )

    def _apply_escalation_upsert(self, con, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        escalation_id = str(payload.get("escalation_id") or "").strip()
        bot_id = str(payload.get("bot_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        if not escalation_id or not bot_id or not session_id:
            return
        con.execute(
            """
            INSERT INTO conversation_escalations(
              escalation_id, bot_id, session_id, visitor_email, details, status, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (escalation_id) DO UPDATE SET
              details = EXCLUDED.details,
              status = EXCLUDED.status
            """,
            (
                escalation_id,
                bot_id,
                session_id,
                str(payload.get("visitor_email") or ""),
                payload.get("details"),
                str(payload.get("status") or "open"),
                str(payload.get("created_at") or _utc_now()),
            ),
        )


_WORKER_TASK: Optional[asyncio.Task] = None
_WORKER: Optional[RuntimePersistenceWorker] = None


async def start_runtime_persistence_worker() -> None:
    global _WORKER_TASK, _WORKER
    if not bool(getattr(config, "CHAT_RUNTIME_REDIS_ENABLED", False)):
        return
    if not bool(getattr(config, "CHAT_RUNTIME_ASYNC_PERSIST_ENABLED", False)):
        return
    if _WORKER_TASK and not _WORKER_TASK.done():
        return
    _WORKER = RuntimePersistenceWorker()
    _WORKER_TASK = asyncio.create_task(_WORKER.run())
    logging.getLogger("runtime_persistence_worker").info("Runtime persistence worker started")


async def stop_runtime_persistence_worker() -> None:
    global _WORKER_TASK, _WORKER
    if _WORKER_TASK and not _WORKER_TASK.done():
        _WORKER_TASK.cancel()
        try:
            await _WORKER_TASK
        except Exception:
            pass
    _WORKER_TASK = None
    if _WORKER is not None:
        try:
            await _WORKER.close()
        except Exception:
            pass
    _WORKER = None
