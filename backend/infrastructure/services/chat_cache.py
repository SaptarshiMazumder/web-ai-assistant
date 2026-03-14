from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional, Sequence

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from common.config import config


@dataclass(frozen=True)
class CacheResult:
    hit: bool
    value: Any = None
    negative: bool = False
    source: str = ""


@dataclass
class _L1Entry:
    envelope: dict[str, Any]
    expires_at: float


_L1: "OrderedDict[str, _L1Entry]" = OrderedDict()
_L1_LOCK = Lock()

_REDIS_SYNC: Optional[Redis] = None
_REDIS_SYNC_READY = False
_REDIS_SYNC_LAST_WARN_AT = 0.0

_REDIS_ASYNC: Optional[AsyncRedis] = None
_REDIS_ASYNC_READY = False
_REDIS_ASYNC_LAST_WARN_AT = 0.0
_INVALIDATION_TASK: Optional[asyncio.Task] = None

_NEGATIVE_FLAG = 1
_POSITIVE_FLAG = 0
_WARN_THROTTLE_SECONDS = 30


def _logger() -> logging.Logger:
    return logging.getLogger("chat_cache")


def _safe_key_part(value: str) -> str:
    return (value or "").strip()


def _cache_enabled() -> bool:
    return bool(getattr(config, "CHAT_CACHE_ENABLED", True))


def _redis_url() -> str:
    return (getattr(config, "REDIS_URL", "") or "").strip() or (getattr(config, "CELERY_BROKER_URL", "") or "").strip()


def _l2_enabled() -> bool:
    return _cache_enabled() and bool(getattr(config, "CHAT_CACHE_L2_ENABLED", True)) and bool(_redis_url())


def _l1_max_items() -> int:
    return int(getattr(config, "CHAT_CACHE_L1_MAX_ITEMS", 5000))


def _namespace() -> str:
    return (getattr(config, "CHAT_CACHE_NAMESPACE", "") or "webai:chatcache:v1").strip()


def _invalidation_channel() -> str:
    return (getattr(config, "CHAT_CACHE_INVALIDATION_CHANNEL", "") or "webai:chatcache:v1:invalidate").strip()


def _key(raw_key: str) -> str:
    return f"{_namespace()}:{raw_key}"


def _throttled_warn(kind: str, msg: str) -> None:
    global _REDIS_SYNC_LAST_WARN_AT, _REDIS_ASYNC_LAST_WARN_AT
    now = time.time()
    if kind == "sync":
        if now - _REDIS_SYNC_LAST_WARN_AT < _WARN_THROTTLE_SECONDS:
            return
        _REDIS_SYNC_LAST_WARN_AT = now
    else:
        if now - _REDIS_ASYNC_LAST_WARN_AT < _WARN_THROTTLE_SECONDS:
            return
        _REDIS_ASYNC_LAST_WARN_AT = now
    _logger().warning(msg)


def _prune_l1_locked(now: float) -> None:
    expired = [k for k, v in _L1.items() if v.expires_at <= now]
    for k in expired:
        _L1.pop(k, None)
    max_items = _l1_max_items()
    while len(_L1) > max_items:
        _L1.popitem(last=False)


def _l1_get_envelope(cache_key: str) -> Optional[dict[str, Any]]:
    now = time.time()
    with _L1_LOCK:
        entry = _L1.get(cache_key)
        if entry is None:
            _prune_l1_locked(now)
            return None
        if entry.expires_at <= now:
            _L1.pop(cache_key, None)
            _prune_l1_locked(now)
            return None
        _L1.move_to_end(cache_key)
        return entry.envelope


def _l1_set_envelope(cache_key: str, envelope: dict[str, Any]) -> None:
    exp = float(envelope.get("exp") or 0)
    now = time.time()
    if exp <= now:
        return
    with _L1_LOCK:
        _L1[cache_key] = _L1Entry(envelope=envelope, expires_at=exp)
        _L1.move_to_end(cache_key)
        _prune_l1_locked(now)


def _l1_invalidate_keys(keys: Sequence[str]) -> None:
    if not keys:
        return
    with _L1_LOCK:
        for cache_key in keys:
            _L1.pop(cache_key, None)


def _l1_invalidate_prefixes(prefixes: Sequence[str]) -> None:
    if not prefixes:
        return
    with _L1_LOCK:
        all_keys = list(_L1.keys())
        for cache_key in all_keys:
            if any(cache_key.startswith(prefix) for prefix in prefixes):
                _L1.pop(cache_key, None)


def _make_envelope(value: Any, ttl_seconds: int, *, negative: bool) -> dict[str, Any]:
    ttl = max(1, int(ttl_seconds or 1))
    return {
        "n": _NEGATIVE_FLAG if negative else _POSITIVE_FLAG,
        "exp": time.time() + ttl,
        "v": None if negative else value,
    }


def _envelope_to_result(envelope: dict[str, Any], *, source: str) -> CacheResult:
    if int(envelope.get("n") or 0) == _NEGATIVE_FLAG:
        return CacheResult(hit=True, value=None, negative=True, source=source)
    return CacheResult(hit=True, value=envelope.get("v"), negative=False, source=source)


def _ensure_sync_redis() -> Optional[Redis]:
    global _REDIS_SYNC, _REDIS_SYNC_READY
    if not _l2_enabled():
        return None
    if _REDIS_SYNC_READY and _REDIS_SYNC is not None:
        return _REDIS_SYNC
    try:
        client = Redis.from_url(_redis_url(), decode_responses=True)
        client.ping()
        _REDIS_SYNC = client
        _REDIS_SYNC_READY = True
        return _REDIS_SYNC
    except Exception as exc:
        _REDIS_SYNC = None
        _REDIS_SYNC_READY = False
        _throttled_warn("sync", f"Chat cache L2 unavailable: {exc}")
        return None


async def _ensure_async_redis() -> Optional[AsyncRedis]:
    global _REDIS_ASYNC, _REDIS_ASYNC_READY
    if not _l2_enabled():
        return None
    if _REDIS_ASYNC_READY and _REDIS_ASYNC is not None:
        return _REDIS_ASYNC
    try:
        client = AsyncRedis.from_url(_redis_url(), decode_responses=True)
        await client.ping()
        _REDIS_ASYNC = client
        _REDIS_ASYNC_READY = True
        return _REDIS_ASYNC
    except Exception as exc:
        if _REDIS_ASYNC is not None:
            try:
                await _REDIS_ASYNC.close()
            except Exception:
                pass
        _REDIS_ASYNC = None
        _REDIS_ASYNC_READY = False
        _throttled_warn("async", f"Chat cache invalidation listener unavailable: {exc}")
        return None


def cache_get_json(cache_key: str) -> CacheResult:
    if not _cache_enabled():
        return CacheResult(hit=False)
    key = _safe_key_part(cache_key)
    if not key:
        return CacheResult(hit=False)

    envelope = _l1_get_envelope(key)
    if envelope is not None:
        return _envelope_to_result(envelope, source="l1")

    redis = _ensure_sync_redis()
    if redis is None:
        return CacheResult(hit=False)

    try:
        raw = redis.get(_key(key))
    except Exception as exc:
        _throttled_warn("sync", f"Chat cache L2 read failed: {exc}")
        return CacheResult(hit=False)
    if raw is None:
        return CacheResult(hit=False)

    try:
        envelope = json.loads(raw)
    except Exception:
        try:
            redis.delete(_key(key))
        except Exception:
            pass
        return CacheResult(hit=False)

    if not isinstance(envelope, dict):
        return CacheResult(hit=False)
    exp = float(envelope.get("exp") or 0)
    if exp <= time.time():
        try:
            redis.delete(_key(key))
        except Exception:
            pass
        return CacheResult(hit=False)

    _l1_set_envelope(key, envelope)
    return _envelope_to_result(envelope, source="l2")


def cache_get_json_l1_only(cache_key: str) -> CacheResult:
    if not _cache_enabled():
        return CacheResult(hit=False)
    key = _safe_key_part(cache_key)
    if not key:
        return CacheResult(hit=False)

    envelope = _l1_get_envelope(key)
    if envelope is None:
        return CacheResult(hit=False)
    return _envelope_to_result(envelope, source="l1")


def cache_set_json(cache_key: str, value: Any, ttl_seconds: int) -> None:
    if not _cache_enabled():
        return
    key = _safe_key_part(cache_key)
    if not key:
        return
    envelope = _make_envelope(value=value, ttl_seconds=ttl_seconds, negative=False)
    _l1_set_envelope(key, envelope)
    redis = _ensure_sync_redis()
    if redis is None:
        return
    try:
        payload = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)
        redis.setex(_key(key), max(1, int(ttl_seconds)), payload)
    except Exception as exc:
        _throttled_warn("sync", f"Chat cache L2 write failed: {exc}")


def cache_set_json_l1_only(cache_key: str, value: Any, ttl_seconds: int) -> None:
    if not _cache_enabled():
        return
    key = _safe_key_part(cache_key)
    if not key:
        return
    envelope = _make_envelope(value=value, ttl_seconds=ttl_seconds, negative=False)
    _l1_set_envelope(key, envelope)


def cache_set_negative(cache_key: str, ttl_seconds: int) -> None:
    if not _cache_enabled():
        return
    key = _safe_key_part(cache_key)
    if not key:
        return
    envelope = _make_envelope(value=None, ttl_seconds=ttl_seconds, negative=True)
    _l1_set_envelope(key, envelope)
    redis = _ensure_sync_redis()
    if redis is None:
        return
    try:
        payload = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)
        redis.setex(_key(key), max(1, int(ttl_seconds)), payload)
    except Exception as exc:
        _throttled_warn("sync", f"Chat cache L2 negative-write failed: {exc}")


def cache_set_negative_l1_only(cache_key: str, ttl_seconds: int) -> None:
    if not _cache_enabled():
        return
    key = _safe_key_part(cache_key)
    if not key:
        return
    envelope = _make_envelope(value=None, ttl_seconds=ttl_seconds, negative=True)
    _l1_set_envelope(key, envelope)


def cache_invalidate_keys(keys: Sequence[str]) -> None:
    if not _cache_enabled():
        return
    unique_keys = [_safe_key_part(k) for k in (keys or []) if _safe_key_part(k)]
    if not unique_keys:
        return
    _l1_invalidate_keys(unique_keys)

    redis = _ensure_sync_redis()
    if redis is None:
        return
    namespaced = [_key(k) for k in unique_keys]
    payload = json.dumps({"keys": unique_keys, "prefixes": []}, separators=(",", ":"), ensure_ascii=False)
    try:
        with redis.pipeline(transaction=False) as pipe:
            if namespaced:
                pipe.delete(*namespaced)
            pipe.publish(_invalidation_channel(), payload)
            pipe.execute()
    except Exception as exc:
        _throttled_warn("sync", f"Chat cache invalidation publish failed: {exc}")


def cache_invalidate_prefixes(prefixes: Sequence[str]) -> None:
    if not _cache_enabled():
        return
    cleaned = [_safe_key_part(p) for p in (prefixes or []) if _safe_key_part(p)]
    if not cleaned:
        return
    _l1_invalidate_prefixes(cleaned)

    redis = _ensure_sync_redis()
    if redis is None:
        return
    keys_to_delete: list[str] = []
    try:
        for prefix in cleaned:
            pattern = _key(f"{prefix}*")
            for full_key in redis.scan_iter(match=pattern, count=200):
                keys_to_delete.append(str(full_key))
        with redis.pipeline(transaction=False) as pipe:
            if keys_to_delete:
                pipe.delete(*keys_to_delete)
            payload = json.dumps({"keys": [], "prefixes": cleaned}, separators=(",", ":"), ensure_ascii=False)
            pipe.publish(_invalidation_channel(), payload)
            pipe.execute()
    except Exception as exc:
        _throttled_warn("sync", f"Chat cache prefix invalidation failed: {exc}")


def claim_webhook_event_once(
    *,
    provider: str,
    scope_id: str,
    event_id: str,
    ttl_seconds: Optional[int] = None,
) -> bool:
    eid = _safe_key_part(event_id)
    if not eid:
        return True
    redis = _ensure_sync_redis()
    if redis is None:
        # Fail-open to preserve functional behavior if Redis is down.
        return True
    ttl = int(ttl_seconds or getattr(config, "CHAT_CACHE_IDEMPOTENCY_TTL_SEC", 86400))
    ttl = max(60, ttl)
    cache_key = _key(cache_key_idempotency_event(provider, scope_id, eid))
    try:
        created = redis.set(cache_key, str(int(time.time())), nx=True, ex=ttl)
        return bool(created)
    except Exception as exc:
        _throttled_warn("sync", f"Webhook idempotency write failed: {exc}")
        return True


async def _invalidation_loop() -> None:
    pubsub = None
    try:
        while True:
            redis = await _ensure_async_redis()
            if redis is None:
                await asyncio.sleep(2.0)
                continue
            try:
                pubsub = redis.pubsub(ignore_subscribe_messages=True)
                await pubsub.subscribe(_invalidation_channel())
                _logger().info("Chat cache invalidation listener subscribed channel=%s", _invalidation_channel())
                while True:
                    msg = await pubsub.get_message(timeout=1.0)
                    if msg is None:
                        await asyncio.sleep(0.05)
                        continue
                    raw = msg.get("data")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    if not isinstance(raw, str):
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    keys = payload.get("keys") if isinstance(payload, dict) else None
                    prefixes = payload.get("prefixes") if isinstance(payload, dict) else None
                    if isinstance(keys, list):
                        _l1_invalidate_keys([_safe_key_part(k) for k in keys if _safe_key_part(k)])
                    if isinstance(prefixes, list):
                        _l1_invalidate_prefixes([_safe_key_part(p) for p in prefixes if _safe_key_part(p)])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _throttled_warn("async", f"Chat cache invalidation listener error: {exc}")
                await asyncio.sleep(1.0)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe(_invalidation_channel())
                    except Exception:
                        pass
                    try:
                        await pubsub.close()
                    except Exception:
                        pass
                    pubsub = None
    except asyncio.CancelledError:
        pass


async def cache_start_listener() -> None:
    global _INVALIDATION_TASK
    if not _l2_enabled():
        return
    if _INVALIDATION_TASK and not _INVALIDATION_TASK.done():
        return
    _INVALIDATION_TASK = asyncio.create_task(_invalidation_loop())


async def cache_stop_listener() -> None:
    global _INVALIDATION_TASK, _REDIS_ASYNC, _REDIS_ASYNC_READY
    if _INVALIDATION_TASK and not _INVALIDATION_TASK.done():
        _INVALIDATION_TASK.cancel()
        try:
            await _INVALIDATION_TASK
        except Exception:
            pass
    _INVALIDATION_TASK = None

    if _REDIS_ASYNC is not None:
        try:
            await _REDIS_ASYNC.close()
        except Exception:
            pass
    _REDIS_ASYNC = None
    _REDIS_ASYNC_READY = False


def cache_key_bot_pk(publishable_key: str) -> str:
    return f"bot:pk:{_safe_key_part(publishable_key)}"


def cache_key_bot_id(bot_id: str) -> str:
    return f"bot:id:{_safe_key_part(bot_id)}"


def cache_key_bot_record(bot_id: str) -> str:
    return f"bot:record:{_safe_key_part(bot_id)}"


def cache_key_verified_hosts(bot_id: str) -> str:
    return f"verified_hosts:{_safe_key_part(bot_id)}"


def cache_key_channel_line(bot_id: str) -> str:
    return f"channel:line:{_safe_key_part(bot_id)}"


def cache_key_channel_ig_bot(bot_id: str) -> str:
    return f"channel:ig:{_safe_key_part(bot_id)}"


def cache_key_channel_ig_page(ig_page_id: str) -> str:
    return f"channel:ig:page:{_safe_key_part(ig_page_id)}"


def cache_key_channel_ig_user(ig_user_id: str) -> str:
    return f"channel:ig:user:{_safe_key_part(ig_user_id)}"


def cache_key_design_line(bot_id: str) -> str:
    return f"design:line:{_safe_key_part(bot_id)}"


def cache_key_corpus(bot_id: str) -> str:
    return f"corpus:{_safe_key_part(bot_id)}"


def cache_key_session_snapshot(channel: str, bot_id: str, external_user_id: str) -> str:
    return f"session:{_safe_key_part(channel)}:{_safe_key_part(bot_id)}:{_safe_key_part(external_user_id)}"


def cache_key_line_user_session(bot_id: str, line_user_id: str) -> str:
    return f"line_user_session:{_safe_key_part(bot_id)}:{_safe_key_part(line_user_id)}"


def cache_key_line_user_session_by_sid(session_id: str) -> str:
    return f"line_user_session:sid:{_safe_key_part(session_id)}"


def cache_key_ig_user_session(bot_id: str, ig_user_id: str) -> str:
    return f"ig_user_session:{_safe_key_part(bot_id)}:{_safe_key_part(ig_user_id)}"


def cache_key_ig_user_session_by_sid(session_id: str) -> str:
    return f"ig_user_session:sid:{_safe_key_part(session_id)}"


def cache_key_session_record(session_id: str) -> str:
    return f"session:id:{_safe_key_part(session_id)}"


def cache_key_history_tail(session_id: str) -> str:
    return f"history_tail:{_safe_key_part(session_id)}"


def cache_key_session_handoff(session_id: str) -> str:
    return f"handoff:{_safe_key_part(session_id)}"


def cache_key_idempotency_event(provider: str, scope_id: str, event_id: str) -> str:
    return f"idemp:{_safe_key_part(provider)}:{_safe_key_part(scope_id)}:{_safe_key_part(event_id)}"
