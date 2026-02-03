import asyncio
import json
import logging
from typing import Dict, Optional, Set

from fastapi import WebSocket
from redis.asyncio import Redis

from common.config import config

_connections: Dict[str, Set[WebSocket]] = {}
_redis: Optional[Redis] = None
_stream_task: Optional[asyncio.Task] = None
_redis_ready: bool = False
_stream_ready: bool = False
_stream_key = (getattr(config, "CONVERSATION_STREAM_KEY", "") or "webai:conversation_events").strip()
_stream_maxlen = int(getattr(config, "CONVERSATION_STREAM_MAXLEN", "10000"))


def _logger() -> logging.Logger:
    return logging.getLogger("conversation_ws")


async def register(session_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.setdefault(session_id, set()).add(websocket)


def unregister(session_id: str, websocket: WebSocket) -> None:
    conns = _connections.get(session_id)
    if not conns:
        return
    conns.discard(websocket)
    if not conns:
        _connections.pop(session_id, None)


async def _send_local(session_id: str, payload: Dict[str, object]) -> None:
    conns = _connections.get(session_id)
    if not conns:
        return
    for ws in list(conns):
        try:
            await ws.send_json(payload)
        except Exception:
            # Drop dead sockets
            conns.discard(ws)


async def _ensure_redis() -> Optional[Redis]:
    global _redis, _redis_ready
    if _redis_ready and _redis:
        return _redis
    url = (getattr(config, "REDIS_URL", "") or "").strip() or config.CELERY_BROKER_URL
    if not url:
        return None
    try:
        _redis = Redis.from_url(url, decode_responses=False)
        await _redis.ping()
        _redis_ready = True
        return _redis
    except Exception as e:
        _logger().warning("Redis unavailable for conversation stream: %s", e)
        _redis_ready = False
        _redis = None
        return None


async def broadcast_message(session_id: str, payload: Dict[str, object]) -> None:
    global _stream_ready
    redis = await _ensure_redis()
    if redis:
        try:
            await redis.xadd(
                _stream_key,
                {"session_id": session_id, "payload": json.dumps(payload)},
                maxlen=_stream_maxlen,
                approximate=True,
            )
            if _stream_ready:
                return
        except Exception as e:
            _logger().warning("Redis stream publish failed, falling back to local: %s", e)
    await _send_local(session_id, payload)


async def _stream_loop() -> None:
    global _stream_ready
    redis = await _ensure_redis()
    if not redis:
        return
    _stream_ready = True
    last_id = "$"
    try:
        while True:
            streams = await redis.xread({_stream_key: last_id}, block=5000, count=200)
            if not streams:
                continue
            for _, entries in streams:
                for entry_id, data in entries:
                    last_id = entry_id
                    sid = data.get(b"session_id") if isinstance(data, dict) else None
                    payload_raw = data.get(b"payload") if isinstance(data, dict) else None
                    if not sid or not payload_raw:
                        continue
                    try:
                        if isinstance(sid, bytes):
                            sid = sid.decode("utf-8", errors="ignore")
                        if isinstance(payload_raw, bytes):
                            payload_raw = payload_raw.decode("utf-8", errors="ignore")
                        payload = json.loads(payload_raw)
                    except Exception:
                        continue
                    await _send_local(str(sid), payload)
    finally:
        _stream_ready = False


async def start_pubsub() -> None:
    global _stream_task
    if _stream_task and not _stream_task.done():
        return
    redis = await _ensure_redis()
    if not redis:
        _logger().info("Conversation WS stream: Redis not configured or unavailable; using local broadcast")
        return
    _stream_task = asyncio.create_task(_stream_loop())
    _logger().info("Conversation WS stream: connected to Redis")


async def stop_pubsub() -> None:
    global _stream_task, _redis, _redis_ready, _stream_ready
    if _stream_task and not _stream_task.done():
        _stream_task.cancel()
        try:
            await _stream_task
        except Exception:
            pass
    _stream_task = None
    _stream_ready = False
    if _redis:
        try:
            await _redis.close()
        except Exception:
            pass
    _redis = None
    _redis_ready = False
