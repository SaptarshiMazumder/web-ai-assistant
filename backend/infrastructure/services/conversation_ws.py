import asyncio
import json
import logging
from typing import Dict, Optional, Set

from fastapi import WebSocket
from redis.asyncio import Redis

from common.config import config

_connections: Dict[str, Set[WebSocket]] = {}
_redis: Optional[Redis] = None
_pubsub_task: Optional[asyncio.Task] = None
_redis_ready: bool = False
_channel_prefix = "webai:conversations:"


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
        _logger().warning("Redis unavailable for conversation pubsub: %s", e)
        _redis_ready = False
        _redis = None
        return None


async def broadcast_message(session_id: str, payload: Dict[str, object]) -> None:
    redis = await _ensure_redis()
    if redis:
        try:
            await redis.publish(_channel_prefix + session_id, json.dumps(payload))
            return
        except Exception as e:
            _logger().warning("Redis publish failed, falling back to local: %s", e)
    await _send_local(session_id, payload)


async def _pubsub_loop() -> None:
    redis = await _ensure_redis()
    if not redis:
        return
    pubsub = redis.pubsub()
    await pubsub.psubscribe(_channel_prefix + "*")
    try:
        async for msg in pubsub.listen():
            if msg is None or msg.get("type") != "pmessage":
                continue
            channel = msg.get("channel")
            data = msg.get("data")
            if not channel or not data:
                continue
            try:
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="ignore")
                session_id = str(channel).split(_channel_prefix, 1)[1]
            except Exception:
                continue
            try:
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                payload = json.loads(data)
            except Exception:
                continue
            await _send_local(session_id, payload)
    finally:
        try:
            await pubsub.close()
        except Exception:
            pass


async def start_pubsub() -> None:
    global _pubsub_task
    if _pubsub_task and not _pubsub_task.done():
        return
    redis = await _ensure_redis()
    if not redis:
        _logger().info("Conversation WS pubsub: Redis not configured or unavailable; using local broadcast")
        return
    _pubsub_task = asyncio.create_task(_pubsub_loop())
    _logger().info("Conversation WS pubsub: connected to Redis")


async def stop_pubsub() -> None:
    global _pubsub_task, _redis, _redis_ready
    if _pubsub_task and not _pubsub_task.done():
        _pubsub_task.cancel()
        try:
            await _pubsub_task
        except Exception:
            pass
    _pubsub_task = None
    if _redis:
        try:
            await _redis.close()
        except Exception:
            pass
    _redis = None
    _redis_ready = False
