from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from common.di.container import bot_service


def _origin_host(origin: Optional[str]) -> str:
    if not origin:
        return ""
    try:
        u = urlparse(origin)
        host = (u.hostname or "").strip().lower()
        return host.split(":")[0]
    except Exception:
        return ""


def _pk_from_path(path: str) -> Optional[str]:
    if not path:
        return None
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "pk":
        return parts[2]
    return None


class DynamicWidgetCORSMiddleware:
    """
    Pure ASGI CORS middleware for widget API endpoints.
    Does NOT extend BaseHTTPMiddleware — streaming responses pass through unbuffered.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = (headers.get(b"origin") or b"").decode("latin-1", errors="ignore")
        origin_host = _origin_host(origin)
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        allow_origin = None
        if origin_host:
            pk = _pk_from_path(path)
            if pk:
                bot = bot_service().get_bot_by_publishable_key(pk)
                if bot:
                    verified = set(bot_service().list_verified_hosts(bot.bot_id))
                    if origin_host in verified:
                        allow_origin = origin

        # Handle preflight
        if method == "OPTIONS":
            resp_headers: dict[str, str] = {}
            if allow_origin:
                resp_headers["access-control-allow-origin"] = allow_origin
                resp_headers["vary"] = "Origin"
                resp_headers["access-control-allow-credentials"] = "true"
                req_headers = (headers.get(b"access-control-request-headers") or b"content-type,authorization").decode(
                    "latin-1", errors="ignore"
                )
                req_method = (headers.get(b"access-control-request-method") or b"POST,GET,OPTIONS").decode(
                    "latin-1", errors="ignore"
                )
                resp_headers["access-control-allow-headers"] = req_headers
                resp_headers["access-control-allow-methods"] = req_method
                resp_headers["access-control-max-age"] = "600"
            resp = Response(status_code=204, headers=resp_headers)
            await resp(scope, receive, send)
            return

        # For non-preflight: inject CORS headers into the response
        if not allow_origin:
            await self.app(scope, receive, send)
            return

        cors_headers = [
            (b"access-control-allow-origin", allow_origin.encode("latin-1")),
            (b"vary", b"Origin"),
            (b"access-control-allow-credentials", b"true"),
        ]
        response_started = False

        async def send_with_cors(message):
            nonlocal response_started
            if message["type"] == "http.response.start" and not response_started:
                response_started = True
                existing = list(message.get("headers", []))
                existing.extend(cors_headers)
                message = {**message, "headers": existing}
            await send(message)

        await self.app(scope, receive, send_with_cors)
