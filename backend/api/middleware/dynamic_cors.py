from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

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


class DynamicWidgetCORSMiddleware(BaseHTTPMiddleware):
    """
    Dynamic CORS for widget API endpoints. We only allow cross-origin XHR from verified
    domains for the specific bot key in the request path.
    """

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        origin_host = _origin_host(origin)

        allow_origin = None
        if origin_host:
            pk = _pk_from_path(request.url.path)
            if pk:
                bot = bot_service().get_bot_by_publishable_key(pk)
                if bot:
                    verified = set(bot_service().list_verified_hosts(bot.bot_id))
                    if origin_host in verified:
                        allow_origin = origin

        if request.method == "OPTIONS":
            resp = Response(status_code=204)
            if allow_origin:
                resp.headers["Access-Control-Allow-Origin"] = allow_origin
                resp.headers["Vary"] = "Origin"
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                req_headers = request.headers.get("access-control-request-headers") or "content-type,authorization"
                req_method = request.headers.get("access-control-request-method") or "POST,GET,OPTIONS"
                resp.headers["Access-Control-Allow-Headers"] = req_headers
                resp.headers["Access-Control-Allow-Methods"] = req_method
                resp.headers["Access-Control-Max-Age"] = "600"
            return resp

        resp = await call_next(request)
        if allow_origin:
            resp.headers["Access-Control-Allow-Origin"] = allow_origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp
