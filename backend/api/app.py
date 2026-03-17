import asyncio
import logging
import os
import signal
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.types import ASGIApp, Receive, Scope, Send

from api.router import api_router
from api.middleware.dynamic_cors import DynamicWidgetCORSMiddleware
from common.config import config
from infrastructure.db.connection import close_connection_pool, ensure_schema_once, get_connection, initialize_connection_pool
from infrastructure.services.conversation_ws import (
    register as register_conversation_ws,
    unregister as unregister_conversation_ws,
    start_pubsub as start_conversation_pubsub,
    stop_pubsub as stop_conversation_pubsub,
)
from infrastructure.services.chat_cache import (
    cache_start_listener,
    cache_stop_listener,
)
from infrastructure.services.runtime_persistence_worker import (
    start_runtime_persistence_worker,
    stop_runtime_persistence_worker,
)
load_dotenv()


def _log_google_creds() -> None:
    uvicorn_logger = logging.getLogger("uvicorn.error")
    try:
        from common.gcp_auth import load_gcp_credentials
        creds, proj = load_gcp_credentials()
        email = getattr(creds, "service_account_email", None)
        if email:
            uvicorn_logger.info("GCP creds: service_account=%s project=%s", email, proj)
        else:
            uvicorn_logger.info("GCP creds: ADC project=%s", proj)
    except Exception as e:
        uvicorn_logger.warning("GCP creds: failed to load: %s", e)


def _dashboard_dist_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "dist"))


def _dashboard_index_path() -> str:
    return os.path.join(_dashboard_dist_path(), "index.html")


def _force_exit(*args, **kwargs):
    print("Force exiting due to Ctrl+C")
    os._exit(0)


def create_app() -> FastAPI:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    signal.signal(signal.SIGINT, _force_exit)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")

    app = FastAPI()
    app.state.db_ready = False

    @app.on_event("startup")
    def _startup_db_check() -> None:
        initialize_connection_pool()
        ensure_schema_once()
        last_exc: Exception | None = None
        for _ in range(4):
            try:
                con = get_connection()
                try:
                    con.execute("SELECT 1")
                finally:
                    con.close()
                app.state.db_ready = True
                return
            except Exception as e:
                last_exc = e
                time.sleep(0.5)
        if last_exc is not None:
            raise HTTPException(status_code=503, detail=f"Database unavailable: {last_exc}") from last_exc
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Pure ASGI middleware — does NOT buffer streaming responses
    # (unlike @app.middleware("http") which uses BaseHTTPMiddleware and kills streaming)
    class _ReadinessGateMiddleware:
        def __init__(self, asgi_app: ASGIApp) -> None:
            self.app = asgi_app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            path = scope.get("path", "")
            if path in {"/live", "/health"}:
                await self.app(scope, receive, send)
                return
            if not app.state.db_ready:
                resp = JSONResponse(status_code=503, content={"detail": "Service not ready"})
                await resp(scope, receive, send)
                return
            await self.app(scope, receive, send)

    app.add_middleware(_ReadinessGateMiddleware)

    @app.on_event("startup")
    async def _startup_log_creds() -> None:
        _log_google_creds()

    @app.on_event("startup")
    async def _startup_conversation_pubsub() -> None:
        await start_conversation_pubsub()

    @app.on_event("startup")
    async def _startup_chat_cache_listener() -> None:
        await cache_start_listener()

    @app.on_event("startup")
    async def _startup_runtime_persistence_worker() -> None:
        await start_runtime_persistence_worker()

    @app.on_event("shutdown")
    async def _shutdown_conversation_pubsub() -> None:
        await stop_conversation_pubsub()

    @app.on_event("shutdown")
    async def _shutdown_chat_cache_listener() -> None:
        await cache_stop_listener()

    @app.on_event("shutdown")
    async def _shutdown_runtime_persistence_worker() -> None:
        await stop_runtime_persistence_worker()

    @app.on_event("shutdown")
    async def _shutdown_db_pool() -> None:
        close_connection_pool()

    if config.REQUIRE_DOMAIN_VERIFICATION:
        app.add_middleware(DynamicWidgetCORSMiddleware)
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    @app.websocket("/ws/conversations/{session_id}")
    async def ws_conversation(session_id: str, websocket):
        ws_logger = logging.getLogger("uvicorn.error")
        ws_logger.info(
            "WS connect attempt session=%s origin=%s client=%s",
            session_id,
            websocket.headers.get("origin"),
            getattr(websocket.client, "host", None),
        )
        try:
            await register_conversation_ws(session_id, websocket)
        except Exception as exc:
            ws_logger.exception("WS accept failed session=%s err=%s", session_id, exc)
            return
        try:
            while True:
                await websocket.receive_text()
        except Exception as exc:
            ws_logger.info("WS disconnected session=%s err=%s", session_id, exc)
        finally:
            unregister_conversation_ws(session_id, websocket)

    app.mount(
        "/widget",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "widget"), html=True),
        name="widget",
    )

    @app.get("/dashboard")
    @app.get("/dashboard/")
    async def dashboard_index():
        index_path = _dashboard_index_path()
        if not os.path.isfile(index_path):
            raise HTTPException(status_code=404, detail="Dashboard not built. Run `npm run build` in /dashboard.")
        return FileResponse(index_path)

    @app.get("/dashboard/{full_path:path}")
    async def dashboard_assets(full_path: str):
        dist_root = _dashboard_dist_path()
        if not os.path.isdir(dist_root):
            raise HTTPException(status_code=404, detail="Dashboard not built. Run `npm run build` in /dashboard.")
        candidate = os.path.abspath(os.path.join(dist_root, full_path))
        if not candidate.startswith(dist_root):
            raise HTTPException(status_code=400, detail="Invalid path")
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        index_path = _dashboard_index_path()
        if not os.path.isfile(index_path):
            raise HTTPException(status_code=404, detail="Dashboard not built. Run `npm run build` in /dashboard.")
        return FileResponse(index_path)

    return app
