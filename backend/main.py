import os, signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from logging_relay import smartqa_log_relay

from api import smart_qa_router
from cors_dynamic import DynamicWidgetCORSMiddleware
from config import config

load_dotenv()

import sys
import asyncio
import logging
if sys.platform.startswith("win"):
    # Playwright / crawl4ai launch browser subprocesses; on Windows this requires
    # the Proactor event loop (Selector loop raises NotImplementedError for subprocess).
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _log_google_creds() -> None:
    """
    Log the exact Google credentials file + identity the backend is configured to use.
    This makes it obvious when uvicorn --reload / different shells are using different env.
    """
    # Use uvicorn's logger so it reliably shows in the terminal.
    uvicorn_logger = logging.getLogger("uvicorn.error")

    path = (config.GOOGLE_APPLICATION_CREDENTIALS or "").strip()
    if not path:
        uvicorn_logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set (config has empty path)")
        print("GCP creds: GOOGLE_APPLICATION_CREDENTIALS is not set")
        return
    if not os.path.exists(path):
        uvicorn_logger.warning("GOOGLE_APPLICATION_CREDENTIALS path does not exist: %s", path)
        print(f"GCP creds: creds_path does not exist: {path}")
        return
    try:
        import google.auth

        creds, proj = google.auth.load_credentials_from_file(path)
        email = getattr(creds, "service_account_email", None)
        if email:
            uvicorn_logger.info("GCP creds: service_account=%s project=%s creds_path=%s", email, proj, path)
            print(f"GCP creds: service_account={email} project={proj} creds_path={path}")
        else:
            uvicorn_logger.info("GCP creds: non-service-account project=%s creds_path=%s", proj, path)
            print(f"GCP creds: non-service-account project={proj} creds_path={path}")
    except Exception as e:
        uvicorn_logger.warning("Failed to load GOOGLE_APPLICATION_CREDENTIALS (%s): %s", path, e)
        print(f"GCP creds: failed to load creds_path={path} err={e}")

def force_exit(*args, **kwargs):
    print("Force exiting due to Ctrl+C")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit)

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable.")

app = FastAPI()

@app.on_event("startup")
async def _startup_log_creds() -> None:
    _log_google_creds()

# For now, do NOT require domain verification. Use permissive CORS.
# When you want to turn verification back on, set REQUIRE_DOMAIN_VERIFICATION=true.
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

# Register endpoints
app.include_router(smart_qa_router)

# Serve embeddable widget assets (loader + iframe page).
app.mount(
    "/widget",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "widget"), html=True),
    name="widget",
)


def _dashboard_dist_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "dist"))


def _dashboard_index_path() -> str:
    return os.path.join(_dashboard_dist_path(), "index.html")


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


@app.websocket("/ws/smartqa-logs")
async def smartqa_logs_ws(websocket: WebSocket):
    await websocket.accept()
    queue = smartqa_log_relay.register()
    try:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect:
        smartqa_log_relay.unregister(queue)