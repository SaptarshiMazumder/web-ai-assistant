import asyncio
import logging
import os
import signal
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.router import api_router
from api.middleware.dynamic_cors import DynamicWidgetCORSMiddleware
from common.config import config

load_dotenv()


def _log_google_creds() -> None:
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

    @app.on_event("startup")
    async def _startup_log_creds() -> None:
        _log_google_creds()

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
