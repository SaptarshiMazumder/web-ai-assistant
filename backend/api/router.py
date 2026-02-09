from fastapi import APIRouter

from api.routes.saas import router as saas_router
from api.routes.health import router as health_router
from api.routes.line_webhook import router as line_router
from api.routes.instagram_webhook import router as instagram_router

api_router = APIRouter()
api_router.include_router(saas_router)
api_router.include_router(health_router)
api_router.include_router(line_router)
api_router.include_router(instagram_router)