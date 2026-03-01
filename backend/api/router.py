from fastapi import APIRouter

from api.routes.saas import router as saas_router
from api.routes.health import router as health_router
from api.routes.line_webhook import router as line_router
from api.routes.instagram_webhook import router as instagram_router
from api.routes.instagram_oauth import router as instagram_oauth_router
from api.routes.assets import router as assets_router
from api.routes.menu_items import router as menu_items_router

api_router = APIRouter()
api_router.include_router(saas_router)
api_router.include_router(health_router)
api_router.include_router(line_router)
api_router.include_router(instagram_router)
api_router.include_router(instagram_oauth_router)
api_router.include_router(assets_router)
api_router.include_router(menu_items_router)