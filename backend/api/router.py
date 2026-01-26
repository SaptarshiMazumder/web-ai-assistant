from fastapi import APIRouter

from api.routes.saas import router as saas_router

api_router = APIRouter()
api_router.include_router(saas_router)
