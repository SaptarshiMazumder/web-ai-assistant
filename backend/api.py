from fastapi import APIRouter

# Aggregate feature routers in a single place for simplicity
smart_qa_router = APIRouter()

from features.smart_site.router import router as smart_router
from features.google_search.router import router as google_router
from features.website_rag.router import router as rag_router

smart_qa_router.include_router(smart_router)
smart_qa_router.include_router(google_router)
smart_qa_router.include_router(rag_router)
