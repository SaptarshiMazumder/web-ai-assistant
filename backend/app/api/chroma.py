from fastapi import APIRouter, Query
from pydantic import BaseModel
import os
from app.config import config

class PageData(BaseModel):
    url: str
    html: str
    domain: str

chroma_router = APIRouter()

@chroma_router.get("/chroma_exists")
async def chroma_exists(domain: str = Query(...)):
    path = f"{config.CHROMA_DB_DIR}{domain}"
    return {"exists": os.path.exists(path)}

@chroma_router.post("/add_page_data")
async def add_page_data(data: PageData):
    # Placeholder for actual implementation
    return {"ok": True}