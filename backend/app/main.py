from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.smart_qa import smart_qa_router
from app.api.chroma import chroma_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints
app.include_router(smart_qa_router)
app.include_router(chroma_router)