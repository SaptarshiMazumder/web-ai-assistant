import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from graph_smart_qa import smartqa_log_relay

from api import qa_router, site_qa_router, smart_qa_router, chroma_router

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints
app.include_router(qa_router)
app.include_router(site_qa_router)
app.include_router(smart_qa_router)
app.include_router(chroma_router)

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