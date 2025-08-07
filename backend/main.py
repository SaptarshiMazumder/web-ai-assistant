import os, signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from logging_relay import smartqa_log_relay

from api import smart_qa_router, chroma_router

load_dotenv()

import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def force_exit(*args, **kwargs):
    print("Force exiting due to Ctrl+C")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit)

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