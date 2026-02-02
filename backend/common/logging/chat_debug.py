import json
import os
from typing import Any, Dict


def _log_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(base_dir, "common", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "chat_debug.log")


def chat_debug_emit(event: Dict[str, Any]) -> None:
    """
    Extremely verbose debug logging for widget chat.
    Prints JSON to terminal AND appends to common/logs/chat_debug.log.
    """
    try:
        t = str(event.get("type") or "")
        trace_id = str(event.get("trace_id") or "")
        if t in ("retrieved_chunk", "filtered_chunk"):
            idx = event.get("idx")
            url = str(event.get("url") or "")
            snippet = str(event.get("snippet") or "")
            label = "RETRIEVED" if t == "retrieved_chunk" else "FILTERED"
            print(f"\nWEB_AI_CHAT_DEBUG_CHUNK {label} trace_id={trace_id} #{idx}\nURL: {url}\n---\n{snippet}\n---\n", flush=True)
        elif t == "gemini_prompt":
            sys_inst = event.get("system_instruction") or ""
            user_msg = event.get("user_message") or ""
            print(f"\nWEB_AI_CHAT_DEBUG gemini_prompt trace_id={trace_id}", flush=True)
            print("--- SYSTEM INSTRUCTION ---", flush=True)
            print(sys_inst, flush=True)
            print("--- USER MESSAGE ---", flush=True)
            print(user_msg, flush=True)
            print("--- END gemini_prompt ---\n", flush=True)
        else:
            try:
                line = "WEB_AI_CHAT_DEBUG " + json.dumps(event, ensure_ascii=False)
            except Exception:
                line = "WEB_AI_CHAT_DEBUG " + str(event)
            print(line, flush=True)
    except Exception:
        pass

    try:
        line = "WEB_AI_CHAT_DEBUG " + json.dumps(event, ensure_ascii=False)
    except Exception:
        line = "WEB_AI_CHAT_DEBUG " + str(event)
    try:
        with open(_log_path(), "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass
