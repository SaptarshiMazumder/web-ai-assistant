import datetime
from app.config import config

def log_llm_prompt(prompt: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("="*40 + "\n")
        f.write(f"[{now}] LLM PROMPT SENT\n")
        f.write("="*40 + "\n")
        f.write(prompt)
        f.write("\n" + "-"*40 + "\n\n")