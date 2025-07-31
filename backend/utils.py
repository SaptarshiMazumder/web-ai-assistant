import re, datetime

def extract_json_from_text(text):
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return arr_match.group(0)
    return None


def clean_markdown(md: str) -> str:
    # Remove [text](url) links, keep just the text
    md = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md)
    # Remove naked URLs
    md = re.sub(r'http[s]?://\S+', '', md)
    # Optionally, remove extra whitespace
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()




PROMPT_LOG_PATH = "llm_prompt_log.txt"

def log_llm_prompt(prompt: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("="*40 + "\n")
        f.write(f"[{now}] LLM PROMPT SENT\n")
        f.write("="*40 + "\n")
        f.write(prompt)
        f.write("\n" + "-"*40 + "\n\n")
