import re

def extract_json_from_text(text):
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return arr_match.group(0)
    return None

def clean_markdown(md: str) -> str:
    md = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md)
    md = re.sub(r'http[s]?://\S+', '', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()