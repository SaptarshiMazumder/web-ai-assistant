import re

def format_for_messaging(text: str) -> str:
    """
    Format Markdown text for messaging platforms like LINE and Instagram.
    
    Performs the following:
    1. Removes bolding (**text** -> text)
    2. Removes italics (*text* -> text)
    3. Converts links ([text](url) -> text: url)
    4. Removes headers (# Header -> Header)
    """
    if not text:
        return ""

    # 1. Convert Links: [Link Text](URL) -> Link Text: URL
    # We use a regex to capture [text](url)
    def replace_link(match):
        label = match.group(1)
        url = match.group(2)
        return f"{label}: {url}"
    
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)

    # 2. Remove Bold: **text** or __text__
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)

    # 3. Remove Italics: *text* or _text_
    # Note: We need to be careful not to match partial bold markers if we did this first, 
    # but since we did bold first, it should be okay. 
    # However, * is also used for lists. We should match pairs.
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)\_(?!\_)(.*?)(?<!_)\_(?!\_)', r'\1', text)

    # 4. Remove Headers: # Header -> Header
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    return text.strip()
