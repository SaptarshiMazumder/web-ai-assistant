import re

def _to_unicode_bold(text: str) -> str:
    """Convert regular text to Unicode bold characters."""
    bold_map = {
        # Uppercase letters
        'A': '𝗔', 'B': '𝗕', 'C': '𝗖', 'D': '𝗗', 'E': '𝗘', 'F': '𝗙', 'G': '𝗚', 'H': '𝗛', 'I': '𝗜', 'J': '𝗝',
        'K': '𝗞', 'L': '𝗟', 'M': '𝗠', 'N': '𝗡', 'O': '𝗢', 'P': '𝗣', 'Q': '𝗤', 'R': '𝗥', 'S': '𝗦', 'T': '𝗧',
        'U': '𝗨', 'V': '𝗩', 'W': '𝗪', 'X': '𝗫', 'Y': '𝗬', 'Z': '𝗭',
        # Lowercase letters
        'a': '𝗮', 'b': '𝗯', 'c': '𝗰', 'd': '𝗱', 'e': '𝗲', 'f': '𝗳', 'g': '𝗴', 'h': '𝗵', 'i': '𝗶', 'j': '𝗷',
        'k': '𝗸', 'l': '𝗹', 'm': '𝗺', 'n': '𝗻', 'o': '𝗼', 'p': '𝗽', 'q': '𝗾', 'r': '𝗿', 's': '𝘀', 't': '𝘁',
        'u': '𝘂', 'v': '𝘃', 'w': '𝘄', 'x': '𝘅', 'y': '𝘆', 'z': '𝘇',
        # Numbers
        '0': '𝟬', '1': '𝟭', '2': '𝟮', '3': '𝟯', '4': '𝟰', '5': '𝟱', '6': '𝟲', '7': '𝟳', '8': '𝟴', '9': '𝟵',
    }
    return ''.join(bold_map.get(char, char) for char in text)

def format_for_messaging(text: str) -> str:
    """
    Format Markdown text for messaging platforms like LINE and Instagram.
    
    Performs the following:
    1. Converts bolding (**text** -> 𝘁𝗲𝘅𝘁 using Unicode bold)
    2. Removes italics (*text* -> text)
    3. Converts links ([text](url) -> text: url)
    4. Removes headers (# Header -> Header)
    5. Removes brackets ([text] -> text)
    """
    if not text:
        return ""

    # 1. Convert Links: [Link Text](URL) -> Link Text: URL
    def replace_link(match):
        label = match.group(1)
        url = match.group(2)
        return f"{label}: {url}"
    
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)

    # 2. Convert Bold to Unicode: **text** or __text__ -> 𝘁𝗲𝘅𝘁
    def replace_bold(match):
        return _to_unicode_bold(match.group(1))
    
    text = re.sub(r'\*\*(.*?)\*\*', replace_bold, text)
    text = re.sub(r'__(.*?)__', replace_bold, text)

    # 3. Remove Italics: *text* or _text_ -> text
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)\_(?!\_)(.*?)(?<!_)\_(?!\_)', r'\1', text)

    # 4. Remove Headers: # Header -> Header
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # 5. Remove standalone brackets: [text] -> text (but only if not part of a markdown link)
    # This pattern matches [text] that is NOT followed by (url)
    text = re.sub(r'\[([^\]]+)\](?!\()', r'\1', text)

    return text.strip()
