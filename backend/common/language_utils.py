import re
from typing import Any


_JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")
_EN_CHAR_RE = re.compile(r"[A-Za-z]")


def normalize_lang(value: Any, fallback: str = "en") -> str:
    raw = str(value or "").strip().lower()
    if raw in ("ja", "jp"):
        return "ja"
    if raw == "en":
        return "en"
    return "ja" if str(fallback or "").strip().lower() in ("ja", "jp") else "en"


def detect_user_language(text: Any, *, fallback: str = "en") -> str:
    normalized_fallback = normalize_lang(fallback)
    sample = str(text or "").strip()
    if not sample:
        return normalized_fallback
    if _JP_CHAR_RE.search(sample):
        return "ja"
    if _EN_CHAR_RE.search(sample):
        return "en"
    return normalized_fallback
