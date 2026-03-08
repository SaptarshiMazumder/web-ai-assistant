from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from domain.platform_profiles import get_default_rag_instruction

RENDER_TARGET_WEB_MARKDOWN = "web_markdown"
RENDER_TARGET_PLAIN_TEXT_CHANNEL = "plain_text_channel"

_URL_SAFE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")
_TRAILING_PUNCTUATION = set(".,;:!?")
_TRAILING_QUOTES = {'"', "'"}
_CLOSING_DELIMITERS = {")": "(", "]": "[", "}": "{"}
_GENERIC_FALLBACK_LABEL = "this page"


@dataclass(frozen=True)
class AnswerLinkCandidate:
    url: str
    label: str = ""
    source_kind: str = "citation"
    domain_key: str = ""


@dataclass(frozen=True)
class NormalizedAnswerLink:
    url: str
    label: str
    source_kind: str


@dataclass(frozen=True)
class NormalizedAnswer:
    text: str
    links: List[NormalizedAnswerLink]


@dataclass(frozen=True)
class _TextSegment:
    text: str


@dataclass(frozen=True)
class _LinkSegment:
    url: str
    label: str
    source_kind: str


@dataclass(frozen=True)
class _MarkdownLinkMatch:
    start: int
    end: int
    label: str
    url: str


def build_answer_link_candidates(
    *,
    reservation_config: Optional[Dict[str, Any]] = None,
    url_bank: Optional[Sequence[Dict[str, Any]]] = None,
    sources: Optional[Iterable[Any]] = None,
) -> List[AnswerLinkCandidate]:
    deduped: Dict[Tuple[str, str, str, str, str], AnswerLinkCandidate] = {}

    def _add(url: Any, *, label: Any = "", source_kind: str = "citation", domain_key: Any = "") -> None:
        normalized_url = _normalize_http_url(url)
        if not normalized_url:
            return
        key = _compare_url_key(normalized_url)
        if key is None:
            return
        candidate = AnswerLinkCandidate(
            url=normalized_url,
            label=str(label or "").strip(),
            source_kind=str(source_kind or "citation").strip() or "citation",
            domain_key=str(domain_key or "").strip().lower(),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            return
        if existing.source_kind == "citation" and candidate.source_kind != "citation":
            deduped[key] = candidate
            return
        if not existing.label and candidate.label:
            deduped[key] = candidate

    if isinstance(reservation_config, dict):
        _add(
            reservation_config.get("url"),
            label=reservation_config.get("link_label"),
            source_kind="reservation",
            domain_key=reservation_config.get("domain_key"),
        )
    if isinstance(url_bank, Sequence):
        for item in url_bank:
            if isinstance(item, dict):
                _add(item.get("url"), label=item.get("label"), source_kind="url_bank")
    if sources is not None:
        for item in sources:
            if isinstance(item, dict):
                _add(item.get("url"), label=item.get("label"), source_kind="citation")
            else:
                _add(getattr(item, "url", None), label=getattr(item, "label", None), source_kind="citation")
    return list(deduped.values())


def normalize_answer_links(
    text: str,
    *,
    candidates: Optional[Sequence[AnswerLinkCandidate]] = None,
    render_target: str = RENDER_TARGET_WEB_MARKDOWN,
) -> NormalizedAnswer:
    if not text:
        return NormalizedAnswer(text=text or "", links=[])

    segments = _parse_segments(str(text), list(candidates or []))
    rendered_parts: List[str] = []
    links: List[NormalizedAnswerLink] = []

    for segment in segments:
        rendered = _render_segment(segment, render_target=render_target)
        if not rendered:
            continue
        if render_target == RENDER_TARGET_PLAIN_TEXT_CHANNEL and rendered_parts:
            if _needs_plain_text_separator(rendered_parts[-1], rendered):
                rendered_parts.append(" ")
        rendered_parts.append(rendered)
        if isinstance(segment, _LinkSegment):
            links.append(
                NormalizedAnswerLink(
                    url=segment.url,
                    label=segment.label,
                    source_kind=segment.source_kind,
                )
            )

    return NormalizedAnswer(text="".join(rendered_parts), links=links)


def _parse_segments(text: str, candidates: Sequence[AnswerLinkCandidate]) -> List[object]:
    segments: List[object] = []
    cursor = 0
    while cursor < len(text):
        markdown_match = _find_next_markdown_link(text, cursor)
        if markdown_match is None:
            segments.extend(_parse_bare_text(text[cursor:], candidates))
            break
        if markdown_match.start > cursor:
            segments.extend(_parse_bare_text(text[cursor:markdown_match.start], candidates))
        normalized = _normalize_markdown_url(markdown_match.url, markdown_match.label, candidates)
        if normalized is None:
            segments.append(_TextSegment(text=text[markdown_match.start:markdown_match.end]))
        else:
            link_segment, trailing_text = normalized
            segments.append(link_segment)
            if trailing_text:
                segments.append(_TextSegment(text=trailing_text))
        cursor = markdown_match.end
    return _merge_text_segments(segments)


def _parse_bare_text(text: str, candidates: Sequence[AnswerLinkCandidate]) -> List[object]:
    if not text:
        return []
    parts: List[object] = []
    cursor = 0
    while cursor < len(text):
        match = re.search(r"https?://", text[cursor:], flags=re.IGNORECASE)
        if match is None:
            parts.append(_TextSegment(text=text[cursor:]))
            break
        url_start = cursor + match.start()
        if url_start > cursor:
            parts.append(_TextSegment(text=text[cursor:url_start]))
        consumed = _consume_url_from_text(text, url_start, candidates)
        if consumed is None:
            parts.append(_TextSegment(text=text[url_start : url_start + 1]))
            cursor = url_start + 1
            continue
        link_segment, end_index, trailing_text = consumed
        parts.append(link_segment)
        if trailing_text:
            parts.append(_TextSegment(text=trailing_text))
        cursor = end_index
    return parts


def _consume_url_from_text(
    text: str,
    start: int,
    candidates: Sequence[AnswerLinkCandidate],
    *,
    original_label: Optional[str] = None,
) -> Optional[Tuple[_LinkSegment, int, str]]:
    end = start
    while end < len(text) and _is_url_char(text[end]):
        end += 1
    if end <= start:
        return None

    raw_token = text[start:end]
    nested_split = _find_nested_url_start(raw_token)
    if nested_split is not None:
        end = start + nested_split
        raw_token = raw_token[:nested_split]

    raw_url, trailing_text = _trim_url_suffix(raw_token)
    if not raw_url:
        return None

    normalized_url, matched_candidate = _canonicalize_candidate_url(raw_url, candidates)
    if not _is_valid_http_url(normalized_url):
        return None

    label = _resolve_link_label(
        normalized_url,
        original_label=original_label,
        matched_candidate=matched_candidate,
    )
    return (
        _LinkSegment(
            url=normalized_url,
            label=label,
            source_kind=matched_candidate.source_kind if matched_candidate else "detected",
        ),
        end,
        trailing_text,
    )


def _normalize_markdown_url(
    raw_url: str,
    original_label: str,
    candidates: Sequence[AnswerLinkCandidate],
) -> Optional[Tuple[_LinkSegment, str]]:
    wrapped = str(raw_url or "").strip()
    consumed = _consume_url_from_text(wrapped, 0, candidates, original_label=original_label)
    if consumed is None:
        return None
    link_segment, end_index, trailing_text = consumed
    return link_segment, trailing_text + wrapped[end_index:]


def _find_next_markdown_link(text: str, start: int) -> Optional[_MarkdownLinkMatch]:
    cursor = start
    while cursor < len(text):
        open_bracket = text.find("[", cursor)
        if open_bracket < 0:
            return None
        match = _parse_markdown_link_at(text, open_bracket)
        if match is not None:
            return match
        cursor = open_bracket + 1
    return None


def _parse_markdown_link_at(text: str, start: int) -> Optional[_MarkdownLinkMatch]:
    if start < 0 or start >= len(text) or text[start] != "[":
        return None
    close_bracket = text.find("]", start + 1)
    if close_bracket < 0 or close_bracket + 1 >= len(text) or text[close_bracket + 1] != "(":
        return None
    depth = 1
    cursor = close_bracket + 2
    while cursor < len(text):
        ch = text[cursor]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return _MarkdownLinkMatch(
                    start=start,
                    end=cursor + 1,
                    label=text[start + 1 : close_bracket],
                    url=text[close_bracket + 2 : cursor],
                )
        cursor += 1
    return None


def _trim_url_suffix(raw_token: str) -> Tuple[str, str]:
    value = str(raw_token or "")
    trailing: List[str] = []

    while value:
        tail = value[-1]
        if tail in _TRAILING_PUNCTUATION or tail in _TRAILING_QUOTES:
            trailing.append(tail)
            value = value[:-1]
            continue
        opener = _CLOSING_DELIMITERS.get(tail)
        if opener and value.count(opener) < value.count(tail):
            trailing.append(tail)
            value = value[:-1]
            continue
        break

    trailing.reverse()
    return value, "".join(trailing)


def _find_nested_url_start(raw_token: str) -> Optional[int]:
    if not raw_token:
        return None
    match = re.search(r"https?://", raw_token[1:], flags=re.IGNORECASE)
    if match is None:
        return None
    index = 1 + match.start()
    if raw_token[index - 1] in {",", ";"}:
        return index
    return None


def _canonicalize_candidate_url(
    url: str,
    candidates: Sequence[AnswerLinkCandidate],
) -> Tuple[str, Optional[AnswerLinkCandidate]]:
    compare_key = _compare_url_key(url)
    for candidate in candidates:
        if compare_key is not None and _compare_url_key(candidate.url) == compare_key:
            return candidate.url, candidate

    parsed = urlsplit(url)
    host = (parsed.netloc or "").strip().lower()
    for candidate in candidates:
        if candidate.source_kind == "reservation" and candidate.domain_key and candidate.domain_key in host:
            return candidate.url, candidate

    return url, None


def _resolve_link_label(
    url: str,
    *,
    original_label: Optional[str] = None,
    matched_candidate: Optional[AnswerLinkCandidate] = None,
) -> str:
    label = re.sub(r"\s+", " ", str(original_label or "")).strip()
    if label and not _is_generic_link_text(label):
        return label
    if matched_candidate and matched_candidate.label:
        return matched_candidate.label
    return _preferred_link_text_from_url(url)


def _preferred_link_text_from_url(url: str) -> str:
    try:
        parsed = urlsplit((url or "").strip())
        if (parsed.scheme or "").lower() not in ("http", "https"):
            return _GENERIC_FALLBACK_LABEL
        path = (parsed.path or "").strip()
        if not path or path == "/":
            return "the website"
        segment = path.rstrip("/").split("/")[-1].strip().lower()
        segment = re.sub(r"[\-_]+", " ", segment)
        segment = re.sub(r"\s+", " ", segment).strip()
        if not segment or segment in {"index", "home", "top"}:
            return _GENERIC_FALLBACK_LABEL
        digit_ratio = sum(1 for ch in segment if ch.isdigit()) / max(1, len(segment))
        if digit_ratio > 0.5 and len(segment) > 4:
            return _GENERIC_FALLBACK_LABEL
        if re.fullmatch(r"[a-f0-9]{6,}", segment):
            return _GENERIC_FALLBACK_LABEL
        mappings = get_default_rag_instruction().get("link_text_mappings") or {}
        if isinstance(mappings, dict) and segment in mappings:
            return str(mappings[segment])
        if 1 <= len(segment) <= 40:
            return f"{segment} page"
    except Exception:
        pass
    return _GENERIC_FALLBACK_LABEL


def _is_generic_link_text(label: str) -> bool:
    if not label:
        return True
    normalized = re.sub(r"\s+", " ", label).strip().lower()
    if not normalized:
        return True
    generic_list = get_default_rag_instruction().get("generic_link_texts") or []
    generic = {str(item).strip().lower() for item in generic_list if str(item).strip()}
    if normalized in generic:
        return True
    if "http://" in normalized or "https://" in normalized:
        return True
    if normalized.startswith("/") and " " not in normalized:
        return True
    if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}(/.*)?", normalized) and " " not in normalized:
        return True
    return False


def _render_segment(segment: object, *, render_target: str) -> str:
    if isinstance(segment, _TextSegment):
        return segment.text
    if not isinstance(segment, _LinkSegment):
        return ""
    if render_target == RENDER_TARGET_PLAIN_TEXT_CHANNEL:
        if segment.label and segment.label != segment.url:
            return f"{segment.label}: {segment.url}"
        return segment.url
    return f"[{segment.label}]({segment.url})"


def _needs_plain_text_separator(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    prev_char = previous[-1]
    next_char = current[0]
    if prev_char.isspace() or next_char.isspace():
        return False
    if next_char in ".,;:!?)]}":
        return False
    if prev_char in "([{":
        return False
    return True


def _merge_text_segments(segments: List[object]) -> List[object]:
    merged: List[object] = []
    for segment in segments:
        if isinstance(segment, _TextSegment):
            if merged and isinstance(merged[-1], _TextSegment):
                merged[-1] = _TextSegment(text=merged[-1].text + segment.text)
            elif segment.text:
                merged.append(segment)
        else:
            merged.append(segment)
    return merged


def _normalize_http_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        return ""
    return url


def _compare_url_key(url: str) -> Optional[Tuple[str, str, str, str, str]]:
    try:
        parsed = urlsplit((url or "").strip())
    except Exception:
        return None
    scheme = (parsed.scheme or "").lower()
    host = (parsed.netloc or "").lower()
    if scheme not in {"http", "https"} or not host:
        return None
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return scheme, host, path, parsed.query or "", parsed.fragment or ""


def _is_valid_http_url(url: str) -> bool:
    compare_key = _compare_url_key(url)
    return compare_key is not None


def _is_url_char(ch: str) -> bool:
    return bool(ch) and ord(ch) <= 127 and ch in _URL_SAFE_CHARS
