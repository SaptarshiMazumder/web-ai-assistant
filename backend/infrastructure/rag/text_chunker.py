"""
Utility to split plain text into overlapping word-count chunks and return
Document objects formatted exactly like pdf_source_tasks.py produces.
"""
from typing import List, Optional

from domain.entities import Document


def chunk_text(
    text: str,
    *,
    title: Optional[str] = None,
    source_url: str,
    max_words: int = 600,
    overlap_words: int = 50,
) -> List[Document]:
    """
    Split *text* into overlapping chunks of at most *max_words* words.
    Each chunk becomes a Document with a metadata header matching the
    format used by pdf_source_tasks.

    Returns an empty list if text is blank.
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    title_str = (title or "").strip() or "Text"
    docs: List[Document] = []
    step = max(1, max_words - overlap_words)
    chunk_index = 0

    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        chunk_text_str = " ".join(chunk_words)

        header_lines = [f"Source: {title_str}"]
        if len(words) > max_words:
            header_lines.append(f"Chunk: {chunk_index + 1}")
        header = "\n".join(header_lines)

        content = f"{header}\n\n{chunk_text_str}"
        chunk_url = f"{source_url}?chunk={chunk_index}"

        docs.append(
            Document(
                url=chunk_url,
                content=content,
                metadata={
                    "source_type": "text",
                    "title": title_str,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words),
                },
            )
        )
        chunk_index += 1

        # Stop after the last chunk (when we've covered all words)
        if start + max_words >= len(words):
            break

    return docs
