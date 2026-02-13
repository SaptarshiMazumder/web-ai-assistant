"""
Topic Extraction Service

Extracts business-relevant topics from website content using LLM (Gemini).
Topics are extracted during crawl and can be re-extracted on-demand from RAG content.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
import vertexai

from common.config import config
from infrastructure.db.repositories import PostgresExtractedTopicRepository

# Config
PROJECT_ID = (config.PROJECT_ID or os.environ.get("PROJECT_ID") or "").strip()
GENAI_LOCATION = (os.environ.get("GENAI_LOCATION") or "global").strip()
RAG_LOCATION = (config.LOCATION or os.environ.get("RAG_LOCATION") or "us-central1").strip()


def _get_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


DEFAULT_MODEL_NAME = (os.environ.get("VERTEX_RAG_MODEL") or "gemini-2.0-flash-001").strip()
TOPIC_MODEL_NAME = (os.environ.get("VERTEX_TOPIC_MODEL") or "").strip() or DEFAULT_MODEL_NAME

TOPIC_CHUNK_CHARS = _get_int_env("TOPIC_CHUNK_CHARS", 25000)
TOPIC_MAX_CHUNKS = _get_int_env("TOPIC_MAX_CHUNKS", 8)
TOPIC_MAX_TOPICS_PER_CHUNK = _get_int_env("TOPIC_MAX_TOPICS_PER_CHUNK", 25)
TOPIC_MAX_TOPICS_TOTAL = _get_int_env("TOPIC_MAX_TOPICS_TOTAL", 60)

# Topic extraction prompts
DEFAULT_CATEGORIES = [
    "product",
    "pricing",
    "shipping",
    "support",
    "policy",
    "location",
    "hours",
    "contact",
    "faq",
    "event",
    "other",
]

TOPIC_DISCOVERY_PROMPT = """Analyze the following website content and extract business-relevant topics that users might ask about.

Rules:
- Return ONLY a JSON array with no markdown fences or commentary.
- Each item should be an object with:
  - "topic": lowercase, 1-3 words
  - "confidence": 0.0 to 1.0 (optional)

Example output:
[
  {{"topic": "refunds", "confidence": 0.95}},
  {{"topic": "pricing plans", "confidence": 0.9}},
  {{"topic": "business hours", "confidence": 0.85}}
]

Extract up to {max_topics} most relevant topics. Be specific and practical.

WEBSITE CONTENT:
"""

TOPIC_CATEGORIZATION_PROMPT = """Group the following topics into 5-10 categories and assign each topic to a category.

Guidelines:
- Categories should be lowercase, 1-3 words.
- Use categories that fit the business domain. Examples only: pricing, faq, refunds, shipping, support.
- Return ONLY a JSON object with no markdown fences or commentary:
  {{
    "categories": ["category1", "category2", "..."],
    "topics": [
      {{"topic": "topic text", "category": "category1"}},
      ...
    ]
  }}
- Every topic must appear exactly once in "topics".
- Each topic's category must be one of "categories".
- Include "other" as a category if needed.

Topics (JSON array):
{topics_json}
"""


def _extract_json_array(text: str) -> Optional[List[Any]]:
    """Extract JSON array from text, handling markdown fences."""
    if not text:
        return None
    
    # Try to find JSON in markdown fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON array
    m = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text, handling markdown fences."""
    if not text:
        return None
    
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    m = re.search(r"({.*})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    return None


def _get_genai_client() -> genai.Client:
    """Get an initialized Gemini client."""
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    return genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)


def _generate_text_with_fallback(
    client: genai.Client,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """Generate text with a cheap model first, then fall back to default if it fails."""
    def _call(model_name: str) -> str:
        resp = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (resp.text or "").strip()

    primary = TOPIC_MODEL_NAME
    try:
        return _call(primary)
    except Exception as e:
        if primary != DEFAULT_MODEL_NAME:
            print(f"[TopicExtraction] Model '{primary}' failed, falling back to '{DEFAULT_MODEL_NAME}': {e}")
            return _call(DEFAULT_MODEL_NAME)
        raise


def _sanitize_category_list(raw_categories: List[Any]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in raw_categories:
        if not isinstance(item, str):
            continue
        label = re.sub(r"\s+", " ", item.strip().lower())
        if not label or len(label) > 40:
            continue
        if label in seen:
            continue
        seen.add(label)
        cleaned.append(label)
    return cleaned


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "by",
    "at",
    "from",
    "is",
    "are",
    "be",
    "how",
    "what",
    "when",
    "where",
}


def _normalize_topic_key(topic: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", topic.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    normalized: List[str] = []
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return " ".join(normalized) or " ".join(tokens) or cleaned.strip()


def _dedupe_topic_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for item in items:
        topic = (item.get("topic") or "").strip()
        if not topic:
            continue
        key = _normalize_topic_key(topic)
        if not key:
            continue
        existing = by_key.get(key)
        if not existing:
            by_key[key] = item
            continue
        # Prefer shorter label to reduce redundancy (e.g., "shipping cost" vs "shipping costs")
        if len(topic) < len((existing.get("topic") or "")):
            by_key[key] = item
    return list(by_key.values())


def _split_content_into_chunks(content: str, max_chars: int, max_chunks: int) -> List[str]:
    if len(content) <= max_chars:
        return [content]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in content.splitlines():
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            if len(chunks) >= max_chunks:
                return chunks
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current and len(chunks) < max_chunks:
        chunks.append("\n".join(current))
    return chunks


def _extract_topic_candidates_from_chunk(
    client: genai.Client,
    content: str,
    max_topics: int,
) -> List[Dict[str, Any]]:
    prompt = TOPIC_DISCOVERY_PROMPT.format(max_topics=max_topics) + content
    raw_text = _generate_text_with_fallback(
        client,
        prompt,
        temperature=0.2,
        max_output_tokens=2048,
    )
    parsed = _extract_json_array(raw_text) or []
    topics: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, str):
            text = item.strip().lower()
            if text:
                topics.append({"topic": text, "confidence": 1.0})
            continue
        if not isinstance(item, dict):
            continue
        text = (item.get("topic") or "").strip().lower()
        if not text:
            continue
        confidence = item.get("confidence", 1.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 1.0
        topics.append({"topic": text, "confidence": confidence})
    return topics


def _categorize_topics(
    client: genai.Client,
    topics: List[str],
) -> Dict[str, str]:
    if not topics:
        return {}
    topics_json = json.dumps(topics)
    prompt = TOPIC_CATEGORIZATION_PROMPT.format(topics_json=topics_json)
    raw_text = _generate_text_with_fallback(
        client,
        prompt,
        temperature=0.1,
        max_output_tokens=2048,
    )
    parsed = _extract_json_object(raw_text) or {}
    categories_raw = parsed.get("categories") if isinstance(parsed, dict) else None
    topics_raw = parsed.get("topics") if isinstance(parsed, dict) else None

    categories = _sanitize_category_list(categories_raw or [])
    if not categories:
        categories = DEFAULT_CATEGORIES.copy()
    if "other" not in categories:
        categories.append("other")

    allowed_categories = set(categories)
    topic_set = set(topics)
    mapping: Dict[str, str] = {}

    if isinstance(topics_raw, list):
        for item in topics_raw:
            if not isinstance(item, dict):
                continue
            topic = (item.get("topic") or "").strip().lower()
            if not topic or topic not in topic_set:
                continue
            category = re.sub(r"\s+", " ", (item.get("category") or "other").strip().lower())
            if not category or category not in allowed_categories:
                category = "other"
            mapping[topic] = category

    # Fill any missing topics as "other"
    for topic in topics:
        if topic not in mapping:
            mapping[topic] = "other"

    return mapping


def _validate_topic(topic: Dict[str, Any], allowed_categories: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Validate and normalize a topic dict."""
    topic_text = (topic.get("topic") or "").strip().lower()
    if not topic_text or len(topic_text) > 100:
        return None
    
    category = re.sub(r"\s+", " ", (topic.get("category") or "other").strip().lower())
    if not category or len(category) > 40:
        category = "other"
    if allowed_categories and category not in allowed_categories:
        category = "other"
    
    confidence = topic.get("confidence", 1.0)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 1.0
    
    return {
        "topic": topic_text,
        "category": category,
        "confidence": confidence,
    }


class TopicExtractionService:
    """Service for extracting topics from website content using LLM."""
    
    def __init__(self):
        self._topic_repo = PostgresExtractedTopicRepository()
    
    def extract_topics_from_content(
        self,
        content: str,
        source_urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from content using Gemini.
        
        Args:
            content: The website content to analyze
            source_urls: Optional list of source URLs for the content
            
        Returns:
            List of extracted topics with topic, category, confidence, source_urls
        """
        if not content or len(content.strip()) < 50:
            return []
        
        try:
            client = _get_genai_client()
            chunks = _split_content_into_chunks(content, TOPIC_CHUNK_CHARS, TOPIC_MAX_CHUNKS)
            print(f"[TopicExtraction] Content {len(content)} chars → {len(chunks)} chunks")
            candidates: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                chunk_candidates = _extract_topic_candidates_from_chunk(
                    client,
                    chunk,
                    TOPIC_MAX_TOPICS_PER_CHUNK,
                )
                print(f"[TopicExtraction] Chunk {i+1}/{len(chunks)}: {len(chunk_candidates)} candidates")
                candidates.extend(chunk_candidates)

            print(f"[TopicExtraction] Total candidates before dedup: {len(candidates)}")
            if not candidates:
                return []

            deduped = _dedupe_topic_candidates(candidates)
            print(f"[TopicExtraction] After dedup: {len(deduped)} topics")
            if not deduped:
                return []
            if len(deduped) > TOPIC_MAX_TOPICS_TOTAL:
                deduped = deduped[:TOPIC_MAX_TOPICS_TOTAL]

            topic_names = [t["topic"] for t in deduped if t.get("topic")]
            topic_names = [t for t in topic_names if t]
            if not topic_names:
                return []

            category_map = _categorize_topics(client, topic_names)
            topics: List[Dict[str, Any]] = []
            for item in deduped:
                topic_text = (item.get("topic") or "").strip().lower()
                if not topic_text:
                    continue
                validated = _validate_topic({
                    "topic": topic_text,
                    "category": category_map.get(topic_text, "other"),
                    "confidence": item.get("confidence", 1.0),
                })
                if validated:
                    validated["source_urls"] = source_urls or []
                    topics.append(validated)

            print(f"[TopicExtraction] Final validated topics: {len(topics)}")
            return topics

        except Exception as e:
            import traceback
            print(f"[TopicExtraction] Error extracting topics: {e}")
            traceback.print_exc()
            return []
    
    def extract_topics_for_bot(
        self,
        *,
        org_id: str,
        bot_id: str,
        content: str,
        source_urls: Optional[List[str]] = None,
        clear_existing: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from content and save them for a bot.
        
        Args:
            org_id: Organization ID
            bot_id: Bot ID
            content: Content to extract topics from
            source_urls: Optional source URLs
            clear_existing: If True, delete existing topics before saving new ones
            
        Returns:
            List of saved ExtractedTopic objects as dicts
        """
        if clear_existing:
            self._topic_repo.delete_all_topics_for_bot(org_id=org_id, bot_id=bot_id)
        
        topics = self.extract_topics_from_content(content, source_urls)
        
        if not topics:
            return []
        
        saved = self._topic_repo.save_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            topics=topics,
        )
        
        return [
            {
                "topic_id": t.topic_id,
                "topic": t.topic,
                "category": t.category,
                "confidence": t.confidence,
                "source_urls": t.source_urls,
                "occurrence_count": t.occurrence_count,
                "is_active": t.is_active,
            }
            for t in saved
        ]
    
    def extract_topics_from_documents(
        self,
        *,
        org_id: str,
        bot_id: str,
        documents: List[Dict[str, Any]],
        clear_existing: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from multiple documents (e.g., from a crawl).
        
        Args:
            org_id: Organization ID
            bot_id: Bot ID
            documents: List of documents with 'content' and 'url' keys
            clear_existing: If True, delete existing topics first
            
        Returns:
            List of saved topic dicts
        """
        if clear_existing:
            self._topic_repo.delete_all_topics_for_bot(org_id=org_id, bot_id=bot_id)
        
        # Combine all document content
        combined_content_parts: List[str] = []
        all_urls: List[str] = []
        
        for doc in documents:
            content = (doc.get("content") or "").strip()
            url = (doc.get("url") or "").strip()
            
            if content:
                combined_content_parts.append(f"--- Source: {url} ---\n{content}\n")
                if url:
                    all_urls.append(url)
        
        combined_content = "\n".join(combined_content_parts)
        
        if not combined_content:
            return []
        
        # Extract topics from combined content
        topics = self.extract_topics_from_content(combined_content, all_urls)
        
        if not topics:
            return []
        
        saved = self._topic_repo.save_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            topics=topics,
        )
        
        return [
            {
                "topic_id": t.topic_id,
                "topic": t.topic,
                "category": t.category,
                "confidence": t.confidence,
                "source_urls": t.source_urls,
                "occurrence_count": t.occurrence_count,
                "is_active": t.is_active,
            }
            for t in saved
        ]
    
    def create_topic(
        self,
        *,
        org_id: str,
        bot_id: str,
        topic: str,
        category: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a single topic (e.g. from Manage Topics UI)."""
        topic_text = (topic or "").strip()
        if not topic_text:
            return None
        saved = self._topic_repo.save_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            topics=[{"topic": topic_text, "category": (category or "").strip() or None, "confidence": 1.0}],
        )
        if not saved:
            return None
        t = saved[0]
        return {
            "topic_id": t.topic_id,
            "topic": t.topic,
            "category": t.category,
            "confidence": t.confidence,
            "source_urls": t.source_urls,
            "occurrence_count": t.occurrence_count,
            "is_active": t.is_active,
            "extracted_at": t.extracted_at,
            "updated_at": t.updated_at,
        }

    def get_topics(
        self,
        *,
        org_id: str,
        bot_id: str,
        active_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get extracted topics for a bot."""
        topics = self._topic_repo.get_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            active_only=active_only,
            limit=limit,
        )
        
        return [
            {
                "topic_id": t.topic_id,
                "topic": t.topic,
                "category": t.category,
                "confidence": t.confidence,
                "source_urls": t.source_urls,
                "source_url": t.source_url,
                "origin": t.origin,
                "occurrence_count": t.occurrence_count,
                "is_active": t.is_active,
                "extracted_at": t.extracted_at,
                "updated_at": t.updated_at,
            }
            for t in topics
        ]

    def update_topic(
        self,
        *,
        topic_id: str,
        is_active: Optional[bool] = None,
        category: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a topic's status or category."""
        updated = self._topic_repo.update_topic(
            topic_id=topic_id,
            is_active=is_active,
            category=category,
        )
        
        if not updated:
            return None
        
        return {
            "topic_id": updated.topic_id,
            "topic": updated.topic,
            "category": updated.category,
            "confidence": updated.confidence,
            "source_urls": updated.source_urls,
            "occurrence_count": updated.occurrence_count,
            "is_active": updated.is_active,
            "extracted_at": updated.extracted_at,
            "updated_at": updated.updated_at,
        }
    
    def delete_topic(self, *, topic_id: str) -> bool:
        """Delete a topic."""
        return self._topic_repo.delete_topic(topic_id=topic_id)

    def sync_url_bank_topics(
        self,
        *,
        org_id: str,
        bot_id: str,
        url_bank: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Sync URL bank entries as topics with origin='url_bank'.
        Each entry {label, url} becomes a topic named by its label with source_url set.
        Existing url_bank topics not in new list are deactivated.
        """
        if not url_bank:
            return []
        topics_to_save = []
        for entry in url_bank:
            label = (entry.get("label") or "").strip()
            url = (entry.get("url") or "").strip()
            if not label:
                continue
            topics_to_save.append({
                "topic": label.lower(),
                "category": "other",
                "confidence": 1.0,
                "source_urls": [url] if url else [],
                "source_url": url or None,
                "origin": "url_bank",
            })
        if not topics_to_save:
            return []
        saved = self._topic_repo.save_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            topics=topics_to_save,
        )
        return [
            {
                "topic_id": t.topic_id,
                "topic": t.topic,
                "category": t.category,
                "confidence": t.confidence,
                "source_urls": t.source_urls,
                "source_url": t.source_url,
                "origin": t.origin,
                "occurrence_count": t.occurrence_count,
                "is_active": t.is_active,
                "extracted_at": t.extracted_at,
                "updated_at": t.updated_at,
            }
            for t in saved
        ]


# Singleton instance
_topic_extraction_service: Optional[TopicExtractionService] = None


def topic_extraction_service() -> TopicExtractionService:
    """Get the singleton TopicExtractionService instance."""
    global _topic_extraction_service
    if _topic_extraction_service is None:
        _topic_extraction_service = TopicExtractionService()
    return _topic_extraction_service
