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
MODEL_NAME = os.environ.get("VERTEX_RAG_MODEL", "gemini-2.0-flash-001")

# Topic extraction prompt
TOPIC_EXTRACTION_PROMPT = """Analyze the following website content and extract business-relevant topics that users might ask about.

Focus on extracting topics like:
- Products or services offered
- Pricing and payment information
- Shipping and delivery
- Returns and refunds
- Hours of operation
- Location and contact information
- Support and help topics
- Policies (privacy, terms, etc.)
- FAQs and common questions
- Events or promotions
- Industry-specific terms

Return ONLY a JSON array with no markdown fences or commentary. Each topic should have:
- "topic": The topic name (lowercase, 1-3 words)
- "category": Category type (one of: product, pricing, shipping, support, policy, location, hours, contact, faq, event, other)
- "confidence": Confidence score from 0.0 to 1.0

Example output:
[
  {"topic": "refunds", "category": "policy", "confidence": 0.95},
  {"topic": "pricing plans", "category": "pricing", "confidence": 0.9},
  {"topic": "business hours", "category": "hours", "confidence": 0.85}
]

Extract up to 30 most relevant topics. Be specific and practical.

WEBSITE CONTENT:
"""


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
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


def _get_genai_client() -> genai.Client:
    """Get an initialized Gemini client."""
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    return genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)


def _validate_topic(topic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize a topic dict."""
    topic_text = (topic.get("topic") or "").strip().lower()
    if not topic_text or len(topic_text) > 100:
        return None
    
    category = (topic.get("category") or "other").strip().lower()
    valid_categories = {"product", "pricing", "shipping", "support", "policy", 
                       "location", "hours", "contact", "faq", "event", "other"}
    if category not in valid_categories:
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
        
        # Truncate content if too long (Gemini has token limits)
        max_chars = 100000  # ~25k tokens
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (content truncated)"
        
        client = _get_genai_client()
        
        prompt = TOPIC_EXTRACTION_PROMPT + content
        
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            
            raw_text = (resp.text or "").strip()
            parsed = _extract_json_array(raw_text)
            
            if not parsed:
                return []
            
            topics: List[Dict[str, Any]] = []
            for item in parsed:
                validated = _validate_topic(item)
                if validated:
                    validated["source_urls"] = source_urls or []
                    topics.append(validated)
            
            return topics[:30]  # Limit to 30 topics
            
        except Exception as e:
            print(f"[TopicExtraction] Error extracting topics: {e}")
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


# Singleton instance
_topic_extraction_service: Optional[TopicExtractionService] = None


def topic_extraction_service() -> TopicExtractionService:
    """Get the singleton TopicExtractionService instance."""
    global _topic_extraction_service
    if _topic_extraction_service is None:
        _topic_extraction_service = TopicExtractionService()
    return _topic_extraction_service
