"""
Topic-Question Mapping Service

Scans conversation messages and maps user questions to extracted topics using
semantic LLM classification (batch). One Gemini Flash call per batch of messages.
Results are stored in topic_question_mappings table.
"""

import json
import os
from typing import Any, Dict, List, Optional

from infrastructure.db.repositories import (
    PostgresExtractedTopicRepository,
    PostgresTopicQuestionRepository,
)

# Batch size: how many messages to classify in a single LLM call
_BATCH_SIZE = 40


def _classify_messages_batch(
    messages: List[Dict[str, str]],  # [{"id": ..., "text": ...}, ...]
    topics: List[Dict[str, str]],    # [{"id": ..., "name": ...}, ...]
) -> Dict[str, List[str]]:
    """
    Ask Gemini to classify a batch of messages against a topic list.
    Returns {message_id: [topic_id, ...]} for messages that matched.
    One LLM call for the whole batch.
    """
    from google import genai
    from google.genai import types

    project = (os.environ.get("PROJECT_ID") or "").strip()
    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    model = (os.environ.get("VERTEX_TOPIC_MODEL") or os.environ.get("VERTEX_RAG_MODEL") or "gemini-2.0-flash-001").strip()

    if not project:
        return {}

    topic_list = "\n".join(f'- id={t["id"]} name="{t["name"]}"' for t in topics)
    msg_list = "\n".join(f'{i+1}. [{m["id"]}] {m["text"][:300]}' for i, m in enumerate(messages))

    prompt = f"""You are classifying customer support messages into topics.

Topics available:
{topic_list}

For each message below, return which topic IDs it relates to (semantically — the message doesn't need to contain the exact topic word).
A message can match 0, 1, or multiple topics. Only include topics that are clearly relevant.

Messages:
{msg_list}

Return ONLY a JSON object mapping message ID to list of matching topic IDs. No commentary.
Example: {{"msg-abc": ["topic-1", "topic-2"], "msg-xyz": []}}"""

    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        text = (response.text or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(text)
        if isinstance(result, dict):
            return {k: v if isinstance(v, list) else [] for k, v in result.items()}
    except Exception as e:
        # Fall back to empty — non-fatal
        print(f"[TopicMapping] LLM batch classification failed: {type(e).__name__}: {e}")
    return {}


class TopicQuestionService:
    """Maps user questions to topics via semantic LLM batch classification."""

    def __init__(self):
        self._topic_repo = PostgresExtractedTopicRepository()
        self._mapping_repo = PostgresTopicQuestionRepository()

    def compute_topic_mappings(
        self,
        *,
        org_id: str,
        bot_id: str,
        max_messages: int = 500,
    ) -> int:
        """
        Classify new user messages against active topics using Gemini.
        Only processes messages newer than the last mapping run.
        Returns number of new mappings created.
        """
        from infrastructure.db.connection import get_connection

        topics = self._topic_repo.get_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            active_only=True,
            limit=200,
        )
        if not topics:
            return 0

        topic_lookup = {t.topic_id: t for t in topics}
        topic_list = [{"id": t.topic_id, "name": t.topic} for t in topics]

        # Only process messages newer than the last mapping
        con = get_connection()
        try:
            last_row = con.execute(
                """
                SELECT matched_at FROM topic_question_mappings
                WHERE bot_id = %s
                ORDER BY matched_at DESC
                LIMIT 1
                """,
                (bot_id,),
            ).fetchone()
            last_mapped_at = last_row[0] if last_row else None

            if last_mapped_at:
                rows = con.execute(
                    """
                    SELECT m.message_id, m.session_id, m.content, m.created_at
                    FROM conversation_messages m
                    WHERE m.bot_id = %s AND m.role = 'user'
                      AND m.created_at > %s
                    ORDER BY m.created_at ASC
                    LIMIT %s
                    """,
                    (bot_id, last_mapped_at, max(1, min(int(max_messages), 5000))),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT m.message_id, m.session_id, m.content, m.created_at
                    FROM conversation_messages m
                    WHERE m.bot_id = %s AND m.role = 'user'
                    ORDER BY m.created_at ASC
                    LIMIT %s
                    """,
                    (bot_id, max(1, min(int(max_messages), 5000))),
                ).fetchall()
        finally:
            con.close()

        if not rows:
            return 0

        # Build per-message lookup for fast access
        msg_meta: Dict[str, Dict] = {
            row[0]: {"session_id": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        }

        # Process in batches to keep each LLM call small
        new_count = 0
        row_list = list(rows)
        for batch_start in range(0, len(row_list), _BATCH_SIZE):
            batch = row_list[batch_start: batch_start + _BATCH_SIZE]
            msg_batch = [
                {"id": r[0], "text": (r[2] or "")[:300]}
                for r in batch
                if r[2] and r[2].strip()
            ]
            if not msg_batch:
                continue

            classifications = _classify_messages_batch(msg_batch, topic_list)

            for msg_id, matched_topic_ids in classifications.items():
                meta = msg_meta.get(msg_id)
                if not meta:
                    continue
                for topic_id in (matched_topic_ids or []):
                    if topic_id not in topic_lookup:
                        continue
                    content = meta["content"] or ""
                    inserted = self._mapping_repo.upsert_mapping(
                        topic_id=topic_id,
                        org_id=org_id,
                        bot_id=bot_id,
                        session_id=meta["session_id"],
                        message_id=msg_id,
                        question_text=content[:500],
                        match_method="llm",
                    )
                    if inserted:
                        new_count += 1

        return new_count

    def get_questions_for_topic(
        self,
        *,
        topic_id: str,
        bot_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return questions linked to a topic, enriched with session info."""
        from infrastructure.db.connection import get_connection

        questions = self._mapping_repo.get_questions_for_topic(
            topic_id=topic_id,
            limit=limit,
        )
        if not questions:
            return []

        session_ids = list({q.session_id for q in questions})
        session_titles: Dict[str, str] = {}
        if session_ids:
            con = get_connection()
            try:
                placeholders = ",".join(["%s"] * len(session_ids))
                rows = con.execute(
                    f"SELECT session_id, title FROM conversation_sessions WHERE session_id IN ({placeholders})",
                    session_ids,
                ).fetchall()
                session_titles = {r[0]: (r[1] or "") for r in (rows or [])}
            finally:
                con.close()

        return [
            {
                "id": q.id,
                "topic_id": q.topic_id,
                "session_id": q.session_id,
                "message_id": q.message_id,
                "question_text": q.question_text,
                "asked_at": q.matched_at,
                "session_title": session_titles.get(q.session_id, ""),
            }
            for q in questions
        ]

    def get_topic_usage_summary(
        self,
        *,
        org_id: str,
        bot_id: str,
    ) -> Dict[str, Any]:
        """Return topic usage counts for the donut chart."""
        topics = self._topic_repo.get_extracted_topics(
            org_id=org_id,
            bot_id=bot_id,
            active_only=True,
            limit=200,
        )
        usage_counts = self._mapping_repo.get_usage_counts(
            org_id=org_id,
            bot_id=bot_id,
        )
        result_topics = []
        total = 0
        for t in topics:
            count = usage_counts.get(t.topic_id, 0)
            result_topics.append({
                "topic_id": t.topic_id,
                "topic": t.topic,
                "category": t.category or "other",
                "question_count": count,
                "source_url": t.source_url,
                "origin": t.origin,
            })
            total += count

        result_topics.sort(key=lambda x: x["question_count"], reverse=True)
        return {"topics": result_topics, "total_questions": total}


_instance: Optional[TopicQuestionService] = None


def topic_question_service() -> TopicQuestionService:
    global _instance
    if _instance is None:
        _instance = TopicQuestionService()
    return _instance
