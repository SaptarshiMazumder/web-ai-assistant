"""
Prompt Generation Service

Analyzes crawled homepage content using Gemini LLM to automatically generate
a professional system prompt for the bot, in the website's natural language.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPT_MODEL = (
    os.environ.get("VERTEX_PROMPT_GEN_MODEL")
    or os.environ.get("VERTEX_ASSET_MODEL")
    or "gemini-2.0-flash-lite-001"
).strip()

_META_PROMPT = """You are an expert prompt engineer. Your job is to write SYSTEM INSTRUCTIONS for an AI chatbot.

IMPORTANT: You are writing INSTRUCTIONS that tell the AI how to behave — NOT a conversation with a user. The output must read like a configuration document using directive language ("You are...", "You must...").

RULES FOR THE OUTPUT:
1. Write everything in the SAME LANGUAGE as the website content.
2. First line: define the AI's identity — "You are [Business Name]'s AI assistant, a [role]."
3. Include a brief "## About the Business" section — MAXIMUM 2 LINES. Just the business name, type (restaurant/hotel/shop/etc.), and location. Do NOT include prices, reviews, ratings, specific menu items, opening hours, or any details that could change.
4. Do NOT include any "Response Rules" or behavioral instructions (these will be added automatically).
5. Do NOT write anything that looks like a chatbot response. No greetings, no "How can I help you?".
6. Keep the ENTIRE output under 100 words.

EXAMPLE (adapt language to match website):
```
You are ExampleCafe's AI assistant, a knowledgeable café guide.

## About the Business
ExampleCafe is a specialty coffee shop in Shibuya, Tokyo, known for single-origin roasts.
```

Website URL: {url}

Website content (homepage only):
{content}

Generate ONLY the Identity line and "About the Business" section. No commentary, no markdown fences."""

_STANDARD_RESPONSE_RULES = """
## Response Rules
- MANDATORY: Detect the language of the user's input and respond in that same language.
- Use bullet points when listing multiple items.
- Provide detailed, helpful explanations.
- If you don't know something, say so and suggest checking the website.
- End responses on a positive, welcoming note.
"""

def generate_prompt_from_content(
    homepage_content: str,
    root_url: str = "",
    business_name: str = "",
) -> Optional[str]:
    """
    Use Gemini to generate a professional system prompt from homepage content.

    Returns the generated prompt string, or None on failure.
    """
    if not homepage_content or not homepage_content.strip():
        logger.warning("[PromptGen] No homepage content provided")
        return None

    project = (os.environ.get("PROJECT_ID") or "").strip()
    if not project:
        logger.warning("[PromptGen] PROJECT_ID not set, cannot generate prompt")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("[PromptGen] google-genai not available")
        return None

    # Truncate content to avoid token limits (keep first ~12k chars — homepage is usually enough)
    max_content = int(os.environ.get("PROMPT_GEN_MAX_CONTENT_CHARS", "12000"))
    content = homepage_content[:max_content]

    # Inject business name into the prompt instructions if provided
    identity_instruction = ""
    if business_name:
        identity_instruction = f"""
IMPORTANT: The business name is "{business_name}". Use EXACTLY this name in the identity line.
"""

    prompt = _META_PROMPT.format(
        url=root_url or "unknown",
        content=content
    )
    
    if business_name:
        # Prepend the identity instruction to the prompt context
        prompt = identity_instruction + "\n" + prompt

    location = (os.environ.get("GENAI_LOCATION") or "global").strip()
    client = genai.Client(vertexai=True, project=project, location=location)

    try:
        response = client.models.generate_content(
            model=_PROMPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        generated = (response.text or "").strip()
        if not generated:
            logger.warning("[PromptGen] LLM returned empty response")
            return None

        # Clean up any markdown fences the LLM might have added
        if generated.startswith("```"):
            lines = generated.splitlines()
            if lines[-1].strip() == "```":
                generated = "\n".join(lines[1:-1]).strip()
            else:
                generated = "\n".join(lines[1:]).strip()

        # Append standard response rules
        generated = generated + _STANDARD_RESPONSE_RULES

        logger.info("[PromptGen] Generated prompt (%d chars) for %s", len(generated), root_url)
        return generated

    except Exception as e:
        logger.warning(
            "[PromptGen] LLM call failed: %s: %s",
            type(e).__name__,
            str(e)[:200],
        )
        return None
