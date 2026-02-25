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
2. Start with a "## Personality" section — define the AI's identity and role. E.g., "You are [Business Name]'s AI assistant, a [role description]." Describe the persona naturally (e.g., a knowledgeable hotel concierge, a friendly restaurant guide, a helpful hospital front desk assistant, etc.).
3. Include a "## About the Business" section — MAXIMUM 2-3 LINES. Describe the business name, what makes it special, its type (restaurant/hotel/shop/etc.), and location. Do NOT include prices, reviews, ratings, specific menu items, opening hours, or any details that could change.
4. Do NOT include any "Response Rules" or behavioral instructions (these will be added automatically).
5. Do NOT write anything that looks like a chatbot response. No greetings, no "How can I help you?".
6. Keep the ENTIRE output under 150 words.

EXAMPLE (adapt language to match website):
```
## Personality
You are ExampleCafe's AI assistant, a knowledgeable and friendly café guide who is passionate about coffee and helping visitors find their perfect brew.

## About the Business
ExampleCafe is a specialty coffee shop in Shibuya, Tokyo, known for its single-origin roasts and cozy atmosphere.
```

Website URL: {url}

Website content (homepage only):
{content}

Generate ONLY the "## Personality" and "## About the Business" sections. No commentary, no markdown fences."""

_STANDARD_RESPONSE_RULES = """
## Response Rules
- MANDATORY: Detect the language of the user's input and respond in that same language.
- Use bullet points when listing multiple items that belong to the same category (eg, items in a menu, list of services, list of places etc, not limited to these).
- You may use markdown bold (**text**) to emphasize important items, but use it sparingly.
- Provide detailed, helpful explanations.
- If you don't know something, say so and suggest checking the website.
- End responses on a positive, welcoming note, and if applicable, ask a logical follow up question that makes sense from the trained content.
"""

_RAG_META_PROMPT = """You are an expert prompt engineer. Your job is to write SYSTEM INSTRUCTIONS for an AI chatbot.

IMPORTANT: You are writing INSTRUCTIONS that tell the AI how to behave — NOT a conversation with a user. The output must read like a configuration document using directive language ("You are...", "You must...").

RULES FOR THE OUTPUT:
1. Write everything in the SAME LANGUAGE as the business content provided.
2. Start with a "## Personality" section — define the AI's identity and role. E.g., "You are [Business Name]'s AI assistant, a [role description]." Describe the persona naturally (e.g., a knowledgeable hotel concierge, a friendly restaurant guide, a helpful hospital front desk assistant, etc.).
3. Include a "## About the Business" section — MAXIMUM 2-3 LINES. Describe the business name, what makes it special, its type (restaurant/hotel/shop/etc.), and location. Do NOT include prices, reviews, ratings, specific menu items, opening hours, or any details that could change.
4. Do NOT include any "Response Rules" or behavioral instructions (these will be added automatically).
5. Do NOT write anything that looks like a chatbot response. No greetings, no "How can I help you?".
6. Keep the ENTIRE output under 150 words.

EXAMPLE:
```
## Personality
You are ExampleCafe's AI assistant, a knowledgeable and friendly café guide who is passionate about coffee and helping visitors find their perfect brew.

## About the Business
ExampleCafe is a specialty coffee shop in Shibuya, Tokyo, known for its single-origin roasts and cozy atmosphere.
```

Business Name: {business_name}

Business content (from trained data):
{content}

Generate ONLY the "## Personality" and "## About the Business" sections. No commentary, no markdown fences."""


_RAG_META_PROMPT_JA = """あなたは専門のプロンプトエンジニアです。AIチャットボットのシステム指示を書くのがあなたの仕事です。

重要：AIの動作を指示する文書を書いています。ユーザーとの会話ではありません。出力は「あなたは...」「...してください」などの指示文として読めるようにしてください。

出力ルール：
1. すべて日本語で書いてください。
2. 「## パーソナリティ」セクションから始め、AIのアイデンティティと役割を定義してください。例：「あなたは{business_name}のAIアシスタントで、[役割の説明]です。」自然にペルソナを説明してください（例：知識豊富なホテルコンシェルジュ、フレンドリーなレストランガイドなど）。
3. 「## ビジネスについて」セクションを含めてください — 最大2〜3行。ビジネス名、特徴、タイプ（レストラン/ホテル/ショップなど）、場所を記述。価格、レビュー、メニュー項目、営業時間などの変動する詳細は含めないでください。
4. 「応答ルール」や行動指示は含めないでください（自動で追加されます）。
5. チャットボットの応答のようなものは書かないでください。挨拶や「何かお手伝いできますか？」は不要です。
6. 出力全体を150語以内に収めてください。

例：
```
## パーソナリティ
あなたはExampleCafeのAIアシスタントで、コーヒーに情熱を持つ知識豊富でフレンドリーなカフェガイドです。

## ビジネスについて
ExampleCafeは東京・渋谷にあるスペシャルティコーヒーショップで、シングルオリジンの焙煎と居心地の良い雰囲気が特徴です。
```

ビジネス名: {business_name}

ビジネスコンテンツ（トレーニングデータより）:
{content}

「## パーソナリティ」と「## ビジネスについて」セクションのみ生成してください。コメントやマークダウンフェンスは不要です。"""

_STANDARD_RESPONSE_RULES_JA = """
## 応答ルール
- 必須：ユーザーの入力言語を検出し、同じ言語で応答してください。
- 同じカテゴリに属する複数の項目をリストする場合は箇条書きを使用してください（例：メニュー項目、サービス一覧、場所のリストなど）。
- 重要な項目を強調するためにマークダウンの太字（**テキスト**）を控えめに使用できます。
- 詳細で役立つ説明を提供してください。
- わからないことがある場合は正直に伝え、ウェブサイトの確認を提案してください。
- 応答は前向きで歓迎的な言葉で締めくくり、適切な場合はトレーニングコンテンツに基づいた論理的なフォローアップの質問をしてください。
"""


def generate_prompt_from_rag_content(
    rag_snippets: list[str],
    business_name: str = "",
    lang: str = "en",
) -> Optional[str]:
    """
    Use Gemini to generate a professional system prompt from RAG snippets.

    Returns the generated prompt string (with response rules appended), or None on failure.
    """
    if not rag_snippets:
        logger.warning("[PromptGen] No RAG snippets provided")
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

    context_block = "\n\n".join(rag_snippets[:4])

    meta_template = _RAG_META_PROMPT_JA if lang == "ja" else _RAG_META_PROMPT
    prompt = meta_template.format(
        business_name=business_name or ("ビジネス" if lang == "ja" else "the business"),
        content=context_block,
    )

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
            logger.warning("[PromptGen] LLM returned empty response for RAG prompt")
            return None

        # Clean up any markdown fences the LLM might have added
        if generated.startswith("```"):
            lines = generated.splitlines()
            if lines[-1].strip() == "```":
                generated = "\n".join(lines[1:-1]).strip()
            else:
                generated = "\n".join(lines[1:]).strip()

        # Append standard response rules (language-appropriate)
        rules = _STANDARD_RESPONSE_RULES_JA if lang == "ja" else _STANDARD_RESPONSE_RULES
        generated = generated + rules

        logger.info("[PromptGen] Generated RAG-based prompt (%d chars, lang=%s) for %s", len(generated), lang, business_name)
        return generated

    except Exception as e:
        logger.warning(
            "[PromptGen] LLM call failed for RAG prompt: %s: %s",
            type(e).__name__,
            str(e)[:200],
        )
        return None


def generate_prompt_from_content(
    homepage_content: str,
    root_url: str = "",
    business_name: str = "",
    lang: str = "en",
) -> Optional[str]:
    """
    Use Gemini to generate a professional system prompt from homepage content.

    Returns the generated prompt string, or None on failure.
    """
    if not homepage_content or not homepage_content.strip():
        logger.warning("[PromptGen] No homepage content provided")
        return None

    normalized_lang = "ja" if str(lang or "").strip().lower() in ("ja", "jp") else "en"

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

    # Inject business name into the prompt instructions if provided (English template only).
    identity_instruction = ""
    if business_name and normalized_lang != "ja":
        identity_instruction = f"""
IMPORTANT: The business name is "{business_name}". Use EXACTLY this name in the identity line.
"""

    if normalized_lang == "ja":
        prompt = _RAG_META_PROMPT_JA.format(
            business_name=business_name or "ビジネス",
            content=content,
        )
    else:
        prompt = _META_PROMPT.format(
            url=root_url or "unknown",
            content=content
        )
    
    if business_name and identity_instruction:
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

        # Append standard response rules (language-appropriate)
        rules = _STANDARD_RESPONSE_RULES_JA if normalized_lang == "ja" else _STANDARD_RESPONSE_RULES
        generated = generated + rules

        logger.info("[PromptGen] Generated prompt (%d chars, lang=%s) for %s", len(generated), normalized_lang, root_url)
        return generated

    except Exception as e:
        logger.warning(
            "[PromptGen] LLM call failed: %s: %s",
            type(e).__name__,
            str(e)[:200],
        )
        return None
