"""
Deterministic default system-instruction builder (no LLM).
"""

from typing import Any, Dict, Optional


def extract_business_type_from_widget_config(widget_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return normalized business type label from widget config, or None."""
    if not isinstance(widget_config, dict):
        return None
    raw = str(widget_config.get("businessType") or "").strip()
    if not raw:
        return None
    normalized = raw.replace("_", " ").replace("-", " ").strip().lower()
    return normalized or None


def build_default_system_instruction(
    *,
    bot_name: str,
    business_type: Optional[str],
    lang: str = "en",
) -> str:
    """
    Build a deterministic system instruction using bot name + optional business type.
    """
    is_ja = str(lang or "").strip().lower() in ("ja", "jp")
    name = (bot_name or "").strip() or ("このビジネス" if is_ja else "this business")
    bt = (business_type or "").strip()

    if is_ja:
        personality = (
            f"あなたは{name}、一流の{bt}アシスタントです。"
            if bt
            else f"あなたは{name}のAIアシスタントです。"
        )
        return (
            "## Personality\n"
            f"{personality}\n\n"
            "## Response Rules\n"
            "- ユーザーの入力言語に合わせて回答してください。\n"
            "- 学習済みコンテンツに基づいて、明確で正確に回答してください。\n"
            "- 不明な点は推測せず、分からないと伝えたうえで確認方法を案内してください。\n"
            "- 必要に応じて、次に取るべき行動を簡潔に提案してください。"
        )

    personality = f"You are {name}, a premier {bt} assistant." if bt else f"You are {name}, an AI assistant."
    return (
        "## Personality\n"
        f"{personality}\n\n"
        "## Response Rules\n"
        "- Respond in the same language as the user.\n"
        "- Use trained content as the source of truth and answer clearly.\n"
        "- If information is missing, say so directly and suggest how to confirm.\n"
        "- When helpful, suggest a clear next step."
    )

