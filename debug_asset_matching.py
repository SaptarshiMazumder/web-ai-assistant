
import re
import os
from typing import List, Dict, Set, Tuple, Optional

# --- Copy of asset_resolver.py logic ---

_MAX_ASSET_CARDS_PER_ANSWER = 2
_MAX_ASSET_CARDS_EXPLICIT_REQUEST = 4

_VISUAL_REQUEST_TERMS = {
    "photo", "photos", "image", "images", "picture", "pictures",
    "pic", "pics", "gallery", "show me", "send me", "share",
    "let me see", "what it looks like",
}
_VISUAL_REQUEST_MANY_TERMS = {
    "all", "more", "many", "several", "multiple",
    "full menu", "whole menu", "entire menu", "more photos", "more images",
}
_VISUAL_SUPPRESS_TERMS = {
    "no image", "no images", "no photo", "no photos",
    "no picture", "no pictures", "without image", "without images",
    "without photo", "without photos",
}
_ASSET_INTENT_TERMS = {
    "menu", "dish", "dishes", "food", "drink", "drinks",
    "beverage", "beverages", "product", "products", "service", "services",
    "package", "packages", "plan", "plans", "room", "rooms",
    "suite", "suites", "facility", "facilities", "amenity", "amenities",
    "location", "locations", "map", "branch", "branches",
    "store", "stores", "item", "items", "option", "options",
    "catalog", "collection",
}
_SPECIAL_SHORT_TOKENS = {"xl", "xxl", "xs"}
_GENERIC_TOKENS = {
    "about", "available", "booking", "cost", "detail", "details",
    "info", "information", "item", "items", "option", "options",
    "price", "product", "products", "service", "services",
}
_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "for", "from", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "the", "to",
    "we", "with", "you", "your",
} | _GENERIC_TOKENS

_QUALIFIER_GROUPS = (
    {"single", "double", "triple", "quad"},
    {"king", "queen", "twin", "full"},
    {"male", "man", "men", "female", "woman", "women", "kid", "kids", "child", "children"},
    {"small", "medium", "large", "xl", "xxl", "xs"},
)

class BotAsset:
    def __init__(self, asset_id, name, description, keywords, image_public_url, link_url, is_active=True, bot_id="test_bot"):
        self.asset_id = asset_id
        self.name = name
        self.description = description
        self.keywords = keywords
        self.image_public_url = image_public_url
        self.link_url = link_url
        self.is_active = is_active
        self.bot_id = bot_id

def _normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\bbed\s+room(s)?\b", r"bedroom\1", t)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _stem_token(token: str) -> str:
    t = token.strip().lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t

def _tokenize(text: str) -> Set[str]:
    norm = _normalize_text(text)
    if not norm:
        return set()
    out: Set[str] = set()
    for raw in norm.split():
        tok = _stem_token(raw)
        if len(tok) < 3 and tok not in _SPECIAL_SHORT_TOKENS:
            continue
        if tok in _STOPWORDS:
            continue
        out.add(tok)
    return out

def _contains_phrase(text_norm: str, phrase: str) -> bool:
    p = _normalize_text(phrase)
    if not p or len(p) < 3:
        return False
    return p in text_norm

def _normalize_url(url: str) -> str:
    # Simplified for testing
    return (url or "").strip().lower()

def _extract_normalized_urls(text: str) -> Set[str]:
    # Simplified
    return set()

def _qualifier_conflict(answer_tokens: Set[str], asset_tokens: Set[str]) -> bool:
    for group in _QUALIFIER_GROUPS:
        q = answer_tokens & group
        a = asset_tokens & group
        if q and a and q.isdisjoint(a):
            return True
    return False

def _asset_answer_score(
    asset: BotAsset,
    answer_norm: str,
    answer_tokens: Set[str],
    answer_urls: Set[str],
) -> int:
    text = " ".join([asset.name or "", asset.description or "", " ".join(asset.keywords or [])]).strip()
    asset_tokens = _tokenize(text)
    if not asset_tokens:
        return 0
    if _qualifier_conflict(answer_tokens, asset_tokens):
        return 0

    score = 0
    url_hit = False
    
    name_phrase = _contains_phrase(answer_norm, asset.name or "")
    if name_phrase:
        score += 10

    keyword_phrase = False
    for kw in asset.keywords or []:
        if _contains_phrase(answer_norm, kw):
            keyword_phrase = True
            score += 6
            break

    overlap = answer_tokens & asset_tokens
    overlap_count = len(overlap)
    if overlap_count >= 2:
        score += 4 + overlap_count
    elif overlap_count == 1 and any(t not in _GENERIC_TOKENS for t in overlap):
        score += 1

    # Require strong evidence from answer text.
    if not (url_hit or name_phrase or keyword_phrase or overlap_count >= 2):
        print(f"  [DEBUG] Failed Match: {asset.name} - No strong evidence (overlap={overlap_count})")
        return 0
    
    print(f"  [DEBUG] Successful Match: {asset.name} - Score={score} (overlap={overlap_count})")
    return score

def _asset_to_card(asset: BotAsset) -> Dict[str, str]:
    return {
        "asset_id": asset.asset_id,
        "name": asset.name,
        "image_url": asset.image_public_url,
        "link_url": asset.link_url or "",
    }

def _match_assets_from_answer(
    answer: str,
    assets: List[BotAsset],
    *,
    max_cards: int = _MAX_ASSET_CARDS_PER_ANSWER,
) -> List[Dict[str, str]]:
    answer_norm = _normalize_text(answer)
    answer_tokens = _tokenize(answer)
    answer_urls = _extract_normalized_urls(answer)

    if not answer_norm:
        return []

    scored: List[Tuple[int, str, BotAsset]] = []
    for a in assets:
        s = _asset_answer_score(a, answer_norm, answer_tokens, answer_urls)
        if s <= 0:
            continue
        scored.append((s, a.asset_id, a))

    # Stable deterministic ordering by score desc then asset_id asc.
    scored.sort(key=lambda it: (-it[0], it[1]))
    out: List[Dict[str, str]] = []
    for _, _, a in scored:
        out.append(_asset_to_card(a))
        if len(out) >= max_cards:
            break
    return out

def _contains_any_term(text_norm: str, terms: Set[str]) -> bool:
    if not text_norm:
        return False
    for term in terms:
        t = (term or "").strip()
        if not t:
            continue
        if " " in t:
            if t in text_norm:
                return True
            continue
        if re.search(rf"\b{re.escape(t)}\b", text_norm):
            return True
    return False

def _query_matches_cards(query_norm: str, cards: List[Dict[str, str]]) -> bool:
    q_tokens = _tokenize(query_norm)
    if not q_tokens:
        return False
    for card in cards:
        name_tokens = _tokenize(card.get("name", ""))
        if q_tokens & name_tokens:
            return True
    return False

def _max_cards_for_query(user_query: Optional[str], cards: List[Dict[str, str]]) -> int:
    query_norm = _normalize_text(user_query or "")
    if not query_norm:
        return 0
    if _contains_any_term(query_norm, _VISUAL_SUPPRESS_TERMS):
        return 0

    explicit_visual = _contains_any_term(query_norm, _VISUAL_REQUEST_TERMS) or bool(
        re.search(
            r"\b(show|send|share|see|view)\b.*\b(photo|image|picture|pic|gallery|menu|product|service|room|suite|location|map)\b",
            query_norm,
        )
    )
    intent_query = _contains_any_term(query_norm, _ASSET_INTENT_TERMS)
    card_match = _query_matches_cards(query_norm, cards)

    print(f"  [DEBUG] Query Check: explicit={explicit_visual}, intent={intent_query}, card_match={card_match}")

    if not (explicit_visual or intent_query or card_match):
        return 0

    return _MAX_ASSET_CARDS_PER_ANSWER


# --- NEW FUNCTIONS START ---

def build_asset_instruction(assets: List[BotAsset]) -> str:
    if not assets:
        return ""

    lines: list[str] = []
    for a in assets:
        desc = (a.description or "").strip()
        kw = ", ".join(a.keywords or [])
        parts = [f"- **{a.name}** (ID: `{a.asset_id}`)"]
        if desc:
            parts.append(f"  Description: {desc}")
        if kw:
            parts.append(f"  Keywords: {kw}")
        if a.link_url:
            parts.append(f"  Link: {a.link_url}")
        lines.append("\n".join(parts))

    asset_list = "\n".join(lines)
    return (
        "\n\nAVAILABLE BUSINESS ASSETS (products/services with images):\n"
        f"{asset_list}\n\n"
        "ASSET IMAGE RULES:\n"
        "- When your answer mentions or discusses any of the above products/services, "
        "include the marker {{asset:ASSET_ID}} at the end of the relevant sentence or paragraph.\n"
        "- Example: 'We have a beautiful Deluxe Room with ocean views. {{asset:room_deluxe}}'\n"
        "- Only include markers for assets that are directly relevant to your answer.\n"
        "- You may include multiple asset markers if discussing multiple products.\n"
        "- Do NOT mention the marker syntax to the user; it will be automatically converted to an image card.\n"
    )

def build_asset_evidence(assets: List[BotAsset]) -> list[dict[str, str]]:
    if not assets:
        return []

    evidence: list[dict[str, str]] = []
    for a in assets:
        desc = (a.description or "").strip()
        kw = ", ".join(a.keywords or [])
        snippet_parts = [f"Product/Service: {a.name}."]
        if desc:
            snippet_parts.append(f"Description: {desc}.")
        if kw:
            snippet_parts.append(f"Related keywords: {kw}.")
        snippet_parts.append(
            f"To show this product's image in the response, include {{{{asset:{a.asset_id}}}}} in your answer."
        )
        evidence.append({
            "url": a.link_url or f"asset:{a.asset_id}",
            "snippet": " ".join(snippet_parts),
        })
    return evidence

def resolve_asset_markers(
    answer: str,
    assets: List[BotAsset],
) -> Tuple[str, List[Dict[str, str]]]:
    matches = list(re.finditer(r"\{\{asset:([a-zA-Z0-9_\-]+)\}\}", answer))
    if not matches:
        return answer, []

    cards = []
    asset_map = {a.asset_id: a for a in assets}
    cleaned_answer = answer
    for m in reversed(matches):
        start, end = m.span()
        asset_id = m.group(1)
        cleaned_answer = cleaned_answer[:start] + cleaned_answer[end:]
        
        if asset_id in asset_map:
            cards.append(_asset_to_card(asset_map[asset_id]))

    unique_cards = []
    seen = set()
    for c in reversed(cards):
        aid = c["asset_id"]
        if aid not in seen:
            seen.add(aid)
            unique_cards.append(c)

    return cleaned_answer.strip(), unique_cards

# --- NEW FUNCTIONS END ---

# --- Test Runner ---

def test_matching(query: str, answer: str, assets: List[BotAsset]):
    print(f"\n--- Testing Query: '{query}' ---")
    print(f"Answer: '{answer}'")
    
    # 1. Match from answer
    matched_cards = _match_assets_from_answer(answer, assets, max_cards=5)
    print(f"Matched Cards (by content): {len(matched_cards)}")
    
    # 2. Check query intent
    allowed_count = _max_cards_for_query(query, matched_cards)
    print(f"Allowed Cards (by intent): {allowed_count}")
    
    if allowed_count == 0 and len(matched_cards) > 0:
        print(">> BLOCKED by query intent check!")

# Define some test assets
assets = [
    BotAsset(
        asset_id="room_deluxe", 
        name="Deluxe Room", 
        description="A spacious room with a king-size bed and ocean view.", 
        keywords=["suite", "ocean view", "luxury", "king bed"], 
        image_public_url="http://example.com/deluxe.jpg", 
        link_url="http://example.com/book/deluxe"
    ),
    BotAsset(
        asset_id="spaghetti", 
        name="Spaghetti Carbonara", 
        description="Classic Italian pasta with eggs, cheese, beacon and black pepper.", 
        keywords=["pasta", "italian", "dinner", "food"], 
        image_public_url="http://example.com/pasta.jpg", 
        link_url="http://example.com/menu/pasta"
    )
]

print("--- Testing Asset Injection & Marker Resolution ---")

print("\n1. Asset Instruction Generation:")
print(build_asset_instruction(assets))

print("\n2. Asset Evidence Generation:")
for e in build_asset_evidence(assets):
    print(f"  - {e}")

print("\n3. Marker Resolution:")
raw_answer = "You see, we have a wonderful Deluxe Room {{asset:room_deluxe}} and also delicious pasta {{asset:spaghetti}}."
clean_answer, cards = resolve_asset_markers(raw_answer, assets)
print(f"  Raw: '{raw_answer}'")
print(f"  Clean: '{clean_answer}'")
print(f"  Cards: {len(cards)}")
for c in cards:
    print(f"    - {c['name']} ({c['asset_id']})")
