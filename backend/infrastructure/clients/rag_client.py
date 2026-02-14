import os, json, re, asyncio
from google import genai
from google.genai import types

from vertexai import rag as vx_rag
import vertexai

import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Callable
import io
import contextlib
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from common.config import config
from infrastructure.rag.url_map import resolve_evidence_urls

# =========================
# Config
# =========================
PROJECT_ID = (config.PROJECT_ID or os.environ.get("PROJECT_ID") or "").strip()
GENAI_LOCATION = (os.environ.get("GENAI_LOCATION") or "global").strip()
RAG_LOCATION = (config.LOCATION or os.environ.get("RAG_LOCATION") or "us-central1").strip()
GCS_BUCKET = (config.GCS_BUCKET or "").strip()

DEFAULT_RAG_CORPUS = (os.environ.get("DEFAULT_RAG_CORPUS") or "").strip()

MODEL_NAME        = os.environ.get("VERTEX_RAG_MODEL", "gemini-2.0-flash-001")
# Keep thinking small to reduce quota pressure and latency for web QA.
THINK_BUDGET      = int(os.environ.get("VERTEX_RAG_THINK_BUDGET", "256"))
# Many Gemini endpoints have ~8k max output token limits; keep a safe default.
MAX_OUTPUT_TOKENS = int(os.environ.get("VERTEX_RAG_MAX_OUTPUT_TOKENS", "2048"))
# Some models/endpoints reject thinking_config. Default off; opt-in via env var.
ENABLE_THINKING   = os.environ.get("VERTEX_RAG_ENABLE_THINKING", "").strip().lower() in ("1", "true", "yes", "y")
# Performance optimization flags (default OFF for speed)
ENABLE_LLM_SUBQUERIES = os.environ.get("ENABLE_LLM_SUBQUERIES", "").strip().lower() in ("1", "true", "yes", "y")
ENABLE_LLM_RERANK = os.environ.get("ENABLE_LLM_RERANK", "").strip().lower() in ("1", "true", "yes", "y")

RETRIEVAL_TOP_K   = 16         # per subquery; increase to 24–32 for broader recall
MAX_SUBQUERIES    = 5
MAX_STEPS         = 1          # keep 1 for simplicity; raise if you want re-plan loops
ONESHOT_TOP_K     = 40         # broader recall for one-shot

# =========================
# Utilities
# =========================
def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def _ctx_text(ctx) -> str:
    t = getattr(ctx, "text", None)
    if t:
        return t
    chunk = getattr(ctx, "chunk", None)
    if chunk and getattr(chunk, "text", None):
        return chunk.text
    if isinstance(ctx, dict):
        return ctx.get("text") or (ctx.get("chunk") or {}).get("text", "") or ""
    return str(ctx)

def _ctx_uri(ctx) -> str:
    for key in ("uri", "source_uri", "gcs_uri", "sourceUrl", "source_url"):
        v = getattr(ctx, key, None)
        if v:
            return v
        if isinstance(ctx, dict) and ctx.get(key):
            return ctx[key]
    return ""

def format_evidence_block(evidence: List[Dict[str, str]], limit: int = 60) -> str:
    lines = []
    for e in evidence[:limit]:
        snip = (e.get("snippet") or "").replace("\n", " ").strip()
        # Strip markdown image syntax — images eat character budget and are invisible to the LLM.
        snip = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", snip)  # ![alt](url)
        snip = re.sub(r"!\[[^\]]*\]", "", snip)           # ![alt] with no href
        # Keep link labels, drop the URL: [Omakase Course](https://...) → Omakase Course
        snip = re.sub(r"\[([^\]]*)\]\(https?://[^)]*\)", r"\1", snip)
        snip = re.sub(r"  +", " ", snip).strip()
        if len(snip) > 1500:
            snip = snip[:1500] + " ..."
        url = e.get("url") or ""
        title = (e.get("title") or "").strip()
        header = f"--- snippet from {url}"
        if title:
            header += f' ("{title}")'
        header += " ---"
        lines.append(f"{header}\n{snip}")
    return "\n\n".join(lines)

def sanitize_answer_citations(text: str) -> str:
    """Post-process LLM answer to strip forbidden citation patterns.

    Removes:
    - Numbered bracket references: [1], [2, 3], [1, 11, 35, 75]
    - Trailing bullet/numbered URL lists
    - Standalone raw URL lines
    - Leftover double spaces / excess blank lines
    """
    if not text:
        return text
    # Rewrite low-quality markdown link text like [source](URL) or [link](URL)
    # to a more descriptive label derived from the URL.
    def _preferred_link_text_from_url(url: str) -> str:
        try:
            p = urlparse((url or "").strip())
            if (p.scheme or "").lower() not in ("http", "https"):
                return "this page"
            path = (p.path or "").strip()
            if not path or path == "/":
                return "the website"
            seg = path.rstrip("/").split("/")[-1].strip().lower()
            seg = re.sub(r"[\-_]+", " ", seg)
            seg = re.sub(r"\s+", " ", seg).strip()
            if not seg or seg in ("index", "home", "top"):
                return "this page"
            # Detect garbage alphanumeric segments (e.g. "13267332" or "a3f9c2b1")
            # If mostly digits or looks like a hash/ID, fall back to a cleaner label
            digit_ratio = sum(1 for c in seg if c.isdigit()) / max(1, len(seg))
            if digit_ratio > 0.5 and len(seg) > 4:
                return "this page"
            if re.fullmatch(r"[a-f0-9]{6,}", seg):  # looks like a hex hash
                return "this page"
            common = {
                "about": "about page",
                "contact": "contact page",
                "faq": "faq page",
                "faqs": "faq page",
                "hours": "hours page",
                "pricing": "pricing page",
                "prices": "pricing page",
                "price": "pricing page",
                "menu": "menu page",
                "services": "services page",
                "service": "services page",
                "booking": "booking page",
                "reserve": "booking page",
                "reservation": "booking page",
                "reservations": "booking page",
                "location": "location page",
                "access": "location page",
            }
            if seg in common:
                return common[seg]
            if 1 <= len(seg) <= 28:
                return f"{seg} page"
        except Exception:
            pass
        return "this page"

    def _is_generic_link_text(label: str) -> bool:
        if not label:
            return True
        s = re.sub(r"\s+", " ", label).strip().lower()
        if not s:
            return True
        generic = {
            "source", "sources", "link", "links", "here", "this", "page", "this page", "the page",
            "website", "the website", "site", "this site", "more", "details", "info", "information",
            "click here", "learn more", "read more",
        }
        if s in generic:
            return True
        if "http://" in s or "https://" in s:
            return True
        if s.startswith("/") and " " not in s:
            return True
        if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}(/.*)?", s) and " " not in s:
            return True
        return False

    def _rewrite_generic_link_text(m: re.Match) -> str:
        label = (m.group(1) or "").strip()
        url = (m.group(2) or "").strip()
        if _is_generic_link_text(label):
            return f"[{_preferred_link_text_from_url(url)}]({url})"
        return m.group(0)

    # Rewrite generic markdown link labels to descriptive ones.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", _rewrite_generic_link_text, text, flags=re.IGNORECASE)
    # Strip ugly PDF/page bracket citations like:
    # [Some_File.pdf page 5], [foo.pdf p.12], [Document page 1]
    # (but do NOT clobber markdown links like [text](url)).
    text = re.sub(r"\[[^\]]*\.pdf[^\]]*\](?!\()", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\bpage\s*\d+[^\]]*\](?!\()", "", text, flags=re.IGNORECASE)
    # Strip numbered bracket references like [1], [1, 2], [1, 11, 35, 75]
    # Negative lookbehind avoids clobbering markdown links like [text](url)
    text = re.sub(r"(?<!\])\s*\[[\d,\s]+\]", "", text)
    # Strip trailing lines that are bullet/numbered URL lists
    # e.g. "- https://...", "* https://...", "1. https://..."
    text = re.sub(r"(?m)^[\s]*[-*•]\s*https?://\S+.*$", "", text)
    text = re.sub(r"(?m)^[\s]*\d+\.\s*https?://\S+.*$", "", text)
    # Strip standalone raw URL lines (a line that is just a URL)
    text = re.sub(r"(?m)^[\s]*https?://\S+[\s]*$", "", text)
    # Clean up excess blank lines (3+ newlines -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Clean up double spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def dedupe_evidence(evidence: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for e in evidence:
        key = (e.get("url", ""), _hash(e.get("snippet", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


_NOISE_URL_SUBSTRINGS = (
    "url_map.json",
)

_NOISE_TEXT_PATTERNS = (
    "bookmarkstylewebapi",
    "bookmarkcouponwebapi",
    "bookmarksalonwebapi",
    "mystylewebapi",
    "mycouponwebapi",
    "mysalonwebapi",
    "doset/",
    "dodelete/",
    "/csp/my/",
    "ブックマーク_",
)


def _is_noise_evidence(url: str, text: str) -> bool:
    """Drop low-signal snippets that pollute retrieval quality."""
    u = (url or "").strip().lower()
    t = (text or "").strip()
    tl = t.lower()

    if not t:
        return True
    if any(token in u for token in _NOISE_URL_SUBSTRINGS):
        return True
    if u.startswith("gs://") and (u.endswith(".json") or "url_map.json" in u):
        return True
    if any(token in tl for token in _NOISE_TEXT_PATTERNS):
        return True

    # Heuristic: snippets that are mostly link endpoints are rarely answer-bearing.
    # Only filter if the chunk is short AND full of links — long chunks with many links
    # are often legitimate content (menus with reservation links, galleries, etc.).
    link_like = tl.count("http://") + tl.count("https://")
    if link_like >= 3 and len(t) < 700:
        return True

    # Very short boilerplate fragments usually hurt reranking.
    if len(t) < 40:
        return True

    return False


def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    if not h:
        return ""
    # Allow passing full URLs too.
    if h.startswith(("http://", "https://")):
        try:
            h = (urlparse(h).hostname or "").lower()
        except Exception:
            pass
    return h.split(":")[0]


def _evidence_matches_host(e: Dict[str, str], host: str) -> bool:
    """
    Enforce per-site demarcation: only accept snippets whose source URL/GCS URI
    clearly belongs to the current host.
    """
    h = _normalize_host(host)
    if not h:
        return True
    u = (e.get("url") or "").lower()
    if not u:
        return False
    # Covers:
    # - direct website URLs: https://example.com/...
    # - GCS URIs from our upload prefixes: gs://.../host=example.com/...
    return (h in u) or (f"host={h}" in u)


# =========================
# Robust JSON extraction
# =========================
def _extract_json_object(text: str):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

def _sanitize_subqueries(raw) -> List[str]:
    if isinstance(raw, dict):
        seq = raw.get("subqueries", [])
    elif isinstance(raw, list):
        seq = raw
    else:
        seq = []
    cleaned = []
    for s in seq:
        if not isinstance(s, str):
            continue
        s2 = s.strip()
        s2 = re.sub(r"^```(?:json)?\s*", "", s2, flags=re.IGNORECASE)
        s2 = re.sub(r"\s*```$", "", s2)
        s2 = s2.strip().strip("-•").strip()
        if s2 in ("{", "}", "[", "]", '"subqueries":', "subqueries:", "```json", "```"):
            continue
        if s2:
            cleaned.append(s2)
    seen = set()
    uniq = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

# =========================
# Complexity Gate
# =========================
def is_complex_question(client: genai.Client, question: str) -> bool:
    judge_prompt = (
        "Classify whether the user's question needs multi-step retrieval/analysis.\n"
        'Return ONLY JSON: {"complex": true|false}\n\n'
        "complex = true for multi-part comparisons, multi-hop references, abstract/vague asks,\n"
        "or when aggregation across multiple pages is likely. Otherwise false.\n\n"
        f"Question: {question}"
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=judge_prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=256),
        ),
    )
    txt = (resp.text or "").strip()
    try:
        data = json.loads(txt)
        return bool(data.get("complex", False))
    except Exception:
        return False  # safe default: avoid over-planning

# =========================
# Query Expansion (local, no LLM)
# =========================
def expand_query_local(question: str) -> List[str]:
    """
    Fast local query expansion without LLM calls.
    Returns 1-2 query variants based on simple keyword extraction.
    """
    q = (question or "").strip()
    if not q:
        return [q]
    
    # Always include the original question
    variants = [q]
    
    # Extract key nouns/phrases (simple word splitting)
    # Remove common stopwords and very short words
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "what", "which", "who", "when", "where",
        "why", "how", "about", "tell", "me", "please", "want", "know"
    }
    
    # Simple tokenization
    words = re.findall(r'\b\w+\b', q.lower())
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]
    
    # If we extracted meaningful keywords, create a keyword-only variant
    if keywords and len(keywords) < len(words):
        keyword_query = " ".join(keywords[:5])  # max 5 keywords
        if keyword_query and keyword_query != q.lower():
            variants.append(keyword_query)
    
    return variants[:MAX_SUBQUERIES]


# =========================
# Planning
# =========================
def plan_subqueries(client: genai.Client, question: str) -> List[str]:
    prompt = (
        "Propose targeted sub-queries that, if answered from a RAG corpus, would answer the user's question.\n"
        "Return ONLY JSON with NO markdown fences or commentary:\n"
        f'{{ "subqueries": ["...", "..."] }}\nLimit to {MAX_SUBQUERIES} subqueries. Avoid duplicates.\n\n'
        f"User question: {question}"
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET),
        ),
    )
    txt = (resp.text or "").strip()
    parsed = _extract_json_object(txt)
    subs = _sanitize_subqueries(parsed)

    if not subs:
        lines = [l.strip() for l in txt.splitlines()]
        lines = [l.strip("-• ").strip() for l in lines if l and not re.fullmatch(r"[{}\[\]`]+", l)]
        subs = _sanitize_subqueries(lines)

    return subs[:MAX_SUBQUERIES]

# =========================
# Retrieval
# =========================
def retrieve_for_subquery(corpus_name: str, subquery: str, top_k: int) -> List[Dict[str, str]]:
    cfg = vx_rag.RagRetrievalConfig(top_k=top_k)
    resp = vx_rag.retrieval_query(
        rag_resources=[vx_rag.RagResource(rag_corpus=corpus_name)],
        text=subquery,
        rag_retrieval_config=cfg,
    )
    ctxs = getattr(resp, "contexts", None)
    try:
        ctx_list = list(ctxs) if ctxs is not None else []
    except TypeError:
        inner = getattr(ctxs, "contexts", None)
        ctx_list = list(inner) if inner is not None else []

    out = []
    for c in ctx_list:
        text = _ctx_text(c).strip()
        url  = _ctx_uri(c)
        if text and not _is_noise_evidence(url, text):
            out.append({"snippet": text, "url": url})
    # De-dupe and cap per URL — raised to 6 so content-rich pages (menus, coupon lists)
    # don't get cut off after the 3rd chunk.
    out = dedupe_evidence(out)
    per_url: Dict[str, int] = {}
    capped: List[Dict[str, str]] = []
    for e in out:
        u = e.get("url", "") or ""
        count = per_url.get(u, 0)
        if count >= 6:
            continue
        per_url[u] = count + 1
        capped.append(e)
    return capped

# =========================
# LLM Reranking
# =========================
def rerank_evidence(
    client: genai.Client,
    question: str,
    evidence: List[Dict[str, str]],
    top_n: int = 15,
) -> List[Dict[str, str]]:
    """
    Use a fast Gemini Flash call to score each evidence chunk by relevance to the
    question, then return the top_n most relevant chunks in ranked order.

    This replaces brittle heuristic noise filtering with actual semantic relevance
    judgment. Works for any language and content type.
    """
    if len(evidence) <= top_n:
        return evidence

    # Build compact snippet representations for scoring (200 chars each is enough
    # for the reranker to judge relevance without inflating the prompt).
    items = []
    for i, e in enumerate(evidence):
        snip = (e.get("snippet") or "")[:200].replace("\n", " ").strip()
        items.append(f"{i}: {snip}")

    prompt = (
        f"Question: {question}\n\n"
        "Rate each numbered snippet 0-10 for how useful it is for answering the question.\n"
        "0=irrelevant noise, 10=directly answers the question.\n"
        "Return ONLY valid JSON, no commentary:\n"
        '{"scores": [{"i": 0, "s": 8}, {"i": 1, "s": 2}, ...]}\n\n'
        + "\n".join(items)
    )

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                # No thinking budget — pure scoring, speed matters here.
            ),
        )
        parsed = _extract_json_object((resp.text or "").strip())
        if isinstance(parsed, dict) and "scores" in parsed:
            score_map = {
                int(item["i"]): int(item["s"])
                for item in parsed["scores"]
                if isinstance(item, dict) and "i" in item and "s" in item
            }
            ranked = sorted(range(len(evidence)), key=lambda i: -score_map.get(i, 5))
            return [evidence[i] for i in ranked[:top_n]]
    except Exception:
        pass  # fallback: return original order truncated

    return evidence[:top_n]


def heuristic_rerank(
    question: str,
    evidence: List[Dict[str, str]],
    top_n: int = 15,
) -> List[Dict[str, str]]:
    """
    Fast heuristic reranking using keyword overlap scoring.
    No LLM call - pure local computation.
    """
    if len(evidence) <= top_n:
        return evidence
    
    # Extract keywords from question (lowercase, remove stopwords)
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "what", "which", "who", "when", "where",
        "why", "how", "about", "tell", "me", "please", "want", "know"
    }
    
    q_words = re.findall(r'\b\w+\b', question.lower())
    q_keywords = set(w for w in q_words if len(w) > 2 and w not in stopwords)
    
    if not q_keywords:
        # No keywords to match, return original order truncated
        return evidence[:top_n]
    
    # Score each evidence chunk
    scored = []
    for e in evidence:
        snippet = (e.get("snippet") or "").lower()
        url = (e.get("url") or "").lower()
        
        # Count keyword matches in snippet
        keyword_matches = sum(1 for kw in q_keywords if kw in snippet)
        
        # Normalize by number of keywords
        keyword_score = keyword_matches / len(q_keywords) if q_keywords else 0
        
        # Slight bonus for longer snippets (more context)
        length_score = min(len(snippet) / 1000.0, 1.0) * 0.2
        
        # Slight bonus if URL path contains query keywords
        url_score = sum(0.1 for kw in q_keywords if kw in url)
        
        total_score = keyword_score + length_score + url_score
        scored.append((total_score, e))
    
    # Sort by score descending, return top N
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_n]]


# =========================
# One-shot mode (fast path)
# =========================
def one_shot_answer(client: genai.Client, question: str, *, rag_corpus: str):
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[types.VertexRagStoreRagResource(rag_corpus=rag_corpus)],
                    similarity_top_k=ONESHOT_TOP_K,
                )
            )
        )
    ]
    cfg_kwargs = dict(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=tools,
        system_instruction=(
            "Answer using the RAG tool. Retrieve before answering. Be concise.\n"
            "CITATION RULES: Cite sources ONLY as inline markdown hyperlinks woven into sentences, e.g. [the website](URL), [this page](URL), or [pricing page](URL).\n"
            "NEVER use 'source' or 'sources' as the markdown link text.\n"
            "NEVER use numbered references like [1], [1, 2], [1, 11, 35]. NEVER list URLs as bullets or append them at the end.\n"
            "NEVER show raw URLs or domains in the text."
        ),
    )
    if ENABLE_THINKING:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=THINK_BUDGET)

    cfg = types.GenerateContentConfig(**cfg_kwargs)
    for ch in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=question)])],
        config=cfg,
    ):
        if ch.candidates and ch.candidates[0].content and ch.candidates[0].content.parts and ch.text:
            print(ch.text, end="")
    print()

# =========================
# Verification + Re-synthesis
# =========================
def verify_answer_supported(client: genai.Client, question: str, evidence: List[Dict[str, str]], answer: str) -> dict:
    ev_block = format_evidence_block(evidence, limit=80)
    prompt = (
        "Verify that EVERY factual sentence in the assistant's answer is supported by the evidence.\n"
        'Return ONLY JSON: {"supported": true|false, "unsupported_sentences": ["..."], "confidence": 0..1}\n\n'
        f"QUESTION:\n{question}\n\nEVIDENCE:\n{ev_block}\n\nANSWER:\n{answer}"
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_budget=256),
        ),
    )
    try:
        return json.loads((resp.text or "").strip())
    except Exception:
        return {"supported": True, "unsupported_sentences": [], "confidence": 0.5}

def resynthesize_grounded(client: genai.Client, question: str, evidence: List[Dict[str, str]]) -> str:
    SYSTEM = (
        "Write a concise, well-structured answer using ONLY the provided evidence snippets. "
        "Avoid claims not present.\n"
        "CITATION RULES: Cite sources ONLY as inline markdown hyperlinks woven into sentences, e.g. [the website](URL), [this page](URL), or [pricing page](URL).\n"
        "NEVER use 'source' or 'sources' as the markdown link text.\n"
        "Link text must be a short descriptive phrase like 'here', 'this page', 'the website', 'about page'. "
        "NEVER put a URL, hostname, or domain as the link text.\n"
        "NEVER use numbered references like [1], [1, 2], [1, 11, 35]. "
        "NEVER list URLs as bullets or append them at the end. NEVER show raw URLs in the text."
    )
    user_block = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{format_evidence_block(evidence)}\n\n"
        "TASK: Produce a grounded answer. If evidence is insufficient, say exactly what is missing."
    )
    cfg = types.GenerateContentConfig(
        temperature=0.2, top_p=0.9, max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET),
        system_instruction=SYSTEM,
    )
    out = []
    for ch in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_block)])],
        config=cfg,
    ):
        if ch.candidates and ch.candidates[0].content and ch.candidates[0].content.parts and ch.text:
            out.append(ch.text)
    return "".join(out).strip()

# =========================
# Analyze (final synthesis)
# =========================
def analyze_with_evidence(client: genai.Client, question: str, evidence: List[Dict[str, str]]) -> str:
    SYSTEM = (
        "You are an advanced research & analysis assistant.\n"
        "Use the provided evidence snippets as primary sources. "
        "Compare, aggregate, deduplicate; compute counts/sums/ratios when useful; check consistency. "
        "Write a concise, well-structured final answer.\n"
        "CITATION RULES: Cite sources ONLY as inline markdown hyperlinks woven into sentences, e.g. [the website](URL), [this page](URL), or [pricing page](URL).\n"
        "NEVER use 'source' or 'sources' as the markdown link text.\n"
        "Link text must be a short descriptive phrase like 'here', 'this page', 'the website', 'about page'. "
        "NEVER put a URL, hostname, or domain as the link text.\n"
        "NEVER use numbered references like [1], [1, 2], [1, 11, 35]. "
        "NEVER list URLs as bullets or append them at the end. NEVER show raw URLs in the text."
    )
    user_block = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE SNIPPETS (with URLs):\n{format_evidence_block(evidence)}\n\n"
        "TASK: Using ONLY the evidence above, produce the best possible answer. "
        "If the evidence is insufficient, clearly state what is missing."
    )
    cfg = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET),
        system_instruction=SYSTEM,
    )
    out = []
    for ch in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_block)])],
        config=cfg,
    ):
        if ch.candidates and ch.candidates[0].content and ch.candidates[0].content.parts and ch.text:
            print(ch.text, end="")  # stream to console
            out.append(ch.text)
    print()
    return "".join(out).strip()

# =========================
# Simple (non-streaming) synthesis
# =========================
DEFAULT_SYSTEM = (
    "Answer the user's question using ONLY the provided evidence snippets.\n"
    "Base your answer on whatever relevant content the snippets contain. "
    "Only say you cannot find the information if NONE of the snippets mention the topic at all.\n"
    "Keep it concise.\n\n"
    "CITATION LINK RULES (follow exactly):\n"
    "- Cite sources ONLY as inline markdown hyperlinks woven naturally into sentences: [link text](URL).\n"
    "- NEVER use 'source' or 'sources' as the markdown link text.\n"
    "- NEVER include bracket citations like [Some_File.pdf page 1] or [Document page 5].\n"
    "- NEVER mention PDF filenames or page numbers in the answer.\n"
    "- NEVER put URL, hostname, or domain inside the brackets. WRONG: [chocozap.jp](URL), [example.com/about](URL).\n"
    "- For root URL (ends with / or has no path): use exactly 'the website'. Example: [the website](https://example.com/).\n"
    "- For other pages: use one of 'here', 'this page', or a phrase from the path (e.g. /about -> 'about page', /products -> 'product page', /parking -> 'parking page').\n"
    "- Valid link text examples: 'here', 'this page', 'the website', 'about page', 'product page', 'parking page'.\n"
    "- ABSOLUTELY FORBIDDEN: numbered references like [1], [2], [1, 2], [1, 11, 35, 75]. Never refer to sources by number.\n"
    "- ABSOLUTELY FORBIDDEN: appending a list of URLs or bullet-point citations at the end of the answer.\n"
    "- ABSOLUTELY FORBIDDEN: showing raw URLs as plain text anywhere in the answer.\n"
    "- Every source reference MUST be an inline [descriptive text](URL) hyperlink within a sentence."
)

# Appended to custom system instructions so the model still stays grounded and cites sources.
GROUNDING_SUFFIX = (
    "\n\nYou must answer using ONLY the provided evidence snippets and cite sources via inline markdown hyperlinks woven into sentences. "
    "CITATION RULES: Link text must be 'the website' for root URLs; for other pages use 'here', 'this page', or path-based phrases like 'about page', 'product page'. "
    "NEVER use 'source' or 'sources' as the markdown link text. "
    "NEVER include bracket citations like [Some_File.pdf page 1]. NEVER mention PDF filenames or page numbers. "
    "NEVER use URL or domain as link text. "
    "ABSOLUTELY FORBIDDEN: numbered references like [1], [2], [1, 2], [1, 11, 35]. Never refer to sources by number. "
    "ABSOLUTELY FORBIDDEN: appending bullet-point URL lists at the end. "
    "ABSOLUTELY FORBIDDEN: showing raw URLs as plain text. "
    "Every source reference must be an inline [descriptive text](URL) link. "
    "Base your answer on whatever relevant content the snippets contain; only say you don't know if the snippets contain no relevant information at all. Keep responses concise."
)


def _build_grounded_prompt(
    question: str,
    evidence: List[Dict[str, str]],
    *,
    system_instruction: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> Dict[str, str]:
    custom = (system_instruction or "").strip()
    if custom:
        system = custom + GROUNDING_SUFFIX
        task_line = "TASK: Answer using the evidence above. Use the tone, style, and persona from your system instructions."
    else:
        system = DEFAULT_SYSTEM
        task_line = "TASK: Write the best possible grounded answer."
    question_section = (
        f"RECENT CONVERSATION:\n{conversation_context}\n\nQUESTION:\n{question}\n\n"
        if (conversation_context or "").strip()
        else f"QUESTION:\n{question}\n\n"
    )
    user_block = (
        question_section
        + f"EVIDENCE SNIPPETS (with URLs):\n{format_evidence_block(evidence, limit=80)}\n\n"
        + task_line
    )
    return {"system": system, "user_block": user_block}


def synthesize_with_evidence(
    client: genai.Client,
    question: str,
    evidence: List[Dict[str, str]],
    *,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    debug_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    conversation_context: Optional[str] = None,
) -> str:
    prompt = _build_grounded_prompt(
        question, evidence, system_instruction=system_instruction, conversation_context=conversation_context
    )
    system = prompt["system"]
    user_block = prompt["user_block"]
    if debug_cb:
        try:
            debug_cb({
                "type": "gemini_prompt",
                "system_instruction": system,
                "user_message": user_block,
            })
        except Exception:
            pass
    temp = temperature if temperature is not None else 0.2
    cfg = types.GenerateContentConfig(
        temperature=temp,
        top_p=0.9,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        system_instruction=system,
    )
    if ENABLE_THINKING:
        cfg.thinking_config = types.ThinkingConfig(thinking_budget=THINK_BUDGET)
    model = (model_name or "").strip() or MODEL_NAME
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_block)])],
        config=cfg,
    )
    return (resp.text or "").strip()


def synthesize_with_evidence_stream(
    client: genai.Client,
    question: str,
    evidence: List[Dict[str, str]],
    *,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    debug_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    conversation_context: Optional[str] = None,
):
    prompt = _build_grounded_prompt(
        question, evidence, system_instruction=system_instruction, conversation_context=conversation_context
    )
    system = prompt["system"]
    user_block = prompt["user_block"]
    if debug_cb:
        try:
            debug_cb({
                "type": "gemini_prompt",
                "system_instruction": system,
                "user_message": user_block,
            })
        except Exception:
            pass
    temp = temperature if temperature is not None else 0.2
    cfg = types.GenerateContentConfig(
        temperature=temp,
        top_p=0.9,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        system_instruction=system,
    )
    if ENABLE_THINKING:
        cfg.thinking_config = types.ThinkingConfig(thinking_budget=THINK_BUDGET)
    model = (model_name or "").strip() or MODEL_NAME
    for ch in client.models.generate_content_stream(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_block)])],
        config=cfg,
    ):
        if ch.candidates and ch.candidates[0].content and ch.candidates[0].content.parts and ch.text:
            yield ch.text

# =========================
# Main
# =========================
def run_vertex_rag(
    question: str,
    *,
    rag_corpus: str = DEFAULT_RAG_CORPUS,
    allowed_host: Optional[str] = None,
    debug_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    conversation_context: Optional[str] = None,
    extra_evidence: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Minimal callable wrapper that reuses the script logic and returns structured output.

    It captures any printed streaming output to assemble the final answer text without
    changing the underlying logic functions.
    """
    def _dbg(evt: Dict[str, Any]) -> None:
        if debug_cb is None:
            return
        try:
            debug_cb(evt)
        except Exception:
            pass

    _dbg(
        {
            "type": "rag_start",
            "rag_corpus": rag_corpus,
            "allowed_host": allowed_host or "",
            "question": question,
            "model": MODEL_NAME,
            "rag_location": RAG_LOCATION,
            "genai_location": GENAI_LOCATION,
        }
    )
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    if not rag_corpus:
        raise RuntimeError("DEFAULT_RAG_CORPUS is not configured")
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)

    sources: List[Dict[str, str]] = []

    # Retrieval pipeline:
    # 1) Expand query (optionally using LLM or local heuristics)
    # 2) Retrieve from corpus for each sub-query in parallel
    # 3) Optionally filter to the allowed host
    # 4) Rerank by relevance (optionally using LLM or heuristics)
    # 5) Synthesize answer ONLY from top-ranked snippets

    # Step 1: query expansion
    if ENABLE_LLM_SUBQUERIES:
        subqueries = plan_subqueries(client, question)
        if not subqueries:
            subqueries = [question]
        elif question not in subqueries:
            subqueries = [question] + subqueries[:2]
        else:
            subqueries = subqueries[:3]
    else:
        # Fast local expansion
        subqueries = expand_query_local(question)
    
    _dbg({"type": "query_expansion", "original": question, "subqueries": subqueries, "llm_enabled": ENABLE_LLM_SUBQUERIES})

    # Step 2: multi-query retrieval (PARALLEL)
    all_evidence: List[Dict[str, str]] = []
    
    def _retrieve_one(sq: str) -> List[Dict[str, str]]:
        _dbg({"type": "retrieval_start", "query": sq, "top_k": RETRIEVAL_TOP_K, "rag_corpus": rag_corpus})
        return retrieve_for_subquery(rag_corpus, sq, top_k=RETRIEVAL_TOP_K)
    
    with ThreadPoolExecutor(max_workers=min(len(subqueries), 5)) as executor:
        futures = [executor.submit(_retrieve_one, sq) for sq in subqueries]
        for future in futures:
            all_evidence.extend(future.result())
    
    evidence = dedupe_evidence(all_evidence)
    bucket_name = GCS_BUCKET.split("/")[0] if GCS_BUCKET else ""
    resolve_evidence_urls(evidence, bucket_name)
    _dbg({"type": "retrieval_done", "evidence_count": len(evidence)})
    for i, e in enumerate(evidence, 1):
        _dbg({"type": "retrieved_chunk", "idx": i, "url": e.get("url", ""), "snippet": e.get("snippet", "")})

    # Step 3: host filtering
    if allowed_host:
        evidence = [e for e in evidence if _evidence_matches_host(e, allowed_host)]
        _dbg({"type": "host_filter_done", "allowed_host": allowed_host, "evidence_count": len(evidence)})
        for i, e in enumerate(evidence, 1):
            _dbg({"type": "filtered_chunk", "idx": i, "url": e.get("url", ""), "snippet": e.get("snippet", "")})

    if extra_evidence:
        evidence = list(extra_evidence) + evidence
        _dbg({"type": "extra_evidence_prepended", "count": len(extra_evidence)})

    if not evidence:
        _dbg({"type": "rag_no_evidence", "answer": "", "sources": []})
        return {
            "answer": "",
            "sources": [],
            "sufficient": False,
            "selected_links": [],
            "visited_urls": [],
        }

    # Step 4: Reranking — score chunks by relevance, keep best 15
    if len(evidence) > 4:
        if ENABLE_LLM_RERANK:
            evidence = rerank_evidence(client, question, evidence, top_n=15)
        else:
            evidence = heuristic_rerank(question, evidence, top_n=15)
        _dbg({"type": "reranking_done", "evidence_count": len(evidence), "llm_enabled": ENABLE_LLM_RERANK})

    # Step 5: Synthesize grounded answer from top-ranked snippets.
    # Actual system + user prompts sent to Gemini are logged via gemini_prompt in synthesize_with_evidence.
    answer = synthesize_with_evidence(
        client,
        question,
        evidence,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        debug_cb=_dbg,
        conversation_context=conversation_context,
    )
    answer = sanitize_answer_citations(answer)
    _dbg({"type": "model_answer", "answer": answer})

    # Prepare sources from evidence
    for e in evidence:
        sources.append({"excerpt": e.get("snippet", ""), "url": e.get("url", "")})

    _dbg({"type": "rag_done", "sources": sources, "sources_count": len(sources)})
    return {
        "answer": answer,
        "sources": sources,
        "sufficient": True,
        "selected_links": [],
        "visited_urls": [],
    }


def run_vertex_rag_stream(
    question: str,
    *,
    rag_corpus: str = DEFAULT_RAG_CORPUS,
    allowed_host: Optional[str] = None,
    debug_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    conversation_context: Optional[str] = None,
    extra_evidence: Optional[List[Dict[str, str]]] = None,
):
    """Stream deltas as they are generated, then emit a final done event with sources."""
    def _dbg(evt: Dict[str, Any]) -> None:
        if debug_cb is None:
            return
        try:
            debug_cb(evt)
        except Exception:
            pass

    _dbg(
        {
            "type": "rag_start",
            "rag_corpus": rag_corpus,
            "allowed_host": allowed_host or "",
            "question": question,
            "model": MODEL_NAME,
            "rag_location": RAG_LOCATION,
            "genai_location": GENAI_LOCATION,
        }
    )
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    if not rag_corpus:
        raise RuntimeError("DEFAULT_RAG_CORPUS is not configured")
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)

    sources: List[Dict[str, str]] = []

    # Step 1: query expansion
    if ENABLE_LLM_SUBQUERIES:
        subqueries = plan_subqueries(client, question)
        if not subqueries:
            subqueries = [question]
        elif question not in subqueries:
            subqueries = [question] + subqueries[:2]
        else:
            subqueries = subqueries[:3]
    else:
        # Fast local expansion
        subqueries = expand_query_local(question)
    
    _dbg({"type": "query_expansion", "original": question, "subqueries": subqueries, "llm_enabled": ENABLE_LLM_SUBQUERIES})

    # Step 2: multi-query retrieval (PARALLEL)
    all_evidence: List[Dict[str, str]] = []
    
    def _retrieve_one(sq: str) -> List[Dict[str, str]]:
        _dbg({"type": "retrieval_start", "query": sq, "top_k": RETRIEVAL_TOP_K, "rag_corpus": rag_corpus})
        return retrieve_for_subquery(rag_corpus, sq, top_k=RETRIEVAL_TOP_K)
    
    with ThreadPoolExecutor(max_workers=min(len(subqueries), 5)) as executor:
        futures = [executor.submit(_retrieve_one, sq) for sq in subqueries]
        for future in futures:
            all_evidence.extend(future.result())
    
    evidence = dedupe_evidence(all_evidence)
    bucket_name = GCS_BUCKET.split("/")[0] if GCS_BUCKET else ""
    resolve_evidence_urls(evidence, bucket_name)
    _dbg({"type": "retrieval_done", "evidence_count": len(evidence)})
    for i, e in enumerate(evidence, 1):
        _dbg({"type": "retrieved_chunk", "idx": i, "url": e.get("url", ""), "snippet": e.get("snippet", "")})

    # Step 3: host filtering
    if allowed_host:
        evidence = [e for e in evidence if _evidence_matches_host(e, allowed_host)]
        _dbg({"type": "host_filter_done", "allowed_host": allowed_host, "evidence_count": len(evidence)})
        for i, e in enumerate(evidence, 1):
            _dbg({"type": "filtered_chunk", "idx": i, "url": e.get("url", ""), "snippet": e.get("snippet", "")})

    if extra_evidence:
        evidence = list(extra_evidence) + evidence
        _dbg({"type": "extra_evidence_prepended", "count": len(extra_evidence)})

    if not evidence:
        _dbg({"type": "rag_no_evidence", "answer": "", "sources": []})
        yield {
            "type": "done",
            "answer": "",
            "sources": [],
            "sufficient": False,
            "selected_links": [],
            "visited_urls": [],
        }
        return

    # Step 4: Reranking
    if len(evidence) > 4:
        if ENABLE_LLM_RERANK:
            evidence = rerank_evidence(client, question, evidence, top_n=15)
        else:
            evidence = heuristic_rerank(question, evidence, top_n=15)
        _dbg({"type": "reranking_done", "evidence_count": len(evidence), "llm_enabled": ENABLE_LLM_RERANK})

    answer_parts: List[str] = []
    for delta in synthesize_with_evidence_stream(
        client,
        question,
        evidence,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        debug_cb=_dbg,
        conversation_context=conversation_context,
    ):
        answer_parts.append(delta)
        yield {"type": "delta", "text": delta}

    answer = sanitize_answer_citations("".join(answer_parts).strip())
    _dbg({"type": "model_answer", "answer": answer})

    for e in evidence:
        sources.append({"excerpt": e.get("snippet", ""), "url": e.get("url", "")})

    _dbg({"type": "rag_done", "sources": sources, "sources_count": len(sources)})
    yield {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "sufficient": True,
        "selected_links": [],
        "visited_urls": [],
    }

def extract_topics_from_titles(titles: List[str]) -> Dict[str, int]:
    """
    Given a list of session titles (user queries), use LLM to cluster them into topics with counts.
    """
    if not titles:
        # Default empty dict
        return {}
    
    # We must ensure client init here
    if not PROJECT_ID:
        # If project ID is not set, we cannot use LLM.
        return {}

    try:
        # We can reuse the same global client if we want, but creating a new one with correct vertexai init is safer
        # to ensure context is clean if run outside the main app context (e.g. celery task).
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)

        # Cap to 500 items to be safe and efficient
        sample = titles[:500]
        
        # Build prompt
        prompt = (
            "Analyze the following user queries from a chatbot session history. "
            "Group them into 5-10 distinct, meaningful high-level topics (e.g., 'Pricing', 'Technical Support', 'Product Info'). "
            "Return a strictly valid JSON object mapping each topic name to the count of queries that belong to it.\n"
            "Rules:\n"
            "1. Topics must be short (2-5 words).\n"
            "2. Ignore simple greetings (hi, hello) unless they are the majority.\n"
            "3. Output ONLY valid JSON: {\"Topic A\": 5, \"Topic B\": 3}\n"
            "4. Do not include markdown code fences (```json or ```).\n\n"
            "User Queries:\n" + "\n".join(f"- {t}" for t in sample)
        )
        
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        
        txt = (resp.text or "").strip()
        # Clean up code fences just in case
        txt = re.sub(r"^```json\s*", "", txt, flags=re.MULTILINE)
        txt = re.sub(r"\s*```$", "", txt, flags=re.MULTILINE)
        
        data = json.loads(txt)
        cleaned = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    cleaned[str(k).strip()] = int(v)
        return cleaned
        
    except Exception as e:
        print(f"[TopicExtraction] LLM error: {e}")
        return {}
