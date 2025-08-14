# smart_rag_console.py
# Usage: python smart_rag_console.py
#
# Flow:
#   - Decide if the question is complex (gate)
#   - If simple: one-shot (internal retrieval; fast & usually solid)
#   - If complex: PLAN (subqueries) → RETRIEVE (per subquery, with logs) → ANALYZE (final synthesis)
#   - Verify the final answer against evidence; if weak, re-synthesize grounded; if still weak, fall back to one-shot

from google import genai
from google.genai import types

from vertexai import rag as vx_rag
import vertexai

import json
import re
import hashlib
from typing import List, Dict, Any

# =========================
# Config
# =========================
PROJECT_ID     = "gen-lang-client-0545494042"
GENAI_LOCATION = "global"      # google.genai client
RAG_LOCATION   = "us-central1" # Vertex RAG region

RAG_CORPUS = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"

MODEL_NAME        = "gemini-2.5-pro"
THINK_BUDGET      = 1024
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
    for i, e in enumerate(evidence[:limit], 1):
        snip = (e.get("snippet") or "").replace("\n", " ").strip()
        if len(snip) > 600:
            snip = snip[:600] + " ..."
        url = e.get("url") or ""
        lines.append(f"{i}. {snip} [{url}]")
    return "\n".join(lines)

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
        if text:
            out.append({"snippet": text, "url": url})
    return out

# =========================
# One-shot mode (fast path)
# =========================
def one_shot_answer(client: genai.Client, question: str):
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS)],
                    similarity_top_k=ONESHOT_TOP_K,
                )
            )
        )
    ]
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=32768,
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_budget=256),
        system_instruction=(
            "Answer using the RAG tool. Retrieve before answering. "
            "Be concise and include 2–6 bullet citations with URLs at the end."
        ),
    )
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
        "Avoid claims not present. Prefer quoting short phrases and naming/including the source URL. "
        "End with 2–6 bullet citations (URLs)."
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
        "After the answer, include 2–6 bullet citations with URLs of the strongest sources."
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
# Main
# =========================
def main():
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)

    print("Smart RAG Console — Complexity Gate → One-shot OR Plan→Retrieve→Analyze")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Query > ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("bye.")
            break

        # Decide path
        complex_q = is_complex_question(client, q)
        if not complex_q:
            # fast path that worked well before
            one_shot_answer(client, q)
            print("-" * 80)
            continue

        # -------- PLAN --------
        subqueries = plan_subqueries(client, q)
        if not subqueries:
            print("[plan] No subqueries; using the original question.")
            subqueries = [q]
        else:
            print(f"[plan] {len(subqueries)} subqueries:")
            for i, s in enumerate(subqueries, 1):
                print(f"  {i}. {s}")

        # -------- RETRIEVE --------
        evidence: List[Dict[str, str]] = []
        for step in range(MAX_STEPS):
            for idx, sq in enumerate(subqueries, 1):
                print(f"\n[retrieve] Subquery {idx}/{len(subqueries)}: {sq}")
                chunks = retrieve_for_subquery(RAG_CORPUS, sq, top_k=RETRIEVAL_TOP_K)
                print(f"[retrieve] Retrieved {len(chunks)} chunk(s).")
                for j, c in enumerate(chunks, 1):
                    preview = (c.get("snippet") or "").replace("\n", " ").strip()
                    if len(preview) > 280:
                        preview = preview[:280] + " ..."
                    print(f"  --- Chunk {j} ---")
                    print(f"  URL: {c.get('url','')}")
                    print(f"  {preview}\n")
                evidence.extend(chunks)
            break  # (set MAX_STEPS > 1 and add re-plan logic if desired)

        evidence = dedupe_evidence(evidence)
        if not evidence:
            print("[analyze] No evidence retrieved. Falling back to one-shot.\n")
            one_shot_answer(client, q)
            print("-" * 80)
            continue

        # -------- ANALYZE --------
        print("[analyze] Synthesizing final answer...\n")
        final_answer = analyze_with_evidence(client, q, evidence)

        # -------- VERIFY --------
        ver = verify_answer_supported(client, q, evidence, final_answer)
        if (not ver.get("supported")) or (ver.get("confidence", 0) < 0.6):
            print("\n[verify] Detected unsupported claims. Re-synthesizing grounded answer...\n")
            fixed = resynthesize_grounded(client, q, evidence)
            if fixed:
                print(fixed)
            else:
                print("[fallback] Using one-shot mode instead:\n")
                one_shot_answer(client, q)

        print("-" * 80)

if __name__ == "__main__":
    main()
