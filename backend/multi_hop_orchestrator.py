"""
Multi-hop Web QA Orchestrator

This module orchestrates a multi-hop, multi-page web question-answering process.
It starts from an initial page, tries to answer a question, and if needed, follows links to other pages,
using an LLM to select the most promising links, until a sufficient answer is found or a budget is exhausted.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from state import AssistantState, WebsiteMultiHopState
from web_qa_service import webpage_answer_node
from web_pages_worker import scrape_and_qa_many, PageQAResult
from utils import generate_rag_answer_from_vertex_ai, get_or_create_rag_corpus
from logging_relay import log
from config import config
import sys, json as _json

# --- Configuration ---
DEFAULT_MAX_HOPS = 3
DEFAULT_K_LINKS = 5
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TOTAL_PAGE_BUDGET = 25  # safety
USE_VERTEX_RAG = False  # Toggle to switch between old and new

# --- Main Orchestration ---
async def multi_hop_qa_orchestrator(
    init_state: WebsiteMultiHopState,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    k_links: int = DEFAULT_K_LINKS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    total_page_budget: int = DEFAULT_TOTAL_PAGE_BUDGET,
) -> Dict[str, Any]:
    """
    Orchestrates multi-hop QA over web pages, starting from the initial state.
    Hops through links using LLM selection until a sufficient answer is found or budget is exhausted.

    High-Level Flow:
    1. Start with the initial page.
    2. Try to answer the question.
    3. If not sufficient, use AI to pick the best links to follow.
    4. Visit those links, try to answer again.
    5. Repeat until a good answer is found or limits are reached.
    6. Return the best answer found.
    """
    # 1. Setup: extract question, domain, and initialize visited set
    question = init_state.question
    original_domain = init_state.original_domain or urlparse(init_state.page_url).netloc if init_state.page_url else ""
    visited: Set[str] = set(init_state.visited_urls or [])

    # 2. Initialize BFS queue with the starting page
    queue: List[Dict[str, Any]] = [{
        "page_url": init_state.page_url,
        "text": init_state.text,
        "links": init_state.links or [],
        "depth": 0
    }]

    best_partial: Optional[PageQAResult] = None
    pages_seen = 0

    # Emit start stage
    try:
        log(_json.dumps({
            "type": "stage",
            "stage": "orchestrator_start",
            "question": question,
            "init_url": init_state.page_url,
            "max_hops": max_hops,
            "k_links": k_links,
            "max_concurrency": max_concurrency,
            "total_page_budget": total_page_budget,
        }))
    except Exception:
        pass

    # 3. Main loop: process pages in the queue (BFS)
    while queue:
        node = queue.pop(0)
        page_url = node["page_url"]
        text = node["text"]
        links = node["links"]
        depth = node["depth"]

        # Mark this page as visited
        if page_url:
            visited.add(page_url)
        # Stop if we've hit the page budget
        if pages_seen >= total_page_budget:
            log("⚠️ Page budget exceeded. Stopping.")
            break
        pages_seen += 1

        log(f"\n📄 [Depth {depth}] QA on: {page_url or '(initial page)'}")
        log(f"🧪 This page had {len(links)} input links.")
        try:
            log(_json.dumps({
                "type": "stage",
                "stage": "page_qa_start",
                "depth": depth,
                "page_url": page_url,
                "input_links": len(links),
            }))
        except Exception:
            pass

        # 4. Try to answer the question using this page
        qa_result = await run_single_page_qa(text=text, question=question, page_url=page_url)
        sufficient = qa_result.sufficient

        # Keep track of the best partial answer (in case we never find a sufficient one)
        if best_partial is None or (qa_result.confidence or 0) > (best_partial.confidence or 0):
            best_partial = qa_result

        log(f"🧪 Sufficient? {sufficient}")
        try:
            log(_json.dumps({
                "type": "stage",
                "stage": "page_qa_done",
                "depth": depth,
                "page_url": page_url,
                "sufficient": bool(sufficient),
                "confidence": getattr(qa_result, "confidence", None),
            }))
        except Exception:
            pass
        if sufficient:
            # If the answer is good enough, return it immediately
            return {
                "answer": qa_result.answer,
                "sources": qa_result.sources,
                "visited_urls": list(visited),
                "sufficient": True
            }

        # 5. If not sufficient, check if we can hop further
        if depth >= max_hops:
            log(f"⛔ Max hops ({max_hops}) reached at depth {depth}. No expansion.")
            continue

        # 6. Filter links: only unvisited, same-domain, http(s) links
        candidate_links = [
            l for l in links
            if l.get("href", "").startswith("http")
            and l["href"] not in visited
            and is_same_domain(l["href"], original_domain)
        ]
        log(f"🔍 Found {len(candidate_links)} valid domain links:")
        try:
            log(_json.dumps({
                "type": "link_candidates",
                "count": len(candidate_links),
                "depth": depth,
                "domain": original_domain,
                "items": [
                    {"text": (l.get('text') or '')[:120], "href": l.get('href')} for l in candidate_links[:50]
                ],
            }))
        except Exception:
            pass

        if not candidate_links:
            log("🔚 No candidate links to expand from this page.")
            continue

        # 7. Use LLM to select the most promising links to follow next
        from web_pages_worker import llm_select_relevant_links_parallel
        try:
            log(_json.dumps({
                "type": "stage",
                "stage": "link_selection_start",
                "depth": depth,
                "k": k_links,
                "candidate_count": len(candidate_links),
            }))
        except Exception:
            pass
        selected_links = await llm_select_relevant_links_parallel(
            question=question,
            links=candidate_links,
            k=k_links,
            log_fn=log,
        )

        # Optionally, log or notify about the selected links
        if selected_links:
            # Stream a friendly LLM message about selected links as typewriter text
            asyncio.create_task(stream_llm_links_message(question, selected_links))
        try:
            log(_json.dumps({
                "type": "stage",
                "stage": "link_selection_done",
                "depth": depth,
                "selected_count": len(selected_links or []),
                "items": selected_links[:50] if selected_links else [],
            }))
        except Exception:
            pass

        if not selected_links:
            log("🤷 LLM returned no useful follow-up links.")
            continue

        log(f"🔗 LLM selected {len(selected_links)} links to fetch in parallel (k={k_links}).")
        try:
            log(_json.dumps({
                "type": "stage",
                "stage": "children_retrieval_start",
                "depth": depth,
                "count": len(selected_links),
                "max_concurrency": max_concurrency,
            }))
        except Exception:
            pass

        # 8. Scrape and run QA on the selected links in parallel
        child_results: List[PageQAResult] = await scrape_and_qa_many(
            selected_links,
            question=question,
            max_concurrency=max_concurrency,
            original_domain=original_domain,
            log_fn=log
        )

        try:
            any_sufficient = any(getattr(cr, "sufficient", False) for cr in child_results)
            log(_json.dumps({
                "type": "stage",
                "stage": "children_retrieval_done",
                "depth": depth,
                "results": len(child_results),
                "any_sufficient": bool(any_sufficient),
            }))
        except Exception:
            pass

        # 9. If any child page has a sufficient answer, return it immediately
        for cr in child_results:
            if cr.url:
                visited.add(cr.url)
            if cr.sufficient:
                log(f"✅ Found sufficient answer at: {cr.url}")
                return {
                    "answer": cr.answer,
                    "sources": cr.sources,
                    "visited_urls": list(visited),
                    "sufficient": True
                }

        # 10. Otherwise, add all child pages to the queue for further exploration
        for cr in child_results:
            if cr.url:
                visited.add(cr.url)
            queue.append({
                "page_url": cr.url,
                "text": cr.text,
                "links": cr.links,
                "depth": depth + 1
            })

    # 11. If we exit the loop without a sufficient answer, return the best partial answer found
    log("❗ No sufficient answer found. Returning best partial.")
    if best_partial:
        return {
            "answer": best_partial.answer,
            "sources": best_partial.sources,
            "visited_urls": list(visited),
            "sufficient": False,
            "confidence": best_partial.confidence,
            "multi_page": False
        }
    else:
        return {
            "answer": "Sorry, I couldn't find a sufficient answer on the pages I visited.",
            "sources": [],
            "visited_urls": list(visited),
            "sufficient": False
        }

# --- Helpers ---
async def run_single_page_qa(text: str, question: str, page_url: Optional[str]) -> PageQAResult:
    """
    Runs QA on a single page using either Vertex AI RAG or the default webpage_answer_node.
    """
    s = AssistantState(text=text, question=question, page_url=page_url or "")
    if USE_VERTEX_RAG:
        print("🧠 Using Vertex AI RAG for answer generation")
        s.answer = generate_rag_answer_from_vertex_ai(s.question)
        s.sufficient = True  # Or add logic to parse confidence later
    else:
        s = await webpage_answer_node(s)
    return PageQAResult(
        url=page_url or "",
        text=text,
        answer=s.answer,
        sources=[],  # No more chunk sources
        sufficient=s.sufficient
    )

def is_same_domain(url: str, original_domain: str) -> bool:
    """
    Returns True if the given URL is in the same domain as the original.
    """
    return urlparse(url).netloc == original_domain

async def format_llm_links_message(question: str, links: list[dict]) -> str:
    """
    Formats a user-facing message about which links are being explored.
    """
    if not links:
        return ""
    from langchain_openai import ChatOpenAI
    from urllib.parse import urlparse
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
    first_url = links[0].get("href", "") if links else ""
    website = urlparse(first_url).netloc if first_url else "website"
    prompt = (
        f"You are a master of the {website}.\n"
        f"User's question: {question}\n"
        "You found the following links which might help answer the user's question:\n"
        + "\n".join(f"- {l['text'] or l['href']}" for l in links[:5]) + "\n"
        "Write a friendly, brief message to the user, explaining that I found these pages where you can find <whatever query they have>"
        "I'm currently browsing through them to see if I can find the info.... "
        "No greetings or introductions needed. Just Make the tone sound like you are currently browsing"
    )
    result = llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()

async def stream_llm_links_message(question: str, links: list[dict]):
    """
    Streams a friendly message about which links are being explored, sending
    incremental deltas over the websocket so the UI can render typewriter-style.
    Falls back to single-shot if streaming fails.
    """
    try:
        from langchain_openai import ChatOpenAI
        from urllib.parse import urlparse
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
        first_url = links[0].get("href", "") if links else ""
        website = urlparse(first_url).netloc if first_url else "website"
        prompt = (
            f"You are a master of the {website}.\n"
            f"User's question: {question}\n"
            "You found the following links which might help answer the user's question:\n"
            + "\n".join(f"- {l['text'] or l['href']}" for l in links[:5]) + "\n"
            "Write a friendly, brief message to the user, explaining that I found these pages where you can find <whatever query they have>"
            "I'm currently browsing through them to see if I can find the info.... "
            "No greetings or introductions needed. Just Make the tone sound like you are currently browsing"
        )

        # Notify UI to reset the LLM links message area for streaming
        log(_json.dumps({
            "type": "llm_links_reset",
            "links": links,
        }))

        accumulated = ""
        try:
            for chunk in llm.stream([{ "role": "user", "content": prompt }]):
                delta_text = getattr(chunk, "content", None) or ""
                if not delta_text:
                    await asyncio.sleep(0)
                    continue
                accumulated += delta_text
                log(_json.dumps({
                    "type": "llm_links_delta",
                    "text": delta_text,
                }))
                await asyncio.sleep(0)
        except Exception:
            # Fallback to non-streaming if stream fails
            result = llm.invoke([{ "role": "user", "content": prompt }])
            accumulated = (result.content or "").strip()
            log(_json.dumps({
                "type": "llm_links_message",
                "message": accumulated,
                "links": links,
            }))
            return

        # Finished streaming
        log(_json.dumps({
            "type": "llm_links_done",
            "links": links,
        }))
    except Exception:
        # Ultimate fallback: try best-effort single shot via existing helper
        try:
            msg = await format_llm_links_message(question, links)
            log(_json.dumps({
                "type": "llm_links_message",
                "message": msg,
                "links": links
            }))
        except Exception:
            pass

# --- RAG Corpus (if needed) ---
rag_corpus = get_or_create_rag_corpus()
