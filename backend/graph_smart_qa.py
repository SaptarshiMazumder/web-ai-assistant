import os
import asyncio
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from graph_qa import answer_node, State
from state import SmartHopState
from smart_parallel import (
    llm_select_relevant_links,
    scrape_and_qa_many,
    PageQAResult,
)
from utils import extract_json_from_text
import sys, json as _json
from logging_relay import log, smartqa_log_relay
from smart_parallel import llm_select_relevant_links_parallel

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

openai_api_key = os.environ.get("OPENAI_API_KEY")

# --- Log relay (keep for compatibility) ---
from asyncio import Queue




DEFAULT_MAX_HOPS = 5
DEFAULT_K_LINKS = 5
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TOTAL_PAGE_BUDGET = 25  # safety

async def _run_page_qa_once(
    text: str,
    question: str,
    page_url: str | None
) -> PageQAResult:
    s = State(text=text, question=question, page_url=page_url or "")
    s = answer_node(s)
    return PageQAResult(
        url=page_url or "",
        text=text,
        answer=s.answer,
        sources=[],  # No more chunk sources
        sufficient=s.sufficient
    )


# async def _is_sufficient(question: str, answer: str) -> bool:
#     prompt = (
#         f"Question: {question}\n"
#         f"Answer given: {answer}\n\n"
#         "Based on the answer, is the user's question fully answered with clear and specific information? "
#         "Reply with only 'YES' or 'NO'"
#     )
#     llm = ChatOpenAI(api_key=openai_api_key, 
#                     #  model="gpt-3.5-turbo", 
#                      model="gpt-4o", 
#                      temperature=0)
#     result = llm.invoke([{"role": "user", "content": prompt}])
#     out = (result.content or "").strip().lower()
#     print(f"⚡ [SUFFICIENCY] {out}")
#     return "yes" in out

def _same_domain(url: str, original_domain: str) -> bool:
    
    return urlparse(url).netloc == original_domain
    

async def smart_qa_runner(
    init_state: SmartHopState,
    *,
    max_hops: int = 3,
    k_links: int = 5,
    max_concurrency: int = 5,
    total_page_budget: int = 25,
) -> Dict[str, Any]:
    question = init_state.question
    original_domain = init_state.original_domain or urlparse(init_state.page_url).netloc
    visited: Set[str] = set(init_state.visited_urls or [])

    queue: List[Dict[str, Any]] = [{
        "page_url": init_state.page_url,
        "text": init_state.text,
        "links": init_state.links or [],
        "depth": 0
    }]

    pages_seen = 0
    best_partial: Optional[PageQAResult] = None

    while queue:
        node = queue.pop(0)
        page_url = node["page_url"]
        text = node["text"]
        links = node["links"]
        depth = node["depth"]

        if page_url:
            visited.add(page_url)

        if pages_seen >= total_page_budget:
            log("⚠️ Page budget exceeded. Stopping.")
            break
        pages_seen += 1

        log(f"\n📄 [Depth {depth}] QA on: {page_url}")
        log(f"🧪 This page had {len(links)} input links.")

        # Run QA on current page
        qa_result = await _run_page_qa_once(text=text, question=question, page_url=page_url)
        log(f"🧪 Sufficient? {qa_result.sufficient} | confidence={qa_result.confidence}%")

        if best_partial is None or (len(qa_result.answer) > len(best_partial.answer)):
            best_partial = qa_result

        if qa_result.sufficient:
            return {
                "answer": qa_result.answer,
                "sources": qa_result.sources,
                "visited_urls": list(visited),
                "sufficient": True,
                "multi_page": False
            }

        # Stop if depth reached
        if depth >= max_hops:
            log(f"⛔ Max hops ({max_hops}) reached at depth {depth}. No expansion.")
            continue

        # Filter and expand links
        candidate_links = [
            l for l in links
            if l.get("href", "").startswith("http")
            and l["href"] not in visited
            and _same_domain(l["href"], original_domain)
        ]
        log(f"🔍 Found {len(candidate_links)} valid domain links:")
        for l in candidate_links:
            print(f"   - {l.get('text', '')[:40]} → {l.get('href')}")

        if not candidate_links:
            log("🔚 No candidate links to expand from this page.")
            continue

        # LLM link selection
        selected_links = await llm_select_relevant_links_parallel(
            question=question,
            links=candidate_links,
            k=k_links,
            log_fn=log,
        )

        if selected_links:
            async def send_llm_links_message():
                msg = await llm_links_message(question, selected_links)
                log(_json.dumps({
                    "type": "llm_links_message",
                    "message": msg,
                    "links": selected_links
                }))
            asyncio.create_task(send_llm_links_message())

        if not selected_links:
            log("🤷 LLM returned no useful follow-up links.")
            continue

        log(f"🔗 LLM selected {len(selected_links)} links to fetch in parallel (k={k_links}).")

        # Scrape and QA all in parallel
        child_results: List[PageQAResult] = await scrape_and_qa_many(
            selected_links,
            question=question,
            max_concurrency=max_concurrency,
            original_domain=original_domain,
            log_fn=log
        )

        # ✅ If any sufficient at this depth → combine & stop
        sufficient_at_this_depth = [r for r in child_results if r.sufficient]
        for r in child_results:
            if r.url:
                visited.add(r.url)

        if sufficient_at_this_depth:
            log(f"✅ Sufficient answers found at depth {depth}, combining and stopping.")
            return await synthesize_final_answer(sufficient_at_this_depth, list(visited))

        # ❌ Otherwise push next hop into queue
        for cr in child_results:
            queue.append({
                "page_url": cr.url,
                "text": cr.text,
                "links": cr.links,
                "depth": depth + 1
            })

    # fallback
    if best_partial:
        return {
            "answer": best_partial.answer,
            "sources": best_partial.sources,
            "visited_urls": list(visited),
            "sufficient": False,
            "multi_page": False
        }

    return {
        "answer": "Sorry, I couldn't find a sufficient answer on the pages I visited.",
        "sources": [],
        "visited_urls": list(visited),
        "sufficient": False,
        "multi_page": False
    }


async def synthesize_final_answer(results: List[PageQAResult], visited_urls: List[str]) -> Dict[str, Any]:
    if len(results) == 1:
        return {
            "answer": results[0].answer,
            "sources": results[0].sources,
            "visited_urls": visited_urls,
            "sufficient": True,
            "multi_page": False
        }

    combined_text = "\n\n---\n\n".join(
        f"[{res.url}]\n{res.answer}" for res in results if res.answer.strip()
    )

    prompt = (
        "Using the following answers from different pages, synthesize a complete, helpful, and well-formatted response to the user's question. "
        "Each answer is from a separate page and may contain partial or overlapping information. Combine all relevant parts. "
        "Do NOT invent anything that isn't in the text.\n\n"
        f"{combined_text}"
    )

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])

    return {
        "answer": result.content.strip(),
        "sources": [{"url": r.url} for r in results],
        "visited_urls": visited_urls,
        "sufficient": True,
        "multi_page": True
    }


async def llm_links_message(question: str, links: list[dict]) -> str:
    if not links:
        return ""
    from langchain_openai import ChatOpenAI
    from urllib.parse import urlparse
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    # Extract website/domain from the first link
    first_url = links[0].get("href", "") if links else ""
    website = urlparse(first_url).netloc if first_url else "website"
    # Compose a nice prompt
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


# --- Compatibility wrapper for api.py ---
class _SmartQAGraphCompat:
    async def ainvoke(self, state: SmartHopState):
        return await smart_qa_runner(state)

smart_qa_graph = _SmartQAGraphCompat()
