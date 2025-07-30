import os
import asyncio
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from graph_qa import enhance_query_node, retrieve_node, answer_node
from state import SmartHopState
from smart_parallel import (
    llm_select_relevant_links,
    scrape_and_qa_many,
    PageQAResult,
)
from utils import extract_json_from_text
import sys, json as _json

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

openai_api_key = os.environ.get("OPENAI_API_KEY")

# --- Log relay (keep for compatibility) ---
from asyncio import Queue
class SmartQALogRelay:
    def __init__(self):
        self.queues: List[Queue] = []

    def register(self):
        q = Queue()
        self.queues.append(q)
        return q

    def unregister(self, q):
        if q in self.queues:
            self.queues.remove(q)

    def log(self, msg: str):
        print(msg)
        for q in self.queues:
            q.put_nowait(msg)

    def clear(self):
        self.queues.clear()
smartqa_log_relay = SmartQALogRelay()

def log(msg: str):
    smartqa_log_relay.log(msg)

DEFAULT_MAX_HOPS = 5
DEFAULT_K_LINKS = 3
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_TOTAL_PAGE_BUDGET = 25  # safety

async def _run_page_qa_once(
    text: str,
    question: str,
    page_url: str | None
) -> PageQAResult:
    print(f"\n⚡ [QA DEBUG] Running QA on: {page_url}")
    class S:
        pass
    s = S()
    s.text = text
    s.question = question
    s.enhanced_query = ""
    s.docs = []
    s.retrieved_docs = []
    s.answer = ""
    s.used_chunks = []
    s.page_url = page_url
    s.sufficient = False

    s = enhance_query_node(s)
    s = retrieve_node(s)
    s = answer_node(s)
    print(f"⚡ [QA DEBUG] Finished QA on: {page_url}\n")
    return PageQAResult(
        url=page_url or "",
        text=text,
        answer=s.answer,
        sources=s.used_chunks,
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
    max_hops: int = DEFAULT_MAX_HOPS,
    k_links: int = DEFAULT_K_LINKS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    total_page_budget: int = DEFAULT_TOTAL_PAGE_BUDGET,
) -> Dict[str, Any]:
    question = init_state.question
    original_domain = init_state.original_domain
    if not original_domain:
        original_domain = urlparse(init_state.page_url).netloc if init_state.page_url else ""
    visited: Set[str] = set(init_state.visited_urls or [])
    hops_used = init_state.hops or 0

    # BFS queue: list of dict nodes that hold page data
    queue: List[Dict[str, Any]] = [{
        "page_url": init_state.page_url,
        "text": init_state.text,
        "links": init_state.links or [],
        "depth": 0
    }]

    best_partial: Optional[PageQAResult] = None
    pages_seen = 0

    while queue:
        node = queue.pop(0)
        page_url = node["page_url"]
        text = node["text"]
        links = node["links"]
        depth = node["depth"]

        # if page_url and page_url in visited:
        #     print(f"⚡ [SKIP] Already visited: {page_url}")
        #     continue
        if page_url:
            visited.add(page_url)

        if pages_seen >= total_page_budget:
            log("⚠️ Page budget exceeded. Stopping.")
            break
        pages_seen += 1

        log(f"\n📄 [Depth {depth}] QA on: {page_url or '(initial page)'}")
        print(f"🧪 This page had {len(links)} input links.")

        # 1) Run QA on this page
        qa_result = await _run_page_qa_once(text=text, question=question, page_url=page_url)
        sufficient = qa_result.sufficient


        # Keep best partial (in case we never get a sufficient one)
        if best_partial is None or (len(qa_result.answer) > len(best_partial.answer)):
            best_partial = qa_result

        log(f"🧪 Sufficient? {sufficient}")
        if sufficient:
            # Return immediately
            return {
                "answer": qa_result.answer,
                "sources": qa_result.sources,
                "visited_urls": list(visited),
                "sufficient": True
            }

        # 2) No? Then expand IF we still have budget/hops
        if depth >= max_hops:
            log(f"⛔ Max hops ({max_hops}) reached at depth {depth}. No expansion.")
            continue

        # Filter links (same domain, unvisited)
        candidate_links = [
            l for l in links
            if l.get("href", "").startswith("http")
            and l["href"] not in visited
            and _same_domain(l["href"], original_domain)
        ]
        print(f"🔍 Found {len(candidate_links)} candidate links:")
        for l in candidate_links:
            print(f"   - {l.get('text', '')[:40]} → {l.get('href')}")

        if not candidate_links:
            log("🔚 No candidate links to expand from this page.")
            continue

        # 3) Let the LLM select the top-k links (no scraping yet!)
        selected_links = llm_select_relevant_links(
            question=question,
            links=candidate_links,
            k=k_links
        )
        if selected_links:
    # Spawn a background task to generate the LLM message and stream when ready
            async def send_llm_links_message():
                msg = await llm_links_message(question, selected_links)
                log(_json.dumps({
                    "type": "llm_links_message",
                    "message": msg,
                    "links": selected_links
                }))
            asyncio.create_task(send_llm_links_message())


        print(f"🧠 LLM selected links: {selected_links}")

        if not selected_links:
            log("🤷 LLM returned no useful follow-up links.")
            continue

        log(f"🔗 LLM selected {len(selected_links)} links to fetch in parallel (k={k_links}).")

        # 4) Scrape + QA those links in parallel
        child_results: List[PageQAResult] = await scrape_and_qa_many(
            selected_links,
            question=question,
            max_concurrency=max_concurrency,
            original_domain=original_domain,
            log_fn=log
        )

        # 5) Check if any are sufficient → early stop
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

        # 6) Otherwise, push children (insufficient ones) back into the queue (BFS)
        for cr in child_results:
            if cr.url:
                visited.add(cr.url)
            queue.append({
                "page_url": cr.url,
                "text": cr.text,
                "links": cr.links,
                "depth": depth + 1
            })

    # If we exit the loop without a sufficient answer:
    log("❗ No sufficient answer found. Returning best partial.")
    if best_partial:
        return {
            "answer": best_partial.answer,
            "sources": best_partial.sources,
            "visited_urls": list(visited),
            "sufficient": False
        }
    else:
        return {
            "answer": "Sorry, I couldn't find a sufficient answer on the pages I visited.",
            "sources": [],
            "visited_urls": list(visited),
            "sufficient": False
        }

async def llm_links_message(question: str, links: list[dict]) -> str:
    if not links:
        return ""
    from langchain_openai import ChatOpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    # Compose a nice prompt
    prompt = (
        f"You are an assistant helping a user find answers on a website.\n"
        f"User's question: {question}\n"
        "You found the following links which might help answer the user's question:\n"
        + "\n".join(f"- {l['text'] or l['href']}" for l in links[:5]) + "\n"
        "Write a friendly, brief message to the user, explaining that you found these links and are now exploring them to find the answer. "
        "Mention that you'll continue to look for the best information. "
        "Keep it short, natural, and relevant to the question."
    )
    result = llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()


# --- Compatibility wrapper for api.py ---
class _SmartQAGraphCompat:
    async def ainvoke(self, state: SmartHopState):
        return await smart_qa_runner(state)

smart_qa_graph = _SmartQAGraphCompat()
