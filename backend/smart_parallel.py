import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from playwright.sync_api import sync_playwright

from langchain_openai import ChatOpenAI

from graph_qa import enhance_query_node, retrieve_node, answer_node
from utils import extract_json_from_text

openai_api_key = os.environ.get("OPENAI_API_KEY")

class PageQAResult(BaseModel):
    url: str
    text: str
    answer: str
    sources: List[Dict[str, Any]]
    sufficient: Optional[bool] = None
    links: List[Dict[str, str]] = []

def llm_select_relevant_links(
    question: str,
    links: List[Dict[str, str]],
    k: int = 3
) -> List[Dict[str, str]]:
    subset = links[:30]
    prompt = (
        f"Question: {question}\n"
        "Here are available links from the page:\n" +
        "\n".join([f"- {l.get('text','').strip()[:80]} ({l.get('href')})" for l in subset]) +
        "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        "Reply with a JSON array of up to 3 objects with 'text' and 'href'."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    output = (result.content or "").strip()

    json_str = extract_json_from_text(output)
    
    data = json.loads(json_str) if json_str else []
    if isinstance(data, list):
        return data[:k]
    return []

def scrape_with_sync_playwright(url, timeout=12000):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout)
            html = page.content()
            final_url = page.url
            browser.close()
            return html, final_url
    except Exception as e:
        print(f"[SCRAPE ERROR] Failed to fetch {url} -- {e}")
        return "", url  # return blank html but the url, so rest of pipeline doesn't break


import asyncio

async def _scrape_with_playwright(url: str, timeout: int = 12000) -> tuple[str, str]:
    print(f"⚡ [SCRAPE] Using Playwright (sync/thread) for: {url}")
    loop = asyncio.get_event_loop()
    html, final_url = await loop.run_in_executor(
        None, scrape_with_sync_playwright, url, timeout
    )
    return html, final_url


def _extract_text_and_links(html: str, base_url: str) -> tuple[str, List[Dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    page_links = [
        {
            "text": a.get_text(strip=True),
            "href": urljoin(base_url, a.get("href"))
        }
        for a in soup.find_all("a", href=True)
        if a.get("href", "").startswith(("http", "/"))
    ]
    page_links = [l for l in page_links if l["href"].startswith("http")]
    return text, page_links

async def scrape_one(url: str) -> Dict[str, Any]:
    html, final_url = await _scrape_with_playwright(url)
    text, links = _extract_text_and_links(html, final_url)
    return {
        "url": final_url,
        "text": text,
        "links": links
    }

async def run_page_qa(
    text: str,
    question: str,
    page_url: str
) -> PageQAResult:
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

    s = enhance_query_node(s)
    s = retrieve_node(s)
    s = answer_node(s)
    return PageQAResult(
        url=page_url,
        text=text,
        answer=s.answer,
        sources=s.used_chunks
    )

async def _is_sufficient(question: str, answer: str) -> bool:
    prompt = (
        f"Question: {question}\n"
        f"Answer given: {answer}\n\n"
        "Based on the answer, is the user's question fully answered with clear and specific information? "
        "Reply with only 'YES' if it is enough, or 'NO' if it is not clear/specific enough."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    return "yes" in (result.content or "").strip().lower()

async def scrape_and_qa_many(
    selected_links: list,
    *,
    question: str,
    max_concurrency: int,
    original_domain: str,
    log_fn=None
):
    sem = asyncio.Semaphore(max_concurrency)
    results = []
    first_sufficient = None

    async def worker(link):
        href = link.get("href")
        text_ = link.get("text", "")
        if log_fn:
            log_fn(f"➡️  Fetching: {href} ({text_[:60]}...)")
        async with sem:
            try:
                scraped = await scrape_one(href)
                qa_res = await run_page_qa(
                    text=scraped["text"],
                    question=question,
                    page_url=scraped["url"]
                )
                qa_res.links = scraped["links"]
                qa_res.sufficient = await _is_sufficient(question, qa_res.answer)
                if log_fn:
                    log_fn(f"   ↳ Done: {href} | sufficient={qa_res.sufficient}")
                return qa_res
            except Exception as e:
                if log_fn:
                    log_fn(f"   ↳ Error on {href}: {e}")
                return PageQAResult(
                    url=href or "",
                    text="",
                    answer="",
                    sources=[],
                    sufficient=False,
                    links=[]
                )

    # Prepare all coroutines
    coros = [worker(l) for l in selected_links]
    # Launch all coroutines as tasks
    pending = {asyncio.create_task(c) for c in coros}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            try:
                result = t.result()
            except Exception as e:
                result = None
            if result:
                results.append(result)
                if result.sufficient:
                    # Cancel all other pending tasks
                    for pt in pending:
                        pt.cancel()
                    # Optionally: await cancellation, or just return
                    return [result]  # Early return, as soon as sufficient found
    # If none sufficient, return all results
    return results