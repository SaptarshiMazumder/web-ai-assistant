import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import sys
import asyncio
from markdownify import markdownify
from urllib.parse import urlparse
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from playwright.sync_api import sync_playwright

from langchain_openai import ChatOpenAI

from graph_qa import answer_node, State

from utils import extract_json_from_text

from vertexai import  rag
from google.cloud import aiplatform
from vertexai.preview import rag

from vertexai.generative_models import GenerativeModel, Tool
from utils import generate_rag_answer_from_vertex_ai

aiplatform.init(project="tour-proj-451201", location="us-central1")

openai_api_key = os.environ.get("OPENAI_API_KEY")

class PageQAResult(BaseModel):
    url: str
    text: str
    answer: str
    sources: List[Dict[str, Any]]
    sufficient: Optional[bool] = None
    links: List[Dict[str, str]] = []
    confidence: Optional[int] = None

import datetime

def save_llm_prompt(prompt: str, filename: str = "llm-prompt.txt"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*40 + "\n")
        f.write(f"[{now}] LLM PROMPT\n")
        f.write("="*40 + "\n")
        f.write(prompt)
        f.write("\n" + "-"*40 + "\n\n")


def llm_select_relevant_links(
    question: str,
    links: List[Dict[str, str]],
    k: int = 3
) -> List[Dict[str, str]]:
    # subset = links[:30]
    prompt = (
        f"The user is on the website {urlparse(links[0].get('href', '')).netloc} and their question is: {question}\n"
        "These are all the links on the page:\n" +
        "\n".join([f"- {l.get('text','').strip()[:80]} ({l.get('href')})" for l in links]) +
        "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        "Use your general knowledge and the context of the question, the website, or similarity, intuition to select the most relevant links.\n"
        "Only select links that you are 90% confident to contain an answer, or links to lead to the answer."
        "If you are not at least 90% confident that a link contains the answer, do not include it."
        "Do not include links that are not relevant to the question, or that you are not confident about.\n"
        "Reply with a JSON array of up to 0-5 objects (max 5, min 0) with 'text' and 'href'."

    )
    # === LOG PROMPT ===
    save_llm_prompt(prompt)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    print(f"LLM SELECT LINKS RESULT: {result}")
    output = (result.content or "").strip()

    json_str = extract_json_from_text(output)
    
    data = json.loads(json_str) if json_str else []
    if isinstance(data, list):
        return data[:k]
    return []

async def llm_select_relevant_links_parallel(
    question: str,
    links: List[Dict[str, str]],
    k: int = 3,
    chunk_size: int = 200,
    max_concurrency: int = 5,
    log_fn: Optional[Callable[[str], None]] = None
) -> List[Dict[str, str]]:
    from langchain_openai import ChatOpenAI
    from utils import extract_json_from_text
    import asyncio

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_on_chunk(chunk: List[Dict[str, str]], chunk_id: int):
        log_fn(f"🧵 Thread {chunk_id} starting with {len(chunk)} available links.")
        # prompt = (
        #     f"The user is on the website {urlparse(chunk[0].get('href', '')).netloc} and their question is: {question}\n"
        #     "These are the links on part of the page:\n" +
        #     "\n".join([f"- {l.get('text','').strip()[:80]} ({l.get('href')})" for l in chunk]) +
        #     "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        #     "Reply with a JSON array of up to 5 objects with 'text' and 'href'."
        # )

        prompt = (
            f"The user is on the website {urlparse(links[0].get('href', '')).netloc} and their question is: {question}\n"
            "These are all the links on the page:\n" +
            "\n".join([f"- {l.get('text','').strip()[:80]} ({l.get('href')})" for l in links]) +
            "\n\nWhich of these links are most likely to contain the answer or helpful information? "
            "Use your general knowledge and the context of the question, the website, or similarity, intuition to select the most relevant links.\n"
            "Only select links that you are 90% confident to contain an answer, or links to lead to the answer."
            "If you are not at least "
            " confident that a link contains the answer, do not include it."
            "Do not include links that are not relevant to the question, or that you are not confident about.\n"
            "Reply with a JSON array of up to 0-5 objects (max 5, min 0) with 'text' and 'href'."

        )

        async with semaphore:
            result = await llm.ainvoke([{"role": "user", "content": prompt}])
            output = (result.content or "").strip()
            json_str = extract_json_from_text(output)
            try:
                links_selected = json.loads(json_str) if json_str else []
                log_fn(f"✅ Thread {chunk_id} found {len(links_selected)} possible links")
                return links_selected
            except Exception as e:
                log_fn(f"❌ Thread {chunk_id} failed to parse JSON: {e}")
                return []

    # Split links into chunks
    chunks = [links[i:i + chunk_size] for i in range(0, len(links), chunk_size)]
    log_fn(f"🌐 Total chunks (threads): {len(chunks)}")
    coros = [run_on_chunk(chunk, idx + 1) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*coros)

    # Flatten and deduplicate
    seen = set()
    combined = []
    for sublist in results:
        for link in sublist:
            href = link.get("href")
            if href and href not in seen:
                combined.append(link)
                seen.add(href)

    return combined[:k]


def scrape_with_sync_playwright(url, timeout=6000):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.route("**/*", lambda route, request: 
                route.abort() if request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_()
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
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

def extract_tables_as_markdown(soup):
    tables = []
    for table in soup.find_all("table"):
        md = []
        # Headers
        headers = []
        for th in table.find_all("th"):
            headers.append(th.get_text(strip=True))
        # Rows
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        # Markdown table formatting
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]
        if headers:
            md.append("| " + " | ".join(headers) + " |")
            md.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            md.append("| " + " | ".join(row) + " |")
        tables.append("\n".join(md))
    return tables

def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, etc.
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    # Remove navbars, footers, headers, menus, popups, sidebars, etc.
    noisy_selectors = [
        "nav", "footer", "aside", "form", "header",
        "[role=navigation]", "[role=contentinfo]", "[role=banner]", "[role=alert]", "[role=dialog]",
        "[class*=nav]", "[class*=footer]", "[class*=header]", "[class*=menu]", "[class*=sidebar]", "[class*=popup]", "[class*=cookie]",
        "[id*=nav]", "[id*=footer]", "[id*=header]", "[id*=menu]", "[id*=sidebar]", "[id*=popup]", "[id*=cookie]",
    ]
    for selector in noisy_selectors:
        for tag in soup.select(selector):
            tag.decompose()

    # Try to extract just the main content
    main = soup.find("main")
    if not main:
        main = soup.find(attrs={"role": "main"})
    if not main:
        main = soup.find("article")
    if not main:
        divs = soup.find_all("div")
        if divs:
            main = max(divs, key=lambda d: len(d.get_text()))
    if main:
        return str(main)
    return str(soup.body) if soup.body else str(soup)

def _extract_text_and_links(html: str, base_url: str) -> tuple[str, List[Dict[str, str]]]:
    clean_html = extract_main_content(html)
    markdown = markdownify(clean_html, heading_style="ATX")
    
    soup = BeautifulSoup(html, "html.parser")
    # Links extraction (unchanged)
    page_links = [
        {
            "text": a.get_text(strip=True),
            "href": urljoin(base_url, a.get("href"))
        }
        for a in soup.find_all("a", href=True)
        if a.get("href", "").startswith(("http", "/"))
    ]
    page_links = [l for l in page_links if l["href"].startswith("http")]
    return markdown, page_links


# Helper functions


import hashlib
RAW_DIR = "raw_pages"
os.makedirs(RAW_DIR, exist_ok=True)
def safe_filename_from_url(url):
    domain = urlparse(url).netloc.replace(":", "_")
    # Hash the full URL for uniqueness
    url_hash = hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]
    return f"{domain}_{url_hash}.txt"

def save_raw_text(url, text):
    fname = safe_filename_from_url(url)
    fpath = os.path.join(RAW_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)


async def scrape_one(url: str) -> Dict[str, Any]:
    html, final_url = await _scrape_with_playwright(url)
    text, links = _extract_text_and_links(html, final_url)

    save_raw_text(final_url, text)
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
    s = State(text=text, question=question, page_url=page_url)
    USE_VERTEX_RAG = False  # Toggle RAG mode

    if USE_VERTEX_RAG:
        print("🧠 Using Vertex AI RAG for answer generation")
        s.answer = generate_rag_answer_from_vertex_ai(s.question)
        s.sufficient = True
        s.confidence = 100  # Static for now, update later if needed
    else:
        s = answer_node(s)

    return PageQAResult(
        url=page_url,
        text=text,
        answer=s.answer,
        sources=[],  # No more chunk sources
        sufficient=s.sufficient,
        confidence=s.confidence
    )

# async def _is_sufficient(question: str, answer: str) -> bool:
#     prompt = (
#         f"Question: {question}\n"
#         f"Answer given: {answer}\n\n"
#         "Based on the answer, is the user's question fully answered with clear and specific information? "
#         "Reply with only 'YES' if it is enough, or 'NO' if it is not clear/specific enough."
#     )
#     llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
#     result = llm.invoke([{"role": "user", "content": prompt}])
#     return "yes" in (result.content or "").strip().lower()

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
        try:
            async with sem:
                scraped = await scrape_one(href)
                qa_res = await run_page_qa(
                    text=scraped["text"],
                    question=question,
                    page_url=scraped["url"]
                )
                qa_res.links = scraped["links"]
                if log_fn:
                    log_fn(f"   ↳ Done: {href} | sufficient={qa_res.sufficient} | confidence={qa_res.confidence}%")
                return qa_res
        except asyncio.CancelledError:
            if log_fn:
                log_fn(f"   ⏹️ Cancelled: {href}")
            raise
        except Exception as e:
            if log_fn:
                log_fn(f"   ↳ Error on {href}: {e}")
            return PageQAResult(
                url=href or "",
                text="",
                answer="",
                sources=[],
                sufficient=False,
                links=[],
                confidence=0  # Optional default
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

# vertex ai rag setup

# --- Create corpus once ---
embedding_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="projects/google/models/text-embedding-005"
    )
)
from utils import get_or_create_rag_corpus
rag_corpus = get_or_create_rag_corpus()

def ingest_content_to_vertex_ai(text: str, url: str):
    """Store content in GCS and import into Vertex AI corpus"""
    from google.cloud import storage
    import uuid

    # Upload to GCS
    bucket_name = "web-ai-dynamic-corpus-bucket"
    file_name = f"scraped_pages/{uuid.uuid4()}.txt"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(text)

    # Import into RAG
    gcs_path = f"gs://{bucket_name}/{file_name}"
    rag.import_files(
        rag_corpus.name,
        paths=[gcs_path],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
        )
    )
    return gcs_path