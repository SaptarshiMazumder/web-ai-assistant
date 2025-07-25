import os
import json
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# New imports for crawling
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    text: str
    question: str

class SiteQARequest(BaseModel):
    question: str
    urls: List[str]

class State(BaseModel):
    text: str
    question: str
    enhanced_query: str = ""
    docs: List[Any] = []
    retrieved_docs: List[Any] = []
    answer: str = ""
    used_chunks: List[Dict[str, Any]] = []

def enhance_query_node(state: State) -> State:
    print("\n=== [EnhanceQuery Node] ===")
    page_text = state.text
    user_question = state.question
    prompt = (
        "You are a query rewriter for search. Only improve or clarify the question if needed, and ONLY use phrases or words that appear in the provided web page content. "
        "If the question is already good, return it unchanged. NEVER invent new topics, courses, or details not present in the page.\n\n"
        f"WEBPAGE CONTENT (first 3000 chars):\n{page_text[:3000]}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        "REWRITTEN QUERY:"
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    enhanced_query = result.content.strip()
    print(f"Enhanced query:\n{enhanced_query}")
    state.enhanced_query = enhanced_query
    return state

def retrieve_node(state: State) -> State:
    print("\n=== [Retrieve Node] ===")
    page_text = state.text
    print(f"Received page_text length: {len(page_text)}")
    enhanced_query = state.enhanced_query
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
    chunks = splitter.split_text(page_text)
    print(f"Number of chunks created: {len(chunks)}")

    docs = []
    pos = 0
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {repr(chunk[:80])}")
        metadata = {"chunk_id": i, "start_char": pos, "end_char": pos + len(chunk)}
        from langchain.schema import Document
        doc = Document(page_content=chunk, metadata=metadata)
        docs.append(doc)
        pos += len(chunk)

    if not docs:
        print("No docs to embed!")
        state.docs = []
        state.retrieved_docs = []
        return state

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = Chroma.from_documents(
        docs, embeddings, collection_name="webpage", persist_directory=None
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    relevant_docs = retriever.get_relevant_documents(enhanced_query or state.question)
    print(f"Number of retrieved docs: {len(relevant_docs)}")

    state.docs = docs
    state.retrieved_docs = relevant_docs
    return state

def answer_node(state: State) -> State:
    print("\n=== [Answer Node] ===")
    question = state.question
    relevant_docs = state.retrieved_docs
    context = "\n\n---\n\n".join([d.page_content for d in relevant_docs])
    prompt = (
    "You are an expert assistant. Using only the content below, answer the user's question as fully and helpfully as possible. "
    "Use your understanding and reasoning, quote, paraphrase, or summarize as appropriate. "
    "If you use information from the content, include a relevant short excerpt from the source as a citation at the end of your answer in this format: (Source: <short excerpt>...) "
    "If you cannot find a direct answer in the content, briefly summarize anything related or useful you did find, "
    "and politely inform the user that the answer does not seem to be present on this page. "
    "Say that they may navigate to a more relevant page on this website, or search online if necessary, "
    "if they think that the information is not present on this page. "
    "Do not hallucinate. Always provide a helpful response.\n\n"
    f"CONTENT:\n{context}\n\n"
    f"USER QUESTION: {question}\n\n"
    "ANSWER:"
    )
    print("\n--- PROMPT TO LLM ---\n")
    print(prompt)
    print("\n--- END OF PROMPT ---\n")

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.2)
    result = llm.invoke([{"role": "user", "content": prompt}])
    answer = result.content.strip()
    def get_excerpt(doc):
        txt = doc.page_content.strip().replace('\n', ' ')
        return txt[:80] + "..." if len(txt) > 80 else txt

    state.answer = answer
    state.used_chunks = [
        {
            "excerpt": get_excerpt(d),
            "full_chunk": d.page_content,
            "start_char": d.metadata.get("start_char"),
            "end_char": d.metadata.get("end_char"),
            "url": d.metadata.get("url"),
            "title": d.metadata.get("title"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in relevant_docs
    ]
    return state

# Existing page QA graph
qa_builder = StateGraph(State)
qa_builder.add_node("EnhanceQuery", enhance_query_node)
qa_builder.add_node("Retrieve", retrieve_node)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("EnhanceQuery", "Retrieve")
qa_builder.add_edge("Retrieve", "Answer")
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("EnhanceQuery")
qa_graph = qa_builder.compile()

@app.post("/ask")
async def ask(request: QARequest):
    state = State(text=request.text, question=request.question)
    result = qa_graph.invoke(state)
    return {
        "answer": result["answer"],
        "enhanced_query": result["enhanced_query"],
        "sources": result["used_chunks"],
    }

# --- NEW: Site-wide QA endpoint ---
def extract_visible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove scripts/styles
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)

def keyword_hits(chunks, question, limit=5):
    qwords = set(question.lower().split())
    scored = []
    for doc in chunks:
        content = doc.page_content.lower()
        score = sum(1 for w in qwords if w in content)
        if score > 0:
            scored.append((score, doc))
    return [doc for score, doc in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]


@app.post("/ask-site")
async def ask_site(request: SiteQARequest):
    urls = request.urls[:10]
    all_chunks = []
    from langchain.schema import Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                title = soup.title.string.strip() if soup.title else url
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "url": url,
                            "title": title,
                            "chunk_id": i
                        }
                    ))
        except Exception as e:
            pass

    if not all_chunks:
        return {"answer": "No content could be retrieved from the provided site pages."}

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = Chroma.from_documents(
        all_chunks, embeddings, collection_name="webpage", persist_directory=None
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    vector_docs = retriever.get_relevant_documents(request.question)
    kw_docs = keyword_hits(all_chunks, request.question, limit=5)

    # Merge, dedupe by content
    seen = set()
    all_docs = []
    for d in (vector_docs + kw_docs):
        sig = (d.metadata.get("url", ""), d.page_content[:40])
        if sig not in seen:
            seen.add(sig)
            all_docs.append(d)

    def chunk_header(doc):
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        return f"[{title}]({url})"

    context = "\n\n---\n\n".join(
        f"{chunk_header(d)}\n{d.page_content}" for d in all_docs
    )

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.2)
    prompt = (
        "You are an expert assistant. Using only the content below (from multiple website pages), answer the user's question as fully and helpfully as possible.\n"
        "If you use information from a chunk, include the page title or URL as a citation at the end of your answer, like (Source: <title or URL>...).\n"
        "If you cannot answer, but there is related information in the content, try your best to answer using the most relevant context.\n"
        "Do not say 'Not found' unless you are certain nothing relevant is present.\n\n"
        f"CONTENT:\n{context}\n\n"
        f"USER QUESTION: {request.question}\n\n"
        "ANSWER:"
    )

    result = llm.invoke([{"role": "user", "content": prompt}])
    answer = result.content.strip()

    def get_excerpt(doc):
        txt = doc.page_content.strip().replace('\n', ' ')
        return txt[:80] + "..." if len(txt) > 80 else txt

    sources = [
    {
        "excerpt": get_excerpt(d),
        "title": d.metadata.get("title", ""),
        "url": d.metadata.get("url", ""),
        "chunk_id": d.metadata.get("chunk_id"),
    }
    for d in all_docs
]
    # return {"answer": answer, "sources": sources, "scanned_urls": urls}
    urls = request.urls[:10]  # LIMIT to 10 for debug! Increase if works.
    contents = []
    for url in urls:
        try:
            print(f"Fetching: {url}")
            resp = requests.get(url, timeout=15)
            if resp.ok:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Get visible text
                text = soup.get_text(separator="\n", strip=True)
                title = soup.title.string.strip() if soup.title else url
                contents.append(f"[{title}]\n{text}")
            else:
                contents.append(f"[{url}] -- Failed with status {resp.status_code}")
        except Exception as e:
            contents.append(f"[{url}] -- Exception: {e}")

    # Join all contents (may be large, but will prove the LLM can answer)
    context = "\n\n---\n\n".join(contents)
    question = request.question

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.2)
    prompt = (
        f"You are a helpful assistant. Using ONLY the following website content, answer the user's question as best as possible. "
        f"If you use information from a specific page, mention the page's title or URL for clarity.\n\n"
        f"CONTENT:\n{context}\n\nUSER QUESTION: {question}\n\nANSWER:"
    )
    print("\n--- PROMPT TO LLM ---\n")
    print(prompt[:1200] + "\n...")  # Print start of prompt, to check size

    result = llm.invoke([{"role": "user", "content": prompt}])
    return {"answer": result.content.strip()}


# ... (other imports/models/nodes unchanged) ...

class SmartQARequest(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str

class SmartHopState(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str
    answer: str = ""
    sources: List[Any] = []
    sufficient: bool = False
    selected_link: Optional[Dict[str, str]] = None
    visited_urls: List[str] = []
    hops: int = 0
    original_domain: str = ""


def extract_json_from_text(text):
    # Look for JSON code block first
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1)
    # Else, look for first [ ... ] in text
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return arr_match.group(0)
    return None

def answer_sufficiency_llm_node(state):
    """
    Uses LLM to decide if the answer is sufficient.
    Returns state with 'sufficient': bool
    """
    print("\n=== [Answer Sufficiency LLM Node] ===")
    prompt = (
        f"Question: {state['question']}\n"
        f"Answer given: {state['answer']}\n\n"
        "Based on the answer, is the user's question fully answered with clear and specific information? "
        "Reply with only 'YES' if it is enough, or 'NO' if it is not clear/specific enough."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    sufficient = "yes" in result.content.strip().lower()
    print(f"LLM says answer sufficient: {sufficient}")
    state["sufficient"] = sufficient
    return state

def llm_select_relevant_links_node(state):
    print("\n=== [LLM Link Selector Node] ===")
    links = state["links"][:30]
    prompt = (
        f"Question: {state['question']}\n"
        "Here are available links from the page:\n" +
        "\n".join([f"- {l['text']} ({l['href']})" for l in links]) +
        "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        "Reply with a JSON array of up to 3 objects with 'text' and 'href'."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    output = result.content.strip()

    json_str = extract_json_from_text(output)
    try:
        if json_str is None:
            raise ValueError("No JSON array found in LLM output")
        selected_links = json.loads(json_str)
        if not isinstance(selected_links, list):
            raise ValueError
        state["selected_links"] = selected_links[:3]
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\n{result.content}")
        state["selected_links"] = []
    return state
def retrieve_and_answer_node(state: SmartHopState) -> SmartHopState:
    s1 = enhance_query_node(State(text=state.text, question=state.question))
    s1 = retrieve_node(s1)
    s1 = answer_node(s1)
    state.answer = s1.answer
    state.sources = s1.used_chunks
    return state

def check_sufficiency_node(state: SmartHopState) -> SmartHopState:
    out = answer_sufficiency_llm_node({
        "question": state.question,
        "answer": state.answer
    })
    state.sufficient = out["sufficient"]
    return state


def pick_next_link_node(state: SmartHopState) -> SmartHopState:
    # Only pick from unvisited, same-domain links
    original_domain = urlparse(state.page_url).netloc if not hasattr(state, "original_domain") else state.original_domain
    if hasattr(state, "original_domain"):
        original_domain = state.original_domain
    else:
        original_domain = urlparse(state.page_url).netloc

    unvisited_links = [
        l for l in state.links
        if l["href"] not in (state.visited_urls or [])
        and urlparse(l["href"]).netloc == original_domain
    ]
    if not unvisited_links:
        state.selected_link = None
        return state
    out = llm_select_relevant_links_node({
        "question": state.question,
        "links": unvisited_links,
    })
    next_links = out.get("selected_links", [])
    state.selected_link = next_links[0] if next_links else None
    return state

def fetch_link_node(state: SmartHopState) -> SmartHopState:
    if not state.selected_link:
        return state
    url = state.selected_link["href"]
    state.visited_urls.append(url)
    try:
        resp = requests.get(url, timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")
        state.text = soup.get_text(separator="\n", strip=True)
        state.page_url = url
        # Extract new links from new page
        page_links = [
            {"text": a.get_text(strip=True), "href": a.get("href")}
            for a in soup.find_all("a", href=True)
            if a.get("href", "").startswith("http")
        ]
        state.links = [l for l in page_links if l["href"].startswith("http")]
        state.hops += 1
    except Exception as e:
        state.selected_link = None  # Can't fetch, stop
    return state

from langgraph.graph import StateGraph, END

def hops_remaining(state: SmartHopState):
    return state.hops < 5

graph = StateGraph(SmartHopState)
graph.add_node("RetrieveAndAnswer", retrieve_and_answer_node)
graph.add_node("CheckSufficiency", check_sufficiency_node)
graph.add_node("PickNextLink", pick_next_link_node)
graph.add_node("FetchLink", fetch_link_node)
graph.add_edge("RetrieveAndAnswer", "CheckSufficiency")
graph.add_conditional_edges(
    "CheckSufficiency",
    lambda s: "end" if s.sufficient or not hops_remaining(s) or not s.links else "pick",
    {
        "end": END,
        "pick": "PickNextLink",
    },
)
graph.add_conditional_edges(
    "PickNextLink",
    lambda s: "end" if s.selected_link is None else "fetch",
    {
        "end": END,
        "fetch": "FetchLink",
    },
)
graph.add_edge("FetchLink", "RetrieveAndAnswer")
graph.set_entry_point("RetrieveAndAnswer")
compiled_hop_graph = graph.compile()



@app.post("/ask-smart")
async def ask_smart(request: SmartQARequest):
    original_domain = urlparse(request.page_url).netloc

    state = SmartHopState(
        text=request.text,
        question=request.question,
        links=request.links,
        page_url=request.page_url,
        visited_urls=[request.page_url],
        hops=0,
        original_domain=original_domain
    )
    result = compiled_hop_graph.invoke(state)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "visited_urls": result["visited_urls"],
        "sufficient": result["sufficient"],
        # Optionally, add: "final_links": result.links
    }
