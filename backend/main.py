import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# New imports for crawling
import requests
from bs4 import BeautifulSoup

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
    "If you cannot answer, but there is related information in the content, try your best to answer using the most relevant context. "
    "Only say 'Not found in the content.' if nothing remotely relevant exists.\n\n"
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
    def get_excerpt(chunk):
        text = chunk.page_content.strip().replace('\n', ' ')
        words = text.split()
        if len(words) > 8:
            return ' '.join(words[:8])
        return text[:80]

    state.answer = answer
    state.used_chunks = [
        {
            "excerpt": get_excerpt(d),
            "full_chunk": d.page_content,
            "start_char": d.metadata["start_char"],
            "end_char": d.metadata["end_char"]
        }
        for d in relevant_docs
    ]
    return state

# Existing page QA graph
builder = StateGraph(State)
builder.add_node("EnhanceQuery", enhance_query_node)
builder.add_node("Retrieve", retrieve_node)
builder.add_node("Answer", answer_node)
builder.add_edge("EnhanceQuery", "Retrieve")
builder.add_edge("Retrieve", "Answer")
builder.add_edge("Answer", END)
builder.set_entry_point("EnhanceQuery")
graph = builder.compile()

@app.post("/ask")
async def ask(request: QARequest):
    state = State(text=request.text, question=request.question)
    result = graph.invoke(state)
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

@app.post("/ask-site")
async def ask_site(request: SiteQARequest):
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