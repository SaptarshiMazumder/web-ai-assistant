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

# --- Load environment variables ---
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

class State(BaseModel):
    text: str
    question: str
    enhanced_query: str = ""
    docs: List[Any] = []
    retrieved_docs: List[Any] = []
    answer: str = ""
    used_chunks: List[Dict[str, Any]] = []

# --- LangGraph Nodes ---

def enhance_query_node(state: State) -> State:
    print("\n=== [EnhanceQuery Node] ===")
    page_text = state.text
    user_question = state.question

    print(f"Original user question:\n{user_question}")

    prompt = (
        "You are a smart assistant. Given the user's question and the web page content, "
        "rewrite the question using terminology and keywords from the web page. "
        "If the question is ambiguous, clarify it. If the question already matches, keep it as is.\n\n"
        f"WEBPAGE CONTENT (first 1200 chars):\n{page_text[:1200]}\n\n"
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
    enhanced_query = state.enhanced_query

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.create_documents([page_text])
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i

    print("\n--- All Chunks ---")
    for d in docs:
        print(f"[Chunk {d.metadata['chunk_id']}]: {d.page_content[:120]}...")

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = Chroma.from_documents(
        docs, embeddings, collection_name="webpage", persist_directory=None
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    relevant_docs = retriever.get_relevant_documents(enhanced_query)

    print("\n--- Retrieved Chunks ---")
    for d in relevant_docs:
        print(f"[Excerpt]: {d.page_content[:120]}...")

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
        "If you truly cannot answer, say 'Not found in the content.'\n\n"
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
    print(f"\n--- Final Answer ---\n{answer}")

    # User-friendly sources: excerpt = first sentence, or first 140 chars, etc
    def get_excerpt(chunk):
        text = chunk.page_content.strip().replace('\n', ' ')
        # Try to find a period for first sentence, fallback to 140 chars
        period = text.find('.')
        if 20 < period < 140:
            return text[:period+1]
        return text[:140] + "..." if len(text) > 140 else text

    state.answer = answer
    state.used_chunks = [
        {
            "excerpt": get_excerpt(d),
            "full_chunk": d.page_content
        } for d in relevant_docs
    ]
    return state

# --- Build the LangGraph ---
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
    # State input for LangGraph
    state = State(text=request.text, question=request.question)
    result = graph.invoke(state)
    return {
        "answer": result["answer"],
        "enhanced_query": result["enhanced_query"],
        "sources": result["used_chunks"],   # Each has an excerpt and full_chunk for debug
    }
