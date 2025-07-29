import os
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from urllib.parse import urlparse

openai_api_key = os.environ.get("OPENAI_API_KEY")

class State(BaseModel):
    text: str
    question: str
    enhanced_query: str = ""
    docs: List[Any] = []
    retrieved_docs: List[Any] = []
    answer: str = ""
    used_chunks: List[Dict[str, Any]] = []

def enhance_query_node(state: State) -> State:
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
    state.enhanced_query = enhanced_query
    return state

def retrieve_node(state: State) -> State:
    page_text = state.text
    enhanced_query = state.enhanced_query
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
    chunks = splitter.split_text(page_text)

    docs = []
    pos = 0
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "start_char": pos,
            "end_char": pos + len(chunk),
            "url": getattr(state, "page_url", None)  # Or state.page_url if it's always set
        }

        from langchain.schema import Document
        doc = Document(page_content=chunk, metadata=metadata)
        docs.append(doc)
        pos += len(chunk)

    if not docs:
        state.docs = []
        state.retrieved_docs = []
        return state

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = Chroma.from_documents(
        docs, embeddings, collection_name="webpage", persist_directory=None
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    relevant_docs = retriever.get_relevant_documents(enhanced_query or state.question)

    state.docs = docs
    state.retrieved_docs = relevant_docs
    return state



def answer_node(state: State) -> State:
    print("Came inside answer_node")
    question = state.question
    relevant_docs = state.retrieved_docs
    print("\n=== Retrieved Chunks ===")
    for i, d in enumerate(relevant_docs):
        url = d.metadata.get("url")
        print(f"[Chunk {i}] (url: {url})")
        print(d.page_content)
        print("------")
    print("======================\n")
    def format_chunk(doc):
        url = doc.metadata.get("url")
        chunk_text = doc.page_content
        if url:
            return f"[Source URL: {url}]\n{chunk_text}"
        return chunk_text

    context = "\n\n---\n\n".join([format_chunk(d) for d in relevant_docs])

    # New prompt: answer + sufficiency
    # prompt = (
    #     "You are an expert assistant. Using only the content below, answer the user's question as fully and helpfully as possible. "
    #     "Use your understanding and reasoning, quote, paraphrase, or summarize as appropriate. "
    #     "If you use information from the content, include a relevant short excerpt from the source as a citation at the end of your answer in this format: (Source: <short excerpt>...) "
    #     "If you cannot find a direct answer in the content, briefly summarize anything related or useful you did find, "
    #     "and politely inform the user that the answer does not seem to be present on this page. "
    #     "Say that they may navigate to a more relevant page on this website, or search online if necessary, "
    #     "ALWAYS provide which page they should visit to find the exact information they are looking for, if possible. "
    #     "If you cite information from a source, always include the full URL shown in the source."
    #     "If you cannot find a direct answer, suggest a related page by URL if present in the content."
    #     "Do not hallucinate."
    #     "If you are confident the answer is specific and fully resolves the question, append the line 'SUFFICIENT: YES' at the end of your answer. "
    #     "If not, append 'SUFFICIENT: NO'.\n\n"
    #     f"CONTENT:\n{context}\n\n"
    #     f"USER QUESTION: {question}\n\n"
    # )

    prompt = (
        "You are an expert assistant. Using only the current webpage content below, answer the user's question by quoting the relevant passage, code block, or table WORD-FOR-WORD, including ALL formatting, indentation, and line breaks. "
        "DO NOT paraphrase, summarize, or shorten ANY part of the quoted answer, unless absolutely necessary. "
        "Prioritize giving the most detailed answer possible by quoting the all relevant text from the content. "
        "If the content has numbers, prices, code, tables, or other specific details, quote them exactly as they appear. "
        "If multiple relevant passages are found, include ALL of them in their entirety, word-for-word. "
        "ALWAYS provide which page they should visit to find the exact information they are looking for, if possible, by saying 'Please visit [page URL] for the full details.' "
        "If you cite information from a source, always include the full URL shown in the source."
        "If no answer is found, summarize anything related, and politely inform the user that the answer does not appear to be present, suggest a related page by URL if present in the content. "
        "Do not hallucinate."
        "At the end, write 'SUFFICIENT: YES' if the answer fully resolves the question, or 'SUFFICIENT: NO' if not.\n\n"
        f"CONTENT:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
    )

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.2)
    result = llm.invoke([{"role": "user", "content": prompt}])
    answer_full = (result.content or "").strip()

    if "\nSUFFICIENT: YES" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: YES")[0].strip()
        sufficient = True
    elif "\nSUFFICIENT: NO" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: NO")[0].strip()
        sufficient = False
    else:
        answer = answer_full
        sufficient = False

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
    # <-- Add this -->
    state.sufficient = sufficient
    return state


qa_builder = StateGraph(State)
qa_builder.add_node("EnhanceQuery", enhance_query_node)
qa_builder.add_node("Retrieve", retrieve_node)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("EnhanceQuery", "Retrieve")
qa_builder.add_edge("Retrieve", "Answer")
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("EnhanceQuery")
qa_graph = qa_builder.compile()
