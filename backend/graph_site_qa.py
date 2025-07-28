import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma

openai_api_key = os.environ.get("OPENAI_API_KEY")

def extract_visible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
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

async def ask_site_handler(request):
    urls = request.urls[:10]
    all_chunks = []
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
        except Exception:
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

    return {"answer": answer, "sources": sources}
