import os
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def get_chroma_dir(url):
    domain = urlparse(url).netloc.replace(":", "_")
    return f"./chroma_dbs/{domain}"

def scrape_with_playwright(url, timeout=12000):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout)
        html = page.content()
        final_url = page.url
        browser.close()
        return html, final_url

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

def chunk_and_save(text, url):
    # Split to chunks (identical to your pipeline)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": url, "chunk_id": i}) for i, c in enumerate(chunks)]

    embeddings = OpenAIEmbeddings()
    persist_dir = get_chroma_dir(url)
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        docs, embeddings,
        collection_name="webpage",
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ ChromaDB stored at: {persist_dir}")

def main():
    url = input("Enter a URL to crawl and embed: ").strip()
    html, final_url = scrape_with_playwright(url)
    text = extract_text_from_html(html)
    print(f"Scraped text length: {len(text)} chars")
    chunk_and_save(text, final_url)

if __name__ == "__main__":
    main()
