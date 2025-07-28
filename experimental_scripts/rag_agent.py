import os
import argparse
import chromadb
from openai import OpenAI
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"          # The directory you used for Chroma
CHROMA_COLLECTION = "docs"             # The collection name you used
OPENAI_MODEL = "gpt-4o"                # Use "gpt-4o" for best results
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set as env variable for security

# --- Helper: build a prompt with context chunks ---
def build_prompt(query, docs):
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n---\nSource {i+1}:\n{doc['document']}\n"
    prompt = (
        f"Answer the user's question using only the provided sources below. "
        f"If the answer is not in the sources, say you don't know.\n\n"
        f"User question: {query}\n\nSources:{context}\n\nAnswer:"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="RAG agent with ChromaDB and GPT-4o")
    parser.add_argument("question", help="The user question to answer")
    parser.add_argument("--db-dir", default=CHROMA_DB_DIR, help="ChromaDB directory")
    parser.add_argument("--collection", default=CHROMA_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    # --- 1. Connect to ChromaDB ---
    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_collection(args.collection)

    # --- 2. Retrieve top-k relevant chunks ---
    # Note: Chroma automatically embeds using the model from insertion
    results = collection.query(
        query_texts=[args.question],
        n_results=args.top_k,
        include=["documents", "metadatas"]
    )

    # Parse out documents (chunks)
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "id": results["ids"][0][i]
        })

    print("---- Retrieved Context ----")
    for d in docs:
        print("Source:", d['metadata'].get('source', ''))
        print(d['document'][:500])
        print("------")
    # --- 3. Build prompt ---
    prompt = build_prompt(args.question, docs)

    # --- 4. Query OpenAI GPT-4o ---
    client_oai = OpenAI(api_key=OPENAI_API_KEY)
    chat_response = client_oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.0,
    )
    answer = chat_response.choices[0].message.content

    # --- 5. Display answer ---
    print("\n=== Answer ===\n")
    print(answer.strip())
    print("\n=== Sources Used ===\n")
    for i, doc in enumerate(docs):
        print(f"[Source {i+1}] {doc['metadata'].get('source','')}")
        headers = doc['metadata'].get('headers','')
        if headers:
            print(f"   Section: {headers}")
    print("")

if __name__ == "__main__":
    main()
