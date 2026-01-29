"""
Delete ALL RAG corpora in the project/location.
Run from backend dir. Loads .env for PROJECT_ID, LOCATION, GOOGLE_APPLICATION_CREDENTIALS.

  cd backend
  python scripts/delete_all_rag_corpora.py
"""
import os
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
os.chdir(_backend_dir)

from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai import rag as vx_rag

def main():
    project = (os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.environ.get("LOCATION") or "us-central1").strip()
    if not project:
        print("Set PROJECT_ID or GOOGLE_CLOUD_PROJECT in .env")
        sys.exit(1)

    vertexai.init(project=project, location=location)
    corpora = list(vx_rag.list_corpora())
    if not corpora:
        print("No RAG corpora found.")
        return

    print(f"Found {len(corpora)} corpus/corpora in {project}/{location}. Deleting...")
    for c in corpora:
        name = getattr(c, "name", None) or str(c)
        try:
            vx_rag.delete_corpus(name=name)
            print(f"  Deleted: {name}")
        except Exception as e:
            print(f"  Failed {name}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
