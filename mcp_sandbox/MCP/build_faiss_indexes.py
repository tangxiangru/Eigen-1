import os
import json
import sys
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

DB_PATH = "paragraphs.json"
UNIFIED_INDEX_DIR = "faiss_unified_index"

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{current_dir}/../configs/llm_call.json", "r") as f:
    llm_call_config = json.load(f)

def build_unified_index():
    """
    Reads the source JSON, iterates through all IDs, aggregates all paragraphs
    into a single list, generates embeddings, builds one unified FAISS index,
    and saves it to disk.
    """
    print(f"\033[34mStarting the unified index building process...\033[0m")
    os.makedirs(UNIFIED_INDEX_DIR, exist_ok=True)

    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=llm_call_config['text-embedding-3-large']["authorization"],
            base_url=llm_call_config['text-embedding-3-large']["url"],
            chunk_size=500,
            timeout=300
        )
    except Exception as e:
        print(f"\031[31mFailed to initialize embedding model: {e}\031[0m", file=sys.stderr)
        return

    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
    except Exception as e:
        print(f"\031[31mError reading or decoding '{DB_PATH}': {e}\031[0m", file=sys.stderr)
        return

    all_docs = []
    print("\033[32mLoading all documents from all IDs into memory...\033[0m")
    for i, p in enumerate(retrieval_data):
        all_docs.append(
            Document(
                page_content=p,
                metadata={"source": f"para_id_{i}"}
            )
        )

    if not all_docs:
        print(f"\033[33mWarning: No documents found in '{DB_PATH}'. Exiting.\033[0m")
        return

    print(f"\033[32mFound {len(all_docs)} total paragraphs. Building unified FAISS index. This may take a while...\033[0m")
    
    try:
        vectorstore = FAISS.from_documents(all_docs, embedding_model)
        
        vectorstore.save_local(UNIFIED_INDEX_DIR)
        
        print(f"\033[32m  -> Successfully built and saved unified index to '{UNIFIED_INDEX_DIR}'\033[0m")

    except Exception as e:
        print(f"\031[31m  -> Failed to build or save the unified index: {e}\031[0m", file=sys.stderr)

    print("\033[34m\nUnified index building process complete.\033[0m")

if __name__ == "__main__":
    build_unified_index()