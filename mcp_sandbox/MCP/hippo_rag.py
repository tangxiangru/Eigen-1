
from typing import Dict, List, Any
from openai import OpenAI
import re
import json
from termcolor import colored
import os
import sys
from typing import List, Dict, Any
import time
from hipporag import HippoRAG
import multiprocessing
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


current_dir = os.path.dirname(os.path.abspath(__file__))

with open(f"{current_dir}/../configs/llm_call.json", "r") as f:
    llm_call_config = json.load(f)
###########################################################################################
########################################### RAG ###########################################
###########################################################################################
class UnifiedRetrieverManager:
    def __init__(self, k: int=5):
        self.k = k
        self.retriever: Any = None
        self.is_initialized = False
        self._is_index_built = False
        self.json_rag_data_path = os.path.join(current_dir, 'paragraphs.json')
        # self.docs = self._load_paragraphs(self.json_rag_data_path)
        self.docs = None
        self.save_dir = os.path.join(current_dir, 'hippo_rag_output')

        model='gpt-4o'
        self.judge_llm = ChatOpenAI(
            model=model,
            api_key=llm_call_config[model]["authorization"],
            base_url=llm_call_config[model]["url"]
        )

    def _initialize(self):
        """
        Loads the single pre-built FAISS index from disk.
        """
        if self.is_initialized:
            return
        
        os.environ["OPENAI_API_KEY"]  = llm_call_config['gpt-4o']["authorization"]
        os.environ["OPENAI_BASE_URL"] = llm_call_config['gpt-4o']["url"]
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]
        
        self.retriever = HippoRAG(save_dir=self.save_dir, 
            llm_model_name='gpt-4o',
            llm_base_url=llm_call_config['gpt-4o']["url"],
            embedding_model_name='text-embedding-3-large',  
            embedding_base_url=llm_call_config['text-embedding-3-large']["url"],
        )
        # self.retriever.index(docs=self.docs)
        
        if not self._is_index_built:
            if self.docs is None:
                self.docs = self._load_paragraphs(self.json_rag_data_path)
            if self.docs:
                self.retriever.index(docs=self.docs)
                self._is_index_built = True
                # 如果内存很紧，可以索引后释放原始 docs 列表引用
                # self.docs = None
                
        self.is_initialized = True

            
    def _load_paragraphs(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[INFO] Loaded {len(data)} paragraphs as docs")
        return data

    def search(self, query: str) -> List[str]:
        if not self.is_initialized:
            self._initialize()

        # start_time = time.time()
        docs = self.retriever.retrieve([query], num_to_retrieve=self.k)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"[RAG Tool DEBUG] Unified search took {elapsed_time:.4f} seconds.", file=sys.stderr)
        # print(docs[0].docs)
        docs = docs[0].docs

        contents = [doc for doc in docs]
        
        prompt_temp = """## Instructions
Please analyze the following problem and its solution. Determine whether the following paragraph is helpful in solving the problem. 
A score of -1 indicates negative help (e.g., misleading information), 
0 indicates irrelevant information, 
1 indicates some positive help (e.g., containing relevant information), 
and 2 indicates very helpful (directly providing key solution ideas or knowledge).

## Query
{query}

## Paragraphs requiring judgment
{content}

Your output only needs to contain a number (-1, 0, 1 or 2):
"""     
        positive_contents = []
        for content in contents:
            msgs = [HumanMessage(prompt_temp.format(query=query, content=content))]
            for i in range(3):
                response = self.judge_llm.invoke(msgs)
                try:
                    score = int(response.content.strip())
                    if score in [-1, 0, 1, 2]:
                        break
                except:
                    score = None
                    msgs.append(response)
                    msgs.append(HumanMessage("Your output only needs to contain a number (-1, 0, 1 or 2). Please try again."))
            if score in [1, 2]:
                positive_contents.append(content)

        return positive_contents


_retriever_instance = None
_lock = multiprocessing.Lock()


def get_retriever_instance() -> UnifiedRetrieverManager:
    global _retriever_instance
    if _retriever_instance is None:
        with _lock:
            if _retriever_instance is None:
                _retriever_instance = UnifiedRetrieverManager(k=5)
    return _retriever_instance

def search_local_documents(query: str) -> list[str]:
    """
    Searches the unified local knowledge base for relevant documents.
    If it fails to find results after 3 attempts, it returns an empty list.
    """
    if not query or not query.strip():
        print("[RAG Tool WARN] Empty query provided to search_local_documents.")
        return []
    
    for attempt in range(3):
        try:
            retriever_instance = get_retriever_instance()
            results = retriever_instance.search(query=query)
            if results:
                return results
            print(f"[RAG Tool WARN] Attempt {attempt + 1}/3: No results found for query: '{query}'. Retrying...")
            time.sleep(0.5)
        except Exception as e:
            import traceback
            error_message = {"status": "error", "details": f"Unhandled exception on attempt {attempt + 1}: {e}\n{traceback.format_exc()}"}
            print(f"\031[31m[RAG Tool ERROR] {error_message}\031[0m", file=sys.stderr)
    
    print(f"[RAG Tool INFO] After 3 attempts, no results were found for query: '{query}'.")
    return []


if __name__ == "__main__":
    query = "disadvantages of air-stable organic radicals in OLEDs"
    results = search_local_documents(query)
    print(f"Results for query '{query}':")
    for idx, doc in enumerate(results):
        print(f"{idx + 1}. {doc[:200]}...")  # Print first 200 characters of each document