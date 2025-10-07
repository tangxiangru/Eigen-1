## Installation

```bash
pip install -r requirements.txt
```

-----

## Configuration

Before running the application, you must configure the necessary API keys and endpoints.

1.  **Web Search API**: Open the file `mcp_sandbox/configs/web_agent.json` and add your `serper_api_key`.

2.  **Large Language Model (LLM) API**: Open the file `mcp_sandbox/configs/llm_call.json` and set the `url` and `authorization` key for your LLM.

-----

## Local RAG Setup

The application uses a local Retrieval-Augmented Generation (RAG) system.

### Method 1: Hippo Rag

This method provides a direct way to use the RAG service.

1.  **Data File**: Ensure your knowledge base documents are located in the following file:
    `mcp_sandbox/MCP/paragraphs.json`

2.  **Configure Server**: Open `mcp_sandbox/MCP/tool_server.py` and ensure line 35 is set to:

    ```python
    from hippo_rag import search_local_documents
    ```

### Method 2: Embedding (FAISS) Retrieval

This method uses a pre-built FAISS vector index for faster retrieval.

1.  **Data File**: Ensure your knowledge base documents are located in the following file:
    `mcp_sandbox/MCP/paragraphs.json`

2.  **Build FAISS Index**: Run the following command from the project root directory to create the vector indexes:

    ```bash
    python mcp_sandbox/MCP/build_faiss_indexes.py
    ```

    *(Note: Indexes will be saved in `mcp_sandbox/MCP/faiss_indexes`)*

3.  **Configure Server**: Open `mcp_sandbox/MCP/tool_server.py` and ensure line 35 is set to:

    ```python
    from rag_tool_emb_llm_judge import search_local_documents
    ```

-----

## Running the Application

The application requires two services to be running in separate terminals.

### Terminal 1: Start the API Proxy

```bash
cd api_proxy
python api_server.py
```

### Terminal 2: Start the Main MCP Server

```bash
cd MCP
bash deploy_server.sh
```

-----

## Testing the Application

See `../llm_agent/tools/tool_manager.py`. It is important to ensure that each tool call returns the correct results.