import json
import os
import sys
import asyncio
import uvicorn
from typing import Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
import logging
import traceback
import threading
import types
from mcp_manager import MCPManager
from io_manage import ThreadOutputManager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pydantic import BaseModel

import time
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))

base_tool_dir = os.path.join(current_dir, 'server', 'BASE-TOOL-Server')
if base_tool_dir not in sys.path:
    sys.path.append(base_tool_dir)

mcp_sandbox_dir = os.path.dirname(current_dir)
api_proxy_dir = os.path.join(mcp_sandbox_dir, 'api_proxy')
if api_proxy_dir not in sys.path:
    sys.path.append(api_proxy_dir)

try:
    from api_utils.web_search_api import serpapi_google_search as web_search
    # from api_utils.web_search_api import serper_google_search as web_search
    from web_agent.web_parse import parse_htmlpage as web_parse_html
    from paper_agent.paper_parse import paper_qa_link as web_parse_pdf
    # from rag_tool_emb_llm_judge import search_local_documents
    from hippo_rag import search_local_documents
    
    print("\033[32m[INFO] Successfully imported base tools from the framework.\033[0m")

    def web_parse(link: str, user_prompt: str, llm: str = "gpt-4.1-nano-2025-04-14"):
        if ".pdf" in link.lower() or 'arxiv.org' in link:
            print("--- Routing to paper parser ---")
            return asyncio.run(web_parse_pdf(link=link, query=user_prompt, llm=llm))
        else:
            print("--- Routing to HTML parser ---")
            return asyncio.run(web_parse_html(url=link, user_prompt=user_prompt, llm=llm))

except ImportError as e:
    print(f"\033[31m[ERROR] Failed to import base tools. Please check sys.path and file locations.\n{e}\033[0m")
    def web_search(*args, **kwargs): return {"error": "web_search tool not found or import failed"}
    def web_parse(*args, **kwargs): return {"error": "web_parse tool not found or import failed"}
    def search_local_documents(*args, **kwargs): return {"error": "search_local_documents tool not found or import failed"}

import inspect
def create_sync_wrapper(async_func):
    """创建一个同步包装器来运行异步函数。"""
    if not inspect.iscoroutinefunction(async_func):
        return async_func

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeRequest(BaseModel):
    code: str
    timeout: Optional[int] = 180  # 默认超时30秒

class CodeResponse(BaseModel):
    output: str
    error: Optional[str]
    execution_time: float

class RagRequest(BaseModel):
    query: str

class RagResponse(BaseModel):
    output: str

  
manager = MCPManager()
app = FastAPI()
output_manager = ThreadOutputManager()
execution_semaphore = asyncio.Semaphore(2000)  
executor = ThreadPoolExecutor(max_workers=1000)  


@app.post("/call_tool/{tool_name}")
async def create_tool_task(tool_name: str, tool_args: Dict):
    print(f"=============={tool_name}===============")
    tool_names = manager.get_toolnames()
    if tool_name not in tool_names:
        raise HTTPException(404, "Tool not found")
    

    result = None
    status = False

    try:
        result = await manager.call_tool(
            tool_name,
            tool_args
        )
        status = True
        
        
        final_result = []
        for item in result:
            try:
                json_obj = json.loads(item)
                final_result.append(json_obj)
            except Exception as e:
                final_result.append(item)
        result = final_result

    except Exception as e:
        result = f"Error: {str(e)}\n\n{traceback.format_exc()}\n\ntool name: {tool_name}\n\n tool args: {tool_args}"
        status = False
        
    finally:
        return {"status": status, "result": result[0]}




async def execute_python_code(code: str, timeout: int) -> Tuple[str, Optional[str], float]:
    
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    try:
        execution_time, output, error = await loop.run_in_executor(executor, _execute_code_safely, code, timeout)
    except Exception as e:
        error = f"Execution failed: {str(e)}"
        output = ""
        logger.error(f"Unexpected error: {error}", exc_info=True)
        execution_time = loop.time() - start_time
        
    return output, error, execution_time

def _execute_code_safely(code: str, timeout: int) -> Tuple[str, Optional[str]]:
    
    logger.info(f"Executing in thread {threading.current_thread().ident}, process {os.getpid()}")

    capture = output_manager.get_capture()

    error = None
    output_value = None
    error_value = None

    start_time = time.time()

    single_executor = ThreadPoolExecutor(max_workers=1)

    def run_code():
        module = types.ModuleType("dynamic_module")
        available_tools = {
            'web_search': web_search,
            'web_parse': web_parse,
            'search_local_documents': search_local_documents,
        }
        wrapped_tools = {
            name: create_sync_wrapper(func) for name, func in available_tools.items()
        }

        module.__dict__.update({
            'print': lambda *args, **kwargs: capture.write(
                ' '.join(str(arg) for arg in args) + ('\n' if kwargs.get('end', '\n') else '')
            ),
            '__builtins__': __builtins__,
            'sys': type('sys', (), {
                'stdout': capture,
                'stderr': capture,
                'stdin': None,
            })(),
        })
        module.__dict__.update(wrapped_tools)
        
        exec(code, module.__dict__)
        return capture.get_stdout(), capture.get_stderr()

    try:
        future = single_executor.submit(run_code)
        output_value, error_value = future.result(timeout=timeout)

    except FutureTimeoutError:
        error = f"Execution timed out after {timeout} seconds"
        logger.warning(f"Code execution timeout: {timeout}s")
        error_value = error

    except SystemExit as se:
        error = f"Code called sys.exit({se.code})"
        if not capture.stderr.closed:
            capture.stderr.write(error)
        logger.warning(f"Code triggered SystemExit: {error}\n\n-----\n{code}")
        error_value = error

    except Exception as e:
        error = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        
        if not capture.stderr.closed:
            capture.stderr.write(error)
        logger.warning(f"Code execution error: {error}\n\n-----\n{code}")
        error_value = error

    finally:
        if not capture.stdout.closed or not capture.stderr.closed:
            if output_value is None:
                output_value = capture.get_stdout()
            if error_value is None or error_value != error:
                error_value = capture.get_stderr()

    execution_time = time.time() - start_time

    return execution_time, output_value, error_value if error_value else None


@app.post("/rag", response_model=RagResponse)
async def rag_handler(request: RagRequest):

        
    try:
        output = await asyncio.get_event_loop().run_in_executor(
            executor,
            search_local_documents,
            request.query
        )
        
        if isinstance(output, (list, dict)):
            output_str = json.dumps(output, ensure_ascii=False)
        else:
            output_str = str(output)
        
        return RagResponse(output=output_str)
        
    except Exception as e:
        logger.error(f"Unexpected server error in RAG: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



@app.post("/execute", response_model=CodeResponse)
async def execute_code_handler(request: CodeRequest):

    if not request.code.strip():
        raise HTTPException(
            status_code=400,
            detail="Code cannot be empty"
        )
    
    logger.info(f"Executing code snippet (timeout: {request.timeout}s)")
    
    try:
        output, error, exec_time = await execute_python_code(
            request.code,
            request.timeout
        )
        
        logger.info(
            f"Code execution completed in {exec_time:.2f}s. "
            f"Output length: {len(output)}, Error: {bool(error)}"
        )
        
        return CodeResponse(
            output=output,
            error=error,
            execution_time=exec_time,
        )
        
    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



if __name__ == "__main__":
    import os

    PORT = os.getenv('PORT', 40001)

    uvicorn.run(
        "tool_server_session:app", 
        host="0.0.0.0", 
        port=int(PORT),
        lifespan="on",
        workers=1,
        reload=True
    )