## FastAPI request call tool

import requests,json
import time
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))    
with open(os.path.join(current_dir, '..', 'configs', 'web_agent.json'), 'r') as f:
    config = json.load(f)

url = config['mcp_server_url']
def call_tool(tool_name: str, tool_args: dict):
    if tool_name is None:
        
        try:
            t1 = time.time()
            resp = requests.get(f"{url}/get_tool")
            result = resp.json()
            t2 = time.time()
            return {
                "tool_result": result,
                "tool_elapsed_time": t2 - t1
            }
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    else:
        try:
            t1 = time.time()
            resp = requests.post(
                f"{url}/call_tool/{tool_name}",
                json=tool_args
            )
            result = resp.json()
            if result["status"]:
                t2 = time.time()
                return {
                    "tool_result": result["result"],
                    "tool_elapsed_time": t2 - t1
                }
            else:
                print(f"Tool error: {result['result']}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None


import time
if __name__ == "__main__":
    # test()
    print(call_tool("web_search", {"query": "what is the multi-modal learning"}))
    