from uuid import uuid4
import requests

class BaseToolManager:
    def __init__(self, url:str):
        self.server_url = url
        self.headers = {
            "Content-Type": "application/json"
        }

    def execute_tool(self, tool_call:str):
        # tool_call = "from tools import *\n" + tool_call
        payload = {
            "code":tool_call
        }
        # print("execution...")
        resp = requests.post(
            f"{self.server_url}/execute",
            headers=self.headers,
            json=payload,
            proxies={"http": None, "https": None}, # remove if error
        )

        return resp.json()
    

if __name__ == "__main__":
    tool_manager = BaseToolManager("<Address>:30008")
    # tool_call = "r = web_search('Watterson estimator bias imputation reference genome')\nprint(r)"
    # tool_call = "r = web_parse('https://en.wikipedia.org/wiki/Python_(programming_language)', 'describe this page')\nprint(r)"
    # tool_call = "r = web_parse('https://arxiv.org/pdf/2502.06148', 'describe this page')\nprint(r)"
    tool_call = "r = search_local_documents('disadvantages of air-stable organic radicals in OLEDs')\nprint(r)"
    result = tool_manager.execute_tool(tool_call)

    print(result)
