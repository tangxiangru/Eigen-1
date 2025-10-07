import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from tool_caller import call_tool
def web_search(query,top_k=10):
    tool_args = {'query':query,'top_k':top_k}
    result = call_tool('web_search', tool_args)
    return result
def web_parse(link,user_prompt,llm='qwen2.5-72b-instruct'):
    tool_args = {'link':link,'user_prompt':user_prompt,'llm':llm}
    result = call_tool('web_parse', tool_args)
    return result