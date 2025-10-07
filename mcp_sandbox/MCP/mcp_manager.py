import os, sys
import json
from typing import List
from mcp_client import MCPClient

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def load_serverlist():
    config_path = os.path.join(current_dir, "config/server_list.json")
    with open(config_path, 'r') as file:
        return json.load(file)

class MCPManager:
    def __init__(self):
        self.client_list: list[MCPClient] = []
        self.tool_client: dict[str, MCPClient] = {}
        self.tool_list: list[dict] = []
        self.is_ready: bool = False

    
    async def ready(self):
        self.is_ready = False
        server_list : List[str] = load_serverlist()

        tmp = []
        for server in server_list:
            is_sse = server.startswith('http')
            if not is_sse:
                tmp.append(os.path.join(current_dir, server))
            else:
                tmp.append(server)

        server_list = tmp
        ready_server = []
        for server in server_list:
            client = MCPClient(venv_path=sys.prefix, server=server)
            try:
                name = await client.connect_to_server()
                tools = await client.get_tools()
                self.client_list.append(client)
                self.tool_list += tools
                for tool in tools:
                    self.tool_client[tool["name"]] = client
                ready_server.append(name)
                print(f"Server {name} ready.")
            except Exception as e:
                print(f"An error occurred while create client: {e}. For server {server}.")
                await client.cleanup()
                continue
        if ready_server:
            self.is_ready = True
        return ready_server
    
    def get_tools(self):
        return self.tool_list
    def get_toolnames(self):
        return self.tool_client.keys()
    def get_status(self):
        return self.is_ready

    async def call_tool(self, tool_name: str, tool_args: dict = None) -> list:
        if not self.tool_client.get(tool_name):
            raise KeyError(f"Tool '{tool_name}' not found in tool list.")
        try:
            result = await self.tool_client[tool_name].call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            raise RuntimeError(e)
    
    async def close(self):
        for client in self.client_list:
            await client.cleanup()