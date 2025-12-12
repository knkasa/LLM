# Modular MCP to create your own MCP.
# pip install modular-mcp
# Another alternative, chuk_mcp library.

from mcp.server import Server
from mcp.types import Tool

server = Server("example-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Does something",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    return {"result": ...}
