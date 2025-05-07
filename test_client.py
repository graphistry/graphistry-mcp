import asyncio
import os
from mcp.client import Client
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client() as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", [tool["name"] for tool in tools])
        
        # Create a simple test graph
        result = await client.call_tool("visualize_graph", {
            "data_format": "edge_list",
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
                {"source": "C", "target": "A"}
            ],
            "title": "Test Triangle Graph"
        })
        
        print(f"\nGraph created successfully!")
        print(f"Graph ID: {result['graph_id']}")
        print(f"URL: {result['url']}")
        print("\nOpen the URL in your browser to view your visualization")

if __name__ == "__main__":
    asyncio.run(main()) 