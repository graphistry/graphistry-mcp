#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import subprocess

# Simple MCP client for testing - uses credentials from environment
async def run_mcp_client():
    # Start the MCP server in the background
    server_process = subprocess.Popen(
        ["python", "run_graphistry_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )
    
    try:
        # First, send a list_tools request
        list_tools_request = {
            "id": "1",
            "type": "list_tools",
        }
        print("Sending list_tools request...")
        server_process.stdin.write(json.dumps(list_tools_request) + "\n")
        server_process.stdin.flush()
        
        # Read the response
        list_tools_response = json.loads(server_process.stdout.readline())
        print(f"Available tools: {[tool['name'] for tool in list_tools_response.get('tools', [])]}")
        
        # Now create a simple graph
        visualize_request = {
            "id": "2",
            "type": "call_tool",
            "name": "visualize_graph",
            "input": {
                "data_format": "edge_list",
                "edges": [
                    {"source": "A", "target": "B"},
                    {"source": "B", "target": "C"},
                    {"source": "C", "target": "D"},
                    {"source": "D", "target": "A"}
                ],
                "title": "Simple Square Graph"
            }
        }
        
        print("Sending visualize_graph request...")
        server_process.stdin.write(json.dumps(visualize_request) + "\n")
        server_process.stdin.flush()
        
        # Read the response
        visualize_response = json.loads(server_process.stdout.readline())
        print(f"Visualization created!")
        print(f"Graph URL: {visualize_response.get('output', {}).get('url')}")
        print(f"Graph ID: {visualize_response.get('output', {}).get('graph_id')}")
        
        return visualize_response.get('output', {}).get('url')
    
    finally:
        print("Closing server process...")
        server_process.terminate()

if __name__ == "__main__":
    url = asyncio.run(run_mcp_client())
    print(f"\nOpen this URL in your browser to view your visualization:\n{url}")