#!/usr/bin/env python3
"""
Simple MCP test client for Graphistry.
Sends a request to visualize a graph and returns the URL.
"""

import json
import subprocess
import sys

def main():
    """Run a simple test of the Graphistry MCP server."""
    print("Starting Graphistry MCP server...")
    
    # Start server process in stdio mode (default)
    process = subprocess.Popen(
        ["python", "run_graphistry_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to initialize
    print("Waiting for server initialization...")
    while True:
        line = process.stderr.readline().strip()
        if not line:
            continue
        print(f"Server: {line}")
        if "ready to process requests" in line:
            break
    
    # Send list_tools request to get available tools
    list_tools_req = {
        "id": "list1",
        "type": "list_tools"
    }
    
    print("\nSending list_tools request...")
    process.stdin.write(json.dumps(list_tools_req) + "\n")
    process.stdin.flush()
    
    # Get response
    response = json.loads(process.stdout.readline())
    tools = [tool["name"] for tool in response.get("tools", [])]
    print(f"Available tools: {tools}")
    
    # Create a simple graph
    print("\nCreating graph visualization...")
    graph_request = {
        "id": "viz1",
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
    
    process.stdin.write(json.dumps(graph_request) + "\n")
    process.stdin.flush()
    
    # Get visualization response
    viz_response = json.loads(process.stdout.readline())
    
    if "output" in viz_response:
        url = viz_response["output"].get("url")
        graph_id = viz_response["output"].get("graph_id")
        
        print("\n✅ Graph visualization created!")
        print(f"Graph ID: {graph_id}")
        print(f"URL: {url}")
        print("\nOpen this URL in your browser to view the visualization.")
    else:
        print(f"\n❌ Error creating visualization: {viz_response}")
    
    # Clean up
    process.terminate()

if __name__ == "__main__":
    main()