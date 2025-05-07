#!/usr/bin/env python3
import requests
import json
from typing import Dict, List, Optional, Any, Union

# Simple HTTP client for testing the Graphistry MCP server
def test_graphistry_http() -> Optional[str]:
    """
    Test the Graphistry MCP server with HTTP requests.
    
    Returns:
        Optional[str]: The URL of the created visualization, or None if creation failed
    """
    # Define the server URL
    base_url: str = "http://localhost:8080"
    
    # Test health endpoint
    try:
        health_response: requests.Response = requests.get(f"{base_url}/health")
        print(f"Health check: {health_response.status_code}")
        print(health_response.json())
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Create a simple graph via the MCP API
    try:
        # Define request structure with proper typing
        edge_list: List[Dict[str, str]] = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "D"},
            {"source": "D", "target": "A"},
            {"source": "A", "target": "E"},
            {"source": "E", "target": "C"}
        ]
        
        mcp_request: Dict[str, Any] = {
            "type": "call_tool",
            "name": "visualize_graph",
            "input": {
                "data_format": "edge_list",
                "edges": edge_list,
                "title": "Simple Graph Example"
            }
        }
        
        mcp_response: requests.Response = requests.post(
            f"{base_url}/api/mcp",
            json=mcp_request
        )
        
        if mcp_response.status_code == 200:
            result: Dict[str, Any] = mcp_response.json()
            print("\nGraph visualization created successfully!")
            
            # Extract URL and graph_id from the response
            output: Dict[str, Any] = result.get('output', {})
            visualization_url: Optional[str] = output.get('url')
            graph_id: Optional[str] = output.get('graph_id')
            
            print(f"URL: {visualization_url}")
            print(f"Graph ID: {graph_id}")
            return visualization_url
        else:
            print(f"Error: {mcp_response.status_code}")
            print(mcp_response.text)
            
    except Exception as e:
        print(f"Graph creation failed: {e}")
    
    return None

if __name__ == "__main__":
    url: Optional[str] = test_graphistry_http()
    if url:
        print(f"\nOpen this URL in your browser to view your visualization:\n{url}")
    else:
        print("\nFailed to create visualization. Make sure the server is running with:")
        print("  python run_graphistry_mcp.py --http 8080")