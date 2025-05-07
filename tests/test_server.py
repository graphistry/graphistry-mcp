"""
Tests for the Graphistry MCP server.
"""
import json
import os
from typing import Dict, Any, List
import pytest
import graphistry
from dotenv import load_dotenv
from graphistry_mcp_server.server import mcp

# Load environment variables from .env file
load_dotenv()

# Set up test environment
password = os.getenv("GRAPHISTRY_PASSWORD")
username = os.getenv("GRAPHISTRY_USERNAME")
if not password or not username:
    raise ValueError("GRAPHISTRY_PASSWORD and GRAPHISTRY_USERNAME must be set in .env file")

print(f"\nInitializing Graphistry with username: {username}")
graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username=username, password=password)

def parse_mcp_response(response) -> Dict[str, Any]:
    """Parse MCP response into a dictionary."""
    if isinstance(response, list) and len(response) > 0:
        content = response[0]
        if hasattr(content, 'text'):
            return json.loads(content.text)
    return response

@pytest.mark.asyncio
async def test_list_tools() -> None:
    """Test that list_tools returns a list of valid tool definitions."""
    print("\nTesting list_tools...")
    tools = await mcp.list_tools()
    print(f"Found {len(tools)} tools")
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check that each tool has the required structure
    for tool in tools:
        print(f"Checking tool: {tool.name}")
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'inputSchema')
    print("✓ list_tools test passed")

@pytest.mark.asyncio
async def test_visualize_graph() -> None:
    """Test that the visualize_graph tool works with basic parameters."""
    print("\nTesting visualize_graph...")
    # Create a simple test graph
    test_params = {
        "data_format": "edge_list",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"}
        ],
        "title": "Test Triangle Graph"
    }
    print(f"Creating graph with params: {test_params}")

    result = await mcp.call_tool("visualize_graph", test_params)
    result_dict = parse_mcp_response(result)
    print(f"Got result: {result_dict}")
    assert isinstance(result_dict, dict)
    assert "graph_id" in result_dict
    assert "title" in result_dict
    assert "url" in result_dict
    print("✓ visualize_graph test passed")

@pytest.mark.asyncio
async def test_get_graph_info() -> None:
    """Test that get_graph_info returns valid information about a graph."""
    print("\nTesting get_graph_info...")
    # First create a graph
    test_params = {
        "data_format": "edge_list",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"}
        ]
    }
    print("Creating test graph...")
    graph_result = await mcp.call_tool("visualize_graph", test_params)
    graph_result_dict = parse_mcp_response(graph_result)
    graph_id = graph_result_dict["graph_id"]
    print(f"Created graph with ID: {graph_id}")
    assert graph_id is not None

    # Now get info about the graph
    print("Getting graph info...")
    info_result = await mcp.call_tool("get_graph_info", {"graph_id": graph_id})
    info_result_dict = parse_mcp_response(info_result)
    print(f"Got graph info: {info_result_dict}")
    assert isinstance(info_result_dict, dict)
    assert "node_count" in info_result_dict
    assert "edge_count" in info_result_dict
    assert info_result_dict["edge_count"] == 2
    print("✓ get_graph_info test passed")

@pytest.mark.asyncio
async def test_apply_layout() -> None:
    """Test that apply_layout successfully changes the graph layout."""
    print("\nTesting apply_layout...")
    # First create a graph
    test_params = {
        "data_format": "edge_list",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "D"}
        ]
    }
    print("Creating test graph...")
    graph_result = await mcp.call_tool("visualize_graph", test_params)
    graph_result_dict = parse_mcp_response(graph_result)
    graph_id = graph_result_dict["graph_id"]
    print(f"Created graph with ID: {graph_id}")
    assert graph_id is not None

    # Now apply a layout
    print("Applying force-directed layout...")
    layout_result = await mcp.call_tool("apply_layout", {
        "graph_id": graph_id,
        "layout": "force_directed"
    })
    layout_result_dict = parse_mcp_response(layout_result)
    print(f"Got layout result: {layout_result_dict}")
    assert isinstance(layout_result_dict, dict)
    assert "graph_id" in layout_result_dict
    assert "url" in layout_result_dict
    print("✓ apply_layout test passed")

@pytest.mark.asyncio
async def test_detect_patterns() -> None:
    """Test that detect_patterns can analyze graph patterns."""
    print("\nTesting detect_patterns...")
    # First create a graph
    test_params = {
        "data_format": "edge_list",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},
            {"source": "D", "target": "E"}
        ]
    }
    print("Creating test graph...")
    graph_result = await mcp.call_tool("visualize_graph", test_params)
    graph_result_dict = parse_mcp_response(graph_result)
    graph_id = graph_result_dict["graph_id"]
    print(f"Created graph with ID: {graph_id}")
    assert graph_id is not None

    # Now detect patterns
    print("Running centrality analysis...")
    pattern_result = await mcp.call_tool("detect_patterns", {
        "graph_id": graph_id,
        "analysis_type": "centrality"
    })
    pattern_result_dict = parse_mcp_response(pattern_result)
    print(f"Got pattern analysis results: {pattern_result_dict}")
    assert isinstance(pattern_result_dict, dict)
    assert "degree_centrality" in pattern_result_dict
    assert "betweenness_centrality" in pattern_result_dict
    assert "closeness_centrality" in pattern_result_dict
    print("✓ detect_patterns test passed")