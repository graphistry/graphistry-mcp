"""
Tests for the Graphistry MCP server.
"""
import json
import os
from typing import Dict, Any
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

def parse_mcp_response(response: Any) -> Dict[str, Any]:
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

    result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
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
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
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
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
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
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
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

@pytest.mark.asyncio
async def test_encode_point_color() -> None:
    print("\nTesting encode_point_color...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "type": "mac"},
            {"source": "B", "target": "C", "type": "macbook"}
        ],
        "title": "Color Test Graph"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    color_result = await mcp.call_tool("encode_point_color", {
        "graph_id": graph_id,
        "column": "type",
        "categorical_mapping": {"mac": "#F99", "macbook": "#99F"},
        "default_mapping": "silver"
    })
    color_dict = parse_mcp_response(color_result)
    print(f"encode_point_color result: {color_dict}")
    assert "graph_id" in color_dict and "url" in color_dict

@pytest.mark.asyncio
async def test_encode_point_size() -> None:
    print("\nTesting encode_point_size...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "score": 10},
            {"source": "B", "target": "C", "score": 20}
        ],
        "title": "Size Test Graph"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    size_result = await mcp.call_tool("encode_point_size", {
        "graph_id": graph_id,
        "column": "score"
    })
    size_dict = parse_mcp_response(size_result)
    print(f"encode_point_size result: {size_dict}")
    assert "graph_id" in size_dict and "url" in size_dict

@pytest.mark.asyncio
async def test_encode_point_icon() -> None:
    print("\nTesting encode_point_icon...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "type": "macbook"},
            {"source": "B", "target": "C", "type": "mac"}
        ],
        "title": "Icon Test Graph"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    icon_result = await mcp.call_tool("encode_point_icon", {
        "graph_id": graph_id,
        "column": "type",
        "categorical_mapping": {"macbook": "laptop", "mac": "desktop"},
        "default_mapping": "question"
    })
    icon_dict = parse_mcp_response(icon_result)
    print(f"encode_point_icon result: {icon_dict}")
    assert "graph_id" in icon_dict and "url" in icon_dict

@pytest.mark.asyncio
async def test_encode_point_badge() -> None:
    print("\nTesting encode_point_badge...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "type": "macbook"},
            {"source": "B", "target": "C", "type": "mac"}
        ],
        "title": "Badge Test Graph"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    badge_result = await mcp.call_tool("encode_point_badge", {
        "graph_id": graph_id,
        "column": "type",
        "position": "TopRight",
        "categorical_mapping": {"macbook": "laptop", "mac": "desktop"},
        "default_mapping": "question"
    })
    badge_dict = parse_mcp_response(badge_result)
    print(f"encode_point_badge result: {badge_dict}")
    assert "graph_id" in badge_dict and "url" in badge_dict

@pytest.mark.asyncio
async def test_apply_ring_categorical_layout() -> None:
    print("\nTesting apply_ring_categorical_layout...")
    # Graphistry expects the ring column ('group') to be present in the nodes table for ring_categorical_layout
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "group": "g1"},
            {"source": "B", "target": "C", "group": "g2"}
        ],
        "nodes": [
            {"id": "A", "group": "g1"},
            {"id": "B", "group": "g2"},
            {"id": "C", "group": "g2"}
        ],
        "node_id": "id",
        "title": "Ring Cat Layout Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_ring_categorical_layout", {
        "graph_id": graph_id,
        "ring_col": "group"
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_ring_categorical_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_apply_group_in_a_box_layout() -> None:
    print("\nTesting apply_group_in_a_box_layout...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"}
        ],
        "nodes": [
            {"id": "A"}, {"id": "B"}, {"id": "C"}
        ],
        "node_id": "id",
        "title": "Group in a Box Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_group_in_a_box_layout", {
        "graph_id": graph_id
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_group_in_a_box_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_apply_modularity_weighted_layout() -> None:
    print("\nTesting apply_modularity_weighted_layout...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"}
        ],
        "nodes": [
            {"id": "A"}, {"id": "B"}, {"id": "C"}
        ],
        "node_id": "id",
        "title": "Modularity Weighted Layout Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_modularity_weighted_layout", {
        "graph_id": graph_id
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_modularity_weighted_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_apply_ring_continuous_layout() -> None:
    print("\nTesting apply_ring_continuous_layout...")
    # Graphistry expects the ring column ('score') to be present in the nodes table for ring_continuous_layout
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "score": 1.0},
            {"source": "B", "target": "C", "score": 2.0}
        ],
        "nodes": [
            {"id": "A", "score": 1.0},
            {"id": "B", "score": 2.0},
            {"id": "C", "score": 2.0}
        ],
        "node_id": "id",
        "title": "Ring Continuous Layout Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_ring_continuous_layout", {
        "graph_id": graph_id,
        "ring_col": "score"
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_ring_continuous_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_apply_time_ring_layout() -> None:
    print("\nTesting apply_time_ring_layout...")
    # Graphistry expects the time column ('created_at') to be present in the nodes table for time_ring_layout
    test_params = {
        "edges": [
            {"source": "A", "target": "B", "created_at": "2024-01-01T00:00:00"},
            {"source": "B", "target": "C", "created_at": "2024-01-02T00:00:00"}
        ],
        "nodes": [
            {"id": "A", "created_at": "2024-01-01T00:00:00"},
            {"id": "B", "created_at": "2024-01-02T00:00:00"},
            {"id": "C", "created_at": "2024-01-02T00:00:00"}
        ],
        "node_id": "id",
        "title": "Time Ring Layout Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_time_ring_layout", {
        "graph_id": graph_id,
        "time_col": "created_at"
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_time_ring_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_apply_tree_layout() -> None:
    print("\nTesting apply_tree_layout...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"}
        ],
        "nodes": [
            {"id": "A"}, {"id": "B"}, {"id": "C"}
        ],
        "node_id": "id",
        "title": "Tree Layout Test"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    layout_result = await mcp.call_tool("apply_tree_layout", {
        "graph_id": graph_id
    })
    layout_dict = parse_mcp_response(layout_result)
    print(f"apply_tree_layout result: {layout_dict}")
    assert "graph_id" in layout_dict and "url" in layout_dict

@pytest.mark.asyncio
async def test_set_graph_settings() -> None:
    print("\nTesting set_graph_settings...")
    test_params = {
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"}
        ],
        "nodes": [
            {"id": "A"}, {"id": "B"}, {"id": "C"}
        ],
        "node_id": "id",
        "title": "Settings Test Graph"
    }
    graph_result = await mcp.call_tool("visualize_graph", {"graph_data": test_params})
    graph_id = parse_mcp_response(graph_result)["graph_id"]
    settings_result = await mcp.call_tool("set_graph_settings", {
        "graph_id": graph_id,
        "url_params": {"pointSize": 0.5, "play": 0}
    })
    settings_dict = parse_mcp_response(settings_result)
    print(f"set_graph_settings result: {settings_dict}")
    assert "graph_id" in settings_dict and "url" in settings_dict