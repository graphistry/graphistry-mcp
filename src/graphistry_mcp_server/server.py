"""
Graphistry MCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
"""

import os
import logging
from typing import Any, Dict, List, Optional
import graphistry
import pandas as pd
import networkx as nx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("graphistry-mcp-server")

# Initialize state
graph_cache: Dict[str, Any] = {}

# Debug: Print environment variables for Graphistry
print(f"[DEBUG] GRAPHISTRY_USERNAME is set: {os.environ.get('GRAPHISTRY_USERNAME') is not None}")
print(f"[DEBUG] GRAPHISTRY_PASSWORD is set: {os.environ.get('GRAPHISTRY_PASSWORD') is not None}")

# Register Graphistry client with credentials and API version 3
print("[DEBUG] Calling graphistry.register() with api=3, protocol=https, server=hub.graphistry.com")
graphistry.register(
    api=3,
    protocol="https",
    server="hub.graphistry.com",
    username=os.environ.get("GRAPHISTRY_USERNAME"),
    password=os.environ.get("GRAPHISTRY_PASSWORD")
)
print("[DEBUG] graphistry.register() call complete")

@mcp.tool()
async def visualize_graph(graph_data: Dict[str, Any], ctx: Optional[Context] = None) -> Dict[str, Any]:
    """
    Visualize a graph using Graphistry's GPU-accelerated renderer.

    Args:
        graph_type (str, optional): Type of graph to visualize. Must be one of "graph" (two-way edges, default), "hypergraph" (many-to-many edges).
        graph_data (dict): Dictionary describing the graph to visualize. Fields:
            - edges (list, required): List of edges, each as a dict with at least 'source' and 'target' keys (e.g., [{"source": "A", "target": "B"}, ...]) and any other columns you want to include in the edge table
            - nodes (list, optional): List of nodes, each as a dict with at least 'id' key (e.g., [{"id": "A"}, ...]) and any other columns you want to include in the node table
            - node_id (str, optional): Column name for node IDs, if nodes are provided, must be provided.
            - source (str, optional): Column name for edge source (default: "source")
            - destination (str, optional): Column name for edge destination (default: "target")
            - columns (list, optional): List of column names for hypergraph edge table, use if graph_type is hypergraph.
            - title (str, optional): Title for the visualization
            - description (str, optional): Description for the visualization
        ctx: MCP context for progress reporting

    Example (graph):
        graph_data = {
            "graph_type": "graph",
            "edges": [
                {"source": "A", "target": "B", "weight": 1},
                {"source": "A", "target": "C", "weight": 2},
                ...
            ],
            "nodes": [
                {"id": "A", "label": "Node A"},
                {"id": "B", "label": "Node B"},
                ...
            ],
            "node_id": "id",
            "source": "source",
            "destination": "target",
            "title": "My Graph",
            "description": "A simple example graph."
        }

    Example (hypergraph):
        graph_data = {
            "graph_type": "hypergraph",
            "edges": [
                {"source": "A", "target": "B", "group": "G1", "weight": 1},
                {"source": "A", "target": "C", "group": "G1", "weight": 1},
                ...
            ],
            "columns": ["source", "target", "group"],
            "title": "My Hypergraph",
            "description": "A simple example hypergraph."
        }
    """
    try:
        if ctx:
            await ctx.info("Initializing graph visualization...")

        graph_type = graph_data.get("graph_type") or "graph"
        edges = graph_data.get("edges")
        nodes = graph_data.get("nodes")
        node_id = graph_data.get("node_id")
        source = graph_data.get("source") or "source"
        destination = graph_data.get("destination") or "target"
        title = graph_data.get("title")
        description = graph_data.get("description")
        columns = graph_data.get("columns", None)

        g = None
        edges_df = None
        nodes_df = None

        if graph_type == "graph":
            if not edges:
                raise ValueError("edges list required for edge_list format")
            edges_df = pd.DataFrame(edges)
            if nodes:
                nodes_df = pd.DataFrame(nodes)
                g = graphistry.edges(edges_df, source=source, destination=destination).nodes(nodes_df, node=node_id)
            else:
                g = graphistry.edges(edges_df, source=source, destination=destination)
            nx_graph = nx.from_pandas_edgelist(edges_df, source=source, target=destination)
        elif graph_type == "hypergraph":
            if not edges:
                raise ValueError("edges list required for hypergraph format")
            edges_df = pd.DataFrame(edges)
            g = graphistry.hypergraph(edges_df, columns)['graph']
            nx_graph = None
        else:
            raise ValueError(f"Unsupported graph_type: {graph_type}")
        g = g.name(title)
        # Generate unique ID and cache
        graph_id = f"graph_{len(graph_cache)}"
        graph_cache[graph_id] = {
            "graph": g,
            "title": title,
            "description": description,
            "edges_df": edges_df,
            "nodes_df": nodes_df,
            "node_id": node_id,
            "source": source,
            "destination": destination,
            "nx_graph": nx_graph
        }
    
        if ctx:
            await ctx.info("Graph visualization complete!")

        return {
            "graph_id": graph_id,
            "title": title,
            "url": g.plot(render=False)
        }
    except Exception as e:
        logger.error(f"Error in visualize_graph: {e}")
        raise

@mcp.tool()
async def get_graph_ids() -> List[str]:
    """Get a list of all stored graph IDs."""
    return list(graph_cache.keys())

@mcp.tool()
async def get_graph_info(graph_id: str) -> Dict[str, Any]:
    """Get information about a stored graph visualization."""
    try:
        if graph_id not in graph_cache:
            raise ValueError(f"Graph not found: {graph_id}")

        graph_data = graph_cache[graph_id]
        edges_df = graph_data["edges_df"]
        source = graph_data["source"]
        destination = graph_data["destination"]

        # Get node and edge counts
        if edges_df is not None:
            node_count = len(set(edges_df[source].unique()) | set(edges_df[destination].unique()))
            edge_count = len(edges_df)
        else:
            node_count = 0
            edge_count = 0

        return {
            "graph_id": graph_id,
            "title": graph_data["title"],
            "description": graph_data["description"],
            "node_count": node_count,
            "edge_count": edge_count
        }
    except Exception as e:
        logger.error(f"Error in get_graph_info: {e}")
        raise

@mcp.tool()
async def apply_layout(graph_id: str, layout: str) -> Dict[str, Any]:
    """Apply a layout algorithm to a graph.
    
    Args:
        graph_id: ID of the graph to apply layout to
        layout: Layout algorithm to apply (force_directed, radial, circle, grid)
    """
    try:
        if graph_id not in graph_cache:
            raise ValueError(f"Graph not found: {graph_id}")

        graph_data = graph_cache[graph_id]
        g = graph_data["graph"]

        # Apply layout using Graphistry's url_params settings
        if layout == "force_directed":
            g = g.settings(url_params={'play': 5000, 'strongGravity': True})
        elif layout == "radial":
            g = g.settings(url_params={'play': 0, 'layout': 'radial'})
        elif layout == "circle":
            g = g.settings(url_params={'play': 0, 'layout': 'circle'})
        elif layout == "grid":
            g = g.settings(url_params={'play': 0, 'layout': 'grid'})
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
        graph_cache[graph_id]["graph"] = g
    
        return {
            "graph_id": graph_id,
            "url": g.plot(render=False)
        }
    except Exception as e:
        logger.error(f"Error in apply_layout: {e}")
        raise

@mcp.tool()
async def detect_patterns(graph_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """
    Identify patterns, communities, and anomalies within graphs. Runs all supported analyses and returns a combined report.

    Args:
        graph_id: ID of the graph to analyze
        ctx: MCP context for progress reporting

    Returns:
        Dictionary with results from all analyses that succeeded. Keys may include:
            - degree_centrality
            - betweenness_centrality
            - closeness_centrality
            - communities (if community detection is available)
            - shortest_path (if path finding is possible)
            - path_length
            - anomalies (if anomaly detection is available)
            - errors (dict of analysis_type -> error message)

    """
    try:
        if graph_id not in graph_cache:
            raise ValueError(f"Graph not found: {graph_id}")

        if ctx:
            await ctx.info("Starting pattern detection (all analyses)...")

        graph_data = graph_cache[graph_id]
        nx_graph = graph_data["nx_graph"]
        edges_df = graph_data["edges_df"]
        source = graph_data["source"]
        destination = graph_data["destination"]

        # Convert to NetworkX graph if needed
        if nx_graph is None and edges_df is not None:
            nx_graph = nx.from_pandas_edgelist(edges_df, source=source, target=destination)

        if nx_graph is None:
            raise ValueError("Graph data not available for analysis")

        results = {}
        errors = {}

        # Centrality
        try:
            results["degree_centrality"] = nx.degree_centrality(nx_graph)
            results["betweenness_centrality"] = nx.betweenness_centrality(nx_graph)
            results["closeness_centrality"] = nx.closeness_centrality(nx_graph)
        except Exception as e:
            errors["centrality"] = str(e)

        # Community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(nx_graph)
            results["communities"] = partition
        except Exception as e:
            errors["community_detection"] = str(e)

        # Path finding (try between first two nodes if possible)
        try:
            nodes = list(nx_graph.nodes())
            if len(nodes) >= 2:
                path = nx.shortest_path(nx_graph, nodes[0], nodes[1])
                results["shortest_path"] = path
                results["path_length"] = len(path) - 1
        except Exception as e:
            errors["path_finding"] = str(e)

        # Anomaly detection (placeholder)
        try:
            # Example: nodes with degree 1 as "anomalies"
            anomalies = [n for n, d in nx_graph.degree() if d == 1]
            results["anomalies"] = anomalies
        except Exception as e:
            errors["anomaly_detection"] = str(e)

        if errors:
            results["errors"] = errors

        if ctx:
            await ctx.info("Pattern detection complete!")

        return results
    except Exception as e:
        logger.error(f"Error in detect_patterns: {e}")
        raise

@mcp.tool()
async def encode_point_color(
    graph_id: str,
    column: str,
    categorical_mapping: Optional[dict] = None,
    default_mapping: Optional[str] = None,
    as_continuous: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Set node color encoding for a graph using Graphistry's encode_point_color API.

    Args:
        graph_id (str): The ID of the graph to modify (from visualize_graph).
        column (str): The node column to use for color encoding (e.g., 'type', 'score').
        categorical_mapping (dict, optional): Map of category values to color codes. Example: {'mac': '#F99', 'macbook': '#99F'}. If not provided, Graphistry will auto-assign colors.
        default_mapping (str, optional): Color code to use for values not in categorical_mapping. Example: 'silver'.
        as_continuous (bool, optional): If True, treat the column as continuous and use a gradient palette. Example: True for numeric columns like 'score'.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        encode_point_color(graph_id, column='type', categorical_mapping={'mac': '#F99', 'macbook': '#99F'}, default_mapping='silver')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    kwargs = {"column": column}
    if categorical_mapping:
        kwargs["categorical_mapping"] = categorical_mapping
    if default_mapping:
        kwargs["default_mapping"] = default_mapping
    if as_continuous:
        kwargs["as_continuous"] = as_continuous
    g = g.encode_point_color(**kwargs)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def encode_point_size(
    graph_id: str,
    column: str,
    categorical_mapping: Optional[dict] = None,
    default_mapping: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Set node size encoding for a graph using Graphistry's encode_point_size API.

    Args:
        graph_id (str): The ID of the graph to modify.
        column (str): The node column to use for size encoding (e.g., 'score', 'type').
        categorical_mapping (dict, optional): Map of category values to sizes. Example: {'mac': 50, 'macbook': 100}. If not provided, Graphistry will auto-assign sizes.
        default_mapping (float, optional): Size to use for values not in categorical_mapping. Example: 20.
        as_continuous (bool, optional): If True, treat the column as continuous and use a size gradient. Example: True for numeric columns like 'score'.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        encode_point_size(graph_id, column='score', as_continuous=True)
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    kwargs = {"column": column}
    if categorical_mapping:
        kwargs["categorical_mapping"] = categorical_mapping
    if default_mapping is not None:
        kwargs["default_mapping"] = default_mapping
    g = g.encode_point_size(**kwargs)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def encode_point_icon(
    graph_id: str,
    column: str,
    categorical_mapping: Optional[dict] = None,
    default_mapping: Optional[str] = None,
    as_text: Optional[bool] = False,
    continuous_binning: Optional[list] = None
) -> Dict[str, Any]:
    """
    Set node icon encoding for a graph using Graphistry's encode_point_icon API.

    Args:
        graph_id (str): The ID of the graph to modify.
        column (str): The node column to use for icon encoding (e.g., 'type', 'origin').
        categorical_mapping (dict, optional): Map of category values to icon names or URLs. Example: {'macbook': 'laptop', 'Canada': 'flag-icon-ca'}. See FontAwesome 4 or ISO country codes for built-ins.
        default_mapping (str, optional): Icon to use for values not in categorical_mapping. Example: 'question'.
        as_text (bool, optional): If True, use text as the icon (for continuous binning or direct text display).
        continuous_binning (list, optional): List of [threshold, icon] pairs for binning continuous values. Example: [[33, 'low'], [66, 'mid'], [None, 'high']].

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        encode_point_icon(graph_id, column='type', categorical_mapping={'macbook': 'laptop', 'Canada': 'flag-icon-ca'}, default_mapping='question')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    kwargs = {"column": column}
    if categorical_mapping:
        kwargs["categorical_mapping"] = categorical_mapping
    if default_mapping:
        kwargs["default_mapping"] = default_mapping
    if as_text:
        kwargs["as_text"] = as_text
    if continuous_binning:
        kwargs["continuous_binning"] = continuous_binning
    g = g.encode_point_icon(**kwargs)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def encode_point_badge(
    graph_id: str,
    column: str,
    position: str = "TopRight",
    categorical_mapping: Optional[dict] = None,
    default_mapping: Optional[str] = None,
    as_text: Optional[bool] = False,
    continuous_binning: Optional[list] = None
) -> Dict[str, Any]:
    """
    Set node badge encoding for a graph using Graphistry's encode_point_badge API.

    Args:
        graph_id (str): The ID of the graph to modify.
        column (str): The node column to use for badge encoding (e.g., 'type', 'origin').
        position (str, optional): Badge position on the node. Example: 'TopRight', 'BottomLeft', etc.
        categorical_mapping (dict, optional): Map of category values to badge icons or images. Example: {'macbook': 'laptop', 'Canada': 'flag-icon-ca'}.
        default_mapping (str, optional): Badge to use for values not in categorical_mapping. Example: 'question'.
        as_text (bool, optional): If True, use text as the badge (for continuous binning or direct text display).
        continuous_binning (list, optional): List of [threshold, badge] pairs for binning continuous values. Example: [[33, None], [66, 'info-circle'], [None, 'exclamation-triangle']].

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        encode_point_badge(graph_id, column='type', position='TopRight', categorical_mapping={'macbook': 'laptop'}, default_mapping='question')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    kwargs = {"column": column, "position": position}
    if categorical_mapping:
        kwargs["categorical_mapping"] = categorical_mapping
    if default_mapping:
        kwargs["default_mapping"] = default_mapping
    if as_text:
        kwargs["as_text"] = as_text
    if continuous_binning:
        kwargs["continuous_binning"] = continuous_binning
    g = g.encode_point_badge(**kwargs)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_ring_categorical_layout(graph_id: str, ring_col: str) -> Dict[str, Any]:
    """
    Apply a categorical ring layout to the graph using Graphistry's ring_categorical_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.
        ring_col (str): The node column to use for determining ring membership (e.g., a categorical attribute like 'type' or 'group').

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_ring_categorical_layout(graph_id, ring_col='type')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.ring_categorical_layout(ring_col)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_group_in_a_box_layout(graph_id: str) -> Dict[str, Any]:
    """
    Apply group-in-a-box layout to the graph using Graphistry's group_in_a_box_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_group_in_a_box_layout(graph_id)
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.group_in_a_box_layout()
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_modularity_weighted_layout(graph_id: str) -> Dict[str, Any]:
    """
    Apply modularity weighted layout to the graph using Graphistry's modularity_weighted_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_modularity_weighted_layout(graph_id)
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.modularity_weighted_layout()
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_ring_continuous_layout(graph_id: str, ring_col: str) -> Dict[str, Any]:
    """
    Apply a continuous ring layout to the graph using Graphistry's ring_continuous_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.
        ring_col (str): The node column to use for determining ring position (should be a continuous/numeric attribute, e.g., 'score').

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_ring_continuous_layout(graph_id, ring_col='score')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.ring_continuous_layout(ring_col)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_time_ring_layout(graph_id: str, time_col: str) -> Dict[str, Any]:
    """
    Apply a time ring layout to the graph using Graphistry's time_ring_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.
        time_col (str): The node column to use for determining ring position (should be a datetime or timestamp attribute, e.g., 'created_at').

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_time_ring_layout(graph_id, time_col='created_at')
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    # Ensure the time_col is datetime64 for Graphistry
    nodes_df = graph_cache[graph_id].get("nodes_df")
    if nodes_df is not None and time_col in nodes_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(nodes_df[time_col]):
            # Coerce to datetime64
            nodes_df[time_col] = pd.to_datetime(nodes_df[time_col], errors="coerce")
            # Update the graph's nodes table
            g = g.nodes(nodes_df)
    g = g.time_ring_layout(time_col)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def apply_tree_layout(graph_id: str) -> Dict[str, Any]:
    """
    Apply a tree (layered hierarchical) layout to the graph using Graphistry's tree_layout API.

    Args:
        graph_id (str): The ID of the graph to modify.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        apply_tree_layout(graph_id)
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.tree_layout()
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

@mcp.tool()
async def set_graph_settings(graph_id: str, url_params: dict) -> Dict[str, Any]:
    """
    Set visualization settings for a graph using Graphistry's settings API.

    Args:
        graph_id (str): The ID of the graph to modify.
        url_params (dict): Dictionary of Graphistry URL parameters to control visualization. Example: {'pointSize': 0.5, 'edgeInfluence': 2, 'play': 0}.

    Returns:
        dict: { 'graph_id': ..., 'url': ... } with the updated visualization URL.

    Example:
        set_graph_settings(graph_id, url_params={'pointSize': 0.5, 'play': 0})
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph not found: {graph_id}")
    g = graph_cache[graph_id]["graph"]
    g = g.settings(url_params=url_params)
    graph_cache[graph_id]["graph"] = g
    return {"graph_id": graph_id, "url": g.plot(render=False)}

if __name__ == "__main__":
    mcp.run()