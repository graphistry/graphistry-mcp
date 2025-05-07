"""
Graphistry MCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
"""

import logging
from typing import Any, Dict, Optional
import os

from dotenv import load_dotenv
load_dotenv()

import graphistry
import pandas as pd
import networkx as nx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context

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

if __name__ == "__main__":
    mcp.run()