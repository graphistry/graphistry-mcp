"""
Graphistry MCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import os

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
graph_cache = {}

# Debug: Print environment variables for Graphistry
print(f"[DEBUG] GRAPHISTRY_USERNAME: {os.environ.get('GRAPHISTRY_USERNAME')}")
print(f"[DEBUG] GRAPHISTRY_PASSWORD: {os.environ.get('GRAPHISTRY_PASSWORD')}")

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
        graph_data (dict): Dictionary describing the graph to visualize. Fields:
            - data_format (str, required): Format of the input data. One of:
                * "edge_list": Use with 'edges' (list of {source, target}) and optional 'nodes' (list of {id, ...})
                * "pandas": Use with 'edges' (list of dicts), 'source' (str), 'destination' (str), and optional 'node_id' (str)
                * "networkx": Use with 'edges' as a networkx.Graph object
            - edges (list, required for edge_list/pandas): List of edges, each as a dict with at least 'source' and 'target' keys (e.g., [{"source": "A", "target": "B"}, ...])
            - nodes (list, optional): List of nodes, each as a dict with at least 'id' key (e.g., [{"id": "A"}, ...])
            - node_id (str, optional): Column name for node IDs (for pandas format)
            - source (str, optional): Column name for edge source (for pandas format)
            - destination (str, optional): Column name for edge destination (for pandas format)
            - title (str, optional): Title for the visualization
            - description (str, optional): Description for the visualization
        ctx: MCP context for progress reporting

    Example:
        graph_data = {
            "data_format": "edge_list",
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "A", "target": "C"},
                ...
            ],
            "nodes": [
                {"id": "A"}, {"id": "B"}, {"id": "C"}
            ],
            "title": "My Graph",
            "description": "A simple example graph."
        }
    """
    try:
        if ctx:
            await ctx.info("Initializing graph visualization...")

        data_format = graph_data.get("data_format")
        edges = graph_data.get("edges")
        nodes = graph_data.get("nodes")
        node_id = graph_data.get("node_id")
        source = graph_data.get("source")
        destination = graph_data.get("destination")
        title = graph_data.get("title")
        description = graph_data.get("description")

        # Handle different input formats
        if data_format == "edge_list":
            if not edges:
                raise ValueError("edges list required for edge_list format")
            df = pd.DataFrame(edges)
            # Ensure source and target columns exist
            if "source" not in df.columns or "target" not in df.columns:
                raise ValueError("edges must contain 'source' and 'target' columns")
            g = graphistry.bind(source="source", destination="target").edges(df)
        elif data_format == "pandas":
            if not (source and destination):
                raise ValueError("source and destination column names required for pandas format")
            df = pd.DataFrame(edges)
            g = graphistry.bind(source=source, destination=destination)
            if node_id:
                g = g.bind(node=node_id)
            g = g.edges(df)
        elif data_format == "networkx":
            g = graphistry.bind().from_networkx(edges)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
        # Generate unique ID and cache
        graph_id = f"graph_{len(graph_cache)}"
        graph_cache[graph_id] = {
            "graph": g,
            "title": title,
            "description": description,
            "edges_df": df if data_format in ["edge_list", "pandas"] else None,
            "nx_graph": edges if data_format == "networkx" else None
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
    """Get information about a stored graph visualization.
    
    Args:
        graph_id: ID of the graph to retrieve information for
    """
    try:
        if graph_id not in graph_cache:
            raise ValueError(f"Graph not found: {graph_id}")

        graph_data = graph_cache[graph_id]
        g = graph_data["graph"]
        edges_df = graph_data["edges_df"]
        nx_graph = graph_data["nx_graph"]

        # Get node and edge counts
        if edges_df is not None:
            node_count = len(set(edges_df["source"].unique()) | set(edges_df["target"].unique()))
            edge_count = len(edges_df)
        elif nx_graph is not None:
            node_count = len(nx_graph.nodes())
            edge_count = len(nx_graph.edges())
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
async def detect_patterns(graph_id: str, 
                        analysis_type: str,
                        options: Optional[Dict[str, Any]] = None,
                        ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Identify patterns, communities, and anomalies within graphs.
    
    Args:
        graph_id: ID of the graph to analyze
        analysis_type: Type of pattern analysis to perform
        options: Additional options for the analysis
        ctx: MCP context for progress reporting
    """
    try:
        if graph_id not in graph_cache:
            raise ValueError(f"Graph not found: {graph_id}")

        if ctx:
            await ctx.info(f"Starting {analysis_type} analysis...")

        graph_data = graph_cache[graph_id]
        nx_graph = graph_data["nx_graph"]
        edges_df = graph_data["edges_df"]

        # Convert to NetworkX graph if needed
        if nx_graph is None and edges_df is not None:
            nx_graph = nx.from_pandas_edgelist(edges_df, source="source", target="target")

        if nx_graph is None:
            raise ValueError("Graph data not available for analysis")

        results = {}
        if analysis_type == "community_detection":
            # Implement community detection
            pass
        elif analysis_type == "centrality":
            results["degree_centrality"] = nx.degree_centrality(nx_graph)
            results["betweenness_centrality"] = nx.betweenness_centrality(nx_graph)
            results["closeness_centrality"] = nx.closeness_centrality(nx_graph)
        elif analysis_type == "path_finding":
            if not options or "source" not in options or "target" not in options:
                raise ValueError("source and target nodes required for path finding")
            try:
                path = nx.shortest_path(nx_graph, options["source"], options["target"])
                results["shortest_path"] = path
                results["path_length"] = len(path) - 1
            except nx.NetworkXNoPath:
                results["error"] = "No path exists between specified nodes"
        elif analysis_type == "anomaly_detection":
            # Implement anomaly detection
            pass
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
        if ctx:
            await ctx.info("Analysis complete!")

        return results
    except Exception as e:
        logger.error(f"Error in detect_patterns: {e}")
        raise

if __name__ == "__main__":
    mcp.run()