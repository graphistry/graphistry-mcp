from enum import Enum
# Record start time globally
import time
SERVER_START_TIME = time.time()
"""
Graphistry FastMCP Server implementation.

This server provides MCP integration for Graphistry's graph visualization platform,
enabling streamable HTTP connections and resumable workflows.

The server focuses on advanced graph insights and investigations, supporting
network analysis, threat detection, and pattern discovery through Graphistry's
GPU-accelerated visualization capabilities.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal, Callable
from pathlib import Path
from functools import lru_cache, wraps

try:
    import pandas as pd
except ImportError:
    import subprocess
    print("Installing pandas via uvx...")
    subprocess.check_call(["uvx", "pip", "install", "pandas"])
    import pandas as pd

try:
    import networkx as nx
except ImportError:
    import subprocess
    print("Installing networkx via uvx...")
    subprocess.check_call(["uvx", "pip", "install", "networkx"])
    import networkx as nx

try:
    from fastmcp import FastMCP
except ImportError:
    import subprocess
    print("Installing fastmcp>=2.2.6 via uvx...")
    subprocess.check_call(["uvx", "pip", "install", "fastmcp>=2.2.6"])
    from fastmcp import FastMCP

try:
    from pydantic import Field, BaseModel
except ImportError:
    import subprocess
    print("Installing pydantic via uvx...")
    subprocess.check_call(["uvx", "pip", "install", "pydantic"])
    from pydantic import Field, BaseModel

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    # Check multiple possible locations for .env file
    possible_paths = [
        Path(".") / ".env",                     # Current directory
        Path("..") / ".env",                    # Parent directory
        Path(__file__).parent.parent.parent / ".env"  # Project root
    ]
    
    loaded = False
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path.absolute()}")
            loaded = True
            break
    
    if not loaded:
        print("No .env file found in any of the searched locations.")
except ImportError:
    import subprocess
    print("Installing python-dotenv via uvx...")
    subprocess.check_call(["pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    
    # Check multiple possible locations for .env file
    possible_paths = [
        Path(".") / ".env",                     # Current directory
        Path("..") / ".env",                    # Parent directory
        Path(__file__).parent.parent.parent / ".env"  # Project root
    ]
    
    loaded = False
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path.absolute()}")
            loaded = True
            break
    
    if not loaded:
        print("No .env file found in any of the searched locations.")

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize graphistry client
try:
    # First try to import real graphistry
    import graphistry
    HAS_GRAPHISTRY = True
    logger.info("Graphistry package available")
    
    # Check for graphistry credentials in environment
    import os
    GRAPHISTRY_USERNAME = os.environ.get("GRAPHISTRY_USERNAME")
    GRAPHISTRY_PASSWORD = os.environ.get("GRAPHISTRY_PASSWORD")
    
    try:
        # Try to use real graphistry with credentials
        if GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD:
            try:
                graphistry.register(
                    api=3,
                    protocol="https",
                    server="hub.graphistry.com",
                    username=GRAPHISTRY_USERNAME,
                    password=GRAPHISTRY_PASSWORD
                )
                logger.info("✅ Graphistry client registered successfully with credentials")
                USE_MOCK = False
            except Exception as e:
                logger.warning(f"❌ Failed to register Graphistry client: {str(e)}")
                logger.warning("Please check your Graphistry credentials and ensure your account is active")
                logger.warning("Falling back to mock Graphistry implementation for development")


    except Exception as mock_error:
        logger.error(f"❌ Failed to set up either real or mock Graphistry: {str(mock_error)}")
        
except ImportError:
    import subprocess
    print("Installing graphistry via uvx...")
    subprocess.check_call(["uvx", "pip", "install", "graphistry"])
    import graphistry
    HAS_GRAPHISTRY = True
    
    # After installing, try again with credentials
    GRAPHISTRY_USERNAME = os.environ.get("GRAPHISTRY_USERNAME")
    GRAPHISTRY_PASSWORD = os.environ.get("GRAPHISTRY_PASSWORD")
    
    try:
        if GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD:
            try:
                graphistry.register(
                    api=3,
                    protocol="https",
                    server="hub.graphistry.com",
                    username=GRAPHISTRY_USERNAME,
                    password=GRAPHISTRY_PASSWORD
                )
                logger.info("✅ Graphistry client registered successfully with credentials")
            except Exception as e:
                logger.warning(f"❌ Failed to register Graphistry client: {str(e)}")
                logger.warning("Falling back to mock Graphistry implementation for development")
    
    except Exception as mock_error:
        logger.error(f"❌ Failed to set up either real or mock Graphistry: {str(mock_error)}")


# Define version
__version__ = "0.2.0"

# Initialize the FastMCP server
mcp = FastMCP(
    name="Graphistry Graph Visualization",
    dependencies=[
        "graphistry",
        "pandas",
        "networkx",
        "python-louvain",
        "uvicorn",
        "pydantic",
    ],
    
    # Optimize performance with increased concurrency
    n_workers=4,
    
    # Add detailed description
    description="GPU-accelerated graph visualization and analysis tools powered by Graphistry"
)

# Initialize optimized in-memory graph cache with LRU eviction

# Maximum size of cache - adjust based on memory requirements
MAX_CACHE_SIZE = 100

class GraphCache:
    def __init__(self, max_size=MAX_CACHE_SIZE):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._get_graph = lru_cache(maxsize=max_size)(self._get_graph_impl)
        self._max_size = max_size
        self._access_order = []
        
    def __getitem__(self, key):
        if key in self.cache:
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self.cache[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        # If cache is full, evict least recently used item
        if len(self.cache) >= self._max_size and key not in self.cache:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self.cache.pop(oldest_key, None)
        
        # Add to cache and update access order
        self.cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def __contains__(self, key):
        return key in self.cache
    
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _get_graph_impl(self, graph_id):
        """Cached graph retrieval implementation"""
        return self.cache.get(graph_id, {}).get("graph", None)
    
    def get_graph(self, graph_id):
        """Get a graph with caching"""
        return self._get_graph(graph_id)
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self._access_order.clear()
        self._get_graph.cache_clear()

# Initialize the graph cache
graph_cache = GraphCache(max_size=MAX_CACHE_SIZE)


class GraphFormat(str, Enum):
    """The format of the input graph data."""
    PANDAS = "pandas"
    NETWORKX = "networkx"
    EDGE_LIST = "edge_list"


class Node(BaseModel):
    """A node in a graph with optional attributes."""
    id: str
    attrs: Dict[str, Any] = Field(default_factory=dict, 
                                  description="Additional node attributes")


class Edge(BaseModel):
    """An edge in a graph with optional attributes."""
    source: str
    target: str
    attrs: Dict[str, Any] = Field(default_factory=dict,
                                 description="Additional edge attributes")


class LayoutOptions(TypedDict, total=False):
    """Options for layout algorithms."""
    play: int
    strongGravity: bool
    layout: str


# Tools Implementation

@mcp.tool()
def visualize_graph(
    data_format: Annotated[GraphFormat, 
                          Field(description="The format of the input data")],
    nodes: Annotated[Optional[List[Node]], 
                    Field(default=None, 
                          description="List of nodes (required for edge_list format)")] = None,
    edges: Annotated[Optional[List[Edge]], 
                    Field(default=None, 
                          description="List of edges (required for edge_list format)")] = None,
    node_id: Annotated[Optional[str], 
                      Field(default=None, 
                            description="Column name for node IDs (for pandas format)")] = None,
    source: Annotated[Optional[str], 
                     Field(default=None, 
                           description="Column name for edge source (for pandas format)")] = None,
    destination: Annotated[Optional[str], 
                          Field(default=None, 
                                description="Column name for edge destination (for pandas format)")] = None,
    title: Annotated[Optional[str], 
                    Field(default="Graph Visualization", 
                          description="Title for the visualization")] = "Graph Visualization",
    description: Annotated[Optional[str], 
                          Field(default="", 
                                description="Description for the visualization")] = "",
) -> Dict[str, Any]:
    """
    Visualize a graph using Graphistry's GPU-accelerated renderer for pattern discovery 
    and relationship analysis.
    """
    logger.info(f"Visualizing graph in {data_format} format")
    
    g: graphistry.Plottable = graphistry.bind()
    
    if data_format == "pandas":
        # Convert edge data to pandas DataFrame
        if edges:
            edges_data = [{"source": e.source, "target": e.target, **e.attrs} for e in edges]
            edges_df = pd.DataFrame(edges_data)
            src = source or "source"
            dst = destination or "target"
            g = g.edges(edges_df, source=src, destination=dst)
        
        # Add nodes if provided
        if nodes:
            nodes_data = [{"id": n.id, **n.attrs} for n in nodes]
            nodes_df = pd.DataFrame(nodes_data)
            nid = node_id or "id"
            g = g.nodes(nodes_df, nid)
    
    elif data_format == "networkx":
        # Create a NetworkX graph from provided nodes and edges
        nx_graph = nx.Graph()
        
        if nodes:
            for node in nodes:
                nx_graph.add_node(node.id, **node.attrs)
        
        if edges:
            for edge in edges:
                nx_graph.add_edge(edge.source, edge.target, **edge.attrs)
        
        g = g.networkx(nx_graph)
    
    elif data_format == "edge_list":
        # Create from simple edge list
        if not edges:
            raise ValueError("Edge list format requires edges parameter")
            
        edges_data = [{"source": e.source, "target": e.target, **e.attrs} for e in edges]
        edges_df = pd.DataFrame(edges_data)
        g = g.edges(edges_df)
    
    else:
        # This should be caught by pydantic but we'll include a check anyway
        raise ValueError(f"Unsupported data format: {data_format}")
    
    # Apply default settings
    g = g.settings(url_params={'play': 5000, 'strongGravity': True})
    
    # Generate a unique ID for this graph
    graph_id = str(uuid.uuid4())
    
    # Store the graph in our cache
    graph_cache[graph_id] = {
        "graph": g,
        "title": title,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "url": g.plot(render=False)
    }
    
    return {
        "graph_id": graph_id,
        "title": title,
        "description": description,
        "url": graph_cache[graph_id]["url"],
        "message": "Graph visualization created successfully"
    }


@mcp.tool()
@lru_cache(maxsize=50)  # Cache results for better performance
def get_graph_info(
    graph_id: Annotated[str, Field(description="ID of the graph to retrieve information for")]
) -> Dict[str, Any]:
    """
    Get information about a stored graph visualization including metrics and statistics.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
    # Try to extract basic graph metrics if possible
    metrics = {}
    try:
        # If we can get a networkx graph, compute some metrics
        if hasattr(g, '_nx'):
            nx_graph = g._nx
            
            # Start with basic metrics that are fast to compute
            metrics = {
                "num_nodes": nx_graph.number_of_nodes(),
                "num_edges": nx_graph.number_of_edges(),
                "density": nx.density(nx_graph),
            }
            
            # Only compute expensive metrics for smaller graphs
            if nx_graph.number_of_nodes() < 1000:
                if nx_graph.number_of_nodes() > 0:
                    # Add connectivity check for non-empty graphs
                    try:
                        metrics["is_connected"] = nx.is_connected(nx_graph)
                    except:
                        # For directed graphs or other special cases
                        metrics["is_connected"] = False
                
                # Compute clustering only for small graphs (expensive operation)
                if nx_graph.number_of_nodes() < 500 and nx_graph.number_of_nodes() > 0:
                    metrics["average_clustering"] = nx.average_clustering(nx_graph)
                
                # Add degree statistics
                degrees = [d for _, d in nx_graph.degree()]
                if degrees:
                    metrics["min_degree"] = min(degrees)
                    metrics["max_degree"] = max(degrees)
                    metrics["avg_degree"] = sum(degrees) / len(degrees)
    except Exception as e:
        logger.warning(f"Could not compute graph metrics: {e}")
    
    return {
        "graph_id": graph_id,
        "title": graph_info["title"],
        "description": graph_info["description"],
        "created_at": graph_info["created_at"],
        "url": graph_info["url"],
        "metrics": metrics
    }


@mcp.tool()
def apply_layout(
    graph_id: Annotated[str, Field(description="ID of the graph to apply layout to")],
    layout: Annotated[str, Field(description="Layout algorithm to apply", 
                               enum=["force_directed", "radial", "circle", "grid"])]
) -> Dict[str, Any]:
    """
    Apply a layout algorithm to a graph for different analysis perspectives.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
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
    
    # Update the graph in cache
    graph_cache[graph_id] = {
        **graph_info,
        "graph": g,
        "url": g.plot(render=False)
    }
    
    return {
        "graph_id": graph_id,
        "title": graph_info["title"],
        "layout": layout,
        "url": graph_cache[graph_id]["url"],
        "message": f"Applied {layout} layout to graph"
    }


class AnalysisType(str, Enum):
    """The type of graph analysis to perform."""
    COMMUNITY_DETECTION = "community_detection"
    CENTRALITY = "centrality"
    PATH_FINDING = "path_finding"
    ANOMALY_DETECTION = "anomaly_detection"


class AnalysisOptions(BaseModel):
    """Options for graph analysis."""
    algorithm: Optional[str] = Field(default="louvain", 
                                   description="Algorithm to use for community detection")
    top_n: Optional[int] = Field(default=5, 
                               description="Number of top nodes to return for centrality")
    source: Optional[str] = Field(default=None, 
                                description="Source node for path finding")
    target: Optional[str] = Field(default=None, 
                                description="Target node for path finding")


@mcp.tool()
def detect_patterns(
    graph_id: Annotated[str, Field(description="ID of the graph to analyze")],
    analysis_type: Annotated[AnalysisType, 
                           Field(description="Type of pattern analysis to perform")],
    options: Annotated[Optional[AnalysisOptions], 
                      Field(default=None, 
                            description="Additional options for the analysis")] = None
) -> Dict[str, Any]:
    """
    Identify interesting patterns, communities, and anomalies within graphs.
    """
    if graph_id not in graph_cache:
        raise ValueError(f"Graph with ID {graph_id} not found")
    
    if options is None:
        options = AnalysisOptions()
    
    graph_info = graph_cache[graph_id]
    g = graph_info["graph"]
    
    # Extract NetworkX graph for analysis
    nx_graph = None
    if hasattr(g, '_nx'):
        nx_graph = g._nx
    else:
        # Try to get a networkx graph from edges
        try:
            edges_df = g._edges
            nx_graph = nx.from_pandas_edgelist(edges_df)
        except Exception:
            raise ValueError("Could not extract a NetworkX graph for analysis")
    
    result = {
        "graph_id": graph_id,
        "analysis_type": analysis_type,
    }
    
    # Perform the requested analysis
    if analysis_type == AnalysisType.COMMUNITY_DETECTION:
        # Detect communities using appropriate algorithm
        algorithm = options.algorithm
        if algorithm == "louvain":
            try:
                try:
                    from community import best_partition
                except ImportError:
                    import subprocess
                    print("Installing python-louvain via uvx...")
                    subprocess.check_call(["uvx", "pip", "install", "python-louvain"])
                    from community import best_partition
                
                partition = best_partition(nx_graph)
                # Convert partition to a format for visualization
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)
                
                # Count nodes in each community
                community_sizes = {comm_id: len(nodes) for comm_id, nodes in communities.items()}
                
                result["communities"] = communities
                result["community_sizes"] = community_sizes
                result["num_communities"] = len(communities)
                
                # Update graph visualization with community colors
                node_attrs = {node: {"community": comm} for node, comm in partition.items()}
                for node, attrs in node_attrs.items():
                    if nx_graph.has_node(node):
                        nx_graph.nodes[node].update(attrs)
                
                # Update the graph in cache with community information
                g = g.networkx(nx_graph)
                graph_cache[graph_id]["graph"] = g
                graph_cache[graph_id]["url"] = g.plot(render=False)
                result["url"] = graph_cache[graph_id]["url"]
                
            except Exception as e:
                result["error"] = f"Error in community detection: {str(e)}"
        else:
            result["error"] = f"Unsupported community detection algorithm: {algorithm}"
    
    elif analysis_type == AnalysisType.CENTRALITY:
        # Calculate various centrality metrics
        try:
            # Get top N nodes by different centrality measures
            top_n = options.top_n
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(nx_graph)
            degree_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(nx_graph)
            betweenness_central_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(nx_graph)
            closeness_central_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            result["centrality"] = {
                "degree": {str(node): value for node, value in degree_central_nodes},
                "betweenness": {str(node): value for node, value in betweenness_central_nodes},
                "closeness": {str(node): value for node, value in closeness_central_nodes}
            }
            
            # Update node attributes with centrality measures
            for node in nx_graph.nodes():
                nx_graph.nodes[node]["degree_centrality"] = degree_centrality.get(node, 0)
                nx_graph.nodes[node]["betweenness_centrality"] = betweenness_centrality.get(node, 0)
                nx_graph.nodes[node]["closeness_centrality"] = closeness_centrality.get(node, 0)
            
            # Update the graph in cache with centrality information
            g = g.networkx(nx_graph)
            graph_cache[graph_id]["graph"] = g
            graph_cache[graph_id]["url"] = g.plot(render=False)
            result["url"] = graph_cache[graph_id]["url"]
            
        except Exception as e:
            result["error"] = f"Error calculating centrality: {str(e)}"
    
    elif analysis_type == AnalysisType.PATH_FINDING:
        # Find paths between nodes
        source = options.source
        target = options.target
        
        if not source or not target:
            result["error"] = "Both source and target nodes are required for path finding"
        else:
            try:
                if nx.has_path(nx_graph, source, target):
                    shortest_path = nx.shortest_path(nx_graph, source, target)
                    result["path"] = shortest_path
                    result["path_length"] = len(shortest_path) - 1
                    
                    # Highlight the path in the visualization
                    path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
                    for u, v in nx_graph.edges():
                        nx_graph.edges[u, v]["on_path"] = False
                    
                    for u, v in path_edges:
                        if nx_graph.has_edge(u, v):
                            nx_graph.edges[u, v]["on_path"] = True
                    
                    # Update the graph in cache with path information
                    g = g.networkx(nx_graph)
                    graph_cache[graph_id]["graph"] = g
                    graph_cache[graph_id]["url"] = g.plot(render=False)
                    result["url"] = graph_cache[graph_id]["url"]
                else:
                    result["path_exists"] = False
                    result["message"] = f"No path exists between {source} and {target}"
            except Exception as e:
                result["error"] = f"Error finding path: {str(e)}"
    
    elif analysis_type == AnalysisType.ANOMALY_DETECTION:
        # Detect anomalies in the graph
        try:
            # Calculate various metrics to identify anomalies
            degrees = dict(nx_graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            std_degree = (sum((d - avg_degree) ** 2 for d in degrees.values()) / len(degrees)) ** 0.5 if degrees else 0
            
            # Identify nodes with unusually high or low degree
            high_degree_threshold = avg_degree + 2 * std_degree
            low_degree_threshold = max(0, avg_degree - 2 * std_degree)
            
            high_degree_nodes = {node: deg for node, deg in degrees.items() if deg > high_degree_threshold}
            isolated_nodes = {node: deg for node, deg in degrees.items() if deg == 0}
            
            result["anomalies"] = {
                "high_degree_nodes": high_degree_nodes,
                "isolated_nodes": isolated_nodes,
                "avg_degree": avg_degree,
                "std_degree": std_degree
            }
            
            # Mark anomalous nodes in the graph
            for node in nx_graph.nodes():
                degree = degrees.get(node, 0)
                nx_graph.nodes[node]["anomaly"] = "high_degree" if degree > high_degree_threshold else ("isolated" if degree == 0 else "normal")
            
            # Update the graph in cache with anomaly information
            g = g.networkx(nx_graph)
            graph_cache[graph_id]["graph"] = g
            graph_cache[graph_id]["url"] = g.plot(render=False)
            result["url"] = graph_cache[graph_id]["url"]
            
        except Exception as e:
            result["error"] = f"Error detecting anomalies: {str(e)}"
    
    else:
        # This should be caught by pydantic but we'll include a check anyway
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    return result


# Health check endpoint
@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    Check the health of the server and return system/resource information.
    
    This tool provides comprehensive health diagnostics including:
    - Server status and uptime
    - Memory usage and system information
    - Cache utilization and performance metrics
    - Port status for HTTP mode
    - Graphistry connection status
    
    Returns:
        Dict with status, uptime, memory usage, and service information.
    """
    # Calculate uptime
    uptime_seconds = time.time() - SERVER_START_TIME
    
    # Get memory usage
    import psutil
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = {
            "rss_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # VMS in MB
            "percent": process.memory_percent()
        }
    except:
        # If psutil fails, provide basic info
        memory_usage = {"available": "psutil not installed"}
        try:
            import subprocess
            print("Installing psutil via uvx for better health monitoring...")
            subprocess.check_call(["uvx", "pip", "install", "psutil"])
        except:
            pass
    
    # Get cache info
    cache_info = {
        "size": len(graph_cache.cache),
        "max_size": graph_cache._max_size,
        "usage_percent": (len(graph_cache.cache) / graph_cache._max_size) * 100 if graph_cache._max_size > 0 else 0
    }
    
    # Get Graphistry connection status
    graphistry_status = {
        "available": HAS_GRAPHISTRY,
        "credentials_configured": bool(GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD)
    }
    
    # Basic system info
    system_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "server_pid": os.getpid()
    }
    
    # Check port status for HTTP mode
    port_info = {
        "http_mode": len(sys.argv) > 1 and sys.argv[1] == "--http",
        "default_port": 8080
    }
    
    if port_info["http_mode"]:
        # Get the actual port in use
        if len(sys.argv) > 2:
            try:
                port_info["current_port"] = int(sys.argv[2])
            except (ValueError, IndexError):
                port_info["current_port"] = 8080
        else:
            port_info["current_port"] = 8080
            
        # Check if default port is available
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 8080))
                port_info["default_port_available"] = True
                s.close()
        except OSError:
            port_info["default_port_available"] = False
    
    return {
        "status": "healthy",
        "version": __version__,
        "uptime_seconds": uptime_seconds,
        "memory_usage": memory_usage,
        "cache_info": cache_info,
        "graphistry_status": graphistry_status,
        "system_info": system_info,
        "port_info": port_info,
        "timestamp": datetime.now().isoformat()
    }


def main() -> None:
    """
    Run the FastMCP server in the appropriate mode based on command line arguments.
    """
    # Record start time for uptime tracking in global scope
    global SERVER_START_TIME
    
    # Use at module level
    SERVER_START_TIME = globals().get('SERVER_START_TIME', time.time())
    
    # Check Graphistry registration status before starting server
    if HAS_GRAPHISTRY:
        if GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD:
            logger.info("Starting server with Graphistry credentials configured")
        else:
            logger.warning("⚠️  Starting server WITHOUT Graphistry credentials")
            logger.warning("For full functionality, please sign up at https://hub.graphistry.com")
            logger.warning("and set GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD environment variables")
    else:
        logger.warning("⚠️  Server starting without Graphistry package installed")
        logger.warning("Visualization capabilities will be limited")
    
    # Determine server mode from command line arguments
    import sys
    import socket
    
    def is_port_in_use(port):
        """Check if a port is already in use by attempting to bind to it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return False
            except socket.error:
                return True
    
    def find_available_port(start_port, max_attempts=20):
        """Find an available port starting from start_port."""
        for port_offset in range(max_attempts):
            port = start_port + port_offset
            if not is_port_in_use(port):
                return port
        return None

    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        port = 8080
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                logger.error(f"Invalid port: {sys.argv[2]}, using default 8080")
        
        # Check if port is already in use
        if is_port_in_use(port):
            # Try to use the suggested port from environment if available
            suggested_port = os.environ.get("GRAPHISTRY_SUGGESTED_PORT")
            if suggested_port and suggested_port.isdigit():
                alt_port = int(suggested_port)
                if not is_port_in_use(alt_port):
                    logger.info(f"Port {port} is in use, switching to alternative port {alt_port}")
                    print(f"Port {port} is in use, switching to alternative port {alt_port}", file=sys.stderr)
                    port = alt_port
                else:
                    # Find another available port
                    alt_port = find_available_port(8081)
                    if alt_port:
                        logger.info(f"Port {port} is in use, switching to alternative port {alt_port}")
                        print(f"Port {port} is in use, switching to alternative port {alt_port}", file=sys.stderr)
                        port = alt_port
                    else:
                        logger.error(f"Port {port} is already in use and no alternative ports are available")
                        print(f"❌ Error: Port {port} is already in use and no alternative ports are available", file=sys.stderr)
                        print(f"Please stop the process using port {port} or specify a different port", file=sys.stderr)
                        sys.exit(1)
            else:
                # Find an available port
                alt_port = find_available_port(8081)
                if alt_port:
                    logger.info(f"Port {port} is in use, switching to alternative port {alt_port}")
                    print(f"Port {port} is in use, switching to alternative port {alt_port}", file=sys.stderr)
                    port = alt_port
                else:
                    logger.error(f"Port {port} is already in use and no alternative ports are available")
                    print(f"❌ Error: Port {port} is already in use and no alternative ports are available", file=sys.stderr)
                    print(f"Please stop the process using port {port} or specify a different port", file=sys.stderr)
                    sys.exit(1)
        
        logger.info(f"Starting Graphistry FastMCP server on HTTP port {port}")
        try:
            import uvicorn
        except ImportError:
            import subprocess
            print("Installing uvicorn via uvx...")
            subprocess.check_call(["uvx", "pip", "install", "uvicorn"])
            import uvicorn
            
        # Print startup information
        logger.info(f"✨ Server ready at http://localhost:{port}")
        print(f"✨ Server started successfully on port {port}", file=sys.stderr)
        print(f"✨ Server URL: http://localhost:{port}", file=sys.stderr)
        # Use public API to get tool count, avoiding _tools attribute
        tool_count = len([t for t in dir(mcp) if callable(getattr(mcp, t)) and not t.startswith('_')])
        logger.info(f"📊 GraphistryMCP v{__version__} initialized with {tool_count} tools")
        
        # Run with optimized settings
        # Use FastMCP's preferred HTTP method
        if hasattr(mcp, 'get_app'):
            app = mcp.get_app()
            
            # Use single worker mode to avoid import string requirement
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=port,
                log_level="info",
                timeout_keep_alive=65
            )
        elif hasattr(mcp, 'sse_app'):
            app = mcp.sse_app
            
            # Use single worker mode to avoid import string requirement
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=port,
                log_level="info",
                timeout_keep_alive=65
            )
        elif hasattr(mcp, 'run_http'):
            # Some versions of FastMCP use run_http method
            mcp.run_http(host="0.0.0.0", port=port)
        else:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            
            # Create a minimal FastAPI wrapper if no HTTP method available
            app = FastAPI(title="Graphistry MCP HTTP Bridge")
            
            @app.post("/mcp")
            async def http_bridge(request: Request):
                data = await request.json()
                input_data = data.get("input", {})
                try:
                    result = await mcp.run(input_data)
                    return JSONResponse(content=result)
                except Exception as e:
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(e)}
                    )
                    
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=port,
                log_level="info",
                timeout_keep_alive=65
            )
    else:
        # Default mode uses stdio transport
        logger.info("Starting Graphistry FastMCP server with stdio transport")
        # Use public API to get tool count, avoiding _tools attribute
        tool_count = len([t for t in dir(mcp) if callable(getattr(mcp, t)) and not t.startswith('_')])
        logger.info(f"📊 GraphistryMCP v{__version__} initialized with {tool_count} tools")
        
        # Check which transport method is available and use it
        if hasattr(mcp, 'run_stdio_async'):
            logger.info("Using run_stdio_async transport method")
            asyncio.run(mcp.run_stdio_async())
        elif hasattr(mcp, 'run'):
            logger.info("Using run transport method")
            asyncio.run(mcp.run())
        else:
            logger.error("No valid transport method found on mcp object")
            print("Error: No valid transport method found on mcp object", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()