"""
Mock Graphistry Viz MCP Server for Talk2Graph development.

Simulates the MCP server that runs inside streamgl-viz workers (PR #3043).
Use this to test Louie's Talk2Graph integration without a running Graphistry instance.

Run standalone:
    uv run python -m graphistry_mcp_server.mock_viz_server

Register as MCP connector on Louie (env var):
    MCP_SERVER_URL=http://localhost:3100/mcp

Or register via CreateConnector API event:
    {"type": "CreateConnector", "connector": {
        "type": "MCP", "name": "Talk2Graph (mock)",
        "config": {"server_url": "http://localhost:3100/mcp"}
    }}
"""

import random
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mock_mcp = FastMCP("graphistry-viz-mock")

# --- Mock session data (cybersecurity dataset) ---

MOCK_SESSIONS: Dict[str, Dict[str, Any]] = {
    "mock-session-001": {
        "session_id": "mock-session-001",
        "numNodes": 1247,
        "numEdges": 3891,
        "title": "Enterprise Network Traffic",
        "description": "Zeek conn.log data from 2026-03-15, 6-hour capture window",
        "nodeColumns": [
            "ip_address", "hostname", "country", "asn", "threat_score",
            "is_malicious", "device_type", "department", "first_seen", "last_seen",
        ],
        "edgeColumns": [
            "src_ip", "dst_ip", "protocol", "service", "bytes_sent", "bytes_recv",
            "duration", "timestamp", "conn_state", "is_encrypted",
        ],
        "active_encodings": {},
        "active_filters": [],
        "node_sample": [
            {"ip_address": "10.0.1.15", "hostname": "ws-jdoe", "country": "US", "asn": "AS64496", "threat_score": 0.12, "is_malicious": False, "device_type": "workstation", "department": "Engineering", "first_seen": "2026-03-15T08:00:12Z", "last_seen": "2026-03-15T14:00:00Z"},
            {"ip_address": "10.0.2.42", "hostname": "srv-db-01", "country": "US", "asn": "AS64496", "threat_score": 0.05, "is_malicious": False, "device_type": "server", "department": "IT", "first_seen": "2026-03-15T00:00:00Z", "last_seen": "2026-03-15T14:00:00Z"},
            {"ip_address": "185.220.101.34", "hostname": None, "country": "DE", "asn": "AS24940", "threat_score": 0.91, "is_malicious": True, "device_type": "unknown", "department": None, "first_seen": "2026-03-15T09:14:33Z", "last_seen": "2026-03-15T13:45:10Z"},
            {"ip_address": "10.0.3.8", "hostname": "ws-asmith", "country": "US", "asn": "AS64496", "threat_score": 0.67, "is_malicious": False, "device_type": "workstation", "department": "Finance", "first_seen": "2026-03-15T08:30:00Z", "last_seen": "2026-03-15T13:59:00Z"},
            {"ip_address": "91.219.237.12", "hostname": None, "country": "RU", "asn": "AS57678", "threat_score": 0.88, "is_malicious": True, "device_type": "unknown", "department": None, "first_seen": "2026-03-15T10:22:17Z", "last_seen": "2026-03-15T12:10:45Z"},
        ],
        "edge_sample": [
            {"src_ip": "10.0.1.15", "dst_ip": "10.0.2.42", "protocol": "TCP", "service": "postgresql", "bytes_sent": 2048, "bytes_recv": 65536, "duration": 12.4, "timestamp": "2026-03-15T08:01:00Z", "conn_state": "SF", "is_encrypted": True},
            {"src_ip": "185.220.101.34", "dst_ip": "10.0.1.15", "protocol": "TCP", "service": "ssh", "bytes_sent": 4096, "bytes_recv": 512, "duration": 0.3, "timestamp": "2026-03-15T09:14:33Z", "conn_state": "REJ", "is_encrypted": False},
            {"src_ip": "10.0.3.8", "dst_ip": "91.219.237.12", "protocol": "TCP", "service": "http", "bytes_sent": 128, "bytes_recv": 32768, "duration": 2.1, "timestamp": "2026-03-15T10:30:00Z", "conn_state": "SF", "is_encrypted": False},
            {"src_ip": "10.0.1.15", "dst_ip": "10.0.3.8", "protocol": "UDP", "service": "dns", "bytes_sent": 64, "bytes_recv": 256, "duration": 0.01, "timestamp": "2026-03-15T08:00:15Z", "conn_state": "SF", "is_encrypted": False},
            {"src_ip": "10.0.3.8", "dst_ip": "185.220.101.34", "protocol": "TCP", "service": "https", "bytes_sent": 512, "bytes_recv": 16384, "duration": 5.7, "timestamp": "2026-03-15T11:00:00Z", "conn_state": "SF", "is_encrypted": True},
        ],
    },
    "mock-session-002": {
        "session_id": "mock-session-002",
        "numNodes": 342,
        "numEdges": 1205,
        "title": "Social Media Influence Network",
        "description": "Twitter/X interaction graph from #CyberSec community, March 2026",
        "nodeColumns": [
            "handle", "display_name", "followers", "following", "verified",
            "account_age_days", "avg_engagement", "bot_score", "community_id", "location",
        ],
        "edgeColumns": [
            "source_handle", "target_handle", "interaction_type", "count",
            "sentiment", "timestamp_first", "timestamp_last",
        ],
        "active_encodings": {},
        "active_filters": [],
        "node_sample": [
            {"handle": "@threat_intel_lab", "display_name": "Threat Intel Lab", "followers": 45200, "following": 312, "verified": True, "account_age_days": 2190, "avg_engagement": 3.2, "bot_score": 0.02, "community_id": 1, "location": "San Francisco, CA"},
            {"handle": "@sec_researcher_99", "display_name": "SecRes99", "followers": 1200, "following": 890, "verified": False, "account_age_days": 365, "avg_engagement": 0.8, "bot_score": 0.15, "community_id": 2, "location": "London, UK"},
            {"handle": "@bot_farm_x42", "display_name": "News Updates Daily", "followers": 50, "following": 5000, "verified": False, "account_age_days": 30, "avg_engagement": 0.01, "bot_score": 0.95, "community_id": 3, "location": None},
            {"handle": "@ciso_weekly", "display_name": "CISO Weekly Digest", "followers": 28400, "following": 150, "verified": True, "account_age_days": 1825, "avg_engagement": 5.1, "bot_score": 0.01, "community_id": 1, "location": "New York, NY"},
            {"handle": "@apt_tracker", "display_name": "APT Tracker", "followers": 8900, "following": 420, "verified": False, "account_age_days": 730, "avg_engagement": 2.4, "bot_score": 0.08, "community_id": 2, "location": "Berlin, DE"},
        ],
        "edge_sample": [
            {"source_handle": "@threat_intel_lab", "target_handle": "@ciso_weekly", "interaction_type": "retweet", "count": 47, "sentiment": 0.8, "timestamp_first": "2026-01-15T10:00:00Z", "timestamp_last": "2026-03-14T18:30:00Z"},
            {"source_handle": "@sec_researcher_99", "target_handle": "@threat_intel_lab", "interaction_type": "reply", "count": 12, "sentiment": 0.6, "timestamp_first": "2026-02-01T14:00:00Z", "timestamp_last": "2026-03-10T09:00:00Z"},
            {"source_handle": "@bot_farm_x42", "target_handle": "@apt_tracker", "interaction_type": "mention", "count": 200, "sentiment": 0.0, "timestamp_first": "2026-03-01T00:00:00Z", "timestamp_last": "2026-03-14T23:59:00Z"},
            {"source_handle": "@apt_tracker", "target_handle": "@sec_researcher_99", "interaction_type": "quote", "count": 5, "sentiment": 0.7, "timestamp_first": "2026-02-20T11:00:00Z", "timestamp_last": "2026-03-12T16:00:00Z"},
            {"source_handle": "@ciso_weekly", "target_handle": "@threat_intel_lab", "interaction_type": "retweet", "count": 31, "sentiment": 0.9, "timestamp_first": "2026-01-20T08:00:00Z", "timestamp_last": "2026-03-13T12:00:00Z"},
        ],
    },
}


def _get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in MOCK_SESSIONS:
        raise ValueError(
            f"Session '{session_id}' not found. "
            f"Available: {list(MOCK_SESSIONS.keys())}"
        )
    return MOCK_SESSIONS[session_id]


# --- MCP Tools (matching PR #3043 tool surface) ---


@mock_mcp.tool()
async def list_sessions() -> Dict[str, Any]:
    """List all active Graphistry visualization sessions."""
    sessions = list(MOCK_SESSIONS.keys())
    return {"sessions": sessions, "count": len(sessions)}


@mock_mcp.tool()
async def get_session_info(
    session_id: str,
    verbosity: str = "brief",
) -> Dict[str, Any]:
    """
    Get information about a Graphistry visualization session.

    Args:
        session_id: The Graphistry session ID
        verbosity: 'brief' for schema only, 'full' to include sample values

    Returns:
        Session metadata including node/edge counts, column names,
        active encodings, and active filters.
    """
    s = _get_session(session_id)
    result: Dict[str, Any] = {
        "session_id": s["session_id"],
        "numNodes": s["numNodes"],
        "numEdges": s["numEdges"],
        "title": s["title"],
        "description": s["description"],
        "nodeColumns": s["nodeColumns"],
        "edgeColumns": s["edgeColumns"],
        "active_encodings": s["active_encodings"],
        "active_filters": s["active_filters"],
    }
    if verbosity == "full":
        result["node_sample"] = s["node_sample"][:3]
        result["edge_sample"] = s["edge_sample"][:3]
    return result


@mock_mcp.tool()
async def set_encoding(
    session_id: str,
    graph_type: str,
    encoding_type: str,
    attribute: str,
    animate: bool = False,
) -> Dict[str, Any]:
    """
    Set a visual encoding (color or size) on nodes or edges based on a data column.

    Args:
        session_id: The Graphistry session ID
        graph_type: 'point' for nodes, 'edge' for edges
        encoding_type: 'color' or 'size'
        attribute: The data column name to encode by
        animate: Whether to animate the transition (default False)
    """
    s = _get_session(session_id)

    columns = s["nodeColumns"] if graph_type == "point" else s["edgeColumns"]
    if attribute not in columns:
        return {
            "success": False,
            "error": f"Column '{attribute}' not found in {graph_type} columns. "
            f"Available: {columns}",
        }

    key = f"{graph_type}_{encoding_type}"
    s["active_encodings"][key] = {"attribute": attribute, "animate": animate}

    return {
        "success": True,
        "message": f"Applied {encoding_type} encoding on {graph_type} by '{attribute}'"
        + (" (animated)" if animate else ""),
        "active_encodings": s["active_encodings"],
    }


@mock_mcp.tool()
async def reset_encoding(
    session_id: str,
    graph_type: str,
    encoding_type: str,
) -> Dict[str, Any]:
    """
    Reset a visual encoding (color or size) on nodes or edges back to the default.

    Args:
        session_id: The Graphistry session ID
        graph_type: 'point' for nodes, 'edge' for edges
        encoding_type: 'color' or 'size'
    """
    s = _get_session(session_id)
    key = f"{graph_type}_{encoding_type}"
    removed = s["active_encodings"].pop(key, None)

    return {
        "success": True,
        "message": f"Reset {encoding_type} encoding on {graph_type}"
        + (f" (was: {removed['attribute']})" if removed else " (was already default)"),
        "active_encodings": s["active_encodings"],
    }


@mock_mcp.tool()
async def add_filter(
    session_id: str,
    attribute: str,
    operator: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Add a filter to the visualization to show only nodes/edges matching a condition.

    Args:
        session_id: The Graphistry session ID
        attribute: The column name to filter on
        operator: Comparison operator (eq, neq, gt, gte, lt, lte, contains)
        value: The value to compare against (string or number)
    """
    s = _get_session(session_id)

    all_columns = s["nodeColumns"] + s["edgeColumns"]
    if attribute not in all_columns:
        return {
            "success": False,
            "error": f"Column '{attribute}' not found. "
            f"Node columns: {s['nodeColumns']}. Edge columns: {s['edgeColumns']}",
        }

    filter_entry = {"attribute": attribute, "operator": operator, "value": value}
    s["active_filters"].append(filter_entry)

    # Simulate reduced counts
    reduction = random.uniform(0.1, 0.5)
    visible_nodes = int(s["numNodes"] * (1 - reduction))
    visible_edges = int(s["numEdges"] * (1 - reduction * 1.3))

    op_symbols = {"eq": "==", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "contains": "contains"}
    op_str = op_symbols.get(operator, operator)

    return {
        "success": True,
        "message": f"Filter applied: {attribute} {op_str} {value}",
        "visible_nodes": visible_nodes,
        "visible_edges": max(0, visible_edges),
        "total_filters": len(s["active_filters"]),
    }


@mock_mcp.tool()
async def reset_filters(session_id: str) -> Dict[str, Any]:
    """
    Clear all filters from the visualization, showing all nodes and edges.

    Args:
        session_id: The Graphistry session ID
    """
    s = _get_session(session_id)
    count = len(s["active_filters"])
    s["active_filters"].clear()

    return {
        "success": True,
        "message": f"Cleared {count} filter(s). Showing all {s['numNodes']} nodes and {s['numEdges']} edges.",
    }


@mock_mcp.tool()
async def get_data_sample(
    session_id: str,
    table: str = "nodes",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Returns sample rows from the node or edge table as JSON.
    Use to inspect data values before applying encodings or filters.

    Args:
        session_id: The Graphistry session ID
        table: 'nodes' or 'edges'
        limit: Number of sample rows to return (1-10, default 5)
    """
    s = _get_session(session_id)
    limit = max(1, min(10, limit))

    if table == "nodes":
        rows = s["node_sample"][:limit]
        columns = s["nodeColumns"]
    elif table == "edges":
        rows = s["edge_sample"][:limit]
        columns = s["edgeColumns"]
    else:
        return {"error": f"Invalid table: {table}. Must be 'nodes' or 'edges'."}

    return {
        "table": table,
        "columns": columns,
        "sample_rows": rows,
        "total_count": s["numNodes"] if table == "nodes" else s["numEdges"],
        "sample_count": len(rows),
    }


def main() -> None:
    """Run the mock viz MCP server."""
    mock_mcp.run()


if __name__ == "__main__":
    main()
