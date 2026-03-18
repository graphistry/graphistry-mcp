# Talk2Graph: MCP Design Notes for the Graphistry Team

This document covers how Louie (GraphistryGPT) integrates with the Talk2Graph MCP server in PR #3043, what works as-is, and design recommendations for the Graphistry team.

## TL;DR

- **Louie needs zero code changes.** The existing MCP plugin discovers tools dynamically from any MCP server URL. Just register the viz MCP endpoint as a connector.
- **The six tools in PR #3043 are the right V1 surface.** Add one more: `get_data_sample`.
- **Animation is Graphistry's concern.** Add an `animate` boolean to `set_encoding` so Louie can request it, but the implementation is entirely on the viz side.
- **V2 direction: collections + GFQL.** Instead of individual `set_encoding`/`add_filter` calls, a single `create_collection` tool that accepts a GFQL query and visual config.

## Architecture: Two MCP Servers

There are two separate MCP servers. They serve different purposes and should not be confused.

### Graphistry Viz MCP (PR #3043, inside `streamgl-viz`)

Runs per-worker inside the Graphistry product. Has direct access to live session state (`nBodiesById`, `workbooksById`, Falcor services). Manipulates the graph the user is currently looking at.

### Graphistry PyGraphistry MCP (this repo, `graphistry-mcp`)

Uses the PyGraphistry Python SDK to create new visualizations on hub.graphistry.com. For external LLM clients (Claude, Cursor, etc.) to build and analyze graphs from scratch.

**Talk2Graph uses the viz MCP.** The PyGraphistry MCP is unrelated to this feature.

## How Louie Connects

Louie's `MCPPlugin` is registered at startup. It handles any MCP connector instance dynamically. When a connector URL is registered, Louie calls `list_tools()`, discovers all available tools, and generates typed Python methods with input validation. No manual tool registration.

### Registering the connector

**Option A: per-org (recommended for production)**

Send a `CreateConnector` event to Louie:
```json
{
  "type": "CreateConnector",
  "connector": {
    "type": "MCP",
    "name": "Graphistry Talk2Graph",
    "description": "Graphistry viz session MCP tools",
    "config": {
      "server_url": "https://<graphistry-host>/mcp/",
      "request_timeout_s": 120.0
    }
  }
}
```

**Option B: system-wide (simpler for dev)**
```bash
MCP_SERVER_URL=https://<graphistry-host>/mcp/
# Restart Louie — connector available to all users automatically
```

Both use existing Louie code paths. The `MCPConnectorConfig` only requires `server_url` (string) and `request_timeout_s` (float, default 120).

## V1 Tool Surface: What to Keep, What to Add

### Current tools (PR #3043) — all good

| Tool | Status | Notes |
|------|--------|-------|
| `list_sessions` | Keep | Fine as-is |
| `get_session_info` | Keep, enhance | Add `verbosity` param (`brief`/`full`), include active encodings and active filters in the response |
| `set_encoding` | Keep, enhance | Add `animate: boolean` parameter (default false) |
| `reset_encoding` | Keep | Fine as-is |
| `add_filter` | Keep | The `addExpression` + `maskDataframe` approach is correct. Consider adding a `table` param (`point`/`edge`) to clarify scope |
| `reset_filters` | Keep | Fine as-is |

### Add for V1: `get_data_sample`

The LLM needs to see actual data values before it can suggest meaningful encodings or filters. Without this, it has to guess column semantics from names alone.

```json
{
  "name": "get_data_sample",
  "description": "Returns sample rows from the node or edge table as JSON. Use to inspect data values before applying encodings or filters.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "session_id": { "type": "string" },
      "table": { "type": "string", "enum": ["nodes", "edges"] },
      "limit": { "type": "number", "minimum": 1, "maximum": 100, "default": 10 }
    },
    "required": ["session_id", "table"]
  }
}
```

### The `animate` flag

From the design discussion: Louie decides what to change, Graphistry decides how to render it. The `animate` flag is an optional hint from the LLM. If you want to default-animate all MCP-driven changes regardless, that is fine too, but having the flag gives the LLM control when a user says "smoothly transition the colors" vs. "just color it."

```json
{
  "session_id": "...",
  "graph_type": "point",
  "encoding_type": "color",
  "attribute": "threat_score",
  "animate": true
}
```

### `get_session_info` enhancements

The LLM needs to know the current visual state to avoid redundant operations and to answer "what am I looking at?" questions. Include:

```json
{
  "session_id": "abc123",
  "numNodes": 1247,
  "numEdges": 3891,
  "title": "Enterprise Network Traffic",
  "nodeColumns": ["ip_address", "hostname", "country", "threat_score", ...],
  "edgeColumns": ["src_ip", "dst_ip", "protocol", "bytes", ...],
  "active_encodings": {
    "point_color": {"attribute": "threat_score"},
    "point_size": {"attribute": "degree"}
  },
  "active_filters": [
    {"attribute": "threat_score", "operator": "gt", "value": 0.5}
  ]
}
```

With `verbosity: "full"`, also include 3-5 sample rows per table so the LLM can see actual values.

## MCP Routing

Each PM2 worker runs its own MCP server. Louie registers one URL. The routing problem (reaching the right worker for a given session) is solved at the nginx/proxy layer, not inside Louie.

Louie does not need to know about workers, PM2, or port assignments. It sends a `session_id` in every tool call, and the infrastructure routes to the correct worker. This is entirely the Graphistry team's problem to solve.

Recommended options:
1. **Nginx route with `X-Session-ID` header** — Louie sends session ID, nginx routes to correct worker
2. **Single MCP gateway with Redis session lookup** — one endpoint proxies to correct worker
3. **Per-worker ports exposed directly** — simplest for single-node dev, does not scale

For V1 single-node deployment, any of these work.

## Session Context Optimization

PR #3043 already does this correctly: on the first chat message, the viz backend gathers session context (columns, counts, metadata) and prepends it to the query before sending to Louie. This saves Louie from needing to call `get_session_info` on the first turn, cutting 2-5 seconds off the response time.

The system prompt in `louieClient.js` tells the LLM to "only use column names that appear in the session context." This is the right constraint for V1.

## V2 Direction: Collections + GFQL

The current V1 pattern (individual `set_encoding`, `add_filter` calls) works but requires multiple round-trips for complex operations. The V2 goal is a single `create_collection` tool:

```json
{
  "name": "create_collection",
  "description": "Create a named collection with a GFQL query and visual configuration",
  "inputSchema": {
    "type": "object",
    "properties": {
      "session_id": { "type": "string" },
      "name": { "type": "string", "description": "Collection name" },
      "gfql_query": { "type": "string", "description": "GFQL query defining the collection" },
      "visual_config": {
        "type": "object",
        "properties": {
          "point_color": { "type": "object", "properties": { "column": { "type": "string" }, "as_continuous": { "type": "boolean" } } },
          "point_size": { "type": "object", "properties": { "column": { "type": "string" }, "as_continuous": { "type": "boolean" } } },
          "animate": { "type": "boolean" }
        }
      }
    },
    "required": ["session_id", "name", "gfql_query"]
  }
}
```

Example: "select all nodes with PageRank > 0.3 and color them red" becomes:
```json
{
  "session_id": "abc123",
  "name": "High PageRank Nodes",
  "gfql_query": "MATCH (n) WITH n, pagerank(n) AS pr WHERE pr > 0.3 WITH n, hop(n, 1) AS nbr RETURN nbr",
  "visual_config": {
    "point_color": { "column": "pagerank", "as_continuous": true },
    "animate": true
  }
}
```

This depends on the collections feature (Manfred's area) being ready for programmatic creation. Until then, V1 tools are sufficient.

## V2: Algorithms as MCP Tools

Expose graph algorithms that operate on the live session:

```json
{
  "name": "run_algorithm",
  "description": "Run a graph algorithm and add results as a new column",
  "inputSchema": {
    "type": "object",
    "properties": {
      "session_id": { "type": "string" },
      "algorithm": { "type": "string", "enum": ["pagerank", "betweenness", "community_detection", "degree"] },
      "output_column": { "type": "string", "description": "Name for the result column" }
    },
    "required": ["session_id", "algorithm"]
  }
}
```

The result column can then be used with `set_encoding` or `add_filter`, enabling chains like: run PageRank, then color by PageRank, then filter to high PageRank nodes.

## Mock Server for Testing

Branch `feat/talk2graph-mock-viz-mcp` in this repo contains a mock viz MCP server at:

```
src/graphistry_mcp_server/mock_viz_server.py
```

It implements all 7 V1 tools with two mock datasets (cybersecurity network traffic and social media influence graph). Useful for:

- Testing Louie integration without a running Graphistry instance
- Validating tool discovery and argument schemas
- Frontend chat widget development against a predictable backend

Run it:
```bash
# stdio mode (direct MCP client testing)
uv run python -m graphistry_mcp_server.mock_viz_server

# HTTP mode (for Louie integration)
uv run fastmcp run src/graphistry_mcp_server/mock_viz_server.py --transport streamable-http --port 3100
```

## Immediate Next Steps

1. **Des + Manfred**: Review this doc, add `get_data_sample` tool and `animate` flag to `set_encoding` in PR #3043
2. **Des + Manfred + Jared**: Short call to align on collections MCP design for V2
3. **Jared**: Test Louie integration against mock server (register `http://localhost:3100/mcp` on local Louie)
4. **Des**: Record a Loom of the current Talk2Graph demo for async review
