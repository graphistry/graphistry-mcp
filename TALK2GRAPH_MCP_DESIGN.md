# Talk2Graph: MCP Design Notes for the Graphistry Team

This document covers how Louie (GraphistryGPT) integrates with the Talk2Graph MCP server in PR #3043, what works as-is, and design recommendations for the Graphistry team.

## TL;DR

- **Louie needs zero code changes.** The existing MCP plugin discovers tools dynamically from any MCP server URL. Just register the viz MCP endpoint as a connector.
- **Fix the system prompt first.** Louie responds like an investigation agent (verbose, verdicts, evidence). Update `TALK2GRAPH_SYSTEM_PROMPT` in `louieClient.js` to make it concise. This is the biggest UX win and it is a ~20 line change on the viz side.
- **Register the MCP server URL.** Tools are built but Louie cannot call them until the connector is registered. One env var or API call.
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

## System Prompt: Critical UX Fix

The current `TALK2GRAPH_SYSTEM_PROMPT` in `louieClient.js` is minimal, and because Louie defaults to its full investigation agent behavior, responses come back with verdict/evidence/investigation-style formatting that is way too verbose for a graph manipulation assistant. This is the single biggest UX issue to fix for V1.

### The problem

Louie's default `LouieAgent` is designed for deep investigations: it produces structured analysis with verdicts, evidence sections, confidence scores, and multi-paragraph explanations. That is exactly the wrong tone for Talk2Graph, where a user says "color by threat score" and expects a one-line confirmation, not a forensic report.

### The fix

Replace the system prompt in `louieClient.js` with something that constrains the response style. This is entirely on the viz side, no Louie changes needed. The system prompt is prepended to the user's query on the first message (before a dthread exists), so it sets the tone for the entire conversation.

**Recommended prompt:**

```javascript
const TALK2GRAPH_SYSTEM_PROMPT = `You are Talk2Graph, a concise graph visualization assistant embedded in Graphistry.

RESPONSE STYLE:
- Answer in 1-3 sentences. Be brief and direct.
- When you use a tool, confirm what you did in one line: "Done — colored nodes by threat_score."
- Do NOT produce investigation reports, verdicts, evidence sections, or confidence scores.
- Do NOT use markdown headers, bullet lists, or structured analysis formats.
- If the user asks a question about the data, give a short factual answer based on the session context.

TOOLS:
- You have MCP tools for manipulating the visualization (set_encoding, reset_encoding, add_filter, reset_filters, get_session_info, get_data_sample).
- Use these tools when the user wants to change colors, sizes, filters, or inspect data.
- Only use column names that appear in the session context.
- When using tools, always include the session_id from the session context.

WHAT YOU ARE NOT:
- You are not an investigation agent. Do not analyze threats or produce reports.
- You are not a search engine. If you do not know something, say so briefly.
- You are a graph manipulation assistant. Help users see their data differently.`;
```

### Why this works without Louie changes

The system prompt is injected by the viz backend in `buildQueryWithContext()` before the query reaches Louie. Louie's `LouieAgent` sees it as part of the user message on the first turn and follows the instructions. On subsequent turns (when a dthread exists), the conversation history carries the tone forward. No agent routing changes, no new agent type, no Louie PR needed.

### V2: Dedicated Talk2Graph agent

For V2, Louie should have a proper `Talk2GraphAgent` with its own prompt and tool routing, rather than relying on system prompt injection. This would:
- Use a lighter LLM (faster responses for simple operations)
- Have graph analytics domain knowledge built into the agent prompt
- Understand GFQL syntax for generating collection queries
- Skip the investigation pipeline entirely

But for V1, the system prompt override in `louieClient.js` is the right lever and it is fast to ship.

## Session Context Optimization

PR #3043 already does this correctly: on the first chat message, the viz backend gathers session context (columns, counts, metadata) and prepends it to the query before sending to Louie. This saves Louie from needing to call `get_session_info` on the first turn, cutting 2-5 seconds off the response time.

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

## Current Status (2026-03-18)

What is working:
- Chat widget is in the right side panel, functional
- Chat proxy hits Louie, gets responses with session context
- Louie can answer questions about the graph (columns, structure, data)

What is not working yet:
- **MCP server URL not registered** — Louie cannot call tools to change the visualization. This is just a config step (see "Registering the connector" above).
- **Responses are too verbose** — Louie produces investigation-style reports instead of brief confirmations. Fix by updating `TALK2GRAPH_SYSTEM_PROMPT` (see "System Prompt: Critical UX Fix" above).

## Immediate Next Steps

**Priority 1 (unblocks everything):**
1. **Des**: Update `TALK2GRAPH_SYSTEM_PROMPT` in `louieClient.js` to the concise prompt above. This is the single biggest UX improvement and is a ~20 line change.
2. **Des or infra**: Register the MCP server URL on the Louie instance so tools actually work. Either set `MCP_SERVER_URL` env var or send `CreateConnector` event.

**Priority 2 (V1 completeness):**
3. **Des + Manfred**: Add `get_data_sample` tool and `animate` flag to `set_encoding` in PR #3043
4. **Des**: Record a Loom of the updated demo (concise responses + working tools) for async review

**Priority 3 (V2 planning):**
5. **Des + Manfred + Jared**: Short call to align on `create_collection` MCP tool design for V2
6. **Jared**: Test Louie integration against mock server in this repo
