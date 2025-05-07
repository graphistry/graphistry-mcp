# Graphistry MCP Integration

GPU-accelerated graph visualization and analytics for Large Language Models using Graphistry and MCP.

## Overview

This project integrates Graphistry's powerful GPU-accelerated graph visualization platform with the Model Control Protocol (MCP), enabling advanced graph analytics capabilities for AI assistants and LLMs. It allows LLMs to visualize and analyze complex network data through a standardized, LLM-friendly interface.

**Key features:**
- GPU-accelerated graph visualization via Graphistry
- Advanced pattern discovery and relationship analysis
- Network analytics (community detection, centrality, path finding, anomaly detection)
- Support for various data formats (Pandas, NetworkX, edge lists)
- LLM-friendly API: single `graph_data` dict for graph tools

## 🚨 Important: Graphistry Registration Required

**This MCP server requires a free Graphistry account to use visualization features.**

1. Sign up for a free account at [hub.graphistry.com](https://hub.graphistry.com)
2. Set your credentials as environment variables or in a `.env` file before starting the server:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   # or create a .env file with:
   # GRAPHISTRY_USERNAME=your_username
   # GRAPHISTRY_PASSWORD=your_password
   ```
   See `.env.example` for a template.

## MCP Configuration (.mcp.json)

To use this project with Cursor or other MCP-compatible tools, you need a `.mcp.json` file in your project root. A template is provided as `.mcp.json.example`.

**Setup:**

```bash
cp .mcp.json.example .mcp.json
```

Edit `.mcp.json` to:
- Set the correct paths for your environment (e.g., project root, Python executable, server script)
- Set your Graphistry credentials (or use environment variables/.env)
- Choose between HTTP and stdio modes:
  - `graphistry-http`: Connects via HTTP (set the `url` to match your server's port)
  - `graphistry`: Connects via stdio (set the `command`, `args`, and `env` as needed)

**Note:**
- `.mcp.json.example` contains both HTTP and stdio configurations. Enable/disable as needed by setting the `disabled` field.
- See `.env.example` for environment variable setup.

## Installation

### Recommended Installation (Python venv + pip)

```bash
# Clone the repository
git clone https://github.com/graphistry/graphistry-mcp.git
cd graphistry-mcp

# Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set up your Graphistry credentials (see above)
```

Or use the setup script:

```bash
./setup-graphistry-mcp.sh
```

## Usage

### Starting the Server

```bash
# Activate your virtual environment if not already active
source .venv/bin/activate

# Start the server (stdio mode)
python run_graphistry_mcp.py

# Or use the start script for HTTP or stdio mode (recommended, sources .env securely)
./start-graphistry-mcp.sh --http 8080
```

### Security & Credential Handling

- The server loads credentials from environment variables or `.env` using [python-dotenv](https://pypi.org/project/python-dotenv/), so you can safely use a `.env` file for local development.
- The `start-graphistry-mcp.sh` script sources `.env` and is the most robust and secure way to launch the server.

### Adding to Cursor (or other LLM tools)

- Add the MCP server to your `.cursor/mcp.json` or equivalent config:
  ```json
  {
    "graphistry": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["/path/to/your/run_graphistry_mcp.py"],
      "env": {
        "GRAPHISTRY_USERNAME": "your_username",
        "GRAPHISTRY_PASSWORD": "your_password"
      },
      "type": "stdio"
    }
  }
  ```
- Make sure the virtual environment is used (either by using the full path to the venv's python, or by activating it before launching).
- If you see errors about API version or missing credentials, double-check your environment variables and registration.

### Example: Visualizing a Graph (LLM-friendly API)

The main tool, `visualize_graph`, now accepts a single `graph_data` dictionary. Example:

```python
{
  "graph_data": {
    "graph_type": "graph",
    "edges": [
      {"source": "A", "target": "B"},
      {"source": "A", "target": "C"},
      {"source": "A", "target": "D"},
      {"source": "A", "target": "E"},
      {"source": "B", "target": "C"},
      {"source": "B", "target": "D"},
      {"source": "B", "target": "E"},
      {"source": "C", "target": "D"},
      {"source": "C", "target": "E"},
      {"source": "D", "target": "E"}
    ],
    "nodes": [
      {"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "E"}
    ],
    "title": "5-node, 10-edge Complete Graph",
    "description": "A complete graph of 5 nodes (K5) where every node is connected to every other node."
  }
}
```

Example (hypergraph):

```python
{
  "graph_data": {
    "graph_type": "hypergraph",
    "edges": [
      {"source": "A", "target": "B", "group": "G1", "weight": 0.7},
      {"source": "A", "target": "C", "group": "G1", "weight": 0.6},
      {"source": "B", "target": "C", "group": "G2", "weight": 0.8},
      {"source": "A", "target": "D", "group": "G2", "weight": 0.5}
    ],
    "columns": ["source", "target", "group"],
    "title": "Test Hypergraph",
    "description": "A simple test hypergraph."
  }
}
```

## Available MCP Tools

The following MCP tools are available for graph visualization, analysis, and manipulation:

- **visualize_graph**: Visualize a graph or hypergraph using Graphistry's GPU-accelerated renderer.
- **get_graph_ids**: List all stored graph IDs in the current session.
- **get_graph_info**: Get metadata (node/edge counts, title, description) for a stored graph.
- **apply_layout**: Apply a standard layout (force_directed, radial, circle, grid) to a graph.
- **detect_patterns**: Run network analysis (centrality, community detection, path finding, anomaly detection).
- **encode_point_color**: Set node color encoding by column (categorical or continuous).
- **encode_point_size**: Set node size encoding by column (categorical or continuous).
- **encode_point_icon**: Set node icon encoding by column (categorical, with icon mapping or binning).
- **encode_point_badge**: Set node badge encoding by column (categorical, with icon mapping or binning).
- **apply_ring_categorical_layout**: Arrange nodes in rings by a categorical column (e.g., group/type).
- **apply_group_in_a_box_layout**: Arrange nodes in group-in-a-box layout (requires igraph).
- **apply_modularity_weighted_layout**: Arrange nodes by modularity-weighted layout (requires igraph).
- **apply_ring_continuous_layout**: Arrange nodes in rings by a continuous column (e.g., score).
- **apply_time_ring_layout**: Arrange nodes in rings by a datetime column (e.g., created_at).
- **apply_tree_layout**: Arrange nodes in a tree (layered hierarchical) layout.
- **set_graph_settings**: Set advanced visualization settings (point size, edge influence, etc.).

## Contributing

PRs and issues welcome! This project is evolving rapidly as we learn more about LLM-driven graph analytics and tool integration.

## License

MIT