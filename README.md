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
2. Set your credentials as environment variables before starting the server:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

Without these credentials, visualization features will not work.

## Installation

### Quick Installation with uv/uvx

This project uses [uv](https://astral.sh/uv) for dependency management, which provides fast and reliable Python package installations.

```bash
# Clone the repository
git clone https://github.com/graphistry/graphistry-mcp.git
cd graphistry-mcp

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uvx pip install -e ".[dev]"

# Set up your Graphistry credentials (obtained from hub.graphistry.com)
export GRAPHISTRY_USERNAME=your_username
export GRAPHISTRY_PASSWORD=your_password
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

# Or use the start script for HTTP or stdio mode
./start-graphistry-mcp.sh --http 8080
```

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
    "data_format": "edge_list",
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

### What We Learned / Caveats & Gotchas

- **Environment Variables:** Always ensure `GRAPHISTRY_USERNAME` and `GRAPHISTRY_PASSWORD` are set in the environment where the server runs. Debug prints can help verify this.
- **Virtual Environment:** Use the venv's Python directly or activate the environment before running the server. This avoids dependency and path issues.
- **LLM Tool Schemas:** LLMs and tool bridges work best with single-argument (dict) APIs. Avoid multi-argument signatures for tools you want to expose to LLMs.
- **Cursor Integration:** When adding to Cursor, use the full path to the venv's Python and ensure all environment variables are set in the config.

## Contributing

PRs and issues welcome! This project is evolving rapidly as we learn more about LLM-driven graph analytics and tool integration.

## License

MIT