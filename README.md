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

- **Environment Variables & .env:** Always ensure `GRAPHISTRY_USERNAME` and `GRAPHISTRY_PASSWORD` are set in the environment or `.env` file. The server loads `.env` automatically using `python-dotenv`.
- **Virtual Environment:** Use the venv's Python directly or activate the environment before running the server. This avoids dependency and path issues.
- **Cursor Integration:** When adding to Cursor, use the full path to the venv's Python and ensure all environment variables are set in the config.

## Contributing

PRs and issues welcome! This project is evolving rapidly as we learn more about LLM-driven graph analytics and tool integration.

## License

MIT