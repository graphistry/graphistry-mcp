# Graphistry MCP Integration

GPU-accelerated graph visualization and analytics for Large Language Models using Graphistry and MCP.

## Overview

This project integrates Graphistry's powerful GPU-accelerated graph visualization platform with the Model Control Protocol (MCP), enabling advanced graph analytics capabilities for AI assistants like Claude. It allows LLMs to visualize and analyze complex network data through a standardized interface.

Key features:
- GPU-accelerated graph visualization via Graphistry
- Advanced pattern discovery and relationship analysis
- Network analytics (community detection, centrality, path finding, anomaly detection)
- Mock implementation for development without credentials
- Support for various data formats (Pandas, NetworkX, edge lists)

## 🚨 Important: Graphistry Registration Required

**This MCP server requires a free Graphistry account to use visualization features.**

1. Sign up for a free account at [hub.graphistry.com](https://hub.graphistry.com)
2. Set your credentials as environment variables before starting the server:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

Without these credentials, certain visualization features will be limited.

## Features

- GPU-accelerated graph visualization via Graphistry
- Advanced pattern discovery and relationship analysis
- Streamable HTTP interface for resumable connections
- Support for various graph data formats (Pandas, NetworkX, edge lists)
- Interactive graph visualization and exploration
- Layout control for different visualization types
- Network investigation and anomaly detection capabilities

## Installation

### Quick Installation with uvx

This project uses [uvx](https://astral.sh/uv) for dependency management, which provides faster and more reliable Python package installations.

```bash
# Clone the repository
git clone https://github.com/bmorphism/graphistry-mcp.git
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

### Automated Setup

Alternatively, use our setup script which handles uvx installation and dependency management:

```bash
# Clone the repository
git clone https://github.com/bmorphism/graphistry-mcp.git
cd graphistry-mcp

# Run the setup script
./setup-graphistry-mcp.sh

# Set up your Graphistry credentials
export GRAPHISTRY_USERNAME=your_username
export GRAPHISTRY_PASSWORD=your_password
```

### Verifying Your Installation

To make sure everything is set up correctly, run our verification script:

```bash
# Make the verification script executable
chmod +x ./verify-installation.sh

# Run the script
./verify-installation.sh
```

This will check for:
1. UV/UVX installation
2. Required Python packages
3. All necessary server files
4. Server startup test
5. Health endpoint verification

If any issues are found, the script will provide guidance on how to fix them.

### Troubleshooting Common Issues

#### Python Version and Virtual Environment Issues

If you encounter Python version-related errors during installation:

1. **Python Version Mismatch**
   ```bash
   ERROR: Package 'graphistry-mcp-server' requires a different Python: 3.9.6 not in '>=3.10'
   ```
   Solution:
   ```bash
   # Remove existing virtual environment
   rm -rf .venv
   
   # Create new environment with explicit Python 3.10 path
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Upgrade pip and install dependencies
   pip install --upgrade pip setuptools wheel
   pip install -e ".[dev]"
   ```

2. **Permission Issues**
   If you see permission errors during installation:
   ```bash
   Defaulting to user installation because normal site-packages is not writeable
   ```
   Solution:
   ```bash
   # Remove the virtual environment
   rm -rf .venv
   
   # Create new environment with proper permissions
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Install with --no-user flag
   pip install --no-user -e ".[dev]"
   ```

3. **Cache Issues**
   If you encounter strange dependency resolution issues:
   ```bash
   # Clear pip cache and install without cache
   pip cache purge
   pip install --no-cache-dir -e ".[dev]"
   ```

#### UV/UVX Installation Issues

If you have trouble with UV/UVX installation:

1. **UV Not Found**
   ```bash
   # Install UV manually
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Add to PATH for current session
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **Permission Denied**
   ```bash
   # Fix permissions for .cargo directory
   chmod -R u+w ~/.cargo
   ```

## Usage

### Graphistry Account Setup

Before using this server, you must:

1. Register for a free account at [hub.graphistry.com](https://hub.graphistry.com)
2. Set your credentials as environment variables:
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

These credentials enable the server to create and access GPU-accelerated visualizations.

### Installing Dependencies

Use our automated dependency installer to ensure all required packages are available:

```bash
# Make the installer executable
chmod +x ./install-deps.sh

# Run the installer
./install-deps.sh
```

This will install all required packages using uv/uvx and create a `.env` file for your credentials if one doesn't exist.

### Starting the server

The server can be run in two modes using our improved FastMCP implementation:

1. Standard stdio mode (for typical MCP clients):
```bash
# Make sure your Graphistry credentials are set before running
python run_graphistry_mcp.py
```

2. HTTP mode (for web-based clients or testing):
```bash
# Make sure your Graphistry credentials are set before running
python run_graphistry_mcp.py --http 8080
```

For convenience, you can also use our startup scripts:

```bash
# Studio mode
./start-graphistry-mcp.sh

# HTTP mode
./start-graphistry-mcp.sh --http 8080
```

### Port Management

The server now includes advanced port management to handle situations where port 8080 is already in use:

1. **Port Conflict Detection**: Automatically checks if port 8080 (or any specified port) is available before starting
2. **Process Identification**: Uses OS-specific commands to identify what process is using the port
3. **Automatic Port Switching**: When in HTTP mode, can automatically select an alternative port if default is busy
4. **Suggested Alternatives**: Provides suggested available ports in case of conflicts
5. **Comprehensive Diagnostics**: Use the `--test-connection` flag to check port availability:
   ```bash
   python run_graphistry_mcp.py --test-connection
   ```

This prevents crashes from port conflicts and helps diagnose issues with lingering processes.

### Available Tools

The Graphistry MCP server provides the following tools for graph insights and investigations:

1. **visualize_graph** - Create a graph visualization from different data formats:
   - Supports pandas, networkx, and edge_list formats
   - Customizable node and edge attributes
   - Returns a unique graph ID and visualization URL
   - Reveals patterns and connections in complex data

2. **get_graph_info** - Retrieve information about a stored graph:
   - Access metadata for previously created visualizations
   - Get the visualization URL for sharing
   - Analyze graph metrics and statistics

3. **apply_layout** - Change the layout algorithm for a graph:
   - Force directed layout for natural clustering
   - Radial layout for hierarchical relationship investigation
   - Circle layout for symmetry analysis
   - Grid layout for structural comparisons

4. **detect_patterns** - Identify interesting patterns within graphs:
   - Community detection for network segmentation
   - Path finding between key entities
   - Centrality metrics for key node identification
   - Anomaly detection for outlier identification

### Advanced Visualization Features

The following additional visualization capabilities are available when properly authenticated with Graphistry:

1. **interactive_exploration** - Interact with live visualizations:
   - Pan, zoom, and explore complex graph structures
   - Click nodes and edges to reveal detailed information
   - Filter and highlight specific patterns within the visualization

2. **graph_embedding** - Generate embeddings for graph analysis:
   - Visualize graph embeddings in 2D or 3D space
   - Identify clusters and patterns through dimensional reduction
   - Compare similarity between different graph structures

3. **time_series_analysis** - Visualize how graphs evolve over time:
   - Play back temporal graph changes as animations
   - Track entity relationships as they form and dissolve
   - Identify patterns in temporal network dynamics

4. **annotation_tools** - Add context to your visualizations:
   - Highlight important nodes or relationships
   - Add explanatory text to visualizations
   - Create shareable, annotated graph stories

### Example Client Usage

```python
import asyncio
import os
from mcp.client import Client
from mcp.client.stdio import stdio_client

# Set up Graphistry credentials before starting client
# These should match the credentials used on your Graphistry account
os.environ["GRAPHISTRY_USERNAME"] = "your_graphistry_username"  # Replace with your username
os.environ["GRAPHISTRY_PASSWORD"] = "your_graphistry_password"  # Replace with your password

async def main():
    async with stdio_client() as client:
        # List available tools
        tools = await client.list_tools()
        
        # Create a simple graph
        result = await client.call_tool("visualize_graph", {
            "data_format": "edge_list",
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
                {"source": "C", "target": "A"}
            ],
            "title": "Triangle Graph"
        })
        
        print(f"Graph created: {result['url']}")
        print("Open this URL in your browser to view your visualization")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Applications

Check out the `examples` directory for more advanced applications:

1. **Identity and Access Management (IAM) Visualization** - An interactive demo showing how to visualize and analyze complex IAM relationships in an organization. Identifies security risks such as dormant accounts with high privileges, privilege escalation paths, and over-privileged service accounts.

To run the IAM demo:
```bash
cd examples
./run_iam_demo.sh
```

> **Note**: The visualization URL will only work if you have properly set up your Graphistry credentials and have an active account at [hub.graphistry.com](https://hub.graphistry.com).

## Docker Usage

You can also run the server in a Docker container with your Graphistry credentials:

```bash
# Build the Docker image
docker build -t graphistry-mcp .

# Run the container with environment variables
docker run -p 8080:8080 \
  -e GRAPHISTRY_USERNAME=your_username \
  -e GRAPHISTRY_PASSWORD=your_password \
  graphistry-mcp
```

These environment variables will be passed to the container, allowing it to authenticate with Graphistry's services.

## Development

```bash
# Install development dependencies
uvx pip install -e ".[dev]"

# Run tests
uvx pytest

# Format code
uvx black .

# Lint code
uvx ruff check .

# Type check
uvx mypy src/
```

### Auto-Installing Dependencies

The server is designed to automatically install any missing dependencies using `uvx`. If you encounter any import errors when running the server, it will attempt to install the required packages on-the-fly.

This behavior ensures that all required packages are available, even if they weren't explicitly installed during the initial setup. You'll see messages in the console when dependencies are being installed automatically.

### Performance Optimizations

The Graphistry MCP server includes several performance optimizations:

1. **LRU Caching**: Frequently used graph operations are cached using an LRU (Least Recently Used) cache to reduce computation time for repeated operations.

2. **Lazy Loading**: Heavy dependencies are loaded on-demand to improve startup time and reduce memory usage when certain features aren't being utilized.

3. **Health Monitoring**: A `/health` endpoint is available in HTTP mode to monitor server status and performance metrics.

To access the health endpoint in HTTP mode:
```bash
curl http://localhost:8080/health
```

This returns a JSON response with server status and performance information.

## Graphistry Authentication Details

### Account Registration

1. Visit [hub.graphistry.com](https://hub.graphistry.com) and sign up for a free account
2. After registration, you'll have access to Graphistry's GPU-accelerated visualization platform
3. Keep your username and password secure for use with this MCP server

### Credential Management

There are several ways to provide your Graphistry credentials:

1. **Environment variables** (recommended):
   ```bash
   export GRAPHISTRY_USERNAME=your_username
   export GRAPHISTRY_PASSWORD=your_password
   ```

2. **Configuration file**:
   Create a `.env` file in the project root:
   ```
   GRAPHISTRY_USERNAME=your_username
   GRAPHISTRY_PASSWORD=your_password
   ```

3. **Command-line arguments** (when starting the server):
   ```bash
   ./start-graphistry-mcp.sh --graphistry-username=your_username --graphistry-password=your_password
   ```

### Troubleshooting Authentication

If you encounter authentication issues:

1. Verify your credentials at [hub.graphistry.com](https://hub.graphistry.com)
2. Check for typos in your username or password
3. Ensure your account has been activated (check your email)
4. Look for error messages in the server logs that might indicate authentication problems

### Rate Limits and Usage

Free Graphistry accounts have certain usage limitations. For high-volume or production usage, consider upgrading to a paid plan at [graphistry.com/plans](https://www.graphistry.com/plans).

## License

MIT