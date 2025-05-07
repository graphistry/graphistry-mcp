#!/bin/bash
# Setup script for Graphistry MCP server using uv/uvx
# This script sets up the environment and installs all dependencies
# for the Graphistry MCP server with fast installation via uv.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Check if installation was successful
    if ! command -v uv &> /dev/null; then
        echo "Failed to install uv. Please install manually: https://astral.sh/uv/install"
        exit 1
    fi
    
    echo "✅ uv installed successfully"
fi

# Create a Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv --python=python3.10
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install project in development mode with all dependencies
echo "Installing dependencies with uvx..."
uvx pip install -e ".[dev]"

# Install key packages explicitly to ensure they're available
echo "Installing key packages..."
uvx pip install "fastmcp>=2.2.6" "graphistry" "pandas" "networkx" "uvicorn" "python-dotenv" "python-louvain" "pydantic" "psutil"

# Make the start script executable
chmod +x start-graphistry-mcp.sh

echo "✅ Setup complete!"
echo "----------------------------"
echo "✨ You can now run the server using:"
echo "./start-graphistry-mcp.sh"
echo "Or for HTTP mode:"
echo "./start-graphistry-mcp.sh --http [port]"
echo ""
echo "📝 Remember to set your Graphistry credentials:"
echo "export GRAPHISTRY_USERNAME=your_username"
echo "export GRAPHISTRY_PASSWORD=your_password"
echo "----------------------------"