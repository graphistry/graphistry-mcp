#!/bin/bash
# Setup script for Graphistry MCP server using uv
# This script installs dependencies using uvx for faster package management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

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
    uv venv .venv
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
uvx pip install fastmcp graphistry pandas networkx uvicorn python-dotenv

echo "✅ Setup complete!"
echo "----------------------------"
echo "✨ You can now run the server using:"
echo "./start-graphistry-mcp.sh"
echo ""
echo "📝 Remember to set your Graphistry credentials:"
echo "export GRAPHISTRY_USERNAME=your_username"
echo "export GRAPHISTRY_PASSWORD=your_password"
echo "----------------------------"