#!/bin/bash

# setup-agent.sh - Script to help agents set up and get started with Graphistry MCP

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}==>${NC} $1"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Function to print errors
print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check if Python 3.10+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        if [[ $(echo "$PYTHON_VERSION" | cut -d. -f1) -ge 3 && $(echo "$PYTHON_VERSION" | cut -d. -f2) -ge 10 ]]; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.10 or higher is required. Found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi
}

# Install uv if not present
install_uv() {
    print_status "Checking for uv..."
    if ! command -v uv &>/dev/null; then
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        print_status "uv is already installed"
    fi
}

# Set up virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    if [ ! -d ".venv" ]; then
        uv venv .venv
    fi
    source .venv/bin/activate
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    uvx pip install -e ".[dev]"
}

# Create example configuration
create_config() {
    print_status "Creating example configuration..."
    if [ ! -f ".mcp.json" ]; then
        cp .mcp.json.example .mcp.json
        print_warning "Created .mcp.json from example. Please update with your credentials."
    fi
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Created .env from example. Please update with your credentials."
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python tests/test_mcp_client.py
}

# Main setup process
main() {
    print_status "Starting Graphistry MCP setup for agents..."
    
    # Check Python version
    check_python
    
    # Install uv
    install_uv
    
    # Set up virtual environment
    setup_venv
    
    # Install dependencies
    install_deps
    
    # Create configuration
    create_config
    
    # Run tests
    run_tests
    
    print_status "Setup complete! Next steps:"
    echo "1. Update .mcp.json with your Graphistry credentials"
    echo "2. Update .env with your Graphistry credentials"
    echo "3. Start the server with: ./start-graphistry-mcp.sh"
    echo "4. Test the server with: python tests/test_mcp_client.py"
}

# Run main function
main 