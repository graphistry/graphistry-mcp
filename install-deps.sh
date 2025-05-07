#!/bin/bash
# Install all dependencies for graphistry-mcp

# Set up colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==== Installing Graphistry MCP Dependencies ====${NC}"

# Check for uv installation
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv not found. Please install uv first:${NC}"
    echo "curl -fsS https://install.python-uv.org | python3" 
    exit 1
fi

# Required packages
PACKAGES=(
    "fastmcp>=2.2.6" 
    "graphistry" 
    "pandas" 
    "networkx" 
    "python-dotenv" 
    "python-louvain" 
    "psutil"
    "pydantic"
    "uvicorn"
)

# Install packages
echo -e "${YELLOW}Installing required packages...${NC}"
uv pip install ${PACKAGES[@]}

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
PYTHON_CHECK=$(python -c "
import sys
missing = []
required = [
    'fastmcp', 'graphistry', 'pandas', 'networkx', 
    'dotenv', 'community', 'psutil', 'pydantic', 'uvicorn'
]
for package in required:
    try:
        __import__(package.replace('-', '_'))
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')
if missing:
    sys.exit(1)
")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All dependencies installed successfully!${NC}"
else
    echo -e "${RED}Some dependencies could not be imported:${NC}"
    echo "$PYTHON_CHECK"
    echo -e "${YELLOW}Try installing them manually:${NC}"
    echo "uv pip install ${PACKAGES[@]}"
    exit 1
fi

echo -e "${GREEN}Set up environment variables in .env file:${NC}"
if [ ! -f .env ]; then
    echo "GRAPHISTRY_USERNAME=your_username" > .env
    echo "GRAPHISTRY_PASSWORD=your_password" >> .env
    echo -e "${YELLOW}Created .env file. Please edit with your credentials.${NC}"
else
    echo -e "${YELLOW}Using existing .env file.${NC}"
fi

echo -e "${GREEN}Graphistry MCP is ready to use!${NC}"
echo -e "Test with: python test_health_check.py"
echo -e "Start server with: python run_graphistry_server.py"