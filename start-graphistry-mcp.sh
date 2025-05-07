#!/bin/bash
# Start the Graphistry MCP server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python environment is active, if not activate it
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Activating Python virtual environment..."
  if [ -d ".venv" ]; then
    source .venv/bin/activate
  else
    echo "Creating virtual environment with uvx..."
    uvx venv .venv
    source .venv/bin/activate
    uvx pip install -e ".[dev]"
  fi
fi

# Check for Graphistry credentials in environment or arguments
if [ -z "$GRAPHISTRY_USERNAME" ] || [ -z "$GRAPHISTRY_PASSWORD" ]; then
  # Check for command line arguments
  GRAPHISTRY_USERNAME_ARG=""
  GRAPHISTRY_PASSWORD_ARG=""
  
  for arg in "$@"; do
    if [[ $arg == --graphistry-username=* ]]; then
      GRAPHISTRY_USERNAME_ARG="${arg#*=}"
    elif [[ $arg == --graphistry-password=* ]]; then
      GRAPHISTRY_PASSWORD_ARG="${arg#*=}"
    fi
  done
  
  if [ -n "$GRAPHISTRY_USERNAME_ARG" ] && [ -n "$GRAPHISTRY_PASSWORD_ARG" ]; then
    export GRAPHISTRY_USERNAME="$GRAPHISTRY_USERNAME_ARG"
    export GRAPHISTRY_PASSWORD="$GRAPHISTRY_PASSWORD_ARG"
  else
    # Check for .env file
    if [ -f ".env" ]; then
      echo "Loading Graphistry credentials from .env file..."
      source .env
    fi
    
    # Final check if credentials exist
    if [ -z "$GRAPHISTRY_USERNAME" ] || [ -z "$GRAPHISTRY_PASSWORD" ]; then
      echo "⚠️  Warning: Graphistry credentials not found ⚠️"
      echo "For full functionality, please set GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD"
      echo "Sign up for a free account at https://hub.graphistry.com if you don't have one"
      echo ""
    fi
  fi
else
  echo "✓ Graphistry credentials found in environment variables"
fi

# Check for mode parameter
MODE="stdio"
PORT=8080

for arg in "$@"; do
  if [ "$arg" == "--http" ]; then
    MODE="http"
  elif [[ $arg =~ ^[0-9]+$ ]] && [ "$MODE" == "http" ]; then
    PORT=$arg
  fi
done

# Set logging level if provided
if [ -n "$LOG_LEVEL" ]; then
  export GRAPHISTRY_LOG_LEVEL=$LOG_LEVEL
else
  export GRAPHISTRY_LOG_LEVEL="INFO"
fi

echo "Starting Graphistry MCP server in $MODE mode..."

if [ "$MODE" == "http" ]; then
  echo "Server will be available at http://localhost:$PORT"
  uvx python src/graphistry_fastmcp/server.py --http $PORT
else
  # stdio mode
  exec uvx python src/graphistry_fastmcp/server.py
fi