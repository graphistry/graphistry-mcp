"""Graphistry MCP Server package."""

from .server import mcp

__version__ = "0.1.0"

def main():
    """Run the Graphistry MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
