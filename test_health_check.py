#!/usr/bin/env python
"""
Simple test script to check Graphistry MCP server health

This script directly imports the server module and executes the health_check tool
to verify that the server is properly configured and all dependencies are available.
"""

import sys
import json
from pathlib import Path

# Ensure the graphistry directory is in the path
sys.path.append(str(Path(__file__).parent.absolute()))

try:
    # Import the server module directly
    from src.graphistry_fastmcp.server import health_check
    
    # Run the health check
    result = health_check()
    
    # Print the result as formatted JSON
    print(json.dumps(result, indent=2))
    
    print("\n✅ Health check completed successfully.")
    print(f"Graphistry status: {'Available' if result['graphistry_status']['available'] else 'Unavailable'}")
    print(f"Credentials configured: {'Yes' if result['graphistry_status']['credentials_configured'] else 'No'}")
    
    # Check memory usage if available
    if 'available' not in result['memory_usage']:
        print(f"Memory usage: {result['memory_usage']['rss_mb']:.2f} MB (RSS)")
    
    # Print uptime
    seconds = result['uptime_seconds']
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(f"Uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
except ImportError as e:
    print(f"❌ Failed to import server module: {e}")
    print("Make sure all dependencies are installed:")
    print("  uvx pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)