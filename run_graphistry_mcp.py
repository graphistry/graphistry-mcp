#!/usr/bin/env python3
"""
Runner script for Graphistry MCP server.
This script is designed to be invoked by the MCP system.
"""

import os
import sys
import logging
import subprocess
import asyncio
import time
import socket
from pathlib import Path

# Record start time for reporting
START_TIME = time.time()

# Load environment variables from .env file at startup
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    try:
        # We'll attempt to import and use dotenv later if needed
        # For now, just note that we found the .env file
        print(f"Found .env file at {dotenv_path}", file=sys.stderr)
    except Exception as e:
        print(f"Note: Found .env file but couldn't process immediately: {e}", file=sys.stderr)

# Create logs directory if it doesn't exist
current_dir = Path(__file__).parent
logs_dir = current_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(logs_dir / 'graphistry-mcp.log'),
    filemode='a'
)
logger = logging.getLogger('graphistry-mcp')

# Function to check if a port is already in use
def is_port_available(port, host='localhost'):
    """
    Check if a port is available to use.
    
    Args:
        port (int): The port number to check
        host (str): The host to check against (default: 'localhost')
        
    Returns:
        bool: True if the port is available, False if it's in use
    """
    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to bind to the port
            s.bind((host, port))
            # If we get here, the port is available
            return True
    except socket.error:
        # Port is already in use
        return False

def find_available_port(start_port=8080, max_attempts=20):
    """
    Find an available port starting from a specified port.
    
    Args:
        start_port (int): The port to start checking from
        max_attempts (int): Maximum number of ports to check
        
    Returns:
        int: First available port found, or None if no port is available
    """
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        if is_port_available(port):
            return port
    return None

# Check if MCP server port is available (commonly uses port 8080)
if not is_port_available(8080):
    logger.warning(f"⚠️ Port 8080 is already in use by another process")
    print(f"⚠️ Warning: Port 8080 is already in use by another process.", file=sys.stderr)
    
    # Find an alternative port
    alt_port = find_available_port(8081)
    if alt_port:
        print(f"💡 Alternative port {alt_port} is available for use if needed.", file=sys.stderr)
        logger.info(f"Alternative port {alt_port} is available for use")
        
        # Set environment variable with alternative port suggestion
        os.environ["GRAPHISTRY_SUGGESTED_PORT"] = str(alt_port)
    else:
        print(f"❌ No alternative ports available in range 8081-8100.", file=sys.stderr)
        logger.warning("No alternative ports available in range 8081-8100")
    
    try:
        # Try to identify what's using the port
        if sys.platform.startswith('linux') or sys.platform == 'darwin':  # Linux or macOS
            logger.info("Attempting to identify process using port 8080...")
            try:
                result = subprocess.run(['lsof', '-i', ':8080'], capture_output=True, text=True)
                if result.stdout:
                    print(f"Process using port 8080:\n{result.stdout}", file=sys.stderr)
                    logger.info(f"Process using port 8080:\n{result.stdout}")
                    print(f"To free the port, you can terminate the process using: kill <PID>", file=sys.stderr)
            except Exception as e:
                logger.error(f"Unable to check what's using port 8080: {e}")
    except Exception as e:
        logger.error(f"Error while checking port usage: {e}")

# Function to install dependencies
def install_dependency(package):
    try:
        logger.info(f"Installing {package} with uvx...")
        print(f"Installing {package} with uvx...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed {package}")
        return True
    except Exception as e:
        logger.error(f"Failed to install {package}: {e}")
        print(f"Failed to install {package}: {e}", file=sys.stderr)
        return False

# Check if we're running in the virtual environment
venv_dir = current_dir / '.venv'
venv_python = venv_dir / 'bin' / 'python'
if venv_dir.exists() and venv_python.exists() and not os.environ.get('GRAPHISTRY_VENV_ACTIVE'):
    logger.info(f"Activating virtual environment at {venv_dir}")
    # Re-execute the script with the virtual environment's Python
    os.environ['GRAPHISTRY_VENV_ACTIVE'] = '1'
    try:
        os.execv(str(venv_python), [str(venv_python), __file__] + sys.argv[1:])
    except Exception as e:
        logger.error(f"Failed to activate virtual environment: {e}")
        print(f"Failed to activate virtual environment: {e}", file=sys.stderr)

logger.info("Running with Python interpreter: " + sys.executable)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Python executable: {sys.executable}", file=sys.stderr)

# Add the src directory to the Python path
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(current_dir))

# Ensure dependencies are installed
required_packages = ["fastmcp>=2.2.6", "graphistry", "pandas", "networkx", "python-dotenv", "pydantic"]
for package in required_packages:
    try:
        if package.startswith("fastmcp"):
            import fastmcp
            logger.info(f"Successfully imported {package}")
        elif package == "graphistry":
            import graphistry
            logger.info(f"Successfully imported {package}")
        elif package == "pandas":
            import pandas
            logger.info(f"Successfully imported {package}")
        elif package == "networkx":
            import networkx
            logger.info(f"Successfully imported {package}")
        elif package == "python-dotenv":
            from dotenv import load_dotenv
            logger.info(f"Successfully imported {package}")
            # Now that dotenv is imported, actually load the .env file
            dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                logger.info(f"Loaded environment variables from {dotenv_path}")
                print(f"Loaded environment variables from {dotenv_path}", file=sys.stderr)
        elif package == "pydantic": 
            from pydantic import BaseModel
            logger.info(f"Successfully imported {package}")
    except ImportError:
        logger.warning(f"Dependency {package} not found, installing...")
        install_dependency(package)

# Now try to run the server
try:
    # Import the FastMCP server module and FastMCP class
    try:
        from fastmcp import FastMCP
        logger.info("Successfully imported FastMCP")
    except ImportError:
        logger.error("Could not import FastMCP - attempting to install...")
        if install_dependency("fastmcp>=2.2.6"):
            from fastmcp import FastMCP
            logger.info("Successfully imported FastMCP after installation")
        else:
            logger.error("Failed to import FastMCP even after installation attempt")
            print("Failed to import FastMCP even after installation attempt", file=sys.stderr)
            sys.exit(1)
    
    # Attempt to import module directly
    sys.path.insert(0, str(current_dir))
    try:
        from src.graphistry_fastmcp.server import main, mcp
        logger.info("Successfully imported main function from graphistry_fastmcp.server")
        module_path = "src.graphistry_fastmcp.server"
    except ImportError:
        try:
            from graphistry_fastmcp.server import main, mcp
            logger.info("Successfully imported main function from direct graphistry_fastmcp.server")
            module_path = "graphistry_fastmcp.server"
        except ImportError as e:
            logger.error(f"Could not import server module: {e}")
            print(f"Could not import server module: {e}", file=sys.stderr)
            logger.error(f"Python path: {sys.path}")
            
            # List all files in src directory to help debug
            try:
                print(f"Contents of src directory:", file=sys.stderr)
                for item in os.listdir(str(src_path)):
                    print(f"  {item}", file=sys.stderr)
                
                graphistry_dir = src_path / 'graphistry_fastmcp'
                if graphistry_dir.exists():
                    print(f"Contents of graphistry_fastmcp directory:", file=sys.stderr)
                    for item in os.listdir(str(graphistry_dir)):
                        print(f"  {item}", file=sys.stderr)
            except Exception as dir_error:
                print(f"Error listing directory contents: {dir_error}", file=sys.stderr)
            
            sys.exit(1)
    
    # Run the main function
    logger.info(f"Starting Graphistry MCP server from module: {module_path}")
    print(f"Starting Graphistry MCP server from module: {module_path}", file=sys.stderr)
    
    # Check if test-connection flag was passed
    if len(sys.argv) > 1 and sys.argv[1] == "--test-connection":
        # Check port availability as part of the test
        print("\n===== PORT AVAILABILITY TEST =====")
        default_port = 8080
        port_status = "✅ Available" if is_port_available(default_port) else "❌ In use by another process"
        print(f"Default port {default_port}: {port_status}")
        
        if not is_port_available(default_port):
            # Check if we can identify the process using the port
            if sys.platform.startswith('linux') or sys.platform == 'darwin':
                try:
                    result = subprocess.run(['lsof', '-i', f':{default_port}'], capture_output=True, text=True)
                    if result.stdout:
                        print(f"\nProcess using port {default_port}:")
                        print(result.stdout)
                except Exception:
                    pass
                    
            # Find and suggest alternative ports
            alt_port = find_available_port(default_port + 1)
            if alt_port:
                print(f"💡 Alternative port {alt_port} is available for use")
                os.environ["GRAPHISTRY_SUGGESTED_PORT"] = str(alt_port)
            else:
                print(f"❌ No alternative ports available in range {default_port+1}-{default_port+20}")
        
        print("\n===== GRAPHISTRY API TEST =====")
        print("Testing connection to Graphistry API...")
        try:
            import graphistry
            print("Successfully imported graphistry")
            
            # Get credentials from environment variables
            env_username = os.environ.get("GRAPHISTRY_USERNAME")
            env_password = os.environ.get("GRAPHISTRY_PASSWORD")
            
            print(f"Environment variables:")
            print(f"GRAPHISTRY_USERNAME found: {'Yes' if env_username else 'No'}")
            print(f"GRAPHISTRY_PASSWORD found: {'Yes' if env_password else 'No'}")
            
            # Get credentials from .env file
            dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
            dotenv_username = None
            dotenv_password = None
            if os.path.exists(dotenv_path):
                from dotenv import load_dotenv, dotenv_values
                env_values = dotenv_values(dotenv_path)
                dotenv_username = env_values.get('GRAPHISTRY_USERNAME')
                dotenv_password = env_values.get('GRAPHISTRY_PASSWORD')
                print(f"Credentials from .env file:")
                print(f"GRAPHISTRY_USERNAME found: {'Yes' if dotenv_username else 'No'}")
                print(f"GRAPHISTRY_PASSWORD found: {'Yes' if dotenv_password else 'No'}")
            
            # Compare credentials
            if env_username and dotenv_username and env_username != dotenv_username:
                print(f"⚠️ WARNING: Username in environment ({env_username}) differs from .env file ({dotenv_username})")
            
            # Try with environment credentials first
            if env_username and env_password:
                print(f"\nTest 1: Using credentials from environment variables - Username: {env_username}")
                try:
                    # Get connection parameters from environment variables with defaults
                    api_version = os.environ.get("GRAPHISTRY_API", "3")
                    protocol = os.environ.get("GRAPHISTRY_PROTOCOL", "https")
                    server = os.environ.get("GRAPHISTRY_HOST", "hub.graphistry.com")
                    client_protocol_hostname = os.environ.get("GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME", f"{protocol}://{server}")
                    
                    # Try to convert api to int if possible
                    try:
                        api_version = int(api_version)
                    except ValueError:
                        api_version = 3
                    
                    print(f"Connecting with: API={api_version}, protocol={protocol}, server={server}")
                    
                    graphistry.register(
                        api=api_version,
                        protocol=protocol,
                        server=server,
                        client_protocol_hostname=client_protocol_hostname,
                        username=env_username,
                        password=env_password
                    )
                    print("✅ Successfully registered with Graphistry API using environment credentials")
                except Exception as auth_error:
                    print(f"❌ Authentication error with environment credentials: {auth_error}")
            else:
                print("\nNo credentials found in environment variables to test.")
            
            # Try with .env file credentials
            if dotenv_username and dotenv_password:
                print(f"\nTest 2: Using credentials from .env file - Username: {dotenv_username}")
                try:
                    # Get connection parameters from environment variables with defaults
                    api_version = os.environ.get("GRAPHISTRY_API", "3")
                    protocol = os.environ.get("GRAPHISTRY_PROTOCOL", "https")
                    server = os.environ.get("GRAPHISTRY_HOST", "hub.graphistry.com")
                    client_protocol_hostname = os.environ.get("GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME", f"{protocol}://{server}")
                    
                    # Try to convert api to int if possible
                    try:
                        api_version = int(api_version)
                    except ValueError:
                        api_version = 3
                    
                    print(f"Connecting with: API={api_version}, protocol={protocol}, server={server}")
                    
                    graphistry.register(
                        api=api_version,
                        protocol=protocol,
                        server=server,
                        client_protocol_hostname=client_protocol_hostname,
                        username=dotenv_username,
                        password=dotenv_password
                    )
                    print("✅ Successfully registered with Graphistry API using .env file credentials")
                except Exception as auth_error:
                    print(f"❌ Authentication error with .env file credentials: {auth_error}")
            else:
                print("\nNo credentials found in .env file to test.")
        except Exception as e:
            print(f"❌ Error testing Graphistry API connection: {e}")
            print(f"Error testing Graphistry API connection: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Server startup took {time.time() - START_TIME:.2f} seconds", file=sys.stderr)
        print("Graphistry MCP server ready to process requests", file=sys.stderr)
    
    # Run the server using the appropriate method
    # We need to handle both run_stdio_async and run methods
    if hasattr(mcp, 'run_stdio_async'):
        logger.info("Using run_stdio_async transport method")
        print("Using run_stdio_async transport method", file=sys.stderr)
        asyncio.run(mcp.run_stdio_async())
    elif hasattr(mcp, 'run'):
        logger.info("Using run transport method")
        print("Using run transport method", file=sys.stderr)
        asyncio.run(mcp.run())
    else:
        logger.error("No valid transport method found on mcp object")
        print("Error: No valid transport method found on mcp object", file=sys.stderr)
        print(f"Available attributes on mcp: {dir(mcp)}", file=sys.stderr)
        sys.exit(1)
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    print(f"Import error: {e}", file=sys.stderr)
    
    # Print Python path to help debug
    logger.error(f"Python path: {sys.path}")
    print(f"Python path: {sys.path}", file=sys.stderr)
    
    # Print sys.prefix to check installation location
    print(f"sys.prefix: {sys.prefix}", file=sys.stderr)
    print(f"sys.base_prefix: {sys.base_prefix}", file=sys.stderr)
    
    sys.exit(1)
        
except Exception as e:
    logger.error(f"Error running server: {e}")
    print(f"Error running server: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)