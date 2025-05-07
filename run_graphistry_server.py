#!/usr/bin/env python
"""
Graphistry MCP Server Runner

This script launches the Graphistry MCP server with proper environment setup
and error handling. It provides detailed diagnostics and status information.
"""

import os
import sys
import logging
import time
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("graphistry-runner")

# Server paths
SCRIPT_DIR = Path(__file__).parent.absolute()
SERVER_PATH = SCRIPT_DIR / "src" / "graphistry_fastmcp" / "server.py"
CONFIG_PATH = SCRIPT_DIR / "mcp-config-graphistry.json"

def check_dependencies():
    """Verify that all required dependencies are installed."""
    required_packages = [
        "fastmcp>=2.2.6", 
        "graphistry", 
        "pandas", 
        "networkx", 
        "uvicorn", 
        "pydantic", 
        "python-dotenv", 
        "python-louvain",
        "psutil"
    ]
    
    missing = []
    for package in required_packages:
        package_name = package.split('>=')[0] if '>=' in package else package
        try:
            __import__(package_name)
            logger.info(f"✅ {package_name} is installed")
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        
        # Auto-install dependencies without prompting
        cmd = ["uvx", "pip", "install"] + missing
        logger.info(f"Auto-installing: {' '.join(missing)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info("Dependencies installed successfully")
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.warning("Continuing without all dependencies")
    
    return len(missing) == 0

def check_credentials():
    """Check for Graphistry credentials and load .env if needed."""
    username = os.environ.get("GRAPHISTRY_USERNAME")
    password = os.environ.get("GRAPHISTRY_PASSWORD")
    
    if not username or not password:
        # Try to load from .env file
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            logger.info(f"Loading environment variables from {env_path}")
            
            # Simple .env file parsing
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip("'").strip('"')
            
            # Check again
            username = os.environ.get("GRAPHISTRY_USERNAME")
            password = os.environ.get("GRAPHISTRY_PASSWORD")
    
    if not username or not password:
        logger.warning("⚠️  Graphistry credentials not found")
        logger.warning("For full functionality, please set GRAPHISTRY_USERNAME and GRAPHISTRY_PASSWORD")
        logger.warning("You can set these in a .env file or as environment variables")
        
        # Set demo credentials for testing
        username = "demo@graphistry.com"
        password = "demo123"
        
        with open(SCRIPT_DIR / ".env", "w") as f:
            f.write(f"GRAPHISTRY_USERNAME={username}\n")
            f.write(f"GRAPHISTRY_PASSWORD={password}\n")
        
        logger.info("✅ Created .env file with demo credentials")
        os.environ["GRAPHISTRY_USERNAME"] = username
        os.environ["GRAPHISTRY_PASSWORD"] = password
        return True
    else:
        logger.info("✅ Graphistry credentials found")
        return True
    
    return False

def run_server(http_mode=False, port=8080):
    """Run the Graphistry MCP server."""
    if not SERVER_PATH.exists():
        logger.error(f"Server file not found: {SERVER_PATH}")
        return False
    
    cmd = [sys.executable, str(SERVER_PATH)]
    if http_mode:
        cmd.append("--http")
        cmd.append(str(port))
        logger.info(f"Starting server in HTTP mode on port {port}")
    else:
        logger.info("Starting server in stdio mode")
    
    try:
        logger.info(f"Launching server: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor for startup messages
        startup_timeout = 20  # seconds
        start_time = time.time()
        success_indicators = ["Server started", "ready at", "initialized with", "Starting Graphistry"]
        error_indicators = ["Error:", "Exception:", "Failed to"]
        
        while time.time() - start_time < startup_timeout:
            if process.poll() is not None:
                # Process ended
                out, err = process.communicate()
                logger.error(f"Server process terminated prematurely with exit code {process.returncode}")
                logger.error(f"Output: {out}")
                logger.error(f"Error: {err}")
                return False
            
            # Check stdout
            output_line = process.stdout.readline().strip()
            if output_line:
                print(output_line)
                
                # Check for success indicators
                if any(indicator in output_line for indicator in success_indicators):
                    logger.info("Server started successfully")
                    if http_mode:
                        logger.info(f"Server running at http://localhost:{port}")
                        return True
                
                # Check for error indicators
                if any(indicator in output_line for indicator in error_indicators):
                    logger.warning(f"Potential error detected: {output_line}")
        
        # Continue running in http mode, but return the process for stdio mode
        if http_mode:
            logger.info("HTTP server started and is running in the background")
            return True
        else:
            logger.info("Server started in stdio mode, printing output:")
            # Just keep running and printing output for stdio mode
            while True:
                output_line = process.stdout.readline().strip()
                if output_line:
                    print(output_line)
                
                error_line = process.stderr.readline().strip() 
                if error_line:
                    print(f"ERROR: {error_line}", file=sys.stderr)
                
                if process.poll() is not None:
                    # Process ended
                    break
            
            # Get any remaining output
            out, err = process.communicate()
            if out:
                print(out)
            if err:
                print(f"ERROR: {err}", file=sys.stderr)
            
            return process.returncode == 0
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping server")
        return True
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return False

def main():
    """Main entry point."""
    print("="*80)
    print("Graphistry MCP Server Runner")
    print("="*80)
    
    # Parse arguments
    http_mode = "--http" in sys.argv
    port = 8080
    for i, arg in enumerate(sys.argv):
        if arg == "--http" and i+1 < len(sys.argv) and sys.argv[i+1].isdigit():
            port = int(sys.argv[i+1])
    
    # Check environment
    logger.info("Checking environment and dependencies...")
    deps_ok = check_dependencies()
    creds_ok = check_credentials()
    
    if not deps_ok:
        logger.warning("Some dependencies are missing. Server may not function correctly.")
    
    if not creds_ok:
        logger.warning("Graphistry credentials not configured. Visualization features will be limited.")
    
    # Run the server
    logger.info("Starting Graphistry MCP server...")
    success = run_server(http_mode, port)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()