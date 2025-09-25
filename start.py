#!/usr/bin/env python3
"""
Quick start script for PainCare AI Management Interface
Use this for development and testing
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def print_status(message):
    print(f"[INFO] {message}")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_error(message):
    print(f"[ERROR] {message}")

def check_requirements():
    """Check if requirements are installed"""
    try:
        import flask
        import flask_socketio
        import psutil
        print_success("Required packages are installed")
        return True
    except ImportError as e:
        print_error(f"Missing required packages: {e}")
        print_status("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print_success("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print_error("Failed to install requirements")
            return False

def start_api_server():
    """Start the API server in background"""
    print_status("Starting API server...")
    try:
        # Start API server as background process
        api_process = subprocess.Popen([
            sys.executable, "run_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Give it time to start
        
        if api_process.poll() is None:
            print_success("API server started on port 8000")
            return api_process
        else:
            print_error("API server failed to start")
            return None
    except Exception as e:
        print_error(f"Failed to start API server: {e}")
        return None

def start_management_server():
    """Start the management server"""
    print_status("Starting management interface...")
    try:
        subprocess.run([sys.executable, "management_server.py"])
    except KeyboardInterrupt:
        print_status("Management server stopped by user")
    except Exception as e:
        print_error(f"Management server error: {e}")

def main():
    print("🚀 PainCare AI - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("management_server.py").exists():
        print_error("Please run this script from the PainCare_Model directory")
        sys.exit(1)
    
    # Check and install requirements
    if not check_requirements():
        sys.exit(1)
    
    # Start API server
    api_process = start_api_server()
    
    try:
        # Start management server (blocking)
        print_status("Management interface will be available at: http://localhost:5000")
        print_status("Press Ctrl+C to stop all services")
        print("=" * 50)
        
        # Open browser after a short delay
        import threading
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5000")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        start_management_server()
        
    finally:
        # Clean up API server
        if api_process and api_process.poll() is None:
            print_status("Stopping API server...")
            api_process.terminate()
            api_process.wait()
        print_success("All services stopped")

if __name__ == "__main__":
    main()