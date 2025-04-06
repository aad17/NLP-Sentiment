#!/usr/bin/env python3
"""
Run Script for Sentiment Analysis Application

This script starts all components of the Sentiment Analysis application in the correct order:
1. Mock API (simulates the fire brigade incident API)
2. Model API (serves sentiment predictions)
3. Streamlit dashboard (visualizes the data)

Each component is started in a separate process.
"""
import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import configuration
from app.utils.config import MOCK_API_PORT, MODEL_API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run')

# List to store running processes
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C to stop all processes."""
    logger.info("Shutting down all processes...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

def run_component(command, name, wait_time=2):
    """Run a component in a separate process."""
    logger.info(f"Starting {name}...")
    
    try:
        # Use Popen instead of run to start process in background
        process = subprocess.Popen(
            command,
            shell=True,
            text=True
        )
        processes.append(process)
        
        # Wait for process to start
        time.sleep(wait_time)
        
        if process.poll() is not None:
            logger.error(f"{name} failed to start.")
            return False
        
        logger.info(f"{name} started successfully (PID: {process.pid})")
        return True
    
    except Exception as e:
        logger.error(f"Error starting {name}: {str(e)}")
        return False

def main():
    """Main function to run all components."""
    parser = argparse.ArgumentParser(description='Run Sentiment Analysis application')
    parser.add_argument('--setup', action='store_true', help='Run setup script before starting components')
    parser.add_argument('--no-dashboard', action='store_true', help='Do not start the dashboard')
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Sentiment Analysis application...")
    logger.info(f"Mock API will run on port {MOCK_API_PORT}")
    logger.info(f"Model API will run on port {MODEL_API_PORT}")
    
    # Run setup script if requested
    if args.setup:
        logger.info("Running setup script...")
        subprocess.run("python setup.py", shell=True, check=True)
    
    # Start Mock API
    if not run_component("python app/api/mock_api.py", "Mock API"):
        logger.error("Failed to start Mock API. Exiting.")
        return
    
    # Start Model API
    if not run_component("python app/api/model_api.py", "Model API"):
        logger.error("Failed to start Model API. Exiting.")
        return
    
    # Start Streamlit dashboard
    if not args.no_dashboard:
        run_component("streamlit run app/dashboard/dashboard.py", "Streamlit Dashboard", wait_time=5)
    
    logger.info("All components started successfully!")
    logger.info("Press Ctrl+C to stop all processes.")
    
    # Keep the script running until Ctrl+C
    try:
        while True:
            # Check if any process has terminated
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    logger.error(f"Process {process.pid} has terminated unexpectedly.")
                    # Remove terminated process from list
                    processes.pop(i)
            
            # Exit if all processes have terminated
            if not processes:
                logger.error("All processes have terminated. Exiting.")
                return
            
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 