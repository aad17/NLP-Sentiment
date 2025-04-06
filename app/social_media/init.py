"""
Social Media Integration Initialization Script

This script initializes all the components for the social media integration.
It can be used to start the mock data generation, API, and batch processing.
"""
import os
import sys
import argparse
import logging
import threading
import time
import subprocess
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_init')

def start_api():
    """Start the mock API in a separate process"""
    api_script = os.path.join(os.path.dirname(__file__), '..', 'api', 'mock_api.py')
    logger.info(f"Starting mock API: {api_script}")
    
    try:
        process = subprocess.Popen([sys.executable, api_script])
        logger.info(f"Mock API started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Error starting mock API: {str(e)}")
        return None

def start_streamlit_auth():
    """Start the social media authentication Streamlit app"""
    auth_script = os.path.join(os.path.dirname(__file__), 'social_media_auth.py')
    logger.info(f"Starting social media authentication app: {auth_script}")
    
    try:
        # Get an available port
        port = 8501
        cmd = [
            "streamlit", "run", auth_script,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(cmd)
        logger.info(f"Streamlit authentication app started with PID: {process.pid}")
        logger.info(f"Access the authentication app at: http://localhost:{port}")
        return process
    except Exception as e:
        logger.error(f"Error starting Streamlit authentication app: {str(e)}")
        return None

def start_streamlit_data():
    """Start the social media data Streamlit app"""
    data_script = os.path.join(os.path.dirname(__file__), 'social_media_data.py')
    logger.info(f"Starting social media data app: {data_script}")
    
    try:
        # Get an available port
        port = 8502
        cmd = [
            "streamlit", "run", data_script,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(cmd)
        logger.info(f"Streamlit data app started with PID: {process.pid}")
        logger.info(f"Access the data app at: http://localhost:{port}")
        return process
    except Exception as e:
        logger.error(f"Error starting Streamlit data app: {str(e)}")
        return None

def start_batch_processor():
    """Start the batch processor in a separate process"""
    processor_script = os.path.join(os.path.dirname(__file__), 'data_processor.py')
    logger.info(f"Starting batch processor: {processor_script}")
    
    try:
        process = subprocess.Popen([sys.executable, processor_script])
        logger.info(f"Batch processor started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Error starting batch processor: {str(e)}")
        return None

def start_all_components():
    """Start all components"""
    processes = {}
    
    # Start API
    processes['api'] = start_api()
    
    # Give API time to start
    time.sleep(2)
    
    # Start batch processor
    processes['processor'] = start_batch_processor()
    
    # Start Streamlit apps
    processes['auth_app'] = start_streamlit_auth()
    processes['data_app'] = start_streamlit_data()
    
    return processes

def stop_processes(processes):
    """Stop all running processes"""
    for name, process in processes.items():
        if process and process.poll() is None:
            logger.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info(f"{name} stopped.")
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} did not terminate gracefully, killing...")
                process.kill()

def initialize_social_media_integration(components=None, port_api=None, port_auth=None, port_data=None):
    """
    Initialize the social media integration components.
    
    Args:
        components (list): List of components to start ('api', 'auth', 'data', 'processor')
        port_api (int): Port for the API
        port_auth (int): Port for the authentication app
        port_data (int): Port for the data app
        
    Returns:
        dict: Dictionary of processes
    """
    if components is None:
        components = ['api', 'processor', 'auth', 'data']
    
    processes = {}
    
    if 'api' in components:
        if port_api:
            os.environ['MOCK_API_PORT'] = str(port_api)
        processes['api'] = start_api()
        
        # Give API time to start
        time.sleep(2)
    
    if 'processor' in components:
        processes['processor'] = start_batch_processor()
    
    if 'auth' in components:
        if port_auth:
            os.environ['STREAMLIT_SERVER_PORT'] = str(port_auth)
        processes['auth_app'] = start_streamlit_auth()
    
    if 'data' in components:
        if port_data:
            os.environ['STREAMLIT_SERVER_PORT'] = str(port_data)
        processes['data_app'] = start_streamlit_data()
    
    return processes

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Initialize social media integration components")
    
    parser.add_argument(
        "--components",
        type=str,
        choices=["all", "api", "auth", "data", "processor"],
        default="all",
        help="Components to initialize"
    )
    
    parser.add_argument(
        "--port-api",
        type=int,
        default=None,
        help="Port for the API"
    )
    
    parser.add_argument(
        "--port-auth",
        type=int,
        default=None,
        help="Port for the authentication app"
    )
    
    parser.add_argument(
        "--port-data",
        type=int,
        default=None,
        help="Port for the data app"
    )
    
    args = parser.parse_args()
    
    components_map = {
        "all": ['api', 'processor', 'auth', 'data'],
        "api": ['api'],
        "auth": ['auth'],
        "data": ['data'],
        "processor": ['processor']
    }
    
    components = components_map.get(args.components, ['api', 'processor', 'auth', 'data'])
    
    processes = initialize_social_media_integration(
        components=components,
        port_api=args.port_api,
        port_auth=args.port_auth,
        port_data=args.port_data
    )
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
            
            # Check if any process has terminated
            for name, process in list(processes.items()):
                if process and process.poll() is not None:
                    logger.warning(f"{name} has terminated unexpectedly (exit code: {process.returncode})")
                    del processes[name]
            
            # If all processes have terminated, exit
            if not processes:
                logger.warning("All processes have terminated, exiting.")
                break
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping all processes...")
    
    finally:
        stop_processes(processes)
        logger.info("All processes stopped, exiting.")

if __name__ == "__main__":
    main() 