#!/usr/bin/env python3
"""
Setup Script for Sentiment Analysis Application

This script performs initial setup for the Sentiment Analysis application:
1. Creates necessary directories
2. Initializes the database
3. Ingests initial data
4. Trains the sentiment model
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup')

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'app/data',
        'app/models',
        'logs'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running: {description}")
    
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        logger.info(f"Command output: {process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up Sentiment Analysis application')
    parser.add_argument('--data-count', type=int, default=50, help='Number of initial data points to ingest')
    args = parser.parse_args()
    
    logger.info("Starting setup for Sentiment Analysis application")
    
    # Create necessary directories
    create_directories()
    
    # Initialize database (this happens automatically when importing the module)
    run_command(
        "python -c 'from app.data.database import init_db; init_db()'",
        "Initializing database"
    )
    
    # Ingest initial data
    if not run_command(
        f"python app/data/data_ingestion.py --count {args.data_count}",
        f"Ingesting initial data ({args.data_count} incidents)"
    ):
        logger.warning("Data ingestion failed, but continuing with setup")
    
    # Train sentiment model
    if not run_command(
        "python app/models/train_model.py",
        "Training sentiment model"
    ):
        logger.error("Model training failed. Dashboard will have limited functionality.")
    
    logger.info("Setup completed successfully!")
    logger.info("\nTo run the application:")
    logger.info("1. Start the mock API:       python app/api/mock_api.py")
    logger.info("2. Start the model API:      python app/api/model_api.py")
    logger.info("3. Launch the dashboard:     streamlit run app/dashboard/dashboard.py")

if __name__ == "__main__":
    main() 