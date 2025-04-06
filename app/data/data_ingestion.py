"""
Data Ingestion Script for Sentiment Analysis Application

This script fetches incident reports from the API and stores them in the database.
It can be run manually or as a scheduled task.
"""
import os
import sys
import logging
import requests
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.config import MOCK_API_URL
from app.data.database import save_incidents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_ingestion')

def fetch_incidents(count=10):
    """
    Fetch incidents from the API.
    
    Args:
        count (int): Number of incidents to fetch
        
    Returns:
        list: List of incident dictionaries
    """
    try:
        response = requests.get(f"{MOCK_API_URL}?count={count}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        incidents = response.json()
        logger.info(f"Fetched {len(incidents)} incidents from API")
        return incidents
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching incidents from API: {str(e)}")
        return []

def process_incidents(count=10):
    """
    Fetch incidents from API and save them to the database.
    
    Args:
        count (int): Number of incidents to process
        
    Returns:
        int: Number of incidents saved
    """
    incidents = fetch_incidents(count)
    if not incidents:
        logger.warning("No incidents to process")
        return 0
    
    # Save incidents to database
    saved_count = save_incidents(incidents)
    logger.info(f"Processed {saved_count} new incidents")
    return saved_count

def main():
    """Main function to run the data ingestion script."""
    parser = argparse.ArgumentParser(description='Fetch and store incident reports')
    parser.add_argument('--count', type=int, default=10, help='Number of incidents to fetch')
    args = parser.parse_args()
    
    logger.info(f"Starting data ingestion script at {datetime.now().isoformat()}")
    
    try:
        count = process_incidents(args.count)
        logger.info(f"Data ingestion completed. Saved {count} new incidents.")
    except Exception as e:
        logger.error(f"Unhandled error in data ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 