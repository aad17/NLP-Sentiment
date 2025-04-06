#!/usr/bin/env python3
"""
System Testing Script for Sentiment Analysis Application

This script tests the essential components of the application
to ensure they're working correctly.
"""
import os
import sys
import logging
import requests
import sqlite3
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system_test')

# Define API endpoints
MOCK_API_URL = "http://localhost:5001/get_incidents"
MODEL_API_URL = "http://localhost:5002/predict"
MOCK_API_HEALTH_URL = "http://localhost:5001/health"
MODEL_API_HEALTH_URL = "http://localhost:5002/health"

# Database path
DB_PATH = os.path.join("app", "data", "incidents.db")

def test_database():
    """Test the database connection and structure."""
    try:
        if not os.path.exists(DB_PATH):
            logger.error(f"Database file not found: {DB_PATH}")
            return False
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check incidents table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'")
        if not cursor.fetchone():
            logger.error("Incidents table not found in database")
            conn.close()
            return False
        
        # Check sentiment table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment'")
        if not cursor.fetchone():
            logger.error("Sentiment table not found in database")
            conn.close()
            return False
        
        # Check if data exists
        cursor.execute("SELECT COUNT(*) FROM incidents")
        incidents_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sentiment")
        sentiment_count = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"Database check passed: Found {incidents_count} incidents and {sentiment_count} sentiment records")
        return True
    
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        return False

def test_mock_api():
    """Test the mock API health and functionality."""
    try:
        # Check health endpoint
        response = requests.get(MOCK_API_HEALTH_URL)
        response.raise_for_status()
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            logger.error(f"Mock API health check failed: {health_data}")
            return False
        
        # Check get_incidents endpoint
        response = requests.get(f"{MOCK_API_URL}?count=2")
        response.raise_for_status()
        
        incidents = response.json()
        if not isinstance(incidents, list) or len(incidents) != 2:
            logger.error(f"Mock API returned unexpected data: {incidents}")
            return False
        
        # Verify incident structure
        required_fields = ['incident_id', 'report', 'timestamp']
        for incident in incidents:
            for field in required_fields:
                if field not in incident:
                    logger.error(f"Incident missing required field '{field}': {incident}")
                    return False
        
        logger.info("Mock API check passed")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Mock API test failed: {str(e)}")
        return False

def test_model_api():
    """Test the model API health and prediction functionality."""
    try:
        # Check health endpoint
        response = requests.get(MODEL_API_HEALTH_URL)
        response.raise_for_status()
        
        health_data = response.json()
        logger.info(f"Model API health response: {health_data}")
        
        if health_data.get("status") != "healthy":
            logger.error(f"Model API health check failed: {health_data}")
            return False
        
        # If model isn't loaded, this test can't proceed
        if not health_data.get("model_loaded", False):
            logger.error("Model not loaded in the Model API")
            return False
        
        # Check prediction endpoint with sample text
        test_text = "Successfully rescued family from burning building with no injuries."
        
        response = requests.post(
            MODEL_API_URL,
            json={"text": test_text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        prediction = response.json()
        if "sentiment" not in prediction or "confidence" not in prediction:
            logger.error(f"Model API returned unexpected prediction: {prediction}")
            return False
        
        logger.info(f"Model API check passed: Prediction={prediction['sentiment']}, Confidence={prediction['confidence']}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Model API test failed: {str(e)}")
        return False

def test_model_files():
    """Test if the model files exist."""
    try:
        model_path = os.path.join("app", "models", "sentiment_model.pkl")
        vectorizer_path = os.path.join("app", "models", "vectorizer.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        if not os.path.exists(vectorizer_path):
            logger.error(f"Vectorizer file not found: {vectorizer_path}")
            return False
        
        model_size = os.path.getsize(model_path) / 1024  # KB
        vectorizer_size = os.path.getsize(vectorizer_path) / 1024  # KB
        
        logger.info(f"Model files check passed: Model={model_size:.1f}KB, Vectorizer={vectorizer_size:.1f}KB")
        return True
    
    except Exception as e:
        logger.error(f"Model files test failed: {str(e)}")
        return False

def main():
    """Main function to run all tests."""
    print("=== Sentiment Analysis System Test ===")
    print()
    
    # Run tests and track results
    test_results = {
        "Database": test_database(),
        "Mock API": test_mock_api(),
        "Model API": test_model_api(),
        "Model Files": test_model_files()
    }
    
    # Print results summary
    print("\n=== Test Results ===")
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:15}: {status}")
    
    # Final verdict
    all_passed = all(test_results.values())
    print("\n=== System Status ===")
    if all_passed:
        print("All system components are working correctly!")
    else:
        print("Some system components have issues. Check the logs for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 