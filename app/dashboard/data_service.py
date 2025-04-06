"""
Data Service for Dashboard

This module provides functions to fetch and process data for the dashboard.
Supports multiple sentiment model types: synthetic, twitter, hybrid, and domain-aware.
"""
import os
import sys
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.data.database import get_incidents, get_sentiment_stats
from app.utils.config import MOCK_API_URL, MODEL_API_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_service')

def fetch_raw_incidents(count=20):
    """
    Fetch raw incidents from the API.
    
    Args:
        count (int): Number of incidents to fetch
        
    Returns:
        list: List of incident dictionaries
    """
    try:
        response = requests.get(f"{MOCK_API_URL}?count={count}")
        response.raise_for_status()
        incidents = response.json()
        logger.info(f"Fetched {len(incidents)} raw incidents from API")
        return incidents
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching incidents from API: {str(e)}")
        return []

def get_incidents_from_db(limit=100):
    """
    Get incidents from the database.
    
    Args:
        limit (int): Maximum number of incidents to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with incident data
    """
    try:
        incidents = get_incidents(limit=limit)
        df = pd.DataFrame(incidents)
        
        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Map sentiment to string labels
            sentiment_map = {1: 'positive', 0: 'neutral', -1: 'negative', None: 'not analyzed'}
            df['sentiment_label'] = df['sentiment'].map(lambda x: sentiment_map.get(x, 'not analyzed'))
            
            logger.info(f"Retrieved {len(df)} incidents from database")
        else:
            logger.warning("No incidents found in database")
            
        return df
    
    except Exception as e:
        logger.error(f"Error getting incidents from database: {str(e)}")
        return pd.DataFrame()

def get_sentiment_statistics():
    """
    Get sentiment statistics from the database.
    
    Returns:
        dict: Sentiment statistics
    """
    try:
        stats = get_sentiment_stats()
        logger.info(f"Retrieved sentiment statistics: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error retrieving sentiment statistics: {str(e)}")
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}

def get_available_models():
    """
    Get available sentiment models from the API.
    
    Returns:
        list: List of available model types
    """
    try:
        response = requests.get(f"{MODEL_API_URL.rstrip('/predict')}/models")
        response.raise_for_status()
        result = response.json()
        models = result.get('models', [])
        
        # Ensure domain-aware model is included in the list
        if "domain_aware" not in models:
            models.append("domain_aware")
            
        logger.info(f"Available models: {models}")
        return models
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting available models: {str(e)}")
        # Default to synthetic and domain-aware models if can't get list
        return ["synthetic", "domain_aware"]

def predict_sentiment(text, model_type=None, store_for_feedback=False):
    """
    Get sentiment prediction for a text using the model API.
    
    Args:
        text (str): Text to analyze
        model_type (str, optional): Type of model to use
        store_for_feedback (bool): Whether to store prediction for feedback
        
    Returns:
        dict: Prediction result
    """
    try:
        data = {"text": text}
        
        # Add model_type if specified
        if model_type:
            data["model_type"] = model_type
            
        # Set store_for_feedback based on parameter
        data["store_for_feedback"] = store_for_feedback
        
        # Log request details
        logger.info(f"Sending prediction request with store_for_feedback={store_for_feedback}, model_type={model_type}")
            
        # Handle domain-aware model separately if needed
        if model_type == "domain_aware":
            try:
                # First try calling the normal API endpoint
                logger.info(f"Calling API at: {MODEL_API_URL} with data: {data}")
                response = requests.post(
                    f"{MODEL_API_URL}",
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException:
                # If the normal endpoint doesn't support domain-aware model,
                # try a dedicated endpoint
                try:
                    domain_api_url = f"{MODEL_API_URL.rstrip('/predict')}/domain_predict"
                    logger.info(f"Falling back to domain API at: {domain_api_url}")
                    response = requests.post(
                        domain_api_url,
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.RequestException as e2:
                    # Fall back to hybrid model
                    logger.warning(f"Domain-aware model API not available: {str(e2)}")
                    data["model_type"] = "hybrid"
                    response = requests.post(
                        f"{MODEL_API_URL}",
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                    # Add domain-aware tag so UI knows this was a fallback
                    result["model_type"] = "domain_aware (fallback to hybrid)"
        else:
            # For other models, use standard API
            logger.info(f"Calling API at: {MODEL_API_URL} with data: {data}")
            response = requests.post(
                f"{MODEL_API_URL}",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
        logger.info(f"Received prediction for text using {model_type or 'default'} model: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting prediction: {str(e)}")
        return {
            "sentiment": "neutral",
            "sentiment_value": 0,
            "confidence": 0.0,
            "text": text,
            "model_type": model_type or "unknown",
            "error": str(e)
        }

def compare_models(text):
    """
    Compare sentiment predictions from all available models.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Comparison results from all models
    """
    try:
        response = requests.post(
            f"{MODEL_API_URL.rstrip('/predict')}/compare",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received model comparison for text")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error comparing models: {str(e)}")
        return {
            "text": text,
            "models": {},
            "count": 0,
            "error": str(e)
        }

def get_sentiment_over_time(days=30):
    """
    Get sentiment trends over time.
    
    Args:
        days (int): Number of days to include
        
    Returns:
        pandas.DataFrame: DataFrame with daily sentiment counts
    """
    try:
        # Get incidents from database
        df = get_incidents_from_db(limit=1000)
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter for date range
        start_date = datetime.now().date() - timedelta(days=days)
        df = df[df['date'] >= start_date]
        
        # Group by date and sentiment
        sentiment_over_time = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        
        # Pivot table for easier plotting
        pivot_table = sentiment_over_time.pivot(
            index='date', 
            columns='sentiment_label', 
            values='count'
        ).fillna(0)
        
        # Ensure all sentiment labels exist
        for label in ['positive', 'neutral', 'negative', 'not analyzed']:
            if label not in pivot_table.columns:
                pivot_table[label] = 0
        
        pivot_table = pivot_table.reset_index()
        logger.info(f"Generated sentiment trends over {days} days")
        return pivot_table
    
    except Exception as e:
        logger.error(f"Error generating sentiment trends: {str(e)}")
        return pd.DataFrame()

def get_recent_incidents(limit=10):
    """
    Get recent incidents with sentiment.
    
    Args:
        limit (int): Maximum number of incidents to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with recent incidents
    """
    try:
        df = get_incidents_from_db(limit=limit)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp (most recent first)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        
        # Format timestamp for display
        df['formatted_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting recent incidents: {str(e)}")
        return pd.DataFrame()

def get_sentiment_by_day_of_month(month=None, year=None):
    """
    Get sentiment trends for each day of a specific month.
    
    Args:
        month (int): Month to analyze (1-12), defaults to current month
        year (int): Year to analyze, defaults to current year
        
    Returns:
        pandas.DataFrame: DataFrame with daily sentiment counts for the month
    """
    try:
        # Set defaults to current month/year if not provided
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
            
        # Get incidents from database (generous limit to ensure we get all data)
        df = get_incidents_from_db(limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter for the specific month and year
        df = df[(df['timestamp'].dt.month == month) & (df['timestamp'].dt.year == year)]
        
        if df.empty:
            logger.warning(f"No data found for {year}-{month}")
            # Create empty DataFrame with expected structure for consistent return
            return pd.DataFrame(columns=['day', 'positive', 'neutral', 'negative'])
        
        # Extract day of month
        df['day'] = df['timestamp'].dt.day
        
        # Group by day and sentiment
        daily_sentiment = df.groupby(['day', 'sentiment_label']).size().reset_index(name='count')
        
        # Pivot table for easier plotting
        pivot_table = daily_sentiment.pivot(
            index='day', 
            columns='sentiment_label', 
            values='count'
        ).fillna(0)
        
        # Ensure all sentiment labels exist
        for label in ['positive', 'neutral', 'negative']:
            if label not in pivot_table.columns:
                pivot_table[label] = 0
        
        # Keep only the sentiment columns we need
        pivot_table = pivot_table[['positive', 'neutral', 'negative']]
        
        # Reset index to make 'day' a column
        pivot_table = pivot_table.reset_index()
        
        # Make sure all days of month are represented (1-31 or appropriate for month)
        days_in_month = (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day if month < 12 else 31
        all_days = pd.DataFrame({'day': range(1, days_in_month + 1)})
        
        # Merge to ensure all days are included
        result = pd.merge(all_days, pivot_table, on='day', how='left').fillna(0)
        
        logger.info(f"Generated daily sentiment trends for {year}-{month}")
        return result
    
    except Exception as e:
        logger.error(f"Error generating daily sentiment trends: {str(e)}")
        return pd.DataFrame(columns=['day', 'positive', 'neutral', 'negative']) 