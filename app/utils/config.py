"""
Configuration settings for the Sentiment Analysis application.
This file contains all the configurable parameters and constants used throughout the application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Configurations
MOCK_API_HOST = os.getenv("MOCK_API_HOST", "http://localhost")
MOCK_API_PORT = os.getenv("MOCK_API_PORT", "5001")
MODEL_API_HOST = os.getenv("MODEL_API_HOST", "http://localhost")
MODEL_API_PORT = os.getenv("MODEL_API_PORT", "5002")

MOCK_API_URL = f"{MOCK_API_HOST}:{MOCK_API_PORT}/get_incidents"
MODEL_API_URL = f"{MODEL_API_HOST}:{MODEL_API_PORT}/predict"

# Database Configuration
DB_PATH = os.path.join(BASE_DIR, "app", "data", "incidents.db")

# Model Configuration
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "app", "models", "vectorizer.pkl")

# Sentiment Labels
SENTIMENT_LABELS = {
    1: "Positive",
    0: "Neutral",
    -1: "Negative"
}

# Dashboard Configuration
DASHBOARD_TITLE = "Fire Brigade Incident Sentiment Analysis"
REFRESH_INTERVAL = 60  # seconds 