"""
Model Training Script for Sentiment Analysis

This script trains a sentiment analysis model on incident reports.
It uses a simple TF-IDF vectorizer and SVM classifier for sentiment classification.
"""
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.text_preprocessing import extract_features, map_sentiment_label
from app.utils.mock_data_generator import (
    POSITIVE_REPORTS, NEUTRAL_REPORTS, NEGATIVE_REPORTS
)
from app.utils.config import MODEL_PATH, VECTORIZER_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model')

def generate_training_data(num_samples=100):
    """
    Generate training data from mock reports.
    
    Args:
        num_samples (int): Number of samples per sentiment class
        
    Returns:
        tuple: X (text) and y (labels) arrays
    """
    # Get a balanced dataset
    positive_texts = POSITIVE_REPORTS * (num_samples // len(POSITIVE_REPORTS) + 1)
    neutral_texts = NEUTRAL_REPORTS * (num_samples // len(NEUTRAL_REPORTS) + 1)
    negative_texts = NEGATIVE_REPORTS * (num_samples // len(NEGATIVE_REPORTS) + 1)
    
    # Truncate to required number of samples
    positive_texts = positive_texts[:num_samples]
    neutral_texts = neutral_texts[:num_samples]
    negative_texts = negative_texts[:num_samples]
    
    # Create X and y arrays
    X = positive_texts + neutral_texts + negative_texts
    y = ([map_sentiment_label('positive')] * num_samples + 
         [map_sentiment_label('neutral')] * num_samples + 
         [map_sentiment_label('negative')] * num_samples)
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    return X, y

def train_sentiment_model(X, y):
    """
    Train a sentiment analysis model.
    
    Args:
        X (list): List of text samples
        y (list): List of sentiment labels
        
    Returns:
        tuple: Trained vectorizer and model
    """
    # Preprocess text
    X_processed = extract_features(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.85
    )
    
    # Create classifier
    classifier = LinearSVC()
    
    # Train vectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train classifier
    classifier.fit(X_train_vectorized, y_train)
    
    # Evaluate model
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    logger.info(f"Classification report:\n{report}")
    
    return vectorizer, classifier

def save_model(vectorizer, model, vectorizer_path=VECTORIZER_PATH, model_path=MODEL_PATH):
    """
    Save trained vectorizer and model to disk.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
        vectorizer_path (str): Path to save vectorizer
        model_path (str): Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save vectorizer and model
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    
    logger.info(f"Vectorizer saved to {vectorizer_path}")
    logger.info(f"Model saved to {model_path}")

def main():
    """Main function to train the sentiment model."""
    logger.info("Starting model training")
    
    # Generate training data
    X, y = generate_training_data(num_samples=50)
    logger.info(f"Generated {len(X)} training samples")
    
    # Train model
    vectorizer, model = train_sentiment_model(X, y)
    
    # Save model
    save_model(vectorizer, model)
    
    logger.info("Model training completed")

if __name__ == "__main__":
    main() 