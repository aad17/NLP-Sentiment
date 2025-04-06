#!/usr/bin/env python3
"""
Emergency Services Sentiment Analysis Model Training

This script trains a sentiment analysis model specifically for emergency services
using the processed datasets. It implements a progressive training approach
starting with the existing fire service model and expanding to all emergency services.
"""
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import utility functions
try:
    from app.utils.text_preprocessing import extract_features
except ImportError:
    # Fallback if app module is not available
    def extract_features(texts):
        """Simple text preprocessing - fallback function."""
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = ' '.join(word for word in text.split() if not word.startswith('http'))
            processed_texts.append(text)
        return processed_texts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('emergency_services_model')

# File paths
DATASET_PATH = "emergency_services_dataset_balanced.csv"
EMERGENCY_DATASET_PATH = "emergency_services_dataset_emergency_balanced.csv"
MODEL_PATH = "app/models/emergency_services_model.pkl"
VECTORIZER_PATH = "app/models/emergency_services_vectorizer.pkl"
FIRE_MODEL_PATH = "app/models/enhanced_sentiment_model.pkl"
FIRE_VECTORIZER_PATH = "app/models/enhanced_vectorizer.pkl"

def load_datasets():
    """
    Load the processed datasets.
    
    Returns:
        tuple: (general_df, emergency_df) - Both datasets
    """
    logger.info(f"Loading datasets from {DATASET_PATH} and {EMERGENCY_DATASET_PATH}")
    
    try:
        # Load balanced dataset
        general_df = pd.read_csv(DATASET_PATH)
        logger.info(f"Loaded {len(general_df)} samples from balanced dataset")
        
        # Load emergency-specific dataset
        emergency_df = pd.read_csv(EMERGENCY_DATASET_PATH)
        logger.info(f"Loaded {len(emergency_df)} samples from emergency dataset")
        
        # Display class distribution
        general_dist = general_df['sentiment'].value_counts().to_dict()
        emergency_dist = emergency_df['sentiment'].value_counts().to_dict()
        
        logger.info(f"General dataset distribution: {general_dist}")
        logger.info(f"Emergency dataset distribution: {emergency_dist}")
        
        return general_df, emergency_df
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def preprocess_data(df):
    """
    Preprocess dataset for model training.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing sets
    """
    logger.info("Preprocessing data for model training")
    
    # Extract features and labels
    X = extract_features(df['text'].values)
    y = df['sentiment'].values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def load_existing_model():
    """
    Load the existing fire service sentiment analysis model.
    
    Returns:
        tuple: (vectorizer, model) - Trained vectorizer and model
    """
    logger.info(f"Loading existing fire service model from {FIRE_MODEL_PATH}")
    
    try:
        vectorizer = joblib.load(FIRE_VECTORIZER_PATH)
        model = joblib.load(FIRE_MODEL_PATH)
        logger.info("Successfully loaded existing model")
        return vectorizer, model
    except Exception as e:
        logger.warning(f"Could not load existing model: {str(e)}. Will train from scratch.")
        return None, None

def train_progressive_model(general_df, emergency_df):
    """
    Train a model using progressive approach, starting with fire service data
    and gradually incorporating other emergency services.
    
    Args:
        general_df (pandas.DataFrame): General Twitter dataset
        emergency_df (pandas.DataFrame): Emergency services dataset
        
    Returns:
        tuple: (vectorizer, model) - Trained vectorizer and model
    """
    logger.info("Starting progressive model training")
    
    # Step 1: Load existing fire service model
    fire_vectorizer, fire_model = load_existing_model()
    
    # Step 2: Create new vectorizer with expanded features
    logger.info("Creating new TF-IDF vectorizer with expanded features")
    vectorizer = TfidfVectorizer(
        max_features=15000,  # Increased from 10000
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2)  # Include bigrams
    )
    
    # Step 3: Create classifier
    classifier = LinearSVC(class_weight='balanced')
    
    # Step 4: Progressive training
    # First, train on emergency-specific data
    logger.info("Starting first stage: Training on emergency-specific data")
    X_emergency_train, X_emergency_test, y_emergency_train, y_emergency_test = preprocess_data(emergency_df)
    
    # Fit vectorizer on emergency data
    X_emergency_vectorized = vectorizer.fit_transform(X_emergency_train)
    
    # Train classifier
    logger.info("Training classifier on emergency data")
    classifier.fit(X_emergency_vectorized, y_emergency_train)
    
    # Evaluate emergency-specific model
    X_emergency_test_vectorized = vectorizer.transform(X_emergency_test)
    emergency_accuracy = accuracy_score(y_emergency_test, classifier.predict(X_emergency_test_vectorized))
    logger.info(f"Emergency-specific model accuracy: {emergency_accuracy:.4f}")
    
    # Step 5: Continue training with general data
    logger.info("Starting second stage: Fine-tuning with general Twitter data")
    
    # Sample from general dataset to avoid overwhelming emergency data
    general_sample_size = min(len(emergency_df) * 2, len(general_df))
    general_sample = general_df.sample(general_sample_size, random_state=42)
    logger.info(f"Using {len(general_sample)} samples from general dataset for fine-tuning")
    
    # Preprocess general data
    X_general_train, X_general_test, y_general_train, y_general_test = preprocess_data(general_sample)
    
    # Transform using the same vectorizer
    X_general_vectorized = vectorizer.transform(X_general_train)
    
    # Train on combined data (use a new classifier to preserve the emergency-specific one)
    logger.info("Training on combined dataset")
    combined_classifier = LinearSVC(class_weight='balanced')
    
    # Combine datasets
    X_combined = np.vstack((X_emergency_vectorized.toarray(), X_general_vectorized.toarray()))
    y_combined = np.concatenate((y_emergency_train, y_general_train))
    
    # Train on combined data
    combined_classifier.fit(X_combined, y_combined)
    
    # Evaluate combined model
    X_general_test_vectorized = vectorizer.transform(X_general_test)
    general_accuracy = accuracy_score(y_general_test, combined_classifier.predict(X_general_test_vectorized))
    logger.info(f"Combined model accuracy on general data: {general_accuracy:.4f}")
    
    emergency_accuracy = accuracy_score(y_emergency_test, combined_classifier.predict(X_emergency_test_vectorized))
    logger.info(f"Combined model accuracy on emergency data: {emergency_accuracy:.4f}")
    
    # Step 6: Compare models and choose the best one
    logger.info("Comparing model performances")
    
    # Output classification reports
    emergency_report = classification_report(
        y_emergency_test, 
        combined_classifier.predict(X_emergency_test_vectorized),
        target_names=['negative', 'neutral', 'positive']
    )
    logger.info(f"Combined model classification report on emergency data:\n{emergency_report}")
    
    general_report = classification_report(
        y_general_test, 
        combined_classifier.predict(X_general_test_vectorized),
        target_names=['negative', 'neutral', 'positive']
    )
    logger.info(f"Combined model classification report on general data:\n{general_report}")
    
    return vectorizer, combined_classifier

def evaluate_model(vectorizer, model, df):
    """
    Evaluate the trained model on a dataset.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
        df (pandas.DataFrame): Dataset to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating model on {len(df)} samples")
    
    # Preprocess data
    X = extract_features(df['text'].values)
    y = df['sentiment'].values
    
    # Vectorize
    X_vectorized = vectorizer.transform(X)
    
    # Predict
    y_pred = model.predict(X_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['negative', 'neutral', 'positive'], output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y, y_pred, target_names=['negative', 'neutral', 'positive'])}")
    logger.info(f"Confusion matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def save_model(vectorizer, model):
    """
    Save the trained vectorizer and model to disk.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save vectorizer and model
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    
    logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Also save with timestamp for versioning
    timestamp = int(time.time())
    version_model_path = f"app/models/emergency_services_model_{timestamp}.pkl"
    version_vectorizer_path = f"app/models/emergency_services_vectorizer_{timestamp}.pkl"
    
    joblib.dump(vectorizer, version_vectorizer_path)
    joblib.dump(model, version_model_path)
    
    logger.info(f"Version backup saved with timestamp {timestamp}")

def main():
    """Main function to train the emergency services sentiment analysis model."""
    try:
        logger.info("Starting emergency services model training")
        
        # Step 1: Load datasets
        general_df, emergency_df = load_datasets()
        
        # Step 2: Train progressive model
        vectorizer, model = train_progressive_model(general_df, emergency_df)
        
        # Step 3: Evaluate model on both datasets
        logger.info("Evaluating model on emergency dataset")
        emergency_metrics = evaluate_model(vectorizer, model, emergency_df)
        
        logger.info("Evaluating model on general dataset")
        # Use a sample of general dataset to speed up evaluation
        general_sample = general_df.sample(min(10000, len(general_df)), random_state=42)
        general_metrics = evaluate_model(vectorizer, model, general_sample)
        
        # Step 4: Save model
        save_model(vectorizer, model)
        
        logger.info("Emergency services model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 