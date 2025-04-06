#!/usr/bin/env python3
"""
Enhanced Model Training Script for Sentiment Analysis

This script trains an enhanced sentiment analysis model with 300 samples per class
and evaluates it on 10 test samples.
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_label, map_sentiment_value
from app.utils.mock_data_generator import (
    POSITIVE_REPORTS, NEUTRAL_REPORTS, NEGATIVE_REPORTS, 
    generate_variant, generate_specific_incident
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_model')

# Model file paths
MODEL_PATH = "app/models/enhanced_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/enhanced_vectorizer.pkl"

def generate_enhanced_training_data(num_samples=300):
    """
    Generate enhanced training data with variants.
    
    Args:
        num_samples (int): Number of samples per sentiment class
        
    Returns:
        tuple: X (text) and y (labels) arrays
    """
    positive_texts = []
    neutral_texts = []
    negative_texts = []
    
    # Generate the enhanced dataset
    for _ in range(num_samples):
        # We use the base templates but create variants
        positive_texts.append(generate_variant(np.random.choice(POSITIVE_REPORTS)))
        neutral_texts.append(generate_variant(np.random.choice(NEUTRAL_REPORTS)))
        negative_texts.append(generate_variant(np.random.choice(NEGATIVE_REPORTS)))
    
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
        max_features=10000,  # Increased from 5000
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
    logger.info(f"Model accuracy on validation set: {accuracy:.4f}")
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    logger.info(f"Classification report on validation set:\n{report}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix on validation set:\n{cm}")
    
    return vectorizer, classifier

def save_model(vectorizer, model):
    """
    Save trained vectorizer and model to disk.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save vectorizer and model
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    
    logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")
    logger.info(f"Model saved to {MODEL_PATH}")

def generate_test_samples():
    """
    Generate 10 test samples with known sentiment labels.
    
    Returns:
        list: List of test samples with expected labels
    """
    test_samples = []
    
    # Generate balanced test samples
    for _ in range(3):
        test_samples.append({
            "text": generate_specific_incident("positive")["report"],
            "expected_sentiment": 1
        })
        
    for _ in range(3):
        test_samples.append({
            "text": generate_specific_incident("neutral")["report"],
            "expected_sentiment": 0
        })
        
    for _ in range(4):
        test_samples.append({
            "text": generate_specific_incident("negative")["report"],
            "expected_sentiment": -1
        })
    
    # Shuffle the test samples
    np.random.shuffle(test_samples)
    
    return test_samples

def predict_sentiment(text, vectorizer, model):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input text
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
        
    Returns:
        tuple: (sentiment_value, sentiment_label, confidence)
    """
    try:
        # Preprocess text
        processed_text = extract_features([text])[0]
        
        # Vectorize text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        sentiment_value = model.predict(text_vectorized)[0]
        
        # Get confidence score (distance from decision boundary)
        confidence = np.abs(model.decision_function(text_vectorized)[0])
        if isinstance(confidence, np.ndarray):
            confidence = np.mean(confidence)
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, confidence / 2.0)
        
        # Map sentiment value to label
        sentiment_label = map_sentiment_value(sentiment_value)
        
        return sentiment_value, sentiment_label, confidence
    
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return 0, "neutral", 0.0

def evaluate_model_on_test_samples(vectorizer, model):
    """
    Evaluate the model on test samples.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
    """
    test_samples = generate_test_samples()
    
    logger.info(f"Evaluating model on {len(test_samples)} test samples")
    
    correct = 0
    results = []
    
    for i, sample in enumerate(test_samples):
        text = sample["text"]
        expected = sample["expected_sentiment"]
        
        # Predict sentiment
        sentiment_value, sentiment_label, confidence = predict_sentiment(text, vectorizer, model)
        
        # Check if prediction is correct
        is_correct = sentiment_value == expected
        if is_correct:
            correct += 1
            
        # Store result
        results.append({
            "index": i+1,
            "text": text,
            "predicted_sentiment": sentiment_value,
            "predicted_label": sentiment_label,
            "expected_sentiment": expected,
            "expected_label": map_sentiment_value(expected),
            "confidence": confidence,
            "correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = correct / len(test_samples)
    logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{len(test_samples)} correct)")
    
    # Print detailed results
    logger.info("\nDetailed test results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        logger.info(f"{result['index']}. {status} Text: {result['text']}")
        logger.info(f"   Expected: {result['expected_label']} ({result['expected_sentiment']}), "
                  f"Predicted: {result['predicted_label']} ({result['predicted_sentiment']}), "
                  f"Confidence: {result['confidence']:.4f}")
        logger.info("")
    
    return accuracy, results

def main():
    """Main function to train and evaluate the enhanced model."""
    logger.info("Starting enhanced model training")
    
    # Generate training data
    X, y = generate_enhanced_training_data(num_samples=300)
    logger.info(f"Generated {len(X)} training samples ({X[:5]})")
    
    # Train model
    vectorizer, model = train_sentiment_model(X, y)
    
    # Save model
    save_model(vectorizer, model)
    
    # Evaluate model on test samples
    accuracy, results = evaluate_model_on_test_samples(vectorizer, model)
    
    logger.info("Enhanced model training and evaluation completed")
    return accuracy, results

if __name__ == "__main__":
    main() 