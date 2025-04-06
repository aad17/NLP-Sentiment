#!/usr/bin/env python3
"""
Twitter Data Sentiment Analysis Model Training

This script trains a sentiment analysis model using Twitter data
and evaluates its performance.
"""
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_label, map_sentiment_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('twitter_model')

# Model file paths
MODEL_PATH = "app/models/twitter_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/twitter_vectorizer.pkl"
TWITTER_DATA_PATH = "Twitter_Data.csv"

def load_twitter_data(csv_path=TWITTER_DATA_PATH, sample_size=None):
    """
    Load Twitter data from CSV file and prepare it for model training.
    
    Args:
        csv_path (str): Path to the CSV file
        sample_size (int): Number of samples per class to use (for balanced dataset)
        
    Returns:
        tuple: X (texts) and y (labels) arrays
    """
    logger.info(f"Loading Twitter data from {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV file")
        
        # Display column names
        logger.info(f"Columns in the dataset: {df.columns.tolist()}")
        
        # Use the specific column names from the Twitter dataset
        text_column = 'clean_text'
        label_column = 'category'
        
        logger.info(f"Using column '{text_column}' for text and '{label_column}' for labels")
        
        # Remove rows with missing text or labels
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        
        if missing_text > 0 or missing_labels > 0:
            logger.warning(f"Found {missing_text} missing text values and {missing_labels} missing label values. Dropping these rows.")
            df = df.dropna(subset=[text_column, label_column])
            logger.info(f"After dropping missing values: {len(df)} rows")
        
        # Map sentiment labels to our format (-1, 0, 1)
        # The Twitter dataset already uses -1, 0, 1 labels
        label_values = df[label_column].unique()
        logger.info(f"Unique sentiment labels in the dataset: {label_values}")
        
        # Create a label mapping based on the values in the dataset
        if df[label_column].dtype == float or df[label_column].dtype == int:
            # Convert float labels to int for consistency
            df['mapped_sentiment'] = df[label_column].round().astype(int)
        else:
            # Map string labels if needed
            label_mapping = {
                '-1': -1, '-1.0': -1, 'negative': -1, 'Negative': -1,
                '0': 0, '0.0': 0, 'neutral': 0, 'Neutral': 0,
                '1': 1, '1.0': 1, 'positive': 1, 'Positive': 1
            }
            
            # Apply mapping
            df['mapped_sentiment'] = df[label_column].astype(str).map(label_mapping)
            
            # Check if mapping was successful
            if df['mapped_sentiment'].isna().sum() > 0:
                logger.warning(f"Could not map {df['mapped_sentiment'].isna().sum()} labels. Filling with neutral sentiment (0).")
                df['mapped_sentiment'] = df['mapped_sentiment'].fillna(0)
        
        # Create balanced dataset if sample_size is specified
        if sample_size is not None:
            negative_samples = df[df['mapped_sentiment'] == -1]
            neutral_samples = df[df['mapped_sentiment'] == 0]
            positive_samples = df[df['mapped_sentiment'] == 1]
            
            logger.info(f"Original dataset distribution: Negative={len(negative_samples)}, "
                       f"Neutral={len(neutral_samples)}, Positive={len(positive_samples)}")
            
            # Check if we have enough samples for each class
            if len(negative_samples) < sample_size or len(neutral_samples) < sample_size or len(positive_samples) < sample_size:
                logger.warning(f"Not enough samples for balanced dataset. Using maximum available: "
                              f"Negative={min(sample_size, len(negative_samples))}, "
                              f"Neutral={min(sample_size, len(neutral_samples))}, "
                              f"Positive={min(sample_size, len(positive_samples))}")
                sample_size_neg = min(sample_size, len(negative_samples))
                sample_size_neu = min(sample_size, len(neutral_samples))
                sample_size_pos = min(sample_size, len(positive_samples))
            else:
                sample_size_neg = sample_size_neu = sample_size_pos = sample_size
            
            # Sample with replacement if needed
            if len(negative_samples) < sample_size_neg:
                negative_samples = resample(negative_samples, replace=True, n_samples=sample_size_neg, random_state=42)
            else:
                negative_samples = negative_samples.sample(sample_size_neg, random_state=42)
                
            if len(neutral_samples) < sample_size_neu:
                neutral_samples = resample(neutral_samples, replace=True, n_samples=sample_size_neu, random_state=42)
            else:
                neutral_samples = neutral_samples.sample(sample_size_neu, random_state=42)
                
            if len(positive_samples) < sample_size_pos:
                positive_samples = resample(positive_samples, replace=True, n_samples=sample_size_pos, random_state=42)
            else:
                positive_samples = positive_samples.sample(sample_size_pos, random_state=42)
            
            # Combine the balanced samples
            df_balanced = pd.concat([negative_samples, neutral_samples, positive_samples])
            
            # Shuffle the dataset
            df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Created balanced dataset with {len(df)} samples: "
                       f"Negative={len(df[df['mapped_sentiment'] == -1])}, "
                       f"Neutral={len(df[df['mapped_sentiment'] == 0])}, "
                       f"Positive={len(df[df['mapped_sentiment'] == 1])}")
        
        # Extract features and labels
        X = df[text_column].values
        y = df['mapped_sentiment'].values
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading Twitter data: {str(e)}")
        raise

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
    logger.info("Preprocessing text...")
    X_processed = extract_features(X)
    
    # Split data into training and testing sets
    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Create TF-IDF vectorizer
    logger.info("Creating and training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.85
    )
    
    # Create classifier
    classifier = LinearSVC()
    
    # Train vectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train classifier
    logger.info("Training classifier...")
    classifier.fit(X_train_vectorized, y_train)
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    logger.info(f"Classification report on test set:\n{report}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix on test set:\n{cm}")
    
    return vectorizer, classifier, (X_test, y_test)

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

def evaluate_on_custom_examples(vectorizer, model):
    """
    Evaluate the model on 10 custom examples.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
    """
    # Define 10 custom test examples (not from training data)
    test_examples = [
        {
            "text": "I love the new fire engine, it's so efficient!",
            "expected": 1  # positive
        },
        {
            "text": "Today's shift was exhausting with multiple emergency calls.",
            "expected": -1  # negative
        },
        {
            "text": "The fire station just received new equipment for training.",
            "expected": 0  # neutral
        },
        {
            "text": "Terrible response time from the emergency services today.",
            "expected": -1  # negative
        },
        {
            "text": "Grateful for the quick action of firefighters during today's incident.",
            "expected": 1  # positive
        },
        {
            "text": "The annual inspection of fire extinguishers is scheduled for next week.",
            "expected": 0  # neutral
        },
        {
            "text": "Disappointed with the lack of resources provided to our department.",
            "expected": -1  # negative
        },
        {
            "text": "Happy to announce we've recruited five new team members this month!",
            "expected": 1  # positive
        },
        {
            "text": "The maintenance schedule for trucks has been updated.",
            "expected": 0  # neutral
        },
        {
            "text": "Frustrated with the outdated protocols we're still using.",
            "expected": -1  # negative
        }
    ]
    
    logger.info(f"Evaluating model on {len(test_examples)} custom examples")
    
    correct = 0
    results = []
    
    for i, example in enumerate(test_examples):
        text = example["text"]
        expected = example["expected"]
        
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
    accuracy = correct / len(test_examples)
    logger.info(f"Custom examples accuracy: {accuracy:.4f} ({correct}/{len(test_examples)} correct)")
    
    # Print detailed results
    logger.info("\nDetailed custom test results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        logger.info(f"{result['index']}. {status} Text: {result['text']}")
        logger.info(f"   Expected: {result['expected_label']} ({result['expected_sentiment']}), "
                  f"Predicted: {result['predicted_label']} ({result['predicted_sentiment']}), "
                  f"Confidence: {result['confidence']:.4f}")
        logger.info("")
    
    # Generate summary by sentiment type
    sentiment_results = {
        "positive": {"total": 0, "correct": 0},
        "neutral": {"total": 0, "correct": 0},
        "negative": {"total": 0, "correct": 0}
    }
    
    for result in results:
        label = result["expected_label"]
        sentiment_results[label]["total"] += 1
        if result["correct"]:
            sentiment_results[label]["correct"] += 1
    
    logger.info("Summary by sentiment type:")
    for label, counts in sentiment_results.items():
        if counts["total"] > 0:
            acc = counts["correct"] / counts["total"] * 100
            logger.info(f"{label.capitalize()}: {counts['correct']}/{counts['total']} correct ({acc:.1f}%)")
    
    return accuracy, results

def main():
    """Main function to train and evaluate the Twitter-based model."""
    logger.info("Starting Twitter data-based model training")
    
    # Load Twitter data
    X, y = load_twitter_data(sample_size=300)  # Use 300 samples per class for balance
    logger.info(f"Loaded {len(X)} samples for training")
    
    # Train model
    vectorizer, model, test_data = train_sentiment_model(X, y)
    
    # Save model
    save_model(vectorizer, model)
    
    # Evaluate on custom examples
    accuracy, results = evaluate_on_custom_examples(vectorizer, model)
    
    logger.info("Twitter-based model training and evaluation completed")
    return vectorizer, model, accuracy, results

if __name__ == "__main__":
    main() 