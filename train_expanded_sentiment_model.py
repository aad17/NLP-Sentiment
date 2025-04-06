#!/usr/bin/env python3
"""
Expanded Sentiment Analysis Model Training

This script trains a sentiment analysis model using multiple Twitter datasets
to create a more robust, domain-general sentiment classifier.
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_label, map_sentiment_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('expanded_sentiment_model')

# Model file paths
MODEL_PATH = "app/models/expanded_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/expanded_vectorizer.pkl"

# Dataset paths
TWITTER_DATA_PATH = "Twitter_Data.csv"
TWITTER_TRAINING_PATH = "twitter_training.csv"
LARGE_DATASET_PATH = "training_1600000_processed_noemoticon.csv"

# Test data output
TEST_RESULTS_PATH = "expanded_model_test_results.md"

def load_twitter_data():
    """
    Load the Twitter_Data.csv dataset.
    
    Returns:
        tuple: X (texts) and y (labels) arrays
    """
    logger.info(f"Loading Twitter_Data.csv dataset")
    
    try:
        # Read CSV file
        df = pd.read_csv(TWITTER_DATA_PATH)
        logger.info(f"Loaded {len(df)} rows from Twitter_Data.csv")
        
        # Use the specific column names from the Twitter dataset
        text_column = 'clean_text'
        label_column = 'category'
        
        # Remove rows with missing text or labels
        df = df.dropna(subset=[text_column, label_column])
        
        # Map labels (-1, 0, 1) to our standardized format
        # The Twitter dataset already uses -1, 0, 1 labels
        df['mapped_sentiment'] = df[label_column]
        
        # Extract features and labels
        X = df[text_column].values
        y = df['mapped_sentiment'].values
        
        # Display label distribution
        label_counts = df['mapped_sentiment'].value_counts()
        logger.info(f"Label distribution in Twitter_Data.csv: {label_counts.to_dict()}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading Twitter_Data.csv: {str(e)}")
        raise

def load_twitter_training():
    """
    Load the twitter_training.csv dataset.
    
    Returns:
        tuple: X (texts) and y (labels) arrays or (None, None) if loading fails
    """
    logger.info(f"Loading twitter_training.csv dataset")
    
    try:
        # Read CSV file
        df = pd.read_csv(TWITTER_TRAINING_PATH)
        logger.info(f"Loaded {len(df)} rows from twitter_training.csv")
        
        # The format seems unusual, let's check
        columns = df.columns.tolist()
        logger.info(f"Columns in twitter_training.csv: {columns}")
        
        # Try to identify text and label columns
        if len(columns) >= 4 and 'Positive' in columns:
            # Assuming format is: ID, domain, sentiment, text
            text_column = columns[3]
            label_column = columns[2]
            
            # Remove rows with missing text
            df = df.dropna(subset=[text_column])
            
            # Map sentiment labels to our standard format
            sentiment_map = {
                'Positive': 1,
                'Negative': -1,
                'Other': 0,
                'Neutral': 0
            }
            
            df['mapped_sentiment'] = df[label_column].map(sentiment_map)
            
            # Drop rows where mapping failed
            df = df.dropna(subset=['mapped_sentiment'])
            df['mapped_sentiment'] = df['mapped_sentiment'].astype(int)
            
            # Extract features and labels
            X = df[text_column].values
            y = df['mapped_sentiment'].values
            
            # Display label distribution
            label_counts = df['mapped_sentiment'].value_counts()
            logger.info(f"Label distribution in twitter_training.csv: {label_counts.to_dict()}")
            
            return X, y
        else:
            logger.warning("Could not identify the correct columns in twitter_training.csv. Skipping this dataset.")
            return None, None
        
    except Exception as e:
        logger.error(f"Error loading twitter_training.csv: {str(e)}")
        logger.warning("Skipping twitter_training.csv dataset")
        return None, None

def load_large_dataset(sample_size=None):
    """
    Load the large Twitter dataset (1.6M samples).
    
    Args:
        sample_size (int): Number of samples per class to use (for balanced dataset)
        
    Returns:
        tuple: X (texts) and y (labels) arrays
    """
    logger.info(f"Loading large Twitter dataset (training_1600000_processed_noemoticon.csv)")
    
    try:
        # Read CSV file with Latin-1 encoding
        df = pd.read_csv(LARGE_DATASET_PATH, encoding='latin-1', header=None)
        logger.info(f"Loaded {len(df)} rows from large Twitter dataset")
        
        # Assuming the format is: sentiment, id, date, query, user, text
        # Where sentiment is 0 (negative), 2 (neutral), or 4 (positive)
        df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        
        # Map sentiment values (0, 2, 4) to our standard format (-1, 0, 1)
        sentiment_map = {
            0: -1,  # Negative
            2: 0,   # Neutral
            4: 1    # Positive
        }
        
        df['mapped_sentiment'] = df['sentiment'].map(sentiment_map)
        
        # Display original label distribution
        label_counts = df['sentiment'].value_counts()
        logger.info(f"Original label distribution: {label_counts.to_dict()}")
        
        # Create balanced dataset if sample_size is specified
        if sample_size is not None:
            negative_samples = df[df['mapped_sentiment'] == -1]
            neutral_samples = df[df['mapped_sentiment'] == 0]
            positive_samples = df[df['mapped_sentiment'] == 1]
            
            # Check if we have enough samples for each class
            sample_size_neg = min(sample_size, len(negative_samples))
            sample_size_neu = min(sample_size, len(neutral_samples))
            sample_size_pos = min(sample_size, len(positive_samples))
            
            # Sample without replacement
            negative_samples = negative_samples.sample(sample_size_neg, random_state=42)
            neutral_samples = neutral_samples.sample(sample_size_neu, random_state=42)
            positive_samples = positive_samples.sample(sample_size_pos, random_state=42)
            
            # Combine the balanced samples
            df = pd.concat([negative_samples, neutral_samples, positive_samples])
            
            # Shuffle the dataset
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Created balanced dataset with {len(df)} samples")
        
        # Display final label distribution
        label_counts = df['mapped_sentiment'].value_counts()
        logger.info(f"Final label distribution: {label_counts.to_dict()}")
        
        # Extract features and labels
        X = df['text'].values
        y = df['mapped_sentiment'].values
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading large Twitter dataset: {str(e)}")
        raise

def combine_datasets(datasets):
    """
    Combine multiple datasets into a single training set.
    
    Args:
        datasets (list): List of (X, y) tuples from different datasets
        
    Returns:
        tuple: Combined X and y arrays
    """
    X_combined = []
    y_combined = []
    
    for X, y in datasets:
        if X is not None and y is not None:
            X_combined.extend(X)
            y_combined.extend(y)
    
    return np.array(X_combined), np.array(y_combined)

def train_sentiment_model(X, y):
    """
    Train a sentiment analysis model.
    
    Args:
        X (list): List of text samples
        y (list): List of sentiment labels
        
    Returns:
        tuple: Trained vectorizer, model and test data
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
        max_features=15000,
        min_df=2,
        max_df=0.85
    )
    
    # Create classifier
    classifier = LinearSVC(class_weight='balanced')
    
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
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    logger.info(f"Model F1 score on test set: {f1:.4f}")
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    logger.info(f"Classification report on test set:\n{report}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix on test set:\n{cm}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/expanded_model_confusion_matrix.png')
    logger.info("Saved confusion matrix plot to plots/expanded_model_confusion_matrix.png")
    
    return vectorizer, classifier, (X_test, y_test, y_pred)

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

def save_test_results(test_data):
    """
    Save test results to a markdown file.
    
    Args:
        test_data: Tuple of (X_test, y_test, y_pred)
    """
    X_test, y_test, y_pred = test_data
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create markdown content
    markdown_content = f"""# Expanded Sentiment Analysis Model Test Results

## Model Information
- **Model Type**: Linear Support Vector Classifier (LinearSVC)
- **Vectorizer**: TF-IDF (max_features=15000)
- **Training Data**: Combined multiple Twitter datasets

## Performance Metrics
- **Accuracy**: {accuracy:.4f}
- **F1 Score (weighted)**: {f1:.4f}

## Classification Report
```
{report}
```

## Confusion Matrix
```
{cm}
```

![Confusion Matrix](plots/expanded_model_confusion_matrix.png)

## Sample Predictions

| True Sentiment | Predicted Sentiment | Text |
|----------------|---------------------|------|
"""
    
    # Add sample predictions (10 samples of each class)
    negative_samples = [(X_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] == -1][:10]
    neutral_samples = [(X_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] == 0][:10]
    positive_samples = [(X_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] == 1][:10]
    
    for text, pred in negative_samples:
        pred_label = "Negative" if pred == -1 else "Neutral" if pred == 0 else "Positive"
        markdown_content += f"| Negative | {pred_label} | {text[:100]}... |\n"
    
    for text, pred in neutral_samples:
        pred_label = "Negative" if pred == -1 else "Neutral" if pred == 0 else "Positive"
        markdown_content += f"| Neutral | {pred_label} | {text[:100]}... |\n"
    
    for text, pred in positive_samples:
        pred_label = "Negative" if pred == -1 else "Neutral" if pred == 0 else "Positive"
        markdown_content += f"| Positive | {pred_label} | {text[:100]}... |\n"
    
    # Write markdown file
    with open(TEST_RESULTS_PATH, 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Test results saved to {TEST_RESULTS_PATH}")

def main():
    """Main function to load data, train model, and save results."""
    try:
        # Load datasets
        twitter_data = load_twitter_data()
        twitter_training = load_twitter_training()
        large_dataset = load_large_dataset(sample_size=50000)  # Use a balanced subset of the large dataset
        
        # Combine datasets
        datasets = [twitter_data, twitter_training, large_dataset]
        X_combined, y_combined = combine_datasets(datasets)
        
        logger.info(f"Combined dataset size: {len(X_combined)} samples")
        unique_labels, counts = np.unique(y_combined, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        logger.info(f"Label distribution in combined dataset: {label_distribution}")
        
        # Train model
        vectorizer, model, test_data = train_sentiment_model(X_combined, y_combined)
        
        # Save model
        save_model(vectorizer, model)
        
        # Save test results
        save_test_results(test_data)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 