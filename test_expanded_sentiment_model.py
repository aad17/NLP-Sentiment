#!/usr/bin/env python3
"""
Test Expanded Sentiment Analysis Model

This script tests the expanded sentiment analysis model on various examples
from different domains to verify its generalization capabilities.
"""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_expanded_model')

# Model file paths
MODEL_PATH = "app/models/expanded_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/expanded_vectorizer.pkl"

# Test data file (emergency services specific examples)
EMERGENCY_TEST_DATA = "emergency_sentiment_test_data.csv"

# Custom examples from different domains
CUSTOM_EXAMPLES = [
    # Emergency services domain
    {"text": "The ambulance arrived quickly and saved my life", "expected": 1},
    {"text": "Had to wait 30 minutes for the ambulance to arrive", "expected": -1},
    {"text": "Fire department response was professional", "expected": 1},
    {"text": "The 911 dispatcher was rude and unhelpful", "expected": -1},
    {"text": "Police officer helped me find my lost child", "expected": 1},
    
    # Tech product domain
    {"text": "This new smartphone has amazing battery life", "expected": 1},
    {"text": "The laptop keeps crashing whenever I open multiple applications", "expected": -1},
    {"text": "I bought a standard mid-range TV for my living room", "expected": 0},
    {"text": "This software update broke all my previously working features", "expected": -1},
    {"text": "The camera quality on this phone exceeds expectations", "expected": 1},
    
    # Restaurant/Food domain
    {"text": "The food was delicious and the service was excellent", "expected": 1},
    {"text": "Our waiter was very attentive and friendly", "expected": 1},
    {"text": "I ordered a hamburger with fries for lunch today", "expected": 0},
    {"text": "The restaurant was dirty and the food was cold", "expected": -1},
    {"text": "We waited over an hour for our food to arrive", "expected": -1},
    
    # Travel domain
    {"text": "The hotel room was spacious and clean with a beautiful view", "expected": 1},
    {"text": "My flight was delayed by three hours with no explanation", "expected": -1},
    {"text": "I took the train from New York to Washington DC", "expected": 0},
    {"text": "The beach was crowded but the weather was perfect", "expected": 1},
    {"text": "The taxi driver took a longer route to increase the fare", "expected": -1},
    
    # General social media
    {"text": "Can't wait for the weekend to start!", "expected": 1},
    {"text": "Just found out my exam was postponed again", "expected": -1},
    {"text": "Going to the grocery store to pick up some milk", "expected": 0},
    {"text": "So upset about the election results", "expected": -1},
    {"text": "My new puppy is the cutest thing ever!", "expected": 1}
]

def load_model():
    """
    Load the trained vectorizer and model.
    
    Returns:
        tuple: Loaded vectorizer and model
    """
    logger.info(f"Loading vectorizer from {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    return vectorizer, model

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
        
        # Predict sentiment
        sentiment_value = model.predict(text_vectorized)[0]
        
        # Convert to label
        sentiment_label = map_sentiment_value(sentiment_value)
        
        # Calculate confidence (distance from decision boundary)
        decision_values = model.decision_function(text_vectorized)[0]
        
        # For multi-class, get confidence from the winning class
        if isinstance(decision_values, np.ndarray):
            confidence = max(abs(decision_values))
        else:
            confidence = abs(decision_values)
        
        return sentiment_value, sentiment_label, confidence
        
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return 0, "neutral", 0.0

def test_on_custom_examples(vectorizer, model):
    """
    Test the model on custom examples from different domains.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
        
    Returns:
        dict: Test results with accuracy per domain
    """
    logger.info("Testing model on custom examples from different domains")
    
    results = []
    domain_examples = {
        "Emergency": CUSTOM_EXAMPLES[:5],
        "Tech": CUSTOM_EXAMPLES[5:10],
        "Restaurant": CUSTOM_EXAMPLES[10:15],
        "Travel": CUSTOM_EXAMPLES[15:20],
        "Social Media": CUSTOM_EXAMPLES[20:25]
    }
    
    domain_results = {}
    
    # Test each domain
    for domain, examples in domain_examples.items():
        correct = 0
        domain_predictions = []
        
        for example in examples:
            text = example["text"]
            expected = example["expected"]
            
            # Predict sentiment
            sentiment_value, sentiment_label, confidence = predict_sentiment(text, vectorizer, model)
            
            # Check if prediction is correct
            is_correct = sentiment_value == expected
            if is_correct:
                correct += 1
            
            # Store result
            result = {
                "domain": domain,
                "text": text,
                "expected": expected,
                "expected_label": map_sentiment_value(expected),
                "predicted": sentiment_value,
                "predicted_label": sentiment_label,
                "confidence": confidence,
                "is_correct": is_correct
            }
            results.append(result)
            domain_predictions.append((expected, sentiment_value))
        
        # Calculate domain accuracy
        domain_accuracy = correct / len(examples)
        domain_results[domain] = {
            "accuracy": domain_accuracy,
            "predictions": domain_predictions
        }
        
        logger.info(f"Domain: {domain}, Accuracy: {domain_accuracy:.2f}")
    
    # Calculate overall accuracy
    overall_correct = sum(1 for result in results if result["is_correct"])
    overall_accuracy = overall_correct / len(results)
    logger.info(f"Overall accuracy on custom examples: {overall_accuracy:.2f}")
    
    return {"results": results, "domain_results": domain_results, "overall_accuracy": overall_accuracy}

def test_on_emergency_data(vectorizer, model):
    """
    Test the model on emergency services domain data.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
        
    Returns:
        dict: Test results on emergency services data
    """
    try:
        logger.info(f"Testing model on emergency services data from {EMERGENCY_TEST_DATA}")
        
        # Load test data
        df = pd.read_csv(EMERGENCY_TEST_DATA)
        
        # Check columns (account for capitalization)
        text_col = "Text" if "Text" in df.columns else "text"
        sentiment_col = "Sentiment" if "Sentiment" in df.columns else "sentiment"
        
        if text_col not in df.columns or sentiment_col not in df.columns:
            logger.error(f"Invalid test data format. Required columns: 'Text'/'text', 'Sentiment'/'sentiment'")
            return None
        
        # Map string sentiment labels to numeric values
        sentiment_map = {
            "Positive": 1,
            "Neutral": 0,
            "Negative": -1
        }
        
        df['mapped_sentiment'] = df[sentiment_col].map(sentiment_map)
        
        # Drop rows where mapping failed
        if df['mapped_sentiment'].isna().sum() > 0:
            logger.warning(f"Could not map {df['mapped_sentiment'].isna().sum()} sentiment values. Dropping these rows.")
            df = df.dropna(subset=['mapped_sentiment'])
        
        # Extract features and labels
        X_test = df[text_col].values
        y_test = df['mapped_sentiment'].values
        
        # Preprocess text
        X_processed = extract_features(X_test)
        
        # Vectorize text
        X_vectorized = vectorizer.transform(X_processed)
        
        # Predict sentiment
        y_pred = model.predict(X_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Accuracy on emergency services data: {accuracy:.4f}")
        logger.info(f"Classification report on emergency services data:\n{report}")
        
        # Get individual predictions for reporting
        predictions = []
        for i in range(len(X_test)):
            predictions.append({
                "text": X_test[i],
                "true_sentiment": y_test[i],
                "predicted_sentiment": y_pred[i],
                "is_correct": y_test[i] == y_pred[i]
            })
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error testing on emergency services data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to test the expanded sentiment model."""
    try:
        # Load model
        vectorizer, model = load_model()
        
        # Test on custom examples
        custom_results = test_on_custom_examples(vectorizer, model)
        
        # Test on emergency services data
        emergency_results = test_on_emergency_data(vectorizer, model)
        
        # Generate markdown report
        markdown_content = "# Expanded Sentiment Model Test Results\n\n"
        
        # Add custom examples results
        markdown_content += "## Custom Examples Results\n\n"
        markdown_content += f"**Overall Accuracy**: {custom_results['overall_accuracy']:.2f}\n\n"
        
        markdown_content += "### Domain-specific Accuracy\n\n"
        markdown_content += "| Domain | Accuracy |\n"
        markdown_content += "|--------|----------|\n"
        
        for domain, result in custom_results["domain_results"].items():
            markdown_content += f"| {domain} | {result['accuracy']:.2f} |\n"
        
        markdown_content += "\n### Sample Predictions\n\n"
        markdown_content += "| Domain | Text | Expected | Predicted | Correct |\n"
        markdown_content += "|--------|------|----------|-----------|--------|\n"
        
        for result in custom_results["results"]:
            correct_mark = "✓" if result["is_correct"] else "✗"
            markdown_content += f"| {result['domain']} | {result['text'][:50]}... | {result['expected_label']} | {result['predicted_label']} | {correct_mark} |\n"
        
        # Add emergency services results
        if emergency_results:
            markdown_content += "\n## Emergency Services Data Results\n\n"
            markdown_content += f"**Accuracy**: {emergency_results['accuracy']:.4f}\n\n"
            
            markdown_content += "### Classification Report\n\n"
            markdown_content += "```\n"
            markdown_content += emergency_results['classification_report']
            markdown_content += "```\n\n"
            
            markdown_content += "### Confusion Matrix\n\n"
            markdown_content += "```\n"
            markdown_content += str(emergency_results['confusion_matrix'])
            markdown_content += "```\n"
        
        # Write markdown file
        with open("expanded_model_domain_test_results.md", "w") as f:
            f.write(markdown_content)
        
        logger.info("Test results saved to expanded_model_domain_test_results.md")
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 