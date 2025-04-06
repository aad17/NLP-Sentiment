#!/usr/bin/env python3
"""
Test the enhanced sentiment model with custom examples that don't follow templates
"""
import os
import sys
import logging
import joblib
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('custom_test')

# Model file paths
MODEL_PATH = "app/models/enhanced_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/enhanced_vectorizer.pkl"

# Custom test examples - these are NOT based on our templates
CUSTOM_EXAMPLES = [
    {
        "text": "Fire team encountered significant water pressure problems while fighting a blaze at 123 Main Street.",
        "expected": -1,  # negative
        "notes": "Problem during operation, should be negative"
    },
    {
        "text": "Monthly equipment checks revealed no issues with breathing apparatus units.",
        "expected": 0,  # neutral
        "notes": "Routine procedure with no positive/negative outcome"
    },
    {
        "text": "Chief Johnson commended the team for their exceptional response to the hospital emergency.",
        "expected": 1,  # positive
        "notes": "Commendation indicates positive sentiment"
    },
    {
        "text": "Due to budget cuts, equipment replacement has been delayed by 6 months.",
        "expected": -1,  # negative
        "notes": "Budget issue causing operational problems"
    },
    {
        "text": "The aerial ladder truck was serviced on Thursday as part of regular maintenance.",
        "expected": 0,  # neutral
        "notes": "Routine maintenance, neutral sentiment"
    },
    {
        "text": "Two firefighters were honored with medals of bravery for the rescue at Johnson Apartments.",
        "expected": 1,  # positive
        "notes": "Awards and honors indicate positive outcome"
    },
    {
        "text": "Smoke was visible from ten miles away as crews arrived at the warehouse fire.",
        "expected": 0,  # neutral
        "notes": "Descriptive statement without sentiment"
    },
    {
        "text": "Heat exhaustion affected three firefighters during the response to the factory fire.",
        "expected": -1,  # negative
        "notes": "Health issues for firefighters indicate negative outcome"
    },
    {
        "text": "Cooperation between police and fire departments facilitated rapid evacuation of senior center.",
        "expected": 1,  # positive
        "notes": "Effective cooperation with good outcome"
    },
    {
        "text": "The search and rescue training program has been expanded to include drone operations.",
        "expected": 0,  # neutral
        "notes": "Informational statement about training, neutral"
    }
]

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

def main():
    """Test the model with custom examples."""
    try:
        # Load model and vectorizer
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model and vectorizer loaded successfully")
        
        correct = 0
        results = []
        
        # Test with custom examples
        for i, example in enumerate(CUSTOM_EXAMPLES):
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
                "correct": is_correct,
                "notes": example["notes"]
            })
        
        # Calculate accuracy
        accuracy = correct / len(CUSTOM_EXAMPLES)
        logger.info(f"Custom examples accuracy: {accuracy:.4f} ({correct}/{len(CUSTOM_EXAMPLES)} correct)")
        
        # Print detailed results
        logger.info("\nDetailed custom test results:")
        for result in results:
            status = "✓" if result["correct"] else "✗"
            logger.info(f"{result['index']}. {status} Text: {result['text']}")
            logger.info(f"   Expected: {result['expected_label']} ({result['expected_sentiment']}), "
                      f"Predicted: {result['predicted_label']} ({result['predicted_sentiment']}), "
                      f"Confidence: {result['confidence']:.4f}")
            logger.info(f"   Notes: {result['notes']}")
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
        
    except Exception as e:
        logger.error(f"Error in custom testing: {str(e)}")
        return 0, []

if __name__ == "__main__":
    main() 