#!/usr/bin/env python3
"""
Test the Twitter-based sentiment model on additional examples.
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
logger = logging.getLogger('test_twitter')

# Model file paths
MODEL_PATH = "app/models/twitter_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/twitter_vectorizer.pkl"

# More test examples specific to fire brigade incidents
ADDITIONAL_TEST_EXAMPLES = [
    {
        "text": "Several firefighters were injured due to roof collapse during yesterday's operation.",
        "expected": -1  # negative
    },
    {
        "text": "Our new thermal imaging cameras are making search and rescue much more effective.",
        "expected": 1  # positive
    },
    {
        "text": "Training session on fire suppression techniques will be held next Tuesday.",
        "expected": 0  # neutral
    },
    {
        "text": "Proud of our team for saving all occupants from the apartment fire last night.",
        "expected": 1  # positive
    },
    {
        "text": "Equipment malfunction prevented timely rescue operation at the factory incident.",
        "expected": -1  # negative
    },
    {
        "text": "Department budget for next fiscal year remains unchanged from current allocation.",
        "expected": 0  # neutral
    },
    {
        "text": "Communication breakdown between units led to confusion during the emergency response.",
        "expected": -1  # negative
    },
    {
        "text": "All stations will participate in the upcoming county-wide drill next month.",
        "expected": 0  # neutral
    },
    {
        "text": "Impressive response time led to successful containment of the chemical spill.",
        "expected": 1  # positive
    },
    {
        "text": "Staffing shortages have left us unable to respond to multiple simultaneous calls.",
        "expected": -1  # negative
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

def evaluate_on_additional_examples():
    """
    Evaluate the model on additional fire brigade specific examples.
    """
    try:
        # Load model and vectorizer
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model and vectorizer loaded successfully")
        
        correct = 0
        results = []
        
        # Test with additional examples
        for i, example in enumerate(ADDITIONAL_TEST_EXAMPLES):
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
        accuracy = correct / len(ADDITIONAL_TEST_EXAMPLES)
        logger.info(f"Additional examples accuracy: {accuracy:.4f} ({correct}/{len(ADDITIONAL_TEST_EXAMPLES)} correct)")
        
        # Print detailed results
        logger.info("\nDetailed additional test results:")
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
        
        # Compare with synthetic model if available
        try:
            synthetic_vectorizer = joblib.load("app/models/enhanced_vectorizer.pkl")
            synthetic_model = joblib.load("app/models/enhanced_sentiment_model.pkl")
            
            synthetic_correct = 0
            synthetic_results = []
            
            for i, example in enumerate(ADDITIONAL_TEST_EXAMPLES):
                text = example["text"]
                expected = example["expected"]
                
                # Predict with synthetic model
                synth_sentiment_value, synth_sentiment_label, synth_confidence = predict_sentiment(
                    text, synthetic_vectorizer, synthetic_model
                )
                
                # Check if prediction is correct
                is_correct = synth_sentiment_value == expected
                if is_correct:
                    synthetic_correct += 1
                    
                # Store result
                synthetic_results.append({
                    "index": i+1,
                    "text": text,
                    "predicted_sentiment": synth_sentiment_value,
                    "predicted_label": synth_sentiment_label,
                    "expected_sentiment": expected,
                    "expected_label": map_sentiment_value(expected),
                    "confidence": synth_confidence,
                    "correct": is_correct
                })
            
            # Calculate accuracy
            synthetic_accuracy = synthetic_correct / len(ADDITIONAL_TEST_EXAMPLES)
            logger.info(f"\nSynthetic model accuracy on same examples: {synthetic_accuracy:.4f} ({synthetic_correct}/{len(ADDITIONAL_TEST_EXAMPLES)} correct)")
            
            # Compare models
            logger.info("\nModel Comparison:")
            logger.info(f"Twitter model: {accuracy:.4f} ({correct}/{len(ADDITIONAL_TEST_EXAMPLES)})")
            logger.info(f"Synthetic model: {synthetic_accuracy:.4f} ({synthetic_correct}/{len(ADDITIONAL_TEST_EXAMPLES)})")
            
            if accuracy > synthetic_accuracy:
                logger.info("Twitter model performs better on these examples")
            elif synthetic_accuracy > accuracy:
                logger.info("Synthetic model performs better on these examples")
            else:
                logger.info("Both models perform equally on these examples")
                
        except Exception as e:
            logger.warning(f"Could not compare with synthetic model: {str(e)}")
        
        return accuracy, results
        
    except Exception as e:
        logger.error(f"Error in additional testing: {str(e)}")
        return 0, []

if __name__ == "__main__":
    evaluate_on_additional_examples() 