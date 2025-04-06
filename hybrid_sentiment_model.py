#!/usr/bin/env python3
"""
Hybrid Sentiment Analysis Model

This script implements a hybrid approach combining the Twitter-based and
synthetic models for improved performance on fire brigade incident reports.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hybrid_model')

# Model file paths
TWITTER_MODEL_PATH = "app/models/twitter_sentiment_model.pkl"
TWITTER_VECTORIZER_PATH = "app/models/twitter_vectorizer.pkl"
SYNTHETIC_MODEL_PATH = "app/models/enhanced_sentiment_model.pkl"
SYNTHETIC_VECTORIZER_PATH = "app/models/enhanced_vectorizer.pkl"
HYBRID_MODEL_PATH = "app/models/hybrid_sentiment_model.pkl"

# Test examples from both datasets
TEST_EXAMPLES = [
    # Original test examples
    {
        "text": "I love the new fire engine, it's so efficient!",
        "expected": 1,  # positive
        "domain_specific": False
    },
    {
        "text": "Today's shift was exhausting with multiple emergency calls.",
        "expected": -1,  # negative
        "domain_specific": True
    },
    {
        "text": "The fire station just received new equipment for training.",
        "expected": 0,  # neutral
        "domain_specific": True
    },
    {
        "text": "Terrible response time from the emergency services today.",
        "expected": -1,  # negative
        "domain_specific": True
    },
    {
        "text": "Grateful for the quick action of firefighters during today's incident.",
        "expected": 1,  # positive
        "domain_specific": True
    },
    # Additional test examples
    {
        "text": "Several firefighters were injured due to roof collapse during yesterday's operation.",
        "expected": -1,  # negative
        "domain_specific": True
    },
    {
        "text": "Our new thermal imaging cameras are making search and rescue much more effective.",
        "expected": 1,  # positive
        "domain_specific": True
    },
    {
        "text": "Training session on fire suppression techniques will be held next Tuesday.",
        "expected": 0,  # neutral
        "domain_specific": True
    },
    {
        "text": "Proud of our team for saving all occupants from the apartment fire last night.",
        "expected": 1,  # positive
        "domain_specific": True
    },
    {
        "text": "Equipment malfunction prevented timely rescue operation at the factory incident.",
        "expected": -1,  # negative
        "domain_specific": True
    },
    # Generic examples
    {
        "text": "This product is amazing, I highly recommend it!",
        "expected": 1,  # positive
        "domain_specific": False
    },
    {
        "text": "The service was terrible and I will never come back.",
        "expected": -1,  # negative
        "domain_specific": False
    },
    {
        "text": "It's an okay option if you don't have alternatives.",
        "expected": 0,  # neutral
        "domain_specific": False
    }
]

class HybridSentimentModel:
    """
    A hybrid model that combines predictions from Twitter and synthetic models
    based on confidence scores and domain specificity.
    """
    
    def __init__(self, twitter_model=None, twitter_vectorizer=None, 
                 synthetic_model=None, synthetic_vectorizer=None):
        """
        Initialize the hybrid model with component models
        
        Args:
            twitter_model: Trained Twitter sentiment model
            twitter_vectorizer: Twitter model vectorizer
            synthetic_model: Trained synthetic sentiment model
            synthetic_vectorizer: Synthetic model vectorizer
        """
        # Load models if not provided
        if twitter_model is None or twitter_vectorizer is None:
            self.load_twitter_model()
        else:
            self.twitter_model = twitter_model
            self.twitter_vectorizer = twitter_vectorizer
            
        if synthetic_model is None or synthetic_vectorizer is None:
            self.load_synthetic_model()
        else:
            self.synthetic_model = synthetic_model
            self.synthetic_vectorizer = synthetic_vectorizer
        
        # Configure weights (can be tuned)
        self.twitter_weight = 0.4
        self.synthetic_weight = 0.6
        
        # Domain specificity threshold
        self.domain_threshold = 0.6
        
        logger.info("Hybrid model initialized")
    
    def load_twitter_model(self):
        """Load the Twitter model and vectorizer"""
        try:
            self.twitter_model = joblib.load(TWITTER_MODEL_PATH)
            self.twitter_vectorizer = joblib.load(TWITTER_VECTORIZER_PATH)
            logger.info("Twitter model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Twitter model: {str(e)}")
            raise
    
    def load_synthetic_model(self):
        """Load the synthetic model and vectorizer"""
        try:
            self.synthetic_model = joblib.load(SYNTHETIC_MODEL_PATH)
            self.synthetic_vectorizer = joblib.load(SYNTHETIC_VECTORIZER_PATH)
            logger.info("Synthetic model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading synthetic model: {str(e)}")
            raise
    
    def predict_twitter(self, text):
        """
        Make prediction using the Twitter model
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment_value, confidence)
        """
        try:
            # Preprocess text
            processed_text = extract_features([text])[0]
            
            # Vectorize text
            text_vectorized = self.twitter_vectorizer.transform([processed_text])
            
            # Make prediction
            sentiment_value = self.twitter_model.predict(text_vectorized)[0]
            
            # Get confidence score
            confidence = np.abs(self.twitter_model.decision_function(text_vectorized)[0])
            if isinstance(confidence, np.ndarray):
                confidence = np.mean(confidence)
            
            # Normalize confidence to 0-1 range
            confidence = min(1.0, confidence / 2.0)
            
            return sentiment_value, confidence
        
        except Exception as e:
            logger.error(f"Error in Twitter prediction: {str(e)}")
            return 0, 0.0
    
    def predict_synthetic(self, text):
        """
        Make prediction using the synthetic model
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment_value, confidence)
        """
        try:
            # Preprocess text
            processed_text = extract_features([text])[0]
            
            # Vectorize text
            text_vectorized = self.synthetic_vectorizer.transform([processed_text])
            
            # Make prediction
            sentiment_value = self.synthetic_model.predict(text_vectorized)[0]
            
            # Get confidence score
            confidence = np.abs(self.synthetic_model.decision_function(text_vectorized)[0])
            if isinstance(confidence, np.ndarray):
                confidence = np.mean(confidence)
            
            # Normalize confidence to 0-1 range
            confidence = min(1.0, confidence / 2.0)
            
            return sentiment_value, confidence
        
        except Exception as e:
            logger.error(f"Error in synthetic prediction: {str(e)}")
            return 0, 0.0
    
    def is_domain_specific(self, text):
        """
        Determine if text is domain-specific (fire brigade related)
        
        Args:
            text (str): Input text
            
        Returns:
            float: Domain specificity score (0-1)
        """
        # List of domain-specific keywords
        domain_keywords = [
            'fire', 'firefighter', 'brigade', 'emergency', 'rescue', 'incident', 
            'response', 'evacuation', 'extinguisher', 'alarm', 'drill', 'station',
            'department', 'thermal', 'equipment', 'safety', 'hazard', 'operation',
            'crew', 'engine', 'ladder', 'hose', 'truck', 'dispatch', 'call'
        ]
        
        # Count matching keywords
        matches = sum(1 for keyword in domain_keywords if keyword.lower() in text.lower())
        
        # Calculate score based on keyword density
        words = text.split()
        if not words:
            return 0.0
        
        # Score is ratio of matching keywords to total words, capped at 1.0
        return min(1.0, matches / (len(words) * 0.5))
    
    def predict(self, text, domain_hint=None):
        """
        Make a hybrid prediction using both models
        
        Args:
            text (str): Input text
            domain_hint (bool, optional): If provided, overrides domain detection
            
        Returns:
            tuple: (sentiment_value, sentiment_label, confidence, model_used)
        """
        # Get predictions from both models
        twitter_sentiment, twitter_confidence = self.predict_twitter(text)
        synthetic_sentiment, synthetic_confidence = self.predict_synthetic(text)
        
        # Determine domain specificity
        if domain_hint is not None:
            is_domain = domain_hint
            domain_score = 1.0 if domain_hint else 0.0
        else:
            domain_score = self.is_domain_specific(text)
            is_domain = domain_score >= self.domain_threshold
        
        # Calculate weighted confidence based on domain specificity
        if is_domain:
            # For domain-specific text, favor synthetic model
            synthetic_weight = self.synthetic_weight * (0.5 + domain_score * 0.5)
            twitter_weight = 1.0 - synthetic_weight
        else:
            # For general text, favor Twitter model
            twitter_weight = self.twitter_weight * (1.5 - domain_score * 0.5)
            synthetic_weight = 1.0 - twitter_weight
        
        # Apply weighted confidence for each prediction
        twitter_weighted = twitter_confidence * twitter_weight
        synthetic_weighted = synthetic_confidence * synthetic_weight
        
        # Make final decision based on weighted confidences
        if twitter_weighted > synthetic_weighted:
            final_sentiment = twitter_sentiment
            confidence = twitter_confidence
            model_used = "twitter"
        else:
            final_sentiment = synthetic_sentiment
            confidence = synthetic_confidence
            model_used = "synthetic"
        
        # Map to label
        sentiment_label = map_sentiment_value(final_sentiment)
        
        # If confidences are both very low, default to neutral
        if twitter_confidence < 0.2 and synthetic_confidence < 0.2:
            final_sentiment = 0
            sentiment_label = "neutral"
            confidence = max(twitter_confidence, synthetic_confidence)
            model_used = "low_confidence_default"
        
        # If models disagree with similar confidence, check which one is neutral
        elif (twitter_sentiment != synthetic_sentiment and 
              abs(twitter_weighted - synthetic_weighted) < 0.1):
            if twitter_sentiment == 0:
                final_sentiment = synthetic_sentiment
                sentiment_label = map_sentiment_value(final_sentiment)
                confidence = synthetic_confidence
                model_used = "synthetic_non_neutral"
            elif synthetic_sentiment == 0:
                final_sentiment = twitter_sentiment
                sentiment_label = map_sentiment_value(final_sentiment)
                confidence = twitter_confidence
                model_used = "twitter_non_neutral"
        
        return final_sentiment, sentiment_label, confidence, model_used
    
    def save(self, filepath=HYBRID_MODEL_PATH):
        """Save the hybrid model"""
        try:
            joblib.dump(self, filepath)
            logger.info(f"Hybrid model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving hybrid model: {str(e)}")
            return False


def evaluate_models():
    """
    Evaluate individual models and the hybrid model on test examples
    """
    try:
        # Initialize hybrid model
        hybrid_model = HybridSentimentModel()
        
        # Prepare results
        results = {
            "twitter": {"correct": 0, "count": 0},
            "synthetic": {"correct": 0, "count": 0},
            "hybrid": {"correct": 0, "count": 0}
        }
        
        detailed_results = []
        
        # Test on examples
        for i, example in enumerate(TEST_EXAMPLES):
            text = example["text"]
            expected = example["expected"]
            domain_specific = example["domain_specific"]
            
            # Get individual model predictions
            twitter_sentiment, twitter_confidence = hybrid_model.predict_twitter(text)
            synthetic_sentiment, synthetic_confidence = hybrid_model.predict_synthetic(text)
            
            # Get hybrid prediction
            hybrid_sentiment, hybrid_label, hybrid_confidence, model_used = hybrid_model.predict(text, domain_specific)
            
            # Check correctness
            twitter_correct = twitter_sentiment == expected
            synthetic_correct = synthetic_sentiment == expected
            hybrid_correct = hybrid_sentiment == expected
            
            # Update counts
            results["twitter"]["count"] += 1
            results["synthetic"]["count"] += 1
            results["hybrid"]["count"] += 1
            
            if twitter_correct:
                results["twitter"]["correct"] += 1
            if synthetic_correct:
                results["synthetic"]["correct"] += 1
            if hybrid_correct:
                results["hybrid"]["correct"] += 1
            
            # Add to detailed results
            detailed_results.append({
                "index": i+1,
                "text": text,
                "expected": expected,
                "expected_label": map_sentiment_value(expected),
                "domain_specific": domain_specific,
                "twitter": {
                    "prediction": twitter_sentiment,
                    "confidence": twitter_confidence,
                    "correct": twitter_correct
                },
                "synthetic": {
                    "prediction": synthetic_sentiment,
                    "confidence": synthetic_confidence,
                    "correct": synthetic_correct
                },
                "hybrid": {
                    "prediction": hybrid_sentiment,
                    "confidence": hybrid_confidence,
                    "model_used": model_used,
                    "correct": hybrid_correct
                }
            })
        
        # Calculate accuracies
        twitter_accuracy = results["twitter"]["correct"] / results["twitter"]["count"] if results["twitter"]["count"] > 0 else 0
        synthetic_accuracy = results["synthetic"]["correct"] / results["synthetic"]["count"] if results["synthetic"]["count"] > 0 else 0
        hybrid_accuracy = results["hybrid"]["correct"] / results["hybrid"]["count"] if results["hybrid"]["count"] > 0 else 0
        
        # Log results
        logger.info(f"Twitter Model Accuracy: {twitter_accuracy:.4f} ({results['twitter']['correct']}/{results['twitter']['count']})")
        logger.info(f"Synthetic Model Accuracy: {synthetic_accuracy:.4f} ({results['synthetic']['correct']}/{results['synthetic']['count']})")
        logger.info(f"Hybrid Model Accuracy: {hybrid_accuracy:.4f} ({results['hybrid']['correct']}/{results['hybrid']['count']})")
        
        # Detailed per-example results
        logger.info("\nDetailed Results:")
        for res in detailed_results:
            logger.info(f"{res['index']}. '{res['text']}' (Expected: {res['expected_label']})")
            
            t_correct = "✓" if res["twitter"]["correct"] else "✗"
            s_correct = "✓" if res["synthetic"]["correct"] else "✗"
            h_correct = "✓" if res["hybrid"]["correct"] else "✗"
            
            logger.info(f"   Twitter: {t_correct} {map_sentiment_value(res['twitter']['prediction'])} ({res['twitter']['confidence']:.4f})")
            logger.info(f"   Synthetic: {s_correct} {map_sentiment_value(res['synthetic']['prediction'])} ({res['synthetic']['confidence']:.4f})")
            logger.info(f"   Hybrid: {h_correct} {map_sentiment_value(res['hybrid']['prediction'])} ({res['hybrid']['confidence']:.4f}) [Using: {res['hybrid']['model_used']}]")
            logger.info("")
        
        # Domain vs Non-domain performance
        domain_results = {model: {"correct": 0, "count": 0} for model in ["twitter", "synthetic", "hybrid"]}
        non_domain_results = {model: {"correct": 0, "count": 0} for model in ["twitter", "synthetic", "hybrid"]}
        
        for res in detailed_results:
            target_dict = domain_results if res["domain_specific"] else non_domain_results
            
            for model in ["twitter", "synthetic", "hybrid"]:
                target_dict[model]["count"] += 1
                if res[model]["correct"]:
                    target_dict[model]["correct"] += 1
        
        # Log domain-specific performance
        logger.info("\nDomain-Specific Performance:")
        for model in ["twitter", "synthetic", "hybrid"]:
            if domain_results[model]["count"] > 0:
                acc = domain_results[model]["correct"] / domain_results[model]["count"]
                logger.info(f"{model.capitalize()} Model: {acc:.4f} ({domain_results[model]['correct']}/{domain_results[model]['count']})")
        
        # Log non-domain performance
        logger.info("\nNon-Domain Performance:")
        for model in ["twitter", "synthetic", "hybrid"]:
            if non_domain_results[model]["count"] > 0:
                acc = non_domain_results[model]["correct"] / non_domain_results[model]["count"]
                logger.info(f"{model.capitalize()} Model: {acc:.4f} ({non_domain_results[model]['correct']}/{non_domain_results[model]['count']})")
        
        # Save hybrid model if it performs well
        if hybrid_accuracy > max(twitter_accuracy, synthetic_accuracy):
            hybrid_model.save()
            logger.info(f"Hybrid model saved as it outperforms individual models")
        
        return hybrid_model, results, detailed_results
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    evaluate_models() 