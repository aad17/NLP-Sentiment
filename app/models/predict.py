"""
Prediction Module for Sentiment Analysis

This module contains functions for loading the trained model
and making sentiment predictions on new incident reports.
Supports multiple model types: synthetic, twitter, hybrid, and expanded.
"""
import os
import sys
import logging
import joblib
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.text_preprocessing import preprocess_text, extract_features, map_sentiment_value
from app.utils.config import MODEL_PATH, VECTORIZER_PATH, SENTIMENT_LABELS

# Define paths for all models
SYNTHETIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "enhanced_sentiment_model.pkl")
SYNTHETIC_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "enhanced_vectorizer.pkl")
TWITTER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "twitter_sentiment_model.pkl")
TWITTER_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "twitter_vectorizer.pkl")
HYBRID_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hybrid_sentiment_model.pkl")
# Add expanded model paths
EXPANDED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "expanded_sentiment_model.pkl")
EXPANDED_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "expanded_vectorizer.pkl")
# Add ensemble model path
ENSEMBLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ensemble_sentiment_model.pkl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('predict')

# Track loaded models
models = {
    "default": {"loaded": False, "model": None, "vectorizer": None},
    "synthetic": {"loaded": False, "model": None, "vectorizer": None},
    "twitter": {"loaded": False, "model": None, "vectorizer": None},
    "hybrid": {"loaded": False, "model": None, "vectorizer": None},
    "expanded": {"loaded": False, "model": None, "vectorizer": None},  # Add expanded model
    "ensemble": {"loaded": False, "model": None, "vectorizer": None}   # Add ensemble model
}

# Load default model (backward compatibility)
try:
    models["default"]["vectorizer"] = joblib.load(VECTORIZER_PATH)
    models["default"]["model"] = joblib.load(MODEL_PATH)
    models["default"]["loaded"] = True
    logger.info("Default model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading default model: {str(e)}")
    logger.warning("Default model not loaded. Run train_model.py first.")

def load_model(model_type="default"):
    """
    Load a specific model type.
    
    Args:
        model_type (str): Type of model to load ('default', 'synthetic', 'twitter', 'hybrid', 'expanded', or 'ensemble')
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    if models[model_type]["loaded"]:
        logger.info(f"{model_type.capitalize()} model already loaded")
        return True
        
    try:
        if model_type == "synthetic":
            models[model_type]["vectorizer"] = joblib.load(SYNTHETIC_VECTORIZER_PATH)
            models[model_type]["model"] = joblib.load(SYNTHETIC_MODEL_PATH)
        elif model_type == "twitter":
            models[model_type]["vectorizer"] = joblib.load(TWITTER_VECTORIZER_PATH)
            models[model_type]["model"] = joblib.load(TWITTER_MODEL_PATH)
        elif model_type == "hybrid":
            models[model_type]["model"] = joblib.load(HYBRID_MODEL_PATH)
            # Hybrid model has its own internal vectorizers
        elif model_type == "expanded":
            models[model_type]["vectorizer"] = joblib.load(EXPANDED_VECTORIZER_PATH)
            models[model_type]["model"] = joblib.load(EXPANDED_MODEL_PATH)
        elif model_type == "ensemble":
            models[model_type]["model"] = joblib.load(ENSEMBLE_MODEL_PATH)
            # Ensemble model doesn't need a vectorizer as it uses other models
        else:
            # Default model already attempted to load at module import
            if not models["default"]["loaded"]:
                return False
            models[model_type]["model"] = models["default"]["model"]
            models[model_type]["vectorizer"] = models["default"]["vectorizer"]
            
        models[model_type]["loaded"] = True
        logger.info(f"{model_type.capitalize()} model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {str(e)}")
        return False

def get_available_models():
    """
    Get a list of available models.
    
    Returns:
        list: List of model types that are available
    """
    available = []
    
    # Check default/synthetic model
    if os.path.exists(SYNTHETIC_MODEL_PATH) and os.path.exists(SYNTHETIC_VECTORIZER_PATH):
        available.append("synthetic")
    
    # Check Twitter model
    if os.path.exists(TWITTER_MODEL_PATH) and os.path.exists(TWITTER_VECTORIZER_PATH):
        available.append("twitter")
    
    # Check hybrid model
    if os.path.exists(HYBRID_MODEL_PATH):
        available.append("hybrid")
        
    # Check expanded model
    if os.path.exists(EXPANDED_MODEL_PATH) and os.path.exists(EXPANDED_VECTORIZER_PATH):
        available.append("expanded")
        
    # Check ensemble model
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        available.append("ensemble")
        
    # Always include default if it's loaded
    if models["default"]["loaded"]:
        available.append("default")
        
    return available

def predict_sentiment(text, model_type="default"):
    """
    Predict sentiment for a given text using the specified model.
    
    Args:
        text (str): Input text
        model_type (str): Type of model to use ('default', 'synthetic', 'twitter', 'hybrid', 'expanded', or 'ensemble')
        
    Returns:
        tuple: (sentiment_value, sentiment_label, confidence, model_used)
    """
    # For backward compatibility
    if model_type not in ["default", "synthetic", "twitter", "hybrid", "expanded", "ensemble"]:
        logger.warning(f"Unknown model type: {model_type}. Using default.")
        model_type = "default"
    
    # Load model if not already loaded
    if not models[model_type]["loaded"]:
        success = load_model(model_type)
        if not success:
            logger.error(f"{model_type.capitalize()} model not loaded. Cannot make predictions.")
            return 0, "neutral", 0.0, None
    
    try:
        # Handle ensemble model separately as it has a different interface
        if model_type == "ensemble":
            ensemble_model = models[model_type]["model"]
            # Get ensemble prediction
            sentiment_value, sentiment_label, confidence, model_used = ensemble_model.predict(text)
            
            logger.info(f"Ensemble model predicted: {sentiment_label} (confidence: {confidence:.2f})")
            return sentiment_value, sentiment_label, confidence, model_used
            
        # Handle hybrid model separately as it has a different interface
        if model_type == "hybrid":
            hybrid_model = models[model_type]["model"]
            # Detect if text is domain-specific
            domain_score = is_domain_specific(text)
            is_domain = domain_score >= 0.6  # Same threshold as in hybrid model
            
            # Get hybrid prediction
            sentiment_value, sentiment_label, confidence, model_used = hybrid_model.predict(text, is_domain)
            
            logger.info(f"Hybrid model predicted: {sentiment_label} using {model_used} (confidence: {confidence:.2f})")
            return sentiment_value, sentiment_label, confidence, model_used
        
        # For other models, use standard prediction
        # Preprocess text (use appropriate preprocessor based on model)
        if model_type == "twitter" or model_type == "expanded":
            processed_text = extract_features([text])[0]
        else:
            processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vectorized = models[model_type]["vectorizer"].transform([processed_text])
        
        # Make prediction
        sentiment_value = models[model_type]["model"].predict(text_vectorized)[0]
        
        # Get confidence score (distance from decision boundary)
        confidence = np.abs(models[model_type]["model"].decision_function(text_vectorized)[0])
        if isinstance(confidence, np.ndarray):
            confidence = np.mean(confidence)
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, confidence / 2.0)
        
        # Map sentiment value to label
        sentiment_label = map_sentiment_value(sentiment_value)
        
        logger.info(f"{model_type.capitalize()} model predicted: {sentiment_label} (confidence: {confidence:.2f})")
        return sentiment_value, sentiment_label, confidence, model_type
    
    except Exception as e:
        logger.error(f"Error predicting sentiment with {model_type} model: {str(e)}")
        return 0, "neutral", 0.0, None

def is_domain_specific(text):
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

def batch_predict(texts, model_type="default"):
    """
    Predict sentiment for a batch of texts using the specified model.
    
    Args:
        texts (list): List of text strings
        model_type (str): Type of model to use ('default', 'synthetic', 'twitter', or 'hybrid')
        
    Returns:
        list: List of prediction tuples (sentiment_value, sentiment_label, confidence, model_used)
    """
    # For backward compatibility
    if model_type not in ["default", "synthetic", "twitter", "hybrid"]:
        logger.warning(f"Unknown model type: {model_type}. Using default.")
        model_type = "default"
    
    # Load model if not already loaded
    if not models[model_type]["loaded"]:
        success = load_model(model_type)
        if not success:
            logger.error(f"{model_type.capitalize()} model not loaded. Cannot make predictions.")
            return [(0, "neutral", 0.0, None)] * len(texts)
    
    try:
        # For hybrid model, process each text individually
        if model_type == "hybrid":
            results = []
            for text in texts:
                results.append(predict_sentiment(text, model_type))
            return results
        
        # For other models, use batch prediction
        # Preprocess texts
        if model_type == "twitter":
            processed_texts = extract_features(texts)
        else:
            processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize texts
        texts_vectorized = models[model_type]["vectorizer"].transform(processed_texts)
        
        # Make predictions
        sentiment_values = models[model_type]["model"].predict(texts_vectorized)
        
        # Get confidence scores
        confidences = np.abs(models[model_type]["model"].decision_function(texts_vectorized))
        if len(confidences.shape) > 1:
            confidences = np.mean(confidences, axis=1)
        
        # Normalize confidences
        confidences = np.minimum(1.0, confidences / 2.0)
        
        # Map sentiment values to labels
        results = [
            (value, map_sentiment_value(value), confidence, model_type)
            for value, confidence in zip(sentiment_values, confidences)
        ]
        
        logger.info(f"Predicted sentiment for {len(texts)} texts using {model_type} model")
        return results
    
    except Exception as e:
        logger.error(f"Error batch predicting sentiment with {model_type} model: {str(e)}")
        return [(0, "neutral", 0.0, None)] * len(texts)

if __name__ == "__main__":
    # Test prediction with all available models
    test_texts = [
        "Successfully rescued family from burning building with no injuries.",
        "Responded to fire alarm which was determined to be a false alarm.",
        "Multiple injuries reported due to building collapse during firefighting operation."
    ]
    
    available_models = get_available_models()
    print(f"Available models: {', '.join(available_models)}")
    
    for model_type in available_models:
        print(f"\n--- {model_type.upper()} MODEL PREDICTIONS ---")
        for text in test_texts:
            value, label, conf, used = predict_sentiment(text, model_type)
            print(f"Text: {text}")
            print(f"Sentiment: {label} ({value})")
            print(f"Confidence: {conf:.2f}")
            if model_type == "hybrid":
                print(f"Model used: {used}")
            print()