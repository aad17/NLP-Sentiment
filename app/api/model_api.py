"""
Sentiment Analysis Model API

This module provides a Flask API for serving sentiment analysis predictions.
Supports multiple model types: synthetic, twitter, hybrid, domain-aware, and expanded.
"""
import os
import sys
import logging
from flask import Flask, request, jsonify
import time
import json

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import model and utilities
from app.models.predict import predict_sentiment, batch_predict, get_available_models
from app.data.database import save_sentiment
from app.utils.config import MODEL_API_PORT

# Check for expanded model
EXPANDED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/expanded_sentiment_model.pkl")
EXPANDED_VECTORIZER_PATH = os.path.join("app/models", "expanded_vectorizer.pkl")
EXPANDED_MODEL_AVAILABLE = os.path.exists(EXPANDED_MODEL_PATH) and os.path.exists(EXPANDED_VECTORIZER_PATH)
if EXPANDED_MODEL_AVAILABLE:
    logging.info(f"Expanded model found at {EXPANDED_MODEL_PATH}")
else:
    logging.warning(f"Expanded model file not found at {EXPANDED_MODEL_PATH}")

# Check for ensemble model
ENSEMBLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/ensemble_sentiment_model.pkl")
ENSEMBLE_MODEL_AVAILABLE = os.path.exists(ENSEMBLE_MODEL_PATH)
if ENSEMBLE_MODEL_AVAILABLE:
    logging.info(f"Ensemble model found at {ENSEMBLE_MODEL_PATH}")
else:
    logging.warning(f"Ensemble model file not found at {ENSEMBLE_MODEL_PATH}")

# Import domain-aware model components if available
try:
    from domain_aware_sentiment import predict_domain_aware_sentiment
    from domain_classifier import predict_domains
    # Explicitly check if the model file exists
    DOMAIN_AWARE_MODEL_PATH = os.path.join("app/models", "domain_aware_sentiment_model.pkl")
    DOMAIN_AWARE_AVAILABLE = os.path.exists(DOMAIN_AWARE_MODEL_PATH) 
    if DOMAIN_AWARE_AVAILABLE:
        logging.info(f"Domain-aware model found at {DOMAIN_AWARE_MODEL_PATH}")
    else:
        logging.warning(f"Domain-aware model file not found at {DOMAIN_AWARE_MODEL_PATH}")
except ImportError:
    DOMAIN_AWARE_AVAILABLE = False
    logging.warning("Domain-aware sentiment model not available. Related endpoints will be limited.")

# Import active learning functionality if available
try:
    from active_learning_framework import (
        init_database, store_expert_feedback, store_prediction_for_feedback,
        load_models
    )
    ACTIVE_LEARNING_AVAILABLE = True
    # Initialize the database connection
    db_engine = init_database()
    sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer = load_models()
except ImportError:
    ACTIVE_LEARNING_AVAILABLE = False
    logging.warning("Active learning framework not available. Feedback functionality will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_api')

# Create Flask application
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: Status of the API and available models
    """
    available_models = get_available_models()
    
    # Include domain-aware model availability
    domain_aware_status = "available" if DOMAIN_AWARE_AVAILABLE else "unavailable"
    active_learning_status = "available" if ACTIVE_LEARNING_AVAILABLE else "unavailable"
    
    return jsonify({
        "status": "healthy",
        "available_models": available_models,
        "domain_aware_model": domain_aware_status,
        "active_learning": active_learning_status
    })

@app.route('/models', methods=['GET'])
def list_models():
    """
    List available models endpoint.
    
    Returns:
        JSON: List of available sentiment models
    """
    available_models = get_available_models()
    
    # Add domain-aware model if available
    if DOMAIN_AWARE_AVAILABLE and "domain_aware" not in available_models:
        available_models.append("domain_aware")
    
    # Add expanded model if available
    if EXPANDED_MODEL_AVAILABLE and "expanded" not in available_models:
        available_models.append("expanded")
    
    # Add ensemble model if available
    if ENSEMBLE_MODEL_AVAILABLE and "ensemble" not in available_models:
        available_models.append("ensemble")
    
    return jsonify({
        "models": available_models,
        "count": len(available_models),
        "default": "ensemble" if "ensemble" in available_models else
                  "domain_aware" if "domain_aware" in available_models else 
                  ("expanded" if "expanded" in available_models else 
                   "synthetic" if "synthetic" in available_models else 
                   available_models[0] if available_models else None)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting sentiment of a single text.
    
    Request Body:
        JSON with 'text' field and optional 'model_type' field
        
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Get model type if specified, default to domain-aware if available
        available_models = get_available_models()
        
        # Add domain-aware model if available
        if DOMAIN_AWARE_AVAILABLE and "domain_aware" not in available_models:
            available_models.append("domain_aware")
            
        # Add expanded model if available
        if EXPANDED_MODEL_AVAILABLE and "expanded" not in available_models:
            available_models.append("expanded")
            
        # Add ensemble model if available
        if ENSEMBLE_MODEL_AVAILABLE and "ensemble" not in available_models:
            available_models.append("ensemble")
            
        if not available_models:
            return jsonify({"error": "No models available"}), 500
            
        if 'model_type' in data and data['model_type'] in available_models:
            model_type = data['model_type']
        else:
            model_type = "ensemble" if ENSEMBLE_MODEL_AVAILABLE else (
                "domain_aware" if DOMAIN_AWARE_AVAILABLE else (
                "expanded" if EXPANDED_MODEL_AVAILABLE else 
                "synthetic" if "synthetic" in available_models else available_models[0]
                )
            )
            
        logger.info(f"Using model type: {model_type}")
        
        # Check if we should store for active learning feedback or just predict
        store_for_feedback = data.get('store_for_feedback', False)
        
        # Handle domain-aware model separately
        if model_type == "domain_aware" and DOMAIN_AWARE_AVAILABLE:
            sentiment_result = predict_domain_aware_sentiment(text)
            
            # Store prediction for potential feedback only if explicitly requested
            entry_id = None
            if ACTIVE_LEARNING_AVAILABLE and store_for_feedback:
                try:
                    entry_id = store_prediction_for_feedback(
                        text, 
                        sentiment_result["sentiment_value"], 
                        sentiment_result["confidence"],
                        sentiment_result["domains"],
                        db_engine
                    )
                    logger.info(f"Stored prediction for active learning feedback with ID: {entry_id}")
                    
                    # Add the entry_id to the response when store_for_feedback is True
                    if entry_id:
                        sentiment_result["entry_id"] = entry_id
                        
                except Exception as e:
                    logger.error(f"Failed to store prediction for feedback: {str(e)}")
            
            # Save to database if incident_id is provided
            if 'incident_id' in data:
                save_sentiment(data['incident_id'], sentiment_result["sentiment_value"], sentiment_result["confidence"])
            
            return jsonify(sentiment_result)
        else:
            # Make prediction with standard models
            sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, model_type)
            
            # Save to database if incident_id is provided
            if 'incident_id' in data:
                save_sentiment(data['incident_id'], sentiment_value, confidence)
            
            response = {
                "sentiment": sentiment_label,
                "sentiment_value": int(sentiment_value),
                "confidence": float(confidence),
                "text": text,
                "model_type": model_type
            }
            
            # Add model_used for hybrid model
            if model_type == "hybrid" and model_used:
                response["model_used"] = model_used
            
            # Store for feedback if requested
            entry_id = None
            if ACTIVE_LEARNING_AVAILABLE and store_for_feedback:
                try:
                    # Use general domain for non-domain-aware models
                    domains = ["general"]
                    entry_id = store_prediction_for_feedback(
                        text, sentiment_value, confidence, domains, db_engine
                    )
                    logger.info(f"Stored standard model prediction for feedback with ID: {entry_id}")
                    
                    # Add the entry_id to the response when store_for_feedback is True
                    if entry_id:
                        response["entry_id"] = entry_id
                        
                except Exception as e:
                    logger.error(f"Failed to store prediction for feedback: {str(e)}")
            
            return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/domain_predict', methods=['POST'])
def domain_predict():
    """
    Endpoint for domain-aware sentiment prediction.
    
    Request Body:
        JSON with 'text' field
        
    Returns:
        JSON with prediction results including domains
    """
    if not DOMAIN_AWARE_AVAILABLE:
        return jsonify({"error": "Domain-aware model not available"}), 404
        
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        store_for_feedback = data.get('store_for_feedback', False)
        
        # Make prediction with domain-aware model
        result = predict_domain_aware_sentiment(text)
        
        # Store prediction for potential feedback only if explicitly requested
        entry_id = None
        if ACTIVE_LEARNING_AVAILABLE and store_for_feedback:
            try:
                entry_id = store_prediction_for_feedback(
                    text, 
                    result["sentiment_value"], 
                    result["confidence"],
                    result["domains"],
                    db_engine
                )
                logger.info(f"Stored domain prediction for active learning feedback with ID: {entry_id}")
                
                # Add the entry_id to the response when store_for_feedback is True
                if entry_id:
                    result["entry_id"] = entry_id
                    
            except Exception as e:
                logger.error(f"Failed to store prediction for feedback: {str(e)}")
        
        # Save to database if incident_id is provided
        if 'incident_id' in data:
            save_sentiment(data['incident_id'], result["sentiment_value"], result["confidence"])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in domain prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/predict_domains', methods=['POST'])
def get_domains():
    """
    Endpoint for predicting domains from text.
    
    Request Body:
        JSON with 'text' field
        
    Returns:
        JSON with domain predictions
    """
    if not DOMAIN_AWARE_AVAILABLE:
        return jsonify({"error": "Domain classifier not available"}), 404
        
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Predict domains
        domains = predict_domains(text)
        
        return jsonify({
            "text": text,
            "domains": domains
        })
        
    except Exception as e:
        logger.error(f"Error in domain prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    """
    Endpoint for providing expert feedback on sentiment predictions.
    
    Request Body:
        JSON with:
        - entry_id: ID of the prediction to provide feedback for
        - corrected_sentiment: Expert-corrected sentiment value (-1, 0, 1)
        
    Returns:
        JSON with success status
    """
    if not ACTIVE_LEARNING_AVAILABLE:
        return jsonify({"error": "Active learning framework not available"}), 404
        
    try:
        data = request.get_json()
        
        if not data or 'entry_id' not in data or 'corrected_sentiment' not in data:
            return jsonify({
                "error": "Missing required fields. Please provide 'entry_id' and 'corrected_sentiment'"
            }), 400
        
        entry_id = data['entry_id']
        corrected_sentiment = int(data['corrected_sentiment'])
        
        # Validate sentiment value
        if corrected_sentiment not in [-1, 0, 1]:
            return jsonify({
                "error": "Invalid 'corrected_sentiment'. Must be -1 (negative), 0 (neutral), or 1 (positive)"
            }), 400
        
        # Store expert feedback
        success = store_expert_feedback(entry_id, corrected_sentiment, db_engine)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Feedback stored successfully",
                "entry_id": entry_id
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to store feedback. Entry ID may not exist",
                "entry_id": entry_id
            }), 404
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_route():
    """
    Endpoint for predicting sentiment of multiple texts.
    
    Request Body:
        JSON with 'texts' field (list of strings) and optional 'model_type' field
        
    Returns:
        JSON with list of prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' field in request"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' field should be a list"}), 400
        
        # Get model type if specified, default to synthetic if available
        available_models = get_available_models()
        if not available_models:
            return jsonify({"error": "No models available"}), 500
            
        if 'model_type' in data and data['model_type'] in available_models:
            model_type = data['model_type']
        else:
            model_type = "synthetic" if "synthetic" in available_models else available_models[0]
            
        logger.info(f"Using model type for batch prediction: {model_type}")
        
        # Make predictions
        predictions = batch_predict(texts, model_type)
        
        # Save to database if incident_ids are provided
        if 'incident_ids' in data and isinstance(data['incident_ids'], list):
            incident_ids = data['incident_ids']
            if len(incident_ids) == len(predictions):
                for incident_id, (value, _, confidence, _) in zip(incident_ids, predictions):
                    save_sentiment(incident_id, value, confidence)
        
        # Format results
        results = []
        for (value, label, conf, model_used), text in zip(predictions, texts):
            result = {
                "sentiment": label,
                "sentiment_value": int(value),
                "confidence": float(conf),
                "text": text
            }
            
            # Add model_used for hybrid model
            if model_type == "hybrid" and model_used:
                result["model_used"] = model_used
                
            results.append(result)
        
        return jsonify({
            "predictions": results,
            "count": len(results),
            "model_type": model_type
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    """
    Endpoint for testing prediction with a sample text.
    
    Query Parameters:
        model_type (optional): The model to use for prediction
    
    Returns:
        JSON with prediction results
    """
    sample_text = "Successfully rescued family from burning building with no injuries."
    
    # Get model type from query parameter if provided
    model_type = request.args.get('model_type', None)
    
    # Check if model type is valid
    available_models = get_available_models()
    if not available_models:
        return jsonify({"error": "No models available"}), 500
        
    if model_type not in available_models:
        model_type = "synthetic" if "synthetic" in available_models else available_models[0]
    
    # Make prediction
    sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(sample_text, model_type)
    
    response = {
        "sentiment": sentiment_label,
        "sentiment_value": int(sentiment_value),
        "confidence": float(confidence),
        "text": sample_text,
        "model_type": model_type
    }
    
    # Add model_used for hybrid model
    if model_type == "hybrid" and model_used:
        response["model_used"] = model_used
    
    return jsonify(response)

@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Endpoint for comparing sentiment predictions across multiple models.
    
    Request Body:
        JSON with 'text' field
        
    Returns:
        JSON with predictions from all available models
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Get available models
        available_models = get_available_models()
        
        # Add domain-aware model if available
        if DOMAIN_AWARE_AVAILABLE and "domain_aware" not in available_models:
            available_models.append("domain_aware")
            
        # Add expanded model if available
        if EXPANDED_MODEL_AVAILABLE and "expanded" not in available_models:
            available_models.append("expanded")
        
        if not available_models:
            return jsonify({"error": "No models available"}), 500
        
        # Get predictions from each model
        results = {}
        for model_type in available_models:
            if model_type == "domain_aware" and DOMAIN_AWARE_AVAILABLE:
                # Use domain-aware model
                prediction = predict_domain_aware_sentiment(text)
                results[model_type] = {
                    "sentiment": prediction["sentiment"],
                    "sentiment_value": prediction["sentiment_value"],
                    "confidence": prediction["confidence"],
                    "domains": prediction.get("domains", ["unknown"])
                }
            else:
                # Use standard model
                sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, model_type)
                results[model_type] = {
                    "sentiment": sentiment_label,
                    "sentiment_value": int(sentiment_value),
                    "confidence": float(confidence)
                }
                
                # Add model_used for hybrid model
                if model_type == "hybrid" and model_used:
                    results[model_type]["model_used"] = model_used
        
        # Return all results
        return jsonify({
            "text": text,
            "models": results,
            "count": len(results)
        })
    
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('MODEL_API_PORT', MODEL_API_PORT))
    logger.info(f"Starting model API on port {port}")
    
    # Log available models
    available_models = get_available_models()
    logger.info(f"Available models: {', '.join(available_models) if available_models else 'None'}")
    
    app.run(host='0.0.0.0', port=port) 