#!/usr/bin/env python3
"""
Active Learning Framework for Emergency Services Sentiment Analysis

This script implements an active learning framework that allows the sentiment
analysis model to improve through expert feedback on uncertain predictions.
"""
import os
import sys
import logging
import json
import joblib
import time
import uuid
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('active_learning')

# File paths and constants
EMERGENCY_MODEL_PATH = "app/models/emergency_services_model.pkl"
EMERGENCY_VECTORIZER_PATH = "app/models/emergency_services_vectorizer.pkl"
DOMAIN_MODEL_PATH = "app/models/domain_classifier_model.pkl"
DOMAIN_VECTORIZER_PATH = "app/models/domain_classifier_vectorizer.pkl"
DOMAIN_BINARIZER_PATH = "app/models/domain_classifier_binarizer.pkl"
FEEDBACK_DB_PATH = "app/data/feedback.db"
FEEDBACK_THRESHOLD = 0.3  # Confidence threshold for requesting feedback

# SQLAlchemy setup
Base = declarative_base()

class FeedbackEntry(Base):
    """SQLAlchemy model for feedback database entries."""
    __tablename__ = 'feedback'
    
    id = Column(String, primary_key=True)
    text = Column(String, nullable=False)
    predicted_sentiment = Column(Integer, nullable=False)
    corrected_sentiment = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=False)
    domains = Column(String, nullable=False)
    reviewed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<FeedbackEntry(id='{self.id}', text='{self.text[:20]}...', confidence={self.confidence_score})>"

def init_database(): 
    """
    Initialize the feedback database.
    """
    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(FEEDBACK_DB_PATH), exist_ok=True)
    
    # Create engine and tables
    engine = create_engine(f'sqlite:///{FEEDBACK_DB_PATH}')
    Base.metadata.create_all(engine)
    
    logger.info(f"Initialized feedback database at {FEEDBACK_DB_PATH}")
    return engine

def load_models():
    """
    Load the sentiment and domain classification models.
    
    Returns:
        tuple: (sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer)
    """
    try:
        sentiment_vectorizer = joblib.load(EMERGENCY_VECTORIZER_PATH)
        sentiment_model = joblib.load(EMERGENCY_MODEL_PATH)
        domain_vectorizer = joblib.load(DOMAIN_VECTORIZER_PATH)
        domain_model = joblib.load(DOMAIN_MODEL_PATH)
        domain_binarizer = joblib.load(DOMAIN_BINARIZER_PATH)
        
        logger.info("Successfully loaded sentiment and domain models")
        return sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def predict_with_confidence(text, vectorizer, model):
    """
    Make sentiment prediction with confidence score.
    
    Args:
        text (str): Input text
        vectorizer: TF-IDF vectorizer
        model: Trained sentiment model
        
    Returns:
        tuple: (predicted_sentiment, confidence_score)
    """
    # Process input
    if not isinstance(text, str):
        text = str(text)
    
    # Vectorize input
    text_vectorized = vectorizer.transform([text])
    
    # Get prediction and decision function scores
    prediction = model.predict(text_vectorized)[0]
    decision_scores = model.decision_function(text_vectorized)[0]
    
    # For multi-class SVM, convert decision scores to probabilities with softmax
    if isinstance(decision_scores, np.ndarray) and len(decision_scores) > 1:
        # Softmax function to convert scores to probabilities
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        confidence = np.max(probabilities)
    else:
        # For binary classification, use absolute value of the decision score
        # and normalize it to [0, 1] range using a sigmoid-like function
        confidence = 1 / (1 + np.exp(-np.abs(decision_scores)))
    
    return prediction, float(confidence)

def predict_domains(text, vectorizer, model, binarizer):
    """
    Predict domains for a given text.
    
    Args:
        text (str): Input text
        vectorizer: Domain TF-IDF vectorizer
        model: Domain classifier model
        binarizer: MultiLabelBinarizer for domain labels
        
    Returns:
        list: Predicted domain labels
    """
    # Process input
    if not isinstance(text, str):
        text = str(text)
    
    # Vectorize input
    text_vectorized = vectorizer.transform([text])
    
    # Predict binary domain labels
    domains_binary = model.predict(text_vectorized)[0]
    
    # Convert binary labels back to domain names
    domains = binarizer.classes_[np.where(domains_binary == 1)[0]].tolist()
    
    # Return general domain if no specific domain is predicted
    return domains if domains else ['general']

def store_prediction_for_feedback(text, sentiment, confidence, domains, engine):
    """
    Store a prediction in the feedback database for potential expert review.
    
    Args:
        text (str): Input text
        sentiment (int): Predicted sentiment (-1, 0, 1)
        confidence (float): Confidence score (0-1)
        domains (list): List of domain labels
        engine: SQLAlchemy database engine
        
    Returns:
        str: Feedback entry ID
    """
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create a new feedback entry
        entry_id = str(uuid.uuid4())
        entry = FeedbackEntry(
            id=entry_id,
            text=text,
            predicted_sentiment=int(sentiment),
            confidence_score=float(confidence),
            domains=json.dumps(domains)
        )
        
        # Add to session and commit
        session.add(entry)
        session.commit()
        
        logger.info(f"Stored prediction for feedback with ID: {entry_id}")
        return entry_id
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing prediction for feedback: {str(e)}")
        raise
    
    finally:
        session.close()

def get_uncertain_predictions(engine, limit=10):
    """
    Get predictions with low confidence for expert review.
    
    Args:
        engine: SQLAlchemy database engine
        limit (int): Maximum number of entries to return
        
    Returns:
        list: List of feedback entries
    """
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Query for unreviewed entries with low confidence
        entries = session.query(FeedbackEntry)\
            .filter(FeedbackEntry.reviewed == False)\
            .filter(FeedbackEntry.confidence_score < FEEDBACK_THRESHOLD)\
            .order_by(FeedbackEntry.confidence_score)\
            .limit(limit)\
            .all()
        
        logger.info(f"Found {len(entries)} uncertain predictions for review")
        return entries
    
    except Exception as e:
        logger.error(f"Error getting uncertain predictions: {str(e)}")
        raise
    
    finally:
        session.close()

def store_expert_feedback(entry_id, corrected_sentiment, engine):
    """
    Store expert feedback for a prediction.
    
    Args:
        entry_id (str): Feedback entry ID
        corrected_sentiment (int): Corrected sentiment value (-1, 0, 1)
        engine: SQLAlchemy database engine
        
    Returns:
        bool: Success status
    """
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Find the entry
        entry = session.query(FeedbackEntry).filter(FeedbackEntry.id == entry_id).first()
        
        if not entry:
            logger.error(f"Feedback entry with ID {entry_id} not found")
            return False
        
        # Update with expert feedback
        entry.corrected_sentiment = corrected_sentiment
        entry.reviewed = True
        entry.updated_at = datetime.now()
        
        # Commit changes
        session.commit()
        
        logger.info(f"Stored expert feedback for entry {entry_id}")
        return True
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing expert feedback: {str(e)}")
        return False
    
    finally:
        session.close()

def get_feedback_dataset(engine, include_all=False):
    """
    Get a dataset of reviewed feedback entries for model retraining.
    
    Args:
        engine: SQLAlchemy database engine
        include_all (bool): Whether to include all entries or only corrected ones
        
    Returns:
        pandas.DataFrame: DataFrame with feedback data
    """
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Base query for reviewed entries
        query = session.query(FeedbackEntry).filter(FeedbackEntry.reviewed == True)
        
        # Optionally filter for only corrected entries
        if not include_all:
            query = query.filter(FeedbackEntry.predicted_sentiment != FeedbackEntry.corrected_sentiment)
        
        # Execute query
        entries = query.all()
        
        # Convert to DataFrame
        data = []
        for entry in entries:
            sentiment = entry.corrected_sentiment
            data.append({
                'id': entry.id,
                'text': entry.text,
                'sentiment': sentiment,
                'domains': json.loads(entry.domains)
            })
        
        df = pd.DataFrame(data)
        
        logger.info(f"Retrieved {len(df)} feedback entries for retraining")
        return df
    
    except Exception as e:
        logger.error(f"Error getting feedback dataset: {str(e)}")
        raise
    
    finally:
        session.close()

def retrain_with_feedback(engine, sentiment_vectorizer, sentiment_model):
    """
    Retrain sentiment model with expert feedback.
    
    Args:
        engine: SQLAlchemy database engine
        sentiment_vectorizer: Current TF-IDF vectorizer
        sentiment_model: Current sentiment model
        
    Returns:
        tuple: (updated_vectorizer, updated_model) if retraining was done
    """
    # Get feedback dataset
    feedback_df = get_feedback_dataset(engine)
    
    # Check if we have enough data for retraining
    if len(feedback_df) < 10:
        logger.info(f"Not enough feedback data for retraining ({len(feedback_df)} entries)")
        return sentiment_vectorizer, sentiment_model
    
    try:
        logger.info(f"Retraining model with {len(feedback_df)} feedback entries")
        
        # Get existing dataset (placeholder - in a real implementation, we would load the original training data)
        # Here we're using a small subset of the feedback data to simulate original data
        # In production, you would load your original training data
        original_texts = feedback_df['text'].values[:5]
        original_labels = feedback_df['sentiment'].values[:5]
        
        # Combine with feedback data
        X = np.concatenate([original_texts, feedback_df['text'].values])
        y = np.concatenate([original_labels, feedback_df['sentiment'].values])
        
        # Vectorize combined data using the existing vectorizer
        X_vectorized = sentiment_vectorizer.transform(X)
        
        # Retrain model
        updated_model = sentiment_model
        updated_model.fit(X_vectorized, y)
        
        # Save updated model
        timestamp = int(time.time())
        updated_model_path = f"app/models/emergency_services_model_feedback_{timestamp}.pkl"
        joblib.dump(updated_model, updated_model_path)
        
        logger.info(f"Model retrained and saved to {updated_model_path}")
        
        return sentiment_vectorizer, updated_model
    
    except Exception as e:
        logger.error(f"Error retraining model with feedback: {str(e)}")
        return sentiment_vectorizer, sentiment_model

def simulate_expert_feedback(engine, sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer):
    """
    Simulate expert feedback for demonstration purposes.
    
    Args:
        engine: SQLAlchemy database engine
        sentiment_vectorizer, sentiment_model: Sentiment models
        domain_vectorizer, domain_model, domain_binarizer: Domain models
    """
    # Example texts
    example_texts = [
        "The firefighters quickly contained the blaze with no injuries reported.",
        "Police officers are investigating a series of break-ins in the neighborhood.",
        "Paramedics treated several patients at the scene of the accident.",
        "The flood waters are rising and evacuation orders have been issued.",
        "The 911 dispatcher remained calm while helping the caller perform CPR."
    ]
    
    # Make predictions and store for feedback
    for text in example_texts:
        sentiment, confidence = predict_with_confidence(text, sentiment_vectorizer, sentiment_model)
        domains = predict_domains(text, domain_vectorizer, domain_model, domain_binarizer)
        
        entry_id = store_prediction_for_feedback(text, sentiment, confidence * 0.5, domains, engine)
        
        # Simulate expert feedback (for demonstration)
        # In a real system, this would be provided by human experts
        if "quickly contained" in text:
            store_expert_feedback(entry_id, 1, engine)  # Positive
        elif "investigating" in text:
            store_expert_feedback(entry_id, 0, engine)  # Neutral
        elif "accident" in text:
            store_expert_feedback(entry_id, -1, engine)  # Negative
        elif "evacuation" in text:
            store_expert_feedback(entry_id, -1, engine)  # Negative
        elif "calm" in text:
            store_expert_feedback(entry_id, 1, engine)  # Positive
    
    logger.info("Simulated expert feedback stored in database")

def main():
    """Main function to demonstrate the active learning framework."""
    try:
        logger.info("Starting active learning framework demonstration")
        
        # Initialize database
        engine = init_database()
        
        # Load models
        sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer = load_models()
        
        # Simulate feedback
        simulate_expert_feedback(engine, sentiment_vectorizer, sentiment_model, 
                               domain_vectorizer, domain_model, domain_binarizer)
        
        # Get uncertain predictions for review
        uncertain_predictions = get_uncertain_predictions(engine)
        
        # Display uncertain predictions
        logger.info("Uncertain predictions for expert review:")
        for prediction in uncertain_predictions:
            logger.info(f"ID: {prediction.id}, Text: {prediction.text[:50]}..., "
                      f"Predicted: {prediction.predicted_sentiment}, "
                      f"Confidence: {prediction.confidence_score:.4f}")
        
        # Retrain model with feedback
        updated_vectorizer, updated_model = retrain_with_feedback(
            engine, sentiment_vectorizer, sentiment_model)
        
        logger.info("Active learning framework demonstration completed")
        
    except Exception as e:
        logger.error(f"Error in active learning framework: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 