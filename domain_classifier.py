#!/usr/bin/env python3
"""
Domain Classification for Emergency Services

This module provides domain classification functionality for emergency services text.
"""
import os
import sys
import logging
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('domain_classifier')

# Define constants
DOMAIN_THRESHOLD = 0.5  # Probability threshold for domain classification

# Define file paths
MODELS_DIR = "app/models"
DOMAIN_MODEL_PATH = os.path.join(MODELS_DIR, "domain_classifier_model.pkl")
DOMAIN_VECTORIZER_PATH = os.path.join(MODELS_DIR, "domain_classifier_vectorizer.pkl")
DOMAIN_BINARIZER_PATH = os.path.join(MODELS_DIR, "domain_classifier_binarizer.pkl")

# Define domains and their related keywords
DOMAIN_KEYWORDS = {
    'fire': ['fire', 'firefighter', 'burn', 'flame', 'smoke', 'arson', 'wildfire', 'extinguish'],
    'police': ['police', 'officer', 'crime', 'arrest', 'law', 'enforcement', 'detective', 'patrol'],
    'ems': ['ambulance', 'paramedic', 'emt', 'emergency medical', 'hospital', 'injury', 'medical', 'patient'],
    'disaster_response': ['disaster', 'hurricane', 'flood', 'earthquake', 'evacuation', 'relief', 'emergency management'],
    'coast_guard': ['coast guard', 'maritime', 'rescue', 'water', 'boat', 'ship', 'drowning', 'ocean']
}

def load_domain_classifier():
    """
    Load the domain classifier model, vectorizer, and binarizer.
    
    Returns:
        tuple: (vectorizer, model, binarizer) or (None, None, None) if loading fails
    """
    try:
        vectorizer = joblib.load(DOMAIN_VECTORIZER_PATH)
        model = joblib.load(DOMAIN_MODEL_PATH)
        binarizer = joblib.load(DOMAIN_BINARIZER_PATH)
        logger.info("Domain classifier loaded successfully")
        return vectorizer, model, binarizer
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Failed to load domain classifier: {str(e)}")
        return create_simple_domain_classifier()

def create_simple_domain_classifier():
    """
    Create a simple domain classifier for when the models are not available.
    
    Returns:
        tuple: (vectorizer, model, binarizer)
    """
    logger.info("Creating simple domain classifier")
    
    # Create a simple TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Initialize with some sample texts to fit vocabulary
    sample_texts = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        sample_texts.extend([f"This is about {kw}" for kw in keywords])
    
    vectorizer.fit(sample_texts)
    
    # Create a simple multi-label classifier
    model = MultiOutputClassifier(OneVsRestClassifier(SVC(probability=True)))
    
    # Create domain labels
    domain_labels = []
    for _ in sample_texts:
        domains_present = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in _ for kw in keywords):
                domains_present.append(domain)
        if not domains_present:
            domains_present = ['general']
        domain_labels.append(domains_present)
    
    # Create binarizer and transform labels
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(domain_labels)
    
    # Fit model
    X = vectorizer.transform(sample_texts)
    model.fit(X, y)
    
    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, DOMAIN_VECTORIZER_PATH)
    joblib.dump(model, DOMAIN_MODEL_PATH)
    joblib.dump(binarizer, DOMAIN_BINARIZER_PATH)
    
    logger.info("Simple domain classifier created and saved")
    return vectorizer, model, binarizer

def predict_domains(text, vectorizer=None, domain_model=None, domain_binarizer=None):
    """
    Predict domains for a given text.
    
    Args:
        text (str): Input text
        vectorizer (TfidfVectorizer, optional): Vectorizer for text
        domain_model (sklearn.base.BaseEstimator, optional): Trained domain classifier model
        domain_binarizer (LabelBinarizer, optional): Binarizer for domain labels
        
    Returns:
        list: Predicted domains
    """
    try:
        # Convert text to lowercase for keyword matching
        text_lower = text.lower() if isinstance(text, str) else str(text).lower()
        
        # Check for keywords before using ML model
        keyword_domains = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                keyword_domains.append(domain)
                
        if keyword_domains:
            logger.info(f"Domains predicted using keywords: {keyword_domains}")
            return keyword_domains
            
        # Load domain classifier if not provided
        if vectorizer is None or domain_model is None or domain_binarizer is None:
            vectorizer, domain_model, domain_binarizer = load_domain_classifier()
            
        # Check if domain classifier is available
        if vectorizer is not None and domain_model is not None and domain_binarizer is not None:
            # Ensure text is a list (required for scikit-learn)
            if isinstance(text, str):
                text_list = [text]
            else:
                text_list = list(text) if hasattr(text, '__iter__') else [str(text)]
                
            # Vectorize text
            text_vectorized = vectorizer.transform(text_list)
            
            # Check shape for prediction
            if text_vectorized.shape[0] != 1:
                raise ValueError(f"Expected 1 sample for prediction, got {text_vectorized.shape[0]}")
                
            # Predict domains
            pred_proba = domain_model.predict_proba(text_vectorized)
            
            # Get domains above threshold
            pred_domains = []
            for i, class_idx in enumerate(pred_proba[0] > 0.5):
                if class_idx:
                    domain = domain_binarizer.classes_[i]
                    pred_domains.append(domain)
                    
            logger.info(f"Domains predicted using ML model: {pred_domains}")
            return pred_domains if pred_domains else ['general']
        else:
            # Fallback to general domain
            logger.warning("Domain classifier not available, defaulting to 'general' domain")
            return ['general']
            
    except Exception as e:
        logger.error(f"Error in domain prediction: {str(e)}")
        return ['general']

def main():
    """Main function to test domain classification."""
    try:
        # Test texts
        test_texts = [
            "Firefighters quickly extinguished the kitchen fire",
            "Police officers arrested the suspect after a brief chase",
            "Paramedics treated multiple patients at the scene of the accident",
            "Coast Guard rescued three people from a sinking boat",
            "Emergency management officials coordinated the evacuation during the flood",
            "I had breakfast this morning"
        ]
        
        # Load or create classifier
        vectorizer, model, binarizer = load_domain_classifier()
        
        # Test predictions
        for text in test_texts:
            domains = predict_domains(text, vectorizer, model, binarizer)
            print(f"Text: '{text}'")
            print(f"Domains: {domains}\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error testing domain classifier: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 