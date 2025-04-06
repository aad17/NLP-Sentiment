#!/usr/bin/env python3
"""
Domain-Aware Sentiment Analysis for Emergency Services

This script implements a domain-aware sentiment analysis model that leverages
domain classification to improve sentiment prediction accuracy across different
emergency service domains.
"""
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('domain_aware_sentiment')

# File paths
DATA_DIR = "app/data"
MODELS_DIR = "app/models"
EMERGENCY_TWEETS_PATH = os.path.join(DATA_DIR, "emergency_tweets_processed.csv")
DOMAIN_MODEL_PATH = os.path.join(MODELS_DIR, "domain_classifier_model.pkl")
DOMAIN_VECTORIZER_PATH = os.path.join(MODELS_DIR, "domain_classifier_vectorizer.pkl")
DOMAIN_BINARIZER_PATH = os.path.join(MODELS_DIR, "domain_classifier_binarizer.pkl")
DOMAIN_AWARE_MODEL_PATH = os.path.join(MODELS_DIR, "domain_aware_sentiment_model.pkl")

# Define domains and their related keywords
DOMAIN_KEYWORDS = {
    'fire': ['fire', 'firefighter', 'burn', 'flame', 'smoke', 'arson', 'wildfire', 'extinguish'],
    'police': ['police', 'officer', 'crime', 'arrest', 'law', 'enforcement', 'detective', 'patrol'],
    'ems': ['ambulance', 'paramedic', 'emt', 'emergency medical', 'hospital', 'injury', 'medical', 'patient'],
    'disaster_response': ['disaster', 'hurricane', 'flood', 'earthquake', 'evacuation', 'relief', 'emergency management'],
    'coast_guard': ['coast guard', 'maritime', 'rescue', 'water', 'boat', 'ship', 'drowning', 'ocean']
}

def load_data():
    """
    Load the emergency tweets dataset.
    
    Returns:
        pandas.DataFrame: Emergency tweets dataset
    """
    logger.info(f"Loading data from {EMERGENCY_TWEETS_PATH}")
    
    try:
        df = pd.read_csv(EMERGENCY_TWEETS_PATH)
        logger.info(f"Loaded {len(df)} tweets")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def assign_domains(df, text_column='text'):
    """
    Assign domains to tweets based on keyword matching.
    
    Args:
        df (pandas.DataFrame): Dataset containing tweets
        text_column (str): Name of the column containing tweet text
        
    Returns:
        pandas.DataFrame: Dataset with domain labels
    """
    logger.info("Assigning domains to tweets")
    
    # Initialize domain columns with 0
    for domain in DOMAIN_KEYWORDS:
        df[domain] = 0
    
    # Convert text to lowercase for case-insensitive matching
    df['text_lower'] = df[text_column].str.lower()
    
    # Assign domains based on keyword presence
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            df.loc[df['text_lower'].str.contains(keyword, na=False), domain] = 1
    
    # Create a multi-label domain column
    df['domains'] = df.apply(
        lambda row: [domain for domain in DOMAIN_KEYWORDS if row[domain] == 1],
        axis=1
    )
    
    # If no domain is assigned, label as 'general'
    df.loc[df['domains'].apply(len) == 0, 'domains'] = [['general']]
    
    # Drop temporary columns
    df.drop('text_lower', axis=1, inplace=True)
    
    logger.info("Domain assignment completed")
    return df

def load_domain_classifier():
    """
    Load pre-trained domain classifier.
    
    Returns:
        tuple: (vectorizer, model, binarizer)
    """
    logger.info("Loading domain classifier")
    
    try:
        vectorizer = joblib.load(DOMAIN_VECTORIZER_PATH)
        model = joblib.load(DOMAIN_MODEL_PATH)
        binarizer = joblib.load(DOMAIN_BINARIZER_PATH)
        
        logger.info("Domain classifier loaded successfully")
        return vectorizer, model, binarizer
    
    except FileNotFoundError:
        logger.warning("Domain classifier not found. Will train a new one.")
        return None, None, None

def train_domain_classifier(df, text_column='text', domains_column='domains'):
    """
    Train a domain classifier.
    
    Args:
        df (pandas.DataFrame): Dataset with domain labels
        text_column (str): Name of the column containing tweet text
        domains_column (str): Name of the column containing domain labels
        
    Returns:
        tuple: (vectorizer, model, binarizer)
    """
    logger.info("Training domain classifier")
    
    # Prepare multi-label binarizer
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(df[domains_column])
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    
    # Train model
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultiOutputClassifier(OneVsRestClassifier(SVC(probability=True))))
    ])
    
    X = df[text_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    logger.info(f"Domain classifier accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, DOMAIN_VECTORIZER_PATH)
    joblib.dump(model, DOMAIN_MODEL_PATH)
    joblib.dump(binarizer, DOMAIN_BINARIZER_PATH)
    
    logger.info("Domain classifier trained and saved successfully")
    return vectorizer, model, binarizer

def predict_domains(text, vectorizer, model, binarizer):
    """
    Predict domains for a given text.
    
    Args:
        text (str): Input text
        vectorizer: Domain vectorizer
        model: Domain model
        binarizer: Domain binarizer
        
    Returns:
        list: Predicted domains
    """
    # Process input
    if not isinstance(text, str):
        text = str(text)
    
    # Predict
    pred_binary = model.predict([text])[0]
    
    # Convert binary prediction to domain labels
    pred_domains = binarizer.classes_[np.where(pred_binary == 1)[0]].tolist()
    
    # Return general domain if no specific domain is predicted
    return pred_domains if pred_domains else ['general']

def predict_domain_aware_sentiment(text):
    """
    Predict sentiment with domain awareness for a given text.
    This function is designed to be called from the API.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Prediction result including sentiment, confidence, and domains
    """
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        # Import domain classifier module only when needed
        try:
            from domain_classifier import predict_domains as domain_predict
            domain_classifier_available = True
        except ImportError:
            domain_classifier_available = False
            logger.warning("Domain classifier module not available, using simple keyword matching")
        
        # Get domains for the text
        if domain_classifier_available:
            domains = domain_predict(text)
        else:
            # Simple keyword matching fallback
            domains = []
            text_lower = text.lower()
            for domain, keywords in DOMAIN_KEYWORDS.items():
                if any(keyword in text_lower for keyword in keywords):
                    domains.append(domain)
            if not domains:
                domains = ["general"]
        
        # Create domain-enriched text
        text_with_domain = f"{text} {' '.join(domains)}"
        
        # Prepare text as a list (required format for sklearn)
        text_list = [text_with_domain]
        
        # Load the domain-aware sentiment model - no fallback to simple model
        if os.path.exists(DOMAIN_AWARE_MODEL_PATH):
            sentiment_model = joblib.load(DOMAIN_AWARE_MODEL_PATH)
            logger.info("Loaded domain-aware sentiment model")
            
            # Use the pipeline to predict sentiment
            sentiment_value = sentiment_model.predict(text_list)[0]
            sentiment_proba = sentiment_model.predict_proba(text_list)[0]
            confidence = float(np.max(sentiment_proba))
            logger.info(f"Model prediction successful: {sentiment_value} with confidence {confidence:.2f}")
        else:
            # If model doesn't exist, return an error rather than creating a simple model
            logger.error(f"Domain-aware model not found at {DOMAIN_AWARE_MODEL_PATH}. Pre-trained model required.")
            raise FileNotFoundError(f"Domain-aware sentiment model not found at {DOMAIN_AWARE_MODEL_PATH}")
                
        # Map sentiment value to label
        sentiment_map = {-1: "negative", 0: "neutral", 1: "positive"}
        sentiment_label = sentiment_map.get(int(sentiment_value), "neutral")
        
        # Return structured prediction
        result = {
            "text": text,
            "sentiment": sentiment_label,
            "sentiment_value": int(sentiment_value),
            "confidence": float(confidence),
            "domains": domains,
            "model_type": "domain_aware"
        }
        
        logger.info(f"Domain-aware sentiment prediction: {sentiment_label} (confidence: {confidence:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Error in domain-aware sentiment prediction: {str(e)}")
        # Return a default response with error information
        return {
            "text": text,
            "sentiment": "neutral",
            "sentiment_value": 0,
            "confidence": 0.33,
            "domains": ["general"],
            "model_type": "domain_aware",
            "error": str(e)
        }

def train_domain_aware_sentiment_model(df, text_column='text', sentiment_column='sentiment'):
    """
    Train domain-aware sentiment models.
    
    Args:
        df (pandas.DataFrame): Dataset with domain and sentiment labels
        text_column (str): Name of the column containing tweet text
        sentiment_column (str): Name of the column containing sentiment labels
        
    Returns:
        dict: Dictionary of domain-specific sentiment models
    """
    logger.info("Training domain-aware sentiment models")
    
    # Load or train domain classifier
    domain_vectorizer, domain_model, domain_binarizer = load_domain_classifier()
    if None in (domain_vectorizer, domain_model, domain_binarizer):
        domain_vectorizer, domain_model, domain_binarizer = train_domain_classifier(df)
    
    # Predict domains for all tweets
    if 'predicted_domains' not in df.columns:
        df['predicted_domains'] = df[text_column].apply(
            lambda x: predict_domains(x, domain_vectorizer, domain_model, domain_binarizer)
        )
    
    # Expand predicted domains to individual columns
    all_domains = set()
    for domains in df['predicted_domains']:
        all_domains.update(domains)
    
    for domain in all_domains:
        df[f'pred_{domain}'] = df['predicted_domains'].apply(lambda x: 1 if domain in x else 0)
    
    # Create domain-specific sentiment models
    domain_models = {}
    
    # Add 'text+domain' feature
    df['text_with_domain'] = df[text_column] + ' ' + df['predicted_domains'].apply(lambda x: ' '.join(x))
    
    # Train a unified model that incorporates domain information
    logger.info("Training unified domain-aware sentiment model")
    
    # Create and train the model
    X = df['text_with_domain']
    y = df[sentiment_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build TF-IDF vectorizer and SVM classifier pipeline
    domain_aware_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )),
        ('classifier', SVC(C=1.0, kernel='linear', probability=True))
    ])
    
    # Train the model
    domain_aware_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = domain_aware_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Domain-aware sentiment model accuracy: {accuracy:.4f}")
    logger.info("\nClassification report:\n" + 
                classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
    
    # Save the model
    joblib.dump(domain_aware_pipeline, DOMAIN_AWARE_MODEL_PATH)
    logger.info(f"Domain-aware sentiment model saved to {DOMAIN_AWARE_MODEL_PATH}")
    
    return domain_aware_pipeline

def evaluate_model(df, domain_vectorizer, domain_model, domain_binarizer, sentiment_model, 
                  text_column='text', sentiment_column='sentiment'):
    """
    Evaluate the domain-aware sentiment model.
    
    Args:
        df (pandas.DataFrame): Test dataset
        domain_vectorizer, domain_model, domain_binarizer: Domain classifier components
        sentiment_model: Domain-aware sentiment model
        text_column (str): Name of the column containing tweet text
        sentiment_column (str): Name of the column containing sentiment labels
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating domain-aware sentiment model")
    
    # Make predictions
    predictions = []
    for _, row in df.iterrows():
        text = row[text_column]
        true_sentiment = row[sentiment_column]
        
        # Use the prediction function that returns a dictionary
        result = predict_domain_aware_sentiment(text)
        
        predictions.append({
            'text': text,
            'true_sentiment': true_sentiment,
            'predicted_sentiment': result['sentiment_value'],
            'confidence': result['confidence'],
            'domains': result['domains']
        })
    
    # Convert to DataFrame for analysis
    pred_df = pd.DataFrame(predictions)
    
    # Overall accuracy
    accuracy = (pred_df['true_sentiment'] == pred_df['predicted_sentiment']).mean()
    
    # Generate classification report
    report = classification_report(
        pred_df['true_sentiment'],
        pred_df['predicted_sentiment'],
        target_names=['negative', 'neutral', 'positive'],
        output_dict=True
    )
    
    # Domain-specific accuracy
    domain_accuracy = {}
    for domain in set(sum(pred_df['domains'].tolist(), [])):
        domain_rows = pred_df[pred_df['domains'].apply(lambda x: domain in x)]
        if len(domain_rows) > 0:
            domain_acc = (domain_rows['true_sentiment'] == domain_rows['predicted_sentiment']).mean()
            domain_accuracy[domain] = {
                'accuracy': domain_acc,
                'count': len(domain_rows)
            }
    
    # Log results
    logger.info(f"Overall accuracy: {accuracy:.4f}")
    logger.info("\nClassification report:\n" + 
                classification_report(pred_df['true_sentiment'], pred_df['predicted_sentiment'], 
                                     target_names=['negative', 'neutral', 'positive']))
    
    for domain, metrics in domain_accuracy.items():
        logger.info(f"{domain} domain accuracy: {metrics['accuracy']:.4f} ({metrics['count']} samples)")
    
    # Create results dictionary
    results = {
        'overall_accuracy': accuracy,
        'classification_report': report,
        'domain_accuracy': domain_accuracy,
        'sample_predictions': predictions[:10]  # Include some sample predictions
    }
    
    return results

def save_results(results, filename="domain_aware_results.json"):
    """
    Save evaluation results to file.
    
    Args:
        results (dict): Evaluation results
        filename (str): Output filename
    """
    output_path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def main():
    """Main function to train and evaluate domain-aware sentiment model."""
    try:
        logger.info("Starting domain-aware sentiment analysis")
        
        # Load data
        df = load_data()
        
        # Assign domains if not already present
        if 'domains' not in df.columns:
            df = assign_domains(df)
        
        # Load or train domain classifier
        domain_vectorizer, domain_model, domain_binarizer = load_domain_classifier()
        if None in (domain_vectorizer, domain_model, domain_binarizer):
            domain_vectorizer, domain_model, domain_binarizer = train_domain_classifier(df)
        
        # Train domain-aware sentiment model
        sentiment_model = train_domain_aware_sentiment_model(df)
        
        # Split data for evaluation
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Evaluate model
        results = evaluate_model(
            test_df,
            domain_vectorizer,
            domain_model,
            domain_binarizer,
            sentiment_model
        )
        
        # Save results
        save_results(results)
        
        logger.info("Domain-aware sentiment analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in domain-aware sentiment analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 