#!/usr/bin/env python3
"""
Real-World Validation for Emergency Services Sentiment Analysis

This script implements a validation pipeline for testing the sentiment analysis
model on real-world emergency service reports and communications.
"""
import os
import sys
import re
import logging
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_world_validation')

# File paths
EMERGENCY_MODEL_PATH = "app/models/emergency_services_model.pkl"
EMERGENCY_VECTORIZER_PATH = "app/models/emergency_services_vectorizer.pkl"
DOMAIN_MODEL_PATH = "app/models/domain_classifier_model.pkl"
DOMAIN_VECTORIZER_PATH = "app/models/domain_classifier_vectorizer.pkl"
DOMAIN_BINARIZER_PATH = "app/models/domain_classifier_binarizer.pkl"
VALIDATION_RESULTS_PATH = "app/data/validation_results.json"
REAL_REPORTS_PATH = "real_world_examples.csv"

# Sample real-world reports (for demonstration)
SAMPLE_REPORTS = [
    # Fire service reports
    {
        "text": "Engine 3 responded to a structure fire at 123 Main St. Fire was contained to the kitchen area. No injuries reported.",
        "sentiment": 0,  # neutral
        "domain": "fire"
    },
    {
        "text": "Multiple units dispatched to a 3-alarm fire at the industrial park. Two firefighters experienced smoke inhalation and were transported to General Hospital.",
        "sentiment": -1,  # negative
        "domain": "fire"
    },
    {
        "text": "Crew successfully rescued two children from apartment fire on Oak Street. Both victims in stable condition.",
        "sentiment": 1,  # positive
        "domain": "fire"
    },
    
    # Police reports
    {
        "text": "Officers responded to a noise complaint at 456 Elm St. Residents were advised to lower music volume.",
        "sentiment": 0,  # neutral
        "domain": "police"
    },
    {
        "text": "Suspect apprehended after foot pursuit through downtown area. Officer Rodriguez sustained minor injuries during the arrest.",
        "sentiment": -1,  # negative
        "domain": "police"
    },
    {
        "text": "Community outreach program resulted in 25 new neighborhood watch groups. Crime statistics show 15% reduction in reported incidents.",
        "sentiment": 1,  # positive
        "domain": "police"
    },
    
    # EMS/Ambulance reports
    {
        "text": "Paramedics responded to cardiac arrest call at 789 Pine Ave. Patient transported to County Hospital.",
        "sentiment": 0,  # neutral
        "domain": "ems"
    },
    {
        "text": "Multiple casualties reported in highway pileup. Air ambulance unable to respond due to poor weather conditions.",
        "sentiment": -1,  # negative
        "domain": "ems"
    },
    {
        "text": "EMT Jenkins performed successful emergency delivery on scene. Mother and newborn in excellent condition upon arrival at hospital.",
        "sentiment": 1,  # positive
        "domain": "ems"
    },
    
    # Disaster response reports
    {
        "text": "Flood waters receding in northern district. Damage assessment teams deployed to affected areas.",
        "sentiment": 0,  # neutral
        "domain": "disaster_response"
    },
    {
        "text": "Evacuation orders in effect for coastal zones A and B. Hurricane expected to make landfall at 0200 hours.",
        "sentiment": -1,  # negative
        "domain": "disaster_response"
    },
    {
        "text": "All residents successfully evacuated from wildfire zone. No structures damaged thanks to firebreak established by forestry teams.",
        "sentiment": 1,  # positive
        "domain": "disaster_response"
    },
    
    # Coast guard reports
    {
        "text": "Coast Guard monitoring oil tanker approaching harbor. Standard security procedures in effect.",
        "sentiment": 0,  # neutral
        "domain": "coast_guard"
    },
    {
        "text": "Search for missing boaters enters third day. Weather conditions hampering rescue efforts.",
        "sentiment": -1,  # negative
        "domain": "coast_guard"
    },
    {
        "text": "Rescue team successfully retrieved all five crew members from capsized vessel. Excellent coordination between air and sea units.",
        "sentiment": 1,  # positive
        "domain": "coast_guard"
    }
]

def load_models():
    """
    Load sentiment and domain classification models.
    
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

def create_synthetic_dataset():
    """
    Create a synthetic dataset of real-world emergency service reports.
    For a real implementation, this would load actual reports from public sources.
    
    Returns:
        pandas.DataFrame: DataFrame of real-world reports
    """
    logger.info("Creating synthetic real-world dataset")
    
    # Convert sample reports to DataFrame
    df = pd.DataFrame(SAMPLE_REPORTS)
    
    # Save to CSV for future use
    df.to_csv(REAL_REPORTS_PATH, index=False)
    
    logger.info(f"Created synthetic dataset with {len(df)} samples")
    return df

def load_real_world_dataset():
    """
    Load the real-world dataset. Creates a synthetic one if not found.
    
    Returns:
        pandas.DataFrame: Dataset of real-world reports
    """
    if os.path.exists(REAL_REPORTS_PATH):
        logger.info(f"Loading real-world dataset from {REAL_REPORTS_PATH}")
        df = pd.read_csv(REAL_REPORTS_PATH)
        logger.info(f"Loaded {len(df)} real-world examples")
    else:
        logger.info(f"Real-world dataset not found. Creating synthetic dataset.")
        df = create_synthetic_dataset()
    
    return df

def predict_sentiment(text, vectorizer, model):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input text
        vectorizer: TF-IDF vectorizer
        model: Sentiment model
        
    Returns:
        int: Predicted sentiment (-1, 0, 1)
    """
    # Process input
    if not isinstance(text, str):
        text = str(text)
    
    # Vectorize input
    text_vectorized = vectorizer.transform([text])
    
    # Get prediction
    prediction = model.predict(text_vectorized)[0]
    
    return int(prediction)

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
    
    # Vectorize input
    text_vectorized = vectorizer.transform([text])
    
    # Predict binary domain labels
    domains_binary = model.predict(text_vectorized)[0]
    
    # Convert binary labels back to domain names
    domains = binarizer.classes_[np.where(domains_binary == 1)[0]].tolist()
    
    # Return general domain if no specific domain is predicted
    return domains if domains else ['general']

def validate_model(df, sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer):
    """
    Validate model performance on real-world examples.
    
    Args:
        df (pandas.DataFrame): Dataset of real-world reports
        sentiment_vectorizer, sentiment_model: Sentiment models
        domain_vectorizer, domain_model, domain_binarizer: Domain models
        
    Returns:
        dict: Validation results
    """
    logger.info("Validating model on real-world examples")
    
    # Make predictions
    df['predicted_sentiment'] = df['text'].apply(lambda x: predict_sentiment(x, sentiment_vectorizer, sentiment_model))
    df['predicted_domains'] = df['text'].apply(lambda x: predict_domains(x, domain_vectorizer, domain_model, domain_binarizer))
    
    # Calculate sentiment accuracy
    sentiment_accuracy = accuracy_score(df['sentiment'], df['predicted_sentiment'])
    logger.info(f"Overall sentiment accuracy: {sentiment_accuracy:.4f}")
    
    # Calculate domain accuracy (exact match)
    domain_exact_match = df.apply(lambda row: row['domain'] in row['predicted_domains'], axis=1).mean()
    logger.info(f"Domain exact match ratio: {domain_exact_match:.4f}")
    
    # Generate sentiment classification report
    sentiment_report = classification_report(
        df['sentiment'], 
        df['predicted_sentiment'],
        target_names=['negative', 'neutral', 'positive'],
        output_dict=True
    )
    
    logger.info(f"Sentiment classification report:\n{classification_report(df['sentiment'], df['predicted_sentiment'], target_names=['negative', 'neutral', 'positive'])}")
    
    # Generate confusion matrix
    cm = confusion_matrix(df['sentiment'], df['predicted_sentiment'])
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Analyze performance by domain
    domain_performance = {}
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        if len(domain_df) > 0:
            domain_accuracy = accuracy_score(domain_df['sentiment'], domain_df['predicted_sentiment'])
            domain_performance[domain] = {
                'accuracy': domain_accuracy,
                'count': len(domain_df),
                'report': classification_report(
                    domain_df['sentiment'], 
                    domain_df['predicted_sentiment'],
                    target_names=['negative', 'neutral', 'positive'],
                    output_dict=True
                )
            }
    
    # Log domain-specific performance
    for domain, metrics in domain_performance.items():
        logger.info(f"{domain.upper()} domain accuracy: {metrics['accuracy']:.4f} ({metrics['count']} samples)")
    
    # Prepare results dictionary
    results = {
        'overall': {
            'sentiment_accuracy': sentiment_accuracy,
            'domain_exact_match': domain_exact_match,
            'sentiment_report': sentiment_report,
            'confusion_matrix': cm.tolist()
        },
        'domain_performance': domain_performance,
        'examples': []
    }
    
    # Add example predictions for analysis
    for _, row in df.iterrows():
        results['examples'].append({
            'text': row['text'],
            'true_sentiment': int(row['sentiment']),
            'predicted_sentiment': int(row['predicted_sentiment']),
            'true_domain': row['domain'],
            'predicted_domains': row['predicted_domains'],
            'correct_sentiment': row['sentiment'] == row['predicted_sentiment'],
            'correct_domain': row['domain'] in row['predicted_domains']
        })
    
    return results

def save_validation_results(results):
    """
    Save validation results to file.
    
    Args:
        results (dict): Validation results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(VALIDATION_RESULTS_PATH), exist_ok=True)
    
    # Save to JSON file
    with open(VALIDATION_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {VALIDATION_RESULTS_PATH}")

def analyze_domain_adaptation_needs(results):
    """
    Analyze validation results to determine domain adaptation needs.
    
    Args:
        results (dict): Validation results
        
    Returns:
        dict: Domain adaptation recommendations
    """
    logger.info("Analyzing domain adaptation needs")
    
    recommendations = {
        'overall': [],
        'domains': {}
    }
    
    # Overall recommendations
    overall_accuracy = results['overall']['sentiment_accuracy']
    if overall_accuracy < 0.7:
        recommendations['overall'].append("Consider retraining the model with more real-world examples")
    
    sentiment_report = results['overall']['sentiment_report']
    for sentiment, metrics in sentiment_report.items():
        if sentiment in ['negative', 'neutral', 'positive']:
            if metrics['f1-score'] < 0.6:
                recommendations['overall'].append(f"Improve {sentiment} sentiment detection (F1: {metrics['f1-score']:.2f})")
    
    # Domain-specific recommendations
    for domain, metrics in results['domain_performance'].items():
        domain_recommendations = []
        
        if metrics['accuracy'] < 0.6:
            domain_recommendations.append(f"Overall accuracy is low ({metrics['accuracy']:.2f})")
        
        report = metrics['report']
        for sentiment, sentiment_metrics in report.items():
            if sentiment in ['negative', 'neutral', 'positive'] and sentiment_metrics['f1-score'] < 0.5:
                domain_recommendations.append(f"Poor {sentiment} sentiment detection (F1: {sentiment_metrics['f1-score']:.2f})")
        
        if domain_recommendations:
            recommendations['domains'][domain] = domain_recommendations
    
    # Log recommendations
    logger.info("Domain adaptation recommendations:")
    for rec in recommendations['overall']:
        logger.info(f"* {rec}")
    
    for domain, domain_recs in recommendations['domains'].items():
        logger.info(f"{domain.upper()} domain:")
        for rec in domain_recs:
            logger.info(f"* {rec}")
    
    return recommendations

def plot_validation_results(results):
    """
    Plot validation results.
    
    Args:
        results (dict): Validation results
    """
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(results['overall']['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("plots/confusion_matrix.png")
    plt.close()
    
    # Plot domain-specific accuracy
    domain_accuracy = {domain: metrics['accuracy'] for domain, metrics in results['domain_performance'].items()}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(domain_accuracy.keys(), domain_accuracy.values())
    
    # Add overall accuracy line
    plt.axhline(y=results['overall']['sentiment_accuracy'], color='r', linestyle='-', label='Overall Accuracy')
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.xlabel('Domain')
    plt.ylabel('Accuracy')
    plt.title('Sentiment Accuracy by Domain')
    plt.legend()
    plt.savefig("plots/domain_accuracy.png")
    plt.close()
    
    # Plot F1 scores by domain and sentiment
    domains = list(results['domain_performance'].keys())
    sentiments = ['negative', 'neutral', 'positive']
    
    f1_scores = {}
    for sentiment in sentiments:
        f1_scores[sentiment] = [results['domain_performance'][domain]['report'][sentiment]['f1-score'] 
                               for domain in domains]
    
    x = np.arange(len(domains))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    
    # Plot bars for each sentiment
    bars1 = plt.bar(x - width, f1_scores['negative'], width, label='Negative')
    bars2 = plt.bar(x, f1_scores['neutral'], width, label='Neutral')
    bars3 = plt.bar(x + width, f1_scores['positive'], width, label='Positive')
    
    plt.xlabel('Domain')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Domain and Sentiment')
    plt.xticks(x, domains)
    plt.ylim(0, 1.1)
    plt.legend()
    
    plt.savefig("plots/f1_scores.png")
    plt.close()
    
    logger.info("Validation result plots saved to 'plots' directory")

def main():
    """Main function to run real-world validation."""
    try:
        logger.info("Starting real-world validation")
        
        # Load models
        sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer = load_models()
        
        # Load or create real-world dataset
        real_world_df = load_real_world_dataset()
        
        # Validate model
        validation_results = validate_model(
            real_world_df, 
            sentiment_vectorizer, 
            sentiment_model,
            domain_vectorizer,
            domain_model,
            domain_binarizer
        )
        
        # Save validation results
        save_validation_results(validation_results)
        
        # Analyze domain adaptation needs
        adaptation_recommendations = analyze_domain_adaptation_needs(validation_results)
        
        # Plot validation results
        plot_validation_results(validation_results)
        
        logger.info("Real-world validation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in real-world validation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 