"""
Text Preprocessing Utilities for Sentiment Analysis

This module contains functions for preprocessing text data
before sentiment analysis.
"""
import re
import string
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('text_preprocessing')

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into string
    processed_text = ' '.join(processed_tokens)
    
    return processed_text

def extract_features(text_list):
    """
    Preprocess a list of text samples.
    
    Args:
        text_list (list): List of text strings
        
    Returns:
        list: List of preprocessed text strings
    """
    return [preprocess_text(text) for text in text_list]

def map_sentiment_label(label):
    """
    Map sentiment label from string to integer.
    
    Args:
        label (str): Sentiment label ('positive', 'neutral', 'negative')
        
    Returns:
        int: Sentiment value (1, 0, -1)
    """
    sentiment_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    return sentiment_map.get(label.lower(), 0)  # Default to neutral

def map_sentiment_value(value):
    """
    Map sentiment value from integer to string.
    
    Args:
        value (int): Sentiment value (1, 0, -1)
        
    Returns:
        str: Sentiment label ('positive', 'neutral', 'negative')
    """
    sentiment_map = {
        1: 'positive',
        0: 'neutral',
        -1: 'negative'
    }
    return sentiment_map.get(value, 'neutral')  # Default to neutral 