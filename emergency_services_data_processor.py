#!/usr/bin/env python3
"""
Emergency Services Data Processor

This script processes the large Twitter dataset for sentiment analysis
expansion to all emergency services. It converts labels and prepares
the data for model training.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('emergency_services_data_processor')

# Path to the large Twitter dataset
TWITTER_DATA_PATH = "training_1600000_processed_noemoticon.csv"
PROCESSED_DATA_PATH = "emergency_services_dataset.csv"

def map_sentiment_labels(label):
    """
    Map original Twitter sentiment labels to our format:
    0 (negative) -> -1
    2 (neutral) -> 0
    4 (positive) -> 1
    
    Args:
        label: Original sentiment label
        
    Returns:
        int: Mapped sentiment value (-1, 0, or 1)
    """
    mapping = {
        0: -1,  # negative
        2: 0,   # neutral
        4: 1    # positive
    }
    return mapping.get(label, 0)  # Default to neutral if unknown

def load_and_process_dataset(csv_path=TWITTER_DATA_PATH, sample_size=None, emergency_filter=False):
    """
    Load the Twitter dataset and process it for sentiment analysis.
    
    Args:
        csv_path (str): Path to the CSV file
        sample_size (int): Number of samples per class to use (for balanced dataset)
        emergency_filter (bool): Whether to filter for emergency services content
        
    Returns:
        pandas.DataFrame: Processed dataset
    """
    logger.info(f"Loading Twitter dataset from {csv_path}")
    
    try:
        # Read CSV file - specify column names as they might not be in the file
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        # Check file size and use appropriate chunking if needed
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # Size in MB
        
        if file_size > 500:  # If file is larger than 500MB
            logger.info(f"Large file detected ({file_size:.2f} MB). Using chunked reading...")
            
            # Process in chunks
            chunk_size = 100000  # Adjust based on available memory
            chunk_list = []
            
            # Read and process chunks
            for chunk in pd.read_csv(csv_path, names=column_names, chunksize=chunk_size, encoding='latin-1'):
                # Process chunk
                chunk['sentiment'] = chunk['target'].apply(map_sentiment_labels)
                
                # If filtering for emergency content
                if emergency_filter:
                    # Add emergency filtering logic here
                    pass
                
                chunk_list.append(chunk)
                logger.info(f"Processed chunk with {len(chunk)} rows")
            
            # Combine all chunks
            df = pd.concat(chunk_list, ignore_index=True)
            logger.info(f"Combined all chunks into a dataframe with {len(df)} rows")
            
        else:
            # For smaller files, read directly
            df = pd.read_csv(csv_path, names=column_names, encoding='latin-1')
            logger.info(f"Loaded {len(df)} rows from CSV file")
            
            # Map sentiment labels
            df['sentiment'] = df['target'].apply(map_sentiment_labels)
            
            # Filter for emergency content if requested
            if emergency_filter:
                # Add emergency filtering logic here
                pass
        
        # Log original class distribution
        logger.info(f"Original target values distribution: {df['target'].value_counts().to_dict()}")
        
        # Display column information
        logger.info(f"Columns in the dataset: {df.columns.tolist()}")
        logger.info(f"Number of samples per class: Negative={len(df[df['sentiment'] == -1])}, "
                   f"Neutral={len(df[df['sentiment'] == 0])}, "
                   f"Positive={len(df[df['sentiment'] == 1])}")
        
        # Create synthetic neutral examples if needed
        if len(df[df['sentiment'] == 0]) < 100:  # Very few or no neutral examples
            logger.warning(f"Only {len(df[df['sentiment'] == 0])} neutral examples found. Creating synthetic neutral class.")
            
            # Method 1: Mix positive and negative samples to create neutral
            negative_samples = df[df['sentiment'] == -1].sample(min(10000, len(df[df['sentiment'] == -1])), random_state=42)
            positive_samples = df[df['sentiment'] == 1].sample(min(10000, len(df[df['sentiment'] == 1])), random_state=42)
            
            # Create synthetic neutral by taking parts of sentences from each
            neutral_samples = []
            for i in range(min(len(negative_samples), len(positive_samples))):
                neg_text = negative_samples.iloc[i]['text']
                pos_text = positive_samples.iloc[i]['text']
                
                # Split and combine texts (simple approach)
                neg_words = neg_text.split()[:len(neg_text.split())//2]
                pos_words = pos_text.split()[len(pos_text.split())//2:]
                
                # Combine to create a neutral-like text
                neutral_text = " ".join(neg_words + pos_words)
                
                neutral_samples.append({
                    'target': 2,  # Original neutral value
                    'id': f"synthetic_{i}",
                    'date': negative_samples.iloc[i]['date'],
                    'flag': negative_samples.iloc[i]['flag'],
                    'user': 'synthetic_user',
                    'text': neutral_text,
                    'sentiment': 0  # Our mapped neutral value
                })
            
            # Create DataFrame from synthetic neutrals
            neutral_df = pd.DataFrame(neutral_samples)
            logger.info(f"Created {len(neutral_df)} synthetic neutral examples")
            
            # Combine with original data
            df = pd.concat([df, neutral_df], ignore_index=True)
            
            # Log updated distribution
            logger.info(f"Updated number of samples per class: Negative={len(df[df['sentiment'] == -1])}, "
                       f"Neutral={len(df[df['sentiment'] == 0])}, "
                       f"Positive={len(df[df['sentiment'] == 1])}")
        
        # Create balanced dataset if sample_size is specified
        if sample_size is not None:
            logger.info(f"Creating balanced dataset with {sample_size} samples per class")
            df_balanced = create_balanced_dataset(df, sample_size)
            return df_balanced
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Twitter data: {str(e)}")
        raise

def create_balanced_dataset(df, sample_size):
    """
    Create a balanced dataset with equal samples per sentiment class.
    
    Args:
        df (pandas.DataFrame): Original dataset
        sample_size (int): Number of samples per class
        
    Returns:
        pandas.DataFrame: Balanced dataset
    """
    negative_samples = df[df['sentiment'] == -1]
    neutral_samples = df[df['sentiment'] == 0]
    positive_samples = df[df['sentiment'] == 1]
    
    logger.info(f"Original dataset distribution: Negative={len(negative_samples)}, "
               f"Neutral={len(neutral_samples)}, Positive={len(positive_samples)}")
    
    # Check if we have enough samples for each class
    if len(negative_samples) < sample_size or len(neutral_samples) < sample_size or len(positive_samples) < sample_size:
        logger.warning(f"Not enough samples for balanced dataset. Using maximum available: "
                      f"Negative={min(sample_size, len(negative_samples))}, "
                      f"Neutral={min(sample_size, len(neutral_samples))}, "
                      f"Positive={min(sample_size, len(positive_samples))}")
        sample_size_neg = min(sample_size, len(negative_samples))
        sample_size_neu = min(sample_size, len(neutral_samples))
        sample_size_pos = min(sample_size, len(positive_samples))
    else:
        sample_size_neg = sample_size_neu = sample_size_pos = sample_size
    
    # Sample with replacement if needed
    if len(negative_samples) < sample_size_neg:
        negative_samples = resample(negative_samples, replace=True, n_samples=sample_size_neg, random_state=42)
    else:
        negative_samples = negative_samples.sample(sample_size_neg, random_state=42)
        
    if len(neutral_samples) < sample_size_neu:
        neutral_samples = resample(neutral_samples, replace=True, n_samples=sample_size_neu, random_state=42)
    else:
        neutral_samples = neutral_samples.sample(sample_size_neu, random_state=42)
        
    if len(positive_samples) < sample_size_pos:
        positive_samples = resample(positive_samples, replace=True, n_samples=sample_size_pos, random_state=42)
    else:
        positive_samples = positive_samples.sample(sample_size_pos, random_state=42)
    
    # Combine the balanced samples
    df_balanced = pd.concat([negative_samples, neutral_samples, positive_samples])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created balanced dataset with {len(df_balanced)} samples: "
               f"Negative={len(df_balanced[df_balanced['sentiment'] == -1])}, "
               f"Neutral={len(df_balanced[df_balanced['sentiment'] == 0])}, "
               f"Positive={len(df_balanced[df_balanced['sentiment'] == 1])}")
    
    return df_balanced

def filter_emergency_services_content(df):
    """
    Filter dataset for emergency services related content.
    
    Args:
        df (pandas.DataFrame): Original dataset
        
    Returns:
        pandas.DataFrame: Filtered dataset
    """
    logger.info("Filtering for emergency services related content")
    
    # Keywords for different emergency services
    emergency_keywords = [
        # Fire services
        'fire', 'firefighter', 'blaze', 'burn', 'flame', 'smoke', 'arson',
        'extinguish', 'hydrant', 'ladder', 'hose', 'firestation', 'firetruck',
        
        # Police services
        'police', 'cop', 'officer', 'patrol', 'sheriff', 'deputy', 'detective',
        'arrest', 'crime', 'criminal', 'law enforcement', 'precinct', 'badge',
        
        # EMS/Ambulance
        'ambulance', 'paramedic', 'emt', 'medic', 'emergency medical',
        'stretcher', 'cpr', 'defibrillator', 'hospital', 'injury', 'wounded',
        
        # Coast Guard / Water Rescue
        'coast guard', 'coastguard', 'lifeguard', 'rescue boat', 'drowning',
        'ocean rescue', 'beach patrol', 'maritime', 'water rescue',
        
        # Disaster Response
        'disaster', 'emergency', 'rescue', 'evacuation', 'evacuate', 'flood',
        'hurricane', 'tornado', 'earthquake', 'wildfire', 'landslide',
        
        # General emergency terms
        '911', 'emergency', 'first responder', 'rescue', 'dispatch', 'crisis',
        'emergency response', 'public safety', 'victims', 'survivors'
    ]
    
    # Create regex pattern for case-insensitive matching
    pattern = '|'.join(emergency_keywords)
    
    # Filter dataframe for rows containing emergency-related keywords
    filtered_df = df[df['text'].str.contains(pattern, case=False, regex=True)]
    
    logger.info(f"Original dataset: {len(df)} rows")
    logger.info(f"Filtered dataset: {len(filtered_df)} rows")
    logger.info(f"Retention rate: {len(filtered_df)/len(df)*100:.2f}%")
    
    # If too few samples remain, reduce filtering strictness
    if len(filtered_df) < 1000:
        logger.warning("Too few samples after filtering. Using less strict criteria.")
        # Implement less strict filtering logic here
        
    return filtered_df

def prepare_train_test_split(df, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pandas.DataFrame): Processed dataset
        test_size (float): Proportion of dataset to include in test split
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting dataset into training ({1-test_size:.0%}) and testing ({test_size:.0%}) sets")
    
    X = df['text'].values
    y = df['sentiment'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(df, output_path=PROCESSED_DATA_PATH):
    """
    Save the processed dataset to CSV.
    
    Args:
        df (pandas.DataFrame): Processed dataset
        output_path (str): Path to save the CSV file
    """
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

def main():
    """Main function to process the Twitter dataset."""
    try:
        logger.info("Starting emergency services data processing")
        
        # Step 1: Load and process the dataset
        df = load_and_process_dataset()
        
        # Step 2: Save an unfiltered, unbalanced version
        save_processed_data(df, "emergency_services_dataset_full.csv")
        
        # Step 3: Filter for emergency services content
        df_emergency = filter_emergency_services_content(df)
        save_processed_data(df_emergency, "emergency_services_dataset_filtered.csv")
        
        # Step 4: Create balanced dataset with 50,000 samples per class
        df_balanced = create_balanced_dataset(df, sample_size=50000)
        save_processed_data(df_balanced, "emergency_services_dataset_balanced.csv")
        
        # Step 5: Create emergency-filtered balanced dataset
        df_emergency_balanced = create_balanced_dataset(df_emergency, sample_size=10000)
        save_processed_data(df_emergency_balanced, "emergency_services_dataset_emergency_balanced.csv")
        
        logger.info("Emergency services data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 