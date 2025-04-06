#!/usr/bin/env python3
"""
Transfer Learning with Transformers for Emergency Services Sentiment Analysis

This script implements transfer learning using pre-trained transformer models
(BERT/RoBERTa) to improve sentiment classification for emergency services tweets.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW, 
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transfer_learning')

# File paths
DATA_DIR = "app/data"
MODELS_DIR = "app/models"
EMERGENCY_TWEETS_PATH = os.path.join(DATA_DIR, "emergency_tweets_processed.csv")
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "transformer_model")
RESULTS_PATH = os.path.join(DATA_DIR, "transformer_results.json")

# Model parameters
MODEL_NAME = "distilbert-base-uncased"  # Smaller BERT model for faster training
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class EmergencyTweetsDataset(Dataset):
    """Dataset for emergency tweets."""
    
    def __init__(self, texts, sentiments, tokenizer, max_length=128):
        """
        Initialize dataset.
        
        Args:
            texts (list): List of tweet texts
            sentiments (list): List of sentiment labels
            tokenizer: Tokenizer for text encoding
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map sentiment values to proper labels for 3-class classification
        # Original values: -1 (negative), 0 (neutral), 1 (positive)
        # Map to: 0 (negative), 1 (neutral), 2 (positive)
        self.sentiment_map = {-1: 0, 0: 1, 1: 2}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = self.sentiments[idx]
        
        # Map sentiment to proper label
        sentiment = self.sentiment_map[sentiment]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
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
        
        # Keep only text and sentiment columns
        df = df[['text', 'sentiment']]
        
        # Drop rows with missing text or sentiment
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Convert sentiment to int if needed
        df['sentiment'] = df['sentiment'].astype(int)
        
        logger.info(f"Loaded {len(df)} tweets")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_dataloaders(df, tokenizer, max_length=128, batch_size=16, test_size=0.2):
    """
    Prepare DataLoaders for training and evaluation.
    
    Args:
        df (pandas.DataFrame): Dataset with text and sentiment columns
        tokenizer: Tokenizer for text encoding
        max_length (int): Maximum sequence length
        batch_size (int): Batch size
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    logger.info("Preparing DataLoaders")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentiment'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['sentiment'])
    
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = EmergencyTweetsDataset(
        train_df['text'].values,
        train_df['sentiment'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = EmergencyTweetsDataset(
        val_df['text'].values,
        val_df['sentiment'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = EmergencyTweetsDataset(
        test_df['text'].values,
        test_df['sentiment'].values,
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader, test_df

def create_model(model_name, num_labels=3):
    """
    Create transformer model for sequence classification.
    
    Args:
        model_name (str): Name of pre-trained model
        num_labels (int): Number of output classes
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Creating model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    return model, tokenizer

def train_model(model, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5, output_dir=MODEL_OUTPUT_DIR):
    """
    Train the transformer model.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        output_dir (str): Directory to save model
        
    Returns:
        Trained model
    """
    logger.info("Training model")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    
    return model

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object
        
    Returns:
        dict: Evaluation metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy
    }

def evaluate_model(model, test_dataloader, test_df):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_dataloader: DataLoader for testing
        test_df: Original test dataframe
        
    Returns:
        dict: Evaluation results
    """
    logger.info("Evaluating model")
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Get predictions
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Map predictions back to original sentiment values
    sentiment_map_reverse = {0: -1, 1: 0, 2: 1}
    all_predictions_mapped = [sentiment_map_reverse[pred] for pred in all_predictions]
    all_labels_mapped = [sentiment_map_reverse[label] for label in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels, 
        all_predictions,
        target_names=['negative', 'neutral', 'positive'],
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nClassification report:\n" + 
               classification_report(all_labels, all_predictions, target_names=['negative', 'neutral', 'positive']))
    
    # Sample predictions
    sample_texts = test_df['text'].values[:10]
    sample_labels = test_df['sentiment'].values[:10]
    sample_predictions = all_predictions_mapped[:10]
    
    sample_results = []
    for text, label, pred in zip(sample_texts, sample_labels, sample_predictions):
        sample_results.append({
            'text': text,
            'true_sentiment': int(label),
            'predicted_sentiment': int(pred),
            'correct': label == pred
        })
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'sample_predictions': sample_results
    }
    
    return results

def compare_with_baseline(transformer_results, baseline_accuracy=None):
    """
    Compare transformer model results with baseline model.
    
    Args:
        transformer_results (dict): Results from transformer model
        baseline_accuracy (float): Accuracy of baseline model
        
    Returns:
        dict: Comparison results
    """
    if baseline_accuracy is None:
        # If no baseline accuracy provided, try to load from file
        try:
            baseline_path = os.path.join(DATA_DIR, "baseline_results.json")
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            baseline_accuracy = baseline_data.get('overall_accuracy', 0.7)  # Default to 0.7 if not found
        except:
            # Use a default baseline accuracy if file not found
            baseline_accuracy = 0.7
            logger.warning(f"Baseline results not found. Using default accuracy: {baseline_accuracy}")
    
    transformer_accuracy = transformer_results['accuracy']
    improvement = transformer_accuracy - baseline_accuracy
    percentage_improvement = (improvement / baseline_accuracy) * 100
    
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"Transformer accuracy: {transformer_accuracy:.4f}")
    logger.info(f"Improvement: {improvement:.4f} ({percentage_improvement:.2f}%)")
    
    comparison = {
        'baseline_accuracy': baseline_accuracy,
        'transformer_accuracy': transformer_accuracy,
        'absolute_improvement': improvement,
        'percentage_improvement': percentage_improvement
    }
    
    return comparison

def save_results(results, comparison, filename=RESULTS_PATH):
    """
    Save evaluation results to file.
    
    Args:
        results (dict): Evaluation results
        comparison (dict): Comparison with baseline
        filename (str): Output filename
    """
    output = {
        'results': results,
        'comparison': comparison
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {filename}")

def plot_results(results, output_dir="plots"):
    """
    Plot evaluation results.
    
    Args:
        results (dict): Evaluation results
        output_dir (str): Directory to save plots
    """
    # Create directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Transformer Model')
    plt.savefig(os.path.join(output_dir, "transformer_confusion_matrix.png"))
    plt.close()
    
    # Plot class-wise metrics
    report = results['classification_report']
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['negative', 'neutral', 'positive']
    
    class_metrics = {}
    for metric in metrics:
        class_metrics[metric] = [report[cls][metric] for cls in classes]
    
    # Plot bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        bars = plt.bar(classes, class_metrics[metric])
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.xlabel('Sentiment Class')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} by Sentiment Class - Transformer Model')
        plt.savefig(os.path.join(output_dir, f"transformer_{metric}.png"))
        plt.close()
    
    logger.info(f"Plots saved to {output_dir}")

def main():
    """Main function to train and evaluate transfer learning model."""
    try:
        logger.info("Starting transfer learning with transformers")
        
        # Load data
        df = load_data()
        
        # Create model and tokenizer
        model, tokenizer = create_model(MODEL_NAME)
        
        # Prepare dataloaders
        train_dataloader, val_dataloader, test_dataloader, test_df = prepare_dataloaders(
            df, tokenizer, MAX_LENGTH, BATCH_SIZE
        )
        
        # Train model
        model = train_model(
            model, 
            train_dataloader, 
            val_dataloader, 
            EPOCHS, 
            LEARNING_RATE, 
            MODEL_OUTPUT_DIR
        )
        
        # Evaluate model
        results = evaluate_model(model, test_dataloader, test_df)
        
        # Compare with baseline
        comparison = compare_with_baseline(results)
        
        # Save results
        save_results(results, comparison)
        
        # Plot results
        plot_results(results)
        
        logger.info("Transfer learning completed successfully")
        
    except Exception as e:
        logger.error(f"Error in transfer learning: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 