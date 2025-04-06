#!/usr/bin/env python3
"""
Sentiment Model Evaluation Script

This script tests all available sentiment models using the provided emergency services
test data and generates a comparative performance report.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

# Import the prediction functions for each model
try:
    from app.models.predict import predict_sentiment as predict_default
except ImportError:
    logger.warning("Default sentiment model not available")
    predict_default = None

try:
    from hybrid_sentiment_model import predict_hybrid_sentiment
except ImportError:
    logger.warning("Hybrid sentiment model not available")
    predict_hybrid_sentiment = None

try:
    from domain_aware_sentiment import predict_domain_aware_sentiment
except ImportError:
    logger.warning("Domain-aware sentiment model not available")
    predict_domain_aware_sentiment = None

# Constants
TEST_DATA_PATH = "emergency_sentiment_test_data.csv"
REPORT_PATH = "model_evaluation_report.md"

# Map sentiment labels for evaluation
SENTIMENT_MAP = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

def load_test_data(file_path):
    """Load test data from CSV file."""
    logger.info(f"Loading test data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df['sentiment_value'] = df['Sentiment'].map(SENTIMENT_MAP)
        logger.info(f"Loaded {len(df)} test samples")
        return df
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def evaluate_default_model(test_data):
    """Evaluate the default sentiment model."""
    if predict_default is None:
        logger.warning("Skipping default model evaluation (model not available)")
        return None
    
    logger.info("Evaluating default sentiment model...")
    results = []
    predictions = []
    
    start_time = time.time()
    for idx, row in test_data.iterrows():
        text = row['Text']
        try:
            result = predict_default(text)
            results.append(result)
            predictions.append(result['sentiment_value'])
        except Exception as e:
            logger.error(f"Error predicting with default model: {str(e)}")
            predictions.append(0)  # Default to neutral for errors
    
    execution_time = time.time() - start_time
    
    y_true = test_data['sentiment_value'].values
    
    metrics = calculate_metrics(y_true, predictions, "Default Model", execution_time, len(test_data))
    return metrics

def evaluate_hybrid_model(test_data):
    """Evaluate the hybrid sentiment model."""
    if predict_hybrid_sentiment is None:
        logger.warning("Skipping hybrid model evaluation (model not available)")
        return None
    
    logger.info("Evaluating hybrid sentiment model...")
    results = []
    predictions = []
    
    start_time = time.time()
    for idx, row in test_data.iterrows():
        text = row['Text']
        try:
            result = predict_hybrid_sentiment(text)
            results.append(result)
            predictions.append(result['sentiment_value'])
        except Exception as e:
            logger.error(f"Error predicting with hybrid model: {str(e)}")
            predictions.append(0)  # Default to neutral for errors
    
    execution_time = time.time() - start_time
    
    y_true = test_data['sentiment_value'].values
    
    metrics = calculate_metrics(y_true, predictions, "Hybrid Model", execution_time, len(test_data))
    return metrics

def evaluate_domain_aware_model(test_data):
    """Evaluate the domain-aware sentiment model."""
    if predict_domain_aware_sentiment is None:
        logger.warning("Skipping domain-aware model evaluation (model not available)")
        return None
    
    logger.info("Evaluating domain-aware sentiment model...")
    results = []
    predictions = []
    domains_detected = []
    
    start_time = time.time()
    for idx, row in test_data.iterrows():
        text = row['Text']
        try:
            result = predict_domain_aware_sentiment(text)
            results.append(result)
            predictions.append(result['sentiment_value'])
            domains_detected.append(result.get('domains', ['general']))
        except Exception as e:
            logger.error(f"Error predicting with domain-aware model: {str(e)}")
            predictions.append(0)  # Default to neutral for errors
            domains_detected.append(['error'])
    
    execution_time = time.time() - start_time
    
    y_true = test_data['sentiment_value'].values
    
    metrics = calculate_metrics(y_true, predictions, "Domain-Aware Model", execution_time, len(test_data))
    
    # Add domain detection information
    all_domains = [domain for sublist in domains_detected for domain in sublist]
    domain_counts = pd.Series(all_domains).value_counts().to_dict()
    metrics['domain_distribution'] = domain_counts
    
    return metrics

def calculate_metrics(y_true, y_pred, model_name, execution_time, total_samples):
    """Calculate performance metrics for a model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Calculate per-class metrics
    per_class_metrics = classification_report(y_true, y_pred, 
                                             target_names=["Negative", "Neutral", "Positive"],
                                             output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    
    # Create results dictionary
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "execution_time": execution_time,
        "samples_per_second": total_samples / execution_time,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {execution_time:.2f}s")
    return metrics

def generate_report(metrics_list, test_data):
    """Generate a markdown report comparing model performance."""
    logger.info(f"Generating report at {REPORT_PATH}")
    
    # Filter out None values (skipped models)
    metrics_list = [m for m in metrics_list if m is not None]
    
    with open(REPORT_PATH, 'w') as f:
        f.write("# Emergency Services Sentiment Analysis Model Evaluation\n\n")
        f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Test Data Overview\n\n")
        f.write(f"- **Total samples**: {len(test_data)}\n")
        f.write(f"- **Class distribution**:\n")
        
        sentiment_counts = test_data['Sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            f.write(f"  - {sentiment}: {count} samples ({count/len(test_data)*100:.1f}%)\n")
        
        f.write("\n## Performance Summary\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | Execution Time | Samples/sec |\n")
        f.write("|-------|----------|-----------|--------|----------|----------------|--------------|\n")
        
        for metrics in metrics_list:
            f.write(f"| {metrics['model_name']} | ")
            f.write(f"{metrics['accuracy']:.4f} | ")
            f.write(f"{metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | ")
            f.write(f"{metrics['f1_score']:.4f} | ")
            f.write(f"{metrics['execution_time']:.2f}s | ")
            f.write(f"{metrics['samples_per_second']:.2f} |\n")
        
        # Detailed per-model analysis
        f.write("\n## Detailed Model Analysis\n\n")
        
        for metrics in metrics_list:
            f.write(f"### {metrics['model_name']}\n\n")
            
            # Per-class metrics
            f.write("#### Per-Class Performance\n\n")
            f.write("| Class | Precision | Recall | F1 Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            
            for cls in ["Negative", "Neutral", "Positive"]:
                cls_metrics = metrics['per_class_metrics'][cls]
                f.write(f"| {cls} | ")
                f.write(f"{cls_metrics['precision']:.4f} | ")
                f.write(f"{cls_metrics['recall']:.4f} | ")
                f.write(f"{cls_metrics['f1-score']:.4f} | ")
                f.write(f"{int(cls_metrics['support'])} |\n")
            
            # Confusion Matrix
            f.write("\n#### Confusion Matrix\n\n")
            f.write("```\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write(f"              Predicted\n")
            f.write(f"              Negative  Neutral  Positive\n")
            f.write(f"Actual Negative {cm[0][0]:8d} {cm[0][1]:8d} {cm[0][2]:8d}\n")
            f.write(f"       Neutral  {cm[1][0]:8d} {cm[1][1]:8d} {cm[1][2]:8d}\n")
            f.write(f"       Positive {cm[2][0]:8d} {cm[2][1]:8d} {cm[2][2]:8d}\n")
            f.write("```\n\n")
            
            # Domain distribution for domain-aware model
            if 'domain_distribution' in metrics:
                f.write("#### Domain Distribution\n\n")
                f.write("| Domain | Count | Percentage |\n")
                f.write("|--------|-------|------------|\n")
                
                total_domains = sum(metrics['domain_distribution'].values())
                for domain, count in metrics['domain_distribution'].items():
                    f.write(f"| {domain} | {count} | {count/total_domains*100:.1f}% |\n")
                
                f.write("\n")
        
        # Error Analysis
        f.write("## Error Analysis\n\n")
        f.write("This section highlights examples where models disagreed with the ground truth.\n\n")
        
        # Find examples where any model made an incorrect prediction
        error_examples = []
        
        for idx, row in test_data.iterrows():
            text = row['Text']
            true_sentiment = row['sentiment_value']
            
            model_predictions = {}
            for metrics in metrics_list:
                model_name = metrics['model_name']
                
                # We need to re-run predictions to get individual results
                try:
                    if model_name == "Default Model" and predict_default:
                        result = predict_default(text)
                        pred_sentiment = result['sentiment_value']
                    elif model_name == "Hybrid Model" and predict_hybrid_sentiment:
                        result = predict_hybrid_sentiment(text)
                        pred_sentiment = result['sentiment_value']
                    elif model_name == "Domain-Aware Model" and predict_domain_aware_sentiment:
                        result = predict_domain_aware_sentiment(text)
                        pred_sentiment = result['sentiment_value']
                    else:
                        continue
                        
                    model_predictions[model_name] = {
                        'prediction': pred_sentiment,
                        'confidence': result.get('confidence', 0)
                    }
                except Exception:
                    continue
            
            # Check if any model made a wrong prediction
            any_error = any(pred['prediction'] != true_sentiment for pred in model_predictions.values())
            if any_error and len(model_predictions) > 0:
                error_examples.append({
                    'text': text,
                    'true_sentiment': true_sentiment,
                    'rationale': row['Rationale'],
                    'predictions': model_predictions
                })
        
        # Limit to first 10 examples to keep the report concise
        error_examples = error_examples[:10]
        
        for i, example in enumerate(error_examples, 1):
            true_sentiment_label = {1: "Positive", 0: "Neutral", -1: "Negative"}[example['true_sentiment']]
            
            f.write(f"### Example {i}\n\n")
            f.write(f"**Text**: {example['text']}\n\n")
            f.write(f"**Ground Truth**: {true_sentiment_label} (Rationale: {example['rationale']})\n\n")
            f.write("**Model Predictions**:\n\n")
            
            for model_name, pred in example['predictions'].items():
                pred_sentiment = {1: "Positive", 0: "Neutral", -1: "Negative"}[pred['prediction']]
                correct = pred['prediction'] == example['true_sentiment']
                f.write(f"- {model_name}: {pred_sentiment} (Confidence: {pred['confidence']:.2f}) - {'✓' if correct else '✗'}\n")
            
            f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Determine best model based on F1 score
        if metrics_list:
            best_model = max(metrics_list, key=lambda x: x['f1_score'])
            f.write(f"Based on the evaluation, the **{best_model['model_name']}** demonstrates the best overall performance ")
            f.write(f"with an F1 score of {best_model['f1_score']:.4f} and accuracy of {best_model['accuracy']:.4f}.\n\n")
            
            strengths = []
            areas_for_improvement = []
            
            # Analyze per-class performance of the best model
            best_pcm = best_model['per_class_metrics']
            best_class = max(["Negative", "Neutral", "Positive"], key=lambda c: best_pcm[c]['f1-score'])
            worst_class = min(["Negative", "Neutral", "Positive"], key=lambda c: best_pcm[c]['f1-score'])
            
            strengths.append(f"Strong performance on {best_class.lower()} sentiment detection (F1: {best_pcm[best_class]['f1-score']:.4f})")
            areas_for_improvement.append(f"Further improvements could be made for {worst_class.lower()} sentiment detection (F1: {best_pcm[worst_class]['f1-score']:.4f})")
            
            # Domain-aware model specific insights
            if 'domain_distribution' in best_model:
                dom_dist = best_model['domain_distribution']
                most_common_domain = max(dom_dist.items(), key=lambda x: x[1])[0]
                if most_common_domain != 'general':
                    strengths.append(f"Effective domain identification with {most_common_domain} being the most detected domain")
            
            # Performance speed
            if best_model['samples_per_second'] > 5:
                strengths.append(f"Efficient processing speed ({best_model['samples_per_second']:.2f} samples/sec)")
            else:
                areas_for_improvement.append("Processing speed could be optimized for real-time applications")
            
            f.write("### Key Strengths\n\n")
            for strength in strengths:
                f.write(f"- {strength}\n")
            
            f.write("\n### Areas for Improvement\n\n")
            for area in areas_for_improvement:
                f.write(f"- {area}\n")
            
            f.write("\n### Recommendations\n\n")
            f.write("1. Continue to enhance training data for the underperforming sentiment classes\n")
            f.write("2. Consider ensemble approaches that leverage the strengths of multiple models\n")
            f.write("3. Focus on improving domain detection for more accurate context-aware sentiment analysis\n")
            
        else:
            f.write("No valid models were evaluated in this test run.\n")
    
    logger.info(f"Report generated successfully at {REPORT_PATH}")

def main():
    """Main function to run the model evaluation."""
    try:
        # Load test data
        test_data = load_test_data(TEST_DATA_PATH)
        
        # Evaluate each model
        default_metrics = evaluate_default_model(test_data)
        hybrid_metrics = evaluate_hybrid_model(test_data)
        domain_aware_metrics = evaluate_domain_aware_model(test_data)
        
        # Collect all metrics
        all_metrics = [
            default_metrics,
            hybrid_metrics,
            domain_aware_metrics
        ]
        
        # Generate report
        generate_report(all_metrics, test_data)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 