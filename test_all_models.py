#!/usr/bin/env python3
"""
Model Comparison Test Script

This script tests all available sentiment models against the emergency_sentiment_test_data_2.csv
dataset and generates a comprehensive comparison report.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.models.predict import predict_sentiment, get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_comparison_test')

# Constants
TEST_DATA_PATH = "emergency_sentiment_test_data_2.csv"
OUTPUT_REPORT_PATH = "model_comparison_report.md"
PLOTS_DIR = "plots"

def setup():
    """Initialize test environment."""
    # Create plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"Created plots directory at {PLOTS_DIR}")

def map_sentiment_text_to_value(sentiment_text):
    """Convert textual sentiment to numeric value."""
    sentiment_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    return sentiment_map.get(sentiment_text, 0)

def load_test_data(data_path=TEST_DATA_PATH):
    """Load and prepare test data."""
    logger.info(f"Loading test data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        
        # Check for required columns
        required_columns = ["Text", "Sentiment"]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in test data. Expected: {required_columns}")
            sys.exit(1)
            
        # Map textual sentiments to numeric values
        df["sentiment_value"] = df["Sentiment"].map(map_sentiment_text_to_value)
        
        logger.info(f"Loaded {len(df)} test samples")
        
        # Display distribution
        sentiment_counts = df["Sentiment"].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        sys.exit(1)

def test_model(model_name, test_data):
    """Test a specific model on the test data and return performance metrics."""
    logger.info(f"Testing {model_name} model...")
    
    texts = test_data["Text"].values
    true_values = test_data["sentiment_value"].values
    
    predictions = []
    confidences = []
    
    # Use standard prediction API
    for text in texts:
        try:
            sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, model_name)
            predictions.append(sentiment_value)
            confidences.append(confidence)
        except Exception as e:
            logger.error(f"Error predicting with {model_name} model: {str(e)}")
            predictions.append(0)
            confidences.append(0.0)
    
    # Convert predictions to numpy array
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    f1 = f1_score(true_values, predictions, average='weighted')
    
    # Get classification report
    report = classification_report(
        true_values, 
        predictions,
        labels=[-1, 0, 1],
        target_names=["Negative", "Neutral", "Positive"], 
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(true_values, predictions, labels=[-1, 0, 1])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name.capitalize()} Model Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"{model_name} model achieved {accuracy:.4f} accuracy, {f1:.4f} F1 score")
    
    return {
        "model_name": model_name,
        "predictions": predictions,
        "confidences": confidences,
        "true_values": true_values,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "plot_path": plot_path
    }

def generate_error_analysis(results_dict):
    """Generate error analysis for each model."""
    error_analysis = {}
    
    for model_name, results in results_dict.items():
        predictions = results["predictions"]
        true_values = results["true_values"]
        confidences = results["confidences"]
        
        # Create errors mask
        errors = predictions != true_values
        error_indices = np.where(errors)[0]
        
        # Count error types
        error_types = {}
        for idx in error_indices:
            true_val = true_values[idx]
            pred_val = predictions[idx]
            error_type = f"{true_val}→{pred_val}"
            
            if error_type not in error_types:
                error_types[error_type] = []
            
            error_types[error_type].append(idx)
        
        # Format error types
        formatted_error_types = {}
        for error_type, indices in error_types.items():
            true_val, pred_val = error_type.split("→")
            true_label = "Negative" if float(true_val) == -1 else ("Neutral" if float(true_val) == 0 else "Positive")
            pred_label = "Negative" if float(pred_val) == -1 else ("Neutral" if float(pred_val) == 0 else "Positive")
            
            formatted_key = f"{true_label} as {pred_label}"
            formatted_error_types[formatted_key] = {
                "count": len(indices),
                "percentage": (len(indices) / len(error_indices)) * 100,
                "avg_confidence": np.mean([confidences[i] for i in indices])
            }
        
        error_analysis[model_name] = {
            "total_errors": int(errors.sum()),
            "error_rate": float(errors.mean()),
            "error_types": formatted_error_types,
            "high_confidence_errors": sum(1 for i in error_indices if confidences[i] > 0.7),
            "low_confidence_correct": sum(1 for i in range(len(predictions)) if not errors[i] and confidences[i] < 0.5)
        }
    
    return error_analysis

def generate_comparison_report(results_dict, error_analysis):
    """Generate a comprehensive markdown report comparing all models."""
    logger.info("Generating model comparison report...")
    
    # Sort models by accuracy
    sorted_models = sorted(
        results_dict.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )
    
    # Start building the report
    md_content = f"""# Sentiment Analysis Model Comparison Report

## Overview

This report compares the performance of all available sentiment models on emergency services text data.

**Test Dataset:** {TEST_DATA_PATH}  
**Number of Samples:** {len(results_dict[sorted_models[0][0]]["true_values"])}  
**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Ranking by Accuracy

| Rank | Model | Accuracy | F1 Score | Error Rate |
|------|-------|----------|----------|------------|
"""
    
    # Add model ranking
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        md_content += f"| {rank} | {model_name.capitalize()} | {results['accuracy']:.4f} | {results['f1_score']:.4f} | {error_analysis[model_name]['error_rate']:.4f} |\n"
    
    md_content += "\n## Detailed Performance by Model\n\n"
    
    # Add detailed performance for each model
    for model_name, results in sorted_models:
        md_content += f"### {model_name.capitalize()} Model\n\n"
        
        # Add metrics
        md_content += f"- **Accuracy:** {results['accuracy']:.4f}\n"
        md_content += f"- **F1 Score:** {results['f1_score']:.4f}\n"
        md_content += f"- **Error Rate:** {error_analysis[model_name]['error_rate']:.4f} ({error_analysis[model_name]['total_errors']} errors)\n\n"
        
        # Add confusion matrix
        md_content += f"**Confusion Matrix:**\n\n"
        md_content += f"![{model_name.capitalize()} Confusion Matrix]({results['plot_path']})\n\n"
        
        # Add class-specific metrics
        md_content += "**Performance by Sentiment Class:**\n\n"
        md_content += "| Sentiment | Precision | Recall | F1-Score | Support |\n"
        md_content += "|-----------|-----------|--------|----------|--------|\n"
        
        report = results["classification_report"]
        for sentiment in ["Negative", "Neutral", "Positive"]:
            md_content += f"| {sentiment} | {report[sentiment]['precision']:.4f} | {report[sentiment]['recall']:.4f} | {report[sentiment]['f1-score']:.4f} | {report[sentiment]['support']} |\n"
        
        # Add error analysis
        md_content += "\n**Error Analysis:**\n\n"
        md_content += "| Error Type | Count | Percentage | Avg Confidence |\n"
        md_content += "|------------|-------|------------|---------------|\n"
        
        for error_type, error_info in error_analysis[model_name]["error_types"].items():
            md_content += f"| {error_type} | {error_info['count']} | {error_info['percentage']:.2f}% | {error_info['avg_confidence']:.4f} |\n"
        
        md_content += f"\n- High Confidence Errors (>0.7): {error_analysis[model_name]['high_confidence_errors']}\n"
        md_content += f"- Low Confidence Correct Predictions (<0.5): {error_analysis[model_name]['low_confidence_correct']}\n\n"
        
        md_content += "---\n\n"
    
    # Add comparative analysis
    md_content += "## Comparative Analysis\n\n"
    
    # Create a bar chart comparing model accuracy
    plt.figure(figsize=(10, 6))
    models = [model for model, _ in sorted_models]
    accuracies = [results["accuracy"] for _, results in sorted_models]
    f1_scores = [results["f1_score"] for _, results in sorted_models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [model.capitalize() for model in models])
    plt.legend()
    plt.tight_layout()
    
    # Save the comparison chart
    comparison_chart_path = os.path.join(PLOTS_DIR, "model_performance_comparison.png")
    plt.savefig(comparison_chart_path)
    plt.close()
    
    md_content += f"![Model Performance Comparison]({comparison_chart_path})\n\n"
    
    # Create a heatmap of sentiment class performance across models
    sentiment_performance = {}
    for model_name, results in sorted_models:
        report = results["classification_report"]
        for sentiment in ["Negative", "Neutral", "Positive"]:
            if sentiment not in sentiment_performance:
                sentiment_performance[sentiment] = {}
            sentiment_performance[sentiment][model_name] = report[sentiment]["f1-score"]
    
    # Convert to DataFrame for heatmap
    df_heatmap = pd.DataFrame(sentiment_performance)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title('F1-Score by Sentiment Class and Model')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(PLOTS_DIR, "sentiment_performance_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    
    md_content += f"![Sentiment Performance Heatmap]({heatmap_path})\n\n"
    
    # Add key insights
    md_content += "## Key Insights\n\n"
    
    # Determine best model overall
    best_model = sorted_models[0][0]
    
    # Determine best model for each sentiment class
    best_by_sentiment = {}
    for sentiment in ["Negative", "Neutral", "Positive"]:
        best_model_for_sentiment = max(
            [(model, results["classification_report"][sentiment]["f1-score"]) for model, results in results_dict.items()],
            key=lambda x: x[1]
        )
        best_by_sentiment[sentiment] = best_model_for_sentiment
    
    # Add insights
    md_content += f"1. **Best Overall Model:** {best_model.capitalize()} with {results_dict[best_model]['accuracy']:.4f} accuracy\n\n"
    
    md_content += "2. **Best Model by Sentiment Class:**\n"
    for sentiment, (model, score) in best_by_sentiment.items():
        md_content += f"   - {sentiment}: {model.capitalize()} (F1-Score: {score:.4f})\n"
    
    md_content += "\n3. **Common Error Patterns:**\n"
    
    # Find common error patterns across models
    all_error_types = set()
    for model_analysis in error_analysis.values():
        all_error_types.update(model_analysis["error_types"].keys())
    
    most_common_errors = {}
    for error_type in all_error_types:
        for model_name, model_analysis in error_analysis.items():
            if error_type in model_analysis["error_types"]:
                if error_type not in most_common_errors:
                    most_common_errors[error_type] = 0
                most_common_errors[error_type] += model_analysis["error_types"][error_type]["count"]
    
    # Get top 3 most common errors
    top_errors = sorted(most_common_errors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for error_type, count in top_errors:
        md_content += f"   - {error_type}: {count} instances across all models\n"
    
    md_content += "\n4. **Confidence Analysis:**\n"
    md_content += "   - Models tend to be " + ("overconfident" if sum(analysis["high_confidence_errors"] for analysis in error_analysis.values()) > 10 else "appropriately calibrated") + " in their predictions\n"
    md_content += "   - " + ("High" if sum(analysis["low_confidence_correct"] for analysis in error_analysis.values()) > 20 else "Low") + " number of correct predictions with low confidence\n\n"
    
    md_content += "## Recommendations\n\n"
    
    md_content += "Based on the analysis above, here are recommendations for model selection and improvement:\n\n"
    
    md_content += f"1. **For General Use:** Use the {best_model.capitalize()} model as it has the best overall performance\n\n"
    
    md_content += "2. **For Specific Sentiment Detection:**\n"
    for sentiment, (model, _) in best_by_sentiment.items():
        md_content += f"   - Use {model.capitalize()} for detecting {sentiment.lower()} sentiment\n"
    
    md_content += "\n3. **Model Improvement Opportunities:**\n"
    md_content += f"   - Focus on improving detection of {top_errors[0][0].split(' as ')[0]} sentiment across all models\n"
    md_content += "   - Consider ensemble methods to combine strengths of different models\n"
    md_content += "   - Add more domain-specific training data for emergency services\n"
    
    # Write to file
    with open(OUTPUT_REPORT_PATH, "w") as f:
        f.write(md_content)
    
    logger.info(f"Comparison report saved to {OUTPUT_REPORT_PATH}")
    
    return md_content

def main():
    """Main function to run tests and generate comparison report."""
    try:
        # Setup environment
        setup()
        
        # Load test data
        test_data = load_test_data()
        
        # Get list of available models
        available_models = get_available_models()
        logger.info(f"Found {len(available_models)} models: {', '.join(available_models)}")
        
        # Test each model
        results_dict = {}
        for model_name in available_models:
            results_dict[model_name] = test_model(model_name, test_data)
        
        # Generate error analysis
        error_analysis = generate_error_analysis(results_dict)
        
        # Generate comparative report
        generate_comparison_report(results_dict, error_analysis)
        
        logger.info("Testing completed successfully!")
        logger.info(f"Results saved to {OUTPUT_REPORT_PATH}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 