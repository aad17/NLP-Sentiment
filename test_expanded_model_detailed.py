#!/usr/bin/env python3
"""
Expanded Sentiment Model Detailed Test

This script provides a detailed analysis of the expanded sentiment model's performance
specifically on emergency services text data.
"""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.text_preprocessing import extract_features, map_sentiment_value
from app.models.predict import predict_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('expanded_model_test')

# Constants
TEST_DATA_PATH = "emergency_sentiment_test_data_2.csv"
OUTPUT_REPORT_PATH = "expanded_model_detailed_report.md"
PLOTS_DIR = "plots"
MODEL_PATH = "app/models/expanded_sentiment_model.pkl"
VECTORIZER_PATH = "app/models/expanded_vectorizer.pkl"

def setup():
    """Initialize test environment."""
    # Create plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"Created plots directory at {PLOTS_DIR}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        logger.error(f"Expanded model files not found at {MODEL_PATH} or {VECTORIZER_PATH}")
        sys.exit(1)
    
    logger.info(f"Found expanded model files")

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

def test_expanded_model(test_data):
    """Test the expanded model on the test data."""
    logger.info(f"Testing expanded sentiment model...")
    
    texts = test_data["Text"].values
    true_values = test_data["sentiment_value"].values
    rationales = test_data["Rationale"].values if "Rationale" in test_data.columns else [""] * len(texts)
    
    predictions = []
    confidences = []
    
    # Use standard prediction API
    for text in texts:
        try:
            sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, "expanded")
            predictions.append(sentiment_value)
            confidences.append(confidence)
        except Exception as e:
            logger.error(f"Error predicting with expanded model: {str(e)}")
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
    
    # Create detailed results object
    results = {
        "predictions": predictions,
        "confidences": confidences,
        "true_values": true_values,
        "rationales": rationales,
        "texts": texts,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm
    }
    
    # Generate confusion matrix plot
    plt.figure(figsize=(10, 8))
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
    plt.title('Expanded Model Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, "expanded_model_confusion_matrix_detailed.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    
    # Group by sentiment
    df_results = pd.DataFrame({
        'true_sentiment': [map_sentiment_value(val) for val in true_values],
        'predicted_sentiment': [map_sentiment_value(val) for val in predictions],
        'confidence': confidences,
        'correct': np.array(true_values) == predictions
    })
    
    # Create a violin plot of confidence scores
    sns.violinplot(x='true_sentiment', y='confidence', hue='correct', 
                   data=df_results, split=True, palette={True: "green", False: "red"})
    
    plt.title('Confidence Distribution by Sentiment and Correctness')
    plt.xlabel('True Sentiment')
    plt.ylabel('Confidence Score')
    plt.xticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
    plt.legend(title='Prediction Correct')
    plt.tight_layout()
    
    # Save plot
    confidence_plot_path = os.path.join(PLOTS_DIR, "expanded_model_confidence_distribution.png")
    plt.savefig(confidence_plot_path)
    plt.close()
    
    logger.info(f"Model achieved {accuracy:.4f} accuracy, {f1:.4f} F1 score")
    
    return results

def analyze_results(results):
    """Perform detailed analysis on the results."""
    predictions = results["predictions"]
    true_values = results["true_values"]
    confidences = results["confidences"]
    texts = results["texts"]
    rationales = results["rationales"]
    
    # Create DataFrame with all prediction info
    df = pd.DataFrame({
        "text": texts,
        "true_value": true_values,
        "predicted_value": predictions,
        "confidence": confidences,
        "rationale": rationales
    })
    
    # Add categorical sentiment labels
    df["true_sentiment"] = df["true_value"].map({1: "Positive", 0: "Neutral", -1: "Negative"})
    df["predicted_sentiment"] = df["predicted_value"].map({1: "Positive", 0: "Neutral", -1: "Negative"})
    df["correct"] = df["true_value"] == df["predicted_value"]
    
    # Error analysis
    errors = df[~df["correct"]].copy()
    error_types = errors.groupby(["true_sentiment", "predicted_sentiment"]).size().reset_index(name="count")
    error_types = error_types.sort_values("count", ascending=False)
    
    # By sentiment category
    by_sentiment = df.groupby("true_sentiment").agg(
        count=("text", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean")
    ).reset_index()
    
    # High vs. low confidence errors
    high_conf_errors = errors[errors["confidence"] > 0.7]
    low_conf_errors = errors[errors["confidence"] < 0.3]
    
    # Sample correct and incorrect predictions
    high_conf_correct = df[(df["correct"]) & (df["confidence"] > 0.8)].sample(min(5, len(df[(df["correct"]) & (df["confidence"] > 0.8)])))
    high_conf_incorrect = errors[errors["confidence"] > 0.7].sample(min(5, len(errors[errors["confidence"] > 0.7])))
    difficult_cases = df[df["confidence"] < 0.4].sample(min(5, len(df[df["confidence"] < 0.4])))
    
    return {
        "error_types": error_types,
        "by_sentiment": by_sentiment,
        "high_conf_errors": high_conf_errors,
        "low_conf_errors": low_conf_errors,
        "high_conf_correct": high_conf_correct,
        "high_conf_incorrect": high_conf_incorrect,
        "difficult_cases": difficult_cases,
        "total_errors": len(errors),
        "error_rate": len(errors) / len(df)
    }

def generate_detailed_report(results, analysis):
    """Generate a detailed markdown report of the expanded model's performance."""
    logger.info("Generating detailed report for expanded model...")
    
    accuracy = results["accuracy"]
    f1_score = results["f1_score"]
    report = results["classification_report"]
    error_types = analysis["error_types"]
    by_sentiment = analysis["by_sentiment"]
    
    md_content = f"""# Expanded Sentiment Model Detailed Analysis

## Overview

This report provides an in-depth analysis of the expanded sentiment model's performance on emergency services text data.

**Test Dataset:** {TEST_DATA_PATH}  
**Number of Samples:** {len(results["texts"])}  
**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Performance Summary

- **Accuracy:** {accuracy:.4f}
- **F1 Score:** {f1_score:.4f}
- **Error Rate:** {analysis["error_rate"]:.4f} ({analysis["total_errors"]} errors out of {len(results["texts"])} samples)

## Performance by Sentiment Class

| Sentiment | Samples | Correct | Accuracy | Avg. Confidence |
|-----------|---------|---------|----------|----------------|
"""
    
    for _, row in by_sentiment.iterrows():
        md_content += f"| {row['true_sentiment']} | {row['count']} | {row['correct']} | {row['accuracy']:.4f} | {row['avg_confidence']:.4f} |\n"
    
    md_content += """
## Classification Report

```
"""
    md_content += f"""              precision    recall  f1-score   support

    Negative       {report["Negative"]["precision"]:.4f}      {report["Negative"]["recall"]:.4f}      {report["Negative"]["f1-score"]:.4f}        {report["Negative"]["support"]}
     Neutral       {report["Neutral"]["precision"]:.4f}      {report["Neutral"]["recall"]:.4f}      {report["Neutral"]["f1-score"]:.4f}        {report["Neutral"]["support"]}
    Positive       {report["Positive"]["precision"]:.4f}      {report["Positive"]["recall"]:.4f}      {report["Positive"]["f1-score"]:.4f}        {report["Positive"]["support"]}

    accuracy                          {report["accuracy"]:.4f}        {report["macro avg"]["support"]}
   macro avg       {report["macro avg"]["precision"]:.4f}      {report["macro avg"]["recall"]:.4f}      {report["macro avg"]["f1-score"]:.4f}        {report["macro avg"]["support"]}
weighted avg       {report["weighted avg"]["precision"]:.4f}      {report["weighted avg"]["recall"]:.4f}      {report["weighted avg"]["f1-score"]:.4f}        {report["weighted avg"]["support"]}
```

## Confusion Matrix

![Confusion Matrix](plots/expanded_model_confusion_matrix_detailed.png)

## Confidence Distribution by Sentiment and Correctness

![Confidence Distribution](plots/expanded_model_confidence_distribution.png)

## Error Analysis

### Most Common Error Types

| True Sentiment | Predicted As | Count | Error Rate |
|----------------|--------------|-------|------------|
"""
    
    for _, row in error_types.iterrows():
        true_sentiment = row["true_sentiment"]
        true_count = by_sentiment[by_sentiment["true_sentiment"] == true_sentiment]["count"].values[0]
        error_rate = row["count"] / true_count
        md_content += f"| {row['true_sentiment']} | {row['predicted_sentiment']} | {row['count']} | {error_rate:.4f} |\n"
    
    md_content += """
### High Confidence Correct Predictions

| Text | True Sentiment | Confidence |
|------|---------------|------------|
"""
    
    for _, row in analysis["high_conf_correct"].iterrows():
        text = row["text"]
        if len(text) > 80:
            text = text[:77] + "..."
        md_content += f"| {text} | {row['true_sentiment']} | {row['confidence']:.4f} |\n"
    
    md_content += """
### High Confidence Incorrect Predictions

| Text | True Sentiment | Predicted As | Confidence | Rationale |
|------|---------------|--------------|------------|-----------|
"""
    
    for _, row in analysis["high_conf_incorrect"].iterrows():
        text = row["text"]
        if len(text) > 80:
            text = text[:77] + "..."
        rationale = row["rationale"]
        if len(rationale) > 80:
            rationale = rationale[:77] + "..."
        md_content += f"| {text} | {row['true_sentiment']} | {row['predicted_sentiment']} | {row['confidence']:.4f} | {rationale} |\n"
    
    md_content += """
### Difficult Cases (Low Confidence)

| Text | True Sentiment | Predicted As | Confidence | Correct |
|------|---------------|--------------|------------|---------|
"""
    
    for _, row in analysis["difficult_cases"].iterrows():
        text = row["text"]
        if len(text) > 80:
            text = text[:77] + "..."
        correct = "✓" if row["correct"] else "✗"
        md_content += f"| {text} | {row['true_sentiment']} | {row['predicted_sentiment']} | {row['confidence']:.4f} | {correct} |\n"
    
    md_content += """
## Insights and Recommendations

Based on the analysis above, here are key insights about the expanded model:

1. **Sentiment Class Performance**: 
   - The model performs best on {best_class} sentiment, with {best_accuracy:.2f}% accuracy
   - The model struggles most with {worst_class} sentiment, with only {worst_accuracy:.2f}% accuracy

2. **Common Misclassifications**:
   - Most errors come from confusing {error_from} for {error_to}
   - This suggests the model may not capture subtle distinctions in emergency context

3. **Confidence Analysis**:
   - Average confidence on correct predictions: {avg_correct_conf:.4f}
   - Average confidence on incorrect predictions: {avg_incorrect_conf:.4f}
   - The confidence gap suggests {conf_gap_insight}

4. **Recommendations**:
   - Fine-tune the model specifically on emergency services data
   - Add more {worst_class} examples to the training data
   - Consider implementing domain-specific preprocessing to capture emergency terminology
   - Calibrate confidence scores to better reflect actual prediction reliability
""".format(
        best_class=by_sentiment.iloc[by_sentiment["accuracy"].argmax()]["true_sentiment"],
        best_accuracy=by_sentiment["accuracy"].max() * 100,
        worst_class=by_sentiment.iloc[by_sentiment["accuracy"].argmin()]["true_sentiment"],
        worst_accuracy=by_sentiment["accuracy"].min() * 100,
        error_from=error_types.iloc[0]["true_sentiment"] if not error_types.empty else "N/A",
        error_to=error_types.iloc[0]["predicted_sentiment"] if not error_types.empty else "N/A",
        avg_correct_conf=df[df["correct"]]["confidence"].mean() if 'df' in locals() else 0,
        avg_incorrect_conf=df[~df["correct"]]["confidence"].mean() if 'df' in locals() else 0,
        conf_gap_insight="the model is reasonably calibrated" if 'df' not in locals() or df[df["correct"]]["confidence"].mean() - df[~df["correct"]]["confidence"].mean() > 0.2 else "the model may be overconfident in incorrect predictions"
    )
    
    # Write to file
    with open(OUTPUT_REPORT_PATH, "w") as f:
        f.write(md_content)
    
    logger.info(f"Detailed report saved to {OUTPUT_REPORT_PATH}")
    
    return md_content

def main():
    """Main function to run test and generate detailed report."""
    try:
        # Setup test environment
        setup()
        
        # Load test data
        test_data = load_test_data()
        
        # Test expanded model
        results = test_expanded_model(test_data)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Generate detailed report
        report = generate_detailed_report(results, analysis)
        
        logger.info("Testing completed successfully!")
        logger.info(f"Results saved to {OUTPUT_REPORT_PATH}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 