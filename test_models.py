#!/usr/bin/env python3
"""
Model Testing Script

This script tests the sentiment analysis models via the API.
It tests all available models on a set of example sentences
and compares their performance.
"""
import os
import sys
import requests
import json
import time
import logging
from tabulate import tabulate

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.config import MODEL_API_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_models')

# Model API URL
MODEL_API_BASE_URL = MODEL_API_URL.replace("/predict", "")

# Test examples covering different domains and sentiments
TEST_EXAMPLES = [
    # Fire brigade domain-specific
    {
        "text": "Successfully rescued family from burning building with no injuries.",
        "expected": "positive"
    },
    {
        "text": "Training session on fire suppression techniques will be held next Tuesday.",
        "expected": "neutral"
    },
    {
        "text": "Several firefighters were injured due to roof collapse during yesterday's operation.",
        "expected": "negative"
    },
    {
        "text": "Our new thermal imaging cameras are making search and rescue much more effective.",
        "expected": "positive"
    },
    {
        "text": "Department budget for next fiscal year remains unchanged from current allocation.",
        "expected": "neutral"
    },
    
    # General, non-domain specific
    {
        "text": "This product is amazing, I highly recommend it!",
        "expected": "positive"
    },
    {
        "text": "It's an okay option if you don't have alternatives.",
        "expected": "neutral"
    },
    {
        "text": "The service was terrible and I will never come back.",
        "expected": "negative"
    },
    
    # Mixed/ambiguous
    {
        "text": "Despite equipment malfunction, the team managed to save all residents.",
        "expected": "mixed"  # Could be interpreted as positive or negative
    },
    {
        "text": "The response time improved but still missed the target goals.",
        "expected": "mixed"  # Could be interpreted as positive or negative
    }
]

def check_api_health():
    """Check if the model API is running."""
    try:
        response = requests.get(f"{MODEL_API_BASE_URL}/health")
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

def get_available_models():
    """Get list of available models from the API."""
    try:
        response = requests.get(f"{MODEL_API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except requests.exceptions.RequestException:
        return []

def test_model(model_type, examples):
    """
    Test a specific model on the given examples.
    
    Args:
        model_type (str): The model type to test
        examples (list): List of test examples
        
    Returns:
        dict: Test results
    """
    results = []
    correct = 0
    total = 0
    
    for example in examples:
        text = example["text"]
        expected = example["expected"]
        
        try:
            # Send request to API
            response = requests.post(
                f"{MODEL_API_BASE_URL}/predict",
                json={"text": text, "model_type": model_type},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Error from API: {response.text}")
                result = {
                    "text": text,
                    "expected": expected,
                    "predicted": "error",
                    "confidence": 0.0,
                    "correct": False,
                    "notes": f"API error: {response.status_code}"
                }
                results.append(result)
                total += 1
                continue
                
            # Parse response
            prediction = response.json()
            sentiment = prediction.get("sentiment", "neutral")
            confidence = prediction.get("confidence", 0.0)
            model_used = prediction.get("model_used", None)
            
            # Check if prediction is correct
            # For mixed examples, any prediction is acceptable
            if expected == "mixed":
                is_correct = True
                notes = "Ambiguous example, any prediction accepted"
            else:
                is_correct = sentiment == expected
                notes = ""
            
            if is_correct:
                correct += 1
            
            # Add model_used note for hybrid model
            if model_type == "hybrid" and model_used:
                notes += f"Used {model_used} model" if not notes else f", used {model_used} model"
            
            result = {
                "text": text,
                "expected": expected,
                "predicted": sentiment,
                "confidence": confidence,
                "correct": is_correct,
                "notes": notes
            }
            
            results.append(result)
            total += 1
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            result = {
                "text": text,
                "expected": expected,
                "predicted": "error",
                "confidence": 0.0,
                "correct": False,
                "notes": f"Request error: {str(e)}"
            }
            results.append(result)
            total += 1
    
    # Calculate accuracy (excluding mixed examples)
    non_mixed = [r for r in results if r["expected"] != "mixed"]
    accuracy = correct / total if total > 0 else 0
    non_mixed_accuracy = sum(1 for r in non_mixed if r["correct"]) / len(non_mixed) if non_mixed else 0
    
    # Calculate domain-specific accuracy
    domain_examples = examples[:5]  # First 5 examples are domain-specific
    domain_results = results[:5]
    domain_correct = sum(1 for r in domain_results if r["correct"])
    domain_accuracy = domain_correct / len(domain_results) if domain_results else 0
    
    # Calculate general accuracy
    general_examples = examples[5:8]  # Examples 6-8 are general
    general_results = results[5:8]
    general_correct = sum(1 for r in general_results if r["correct"])
    general_accuracy = general_correct / len(general_results) if general_results else 0
    
    return {
        "model_type": model_type,
        "results": results,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "non_mixed_accuracy": non_mixed_accuracy,
        "domain_accuracy": domain_accuracy,
        "general_accuracy": general_accuracy
    }

def print_results(test_results):
    """Print test results in a formatted table."""
    for model_result in test_results:
        model_type = model_result["model_type"]
        results = model_result["results"]
        accuracy = model_result["accuracy"] * 100
        domain_accuracy = model_result["domain_accuracy"] * 100
        general_accuracy = model_result["general_accuracy"] * 100
        
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 80}")
        print(f"Overall Accuracy: {accuracy:.1f}%")
        print(f"Domain-Specific Accuracy: {domain_accuracy:.1f}%")
        print(f"General Text Accuracy: {general_accuracy:.1f}%")
        print(f"{'-' * 80}")
        
        # Prepare table data
        table_data = []
        for result in results:
            status = "✓" if result["correct"] else "✗"
            confidence = f"{result['confidence'] * 100:.1f}%"
            row = [
                status,
                result["text"][:50] + ("..." if len(result["text"]) > 50 else ""),
                result["expected"],
                result["predicted"],
                confidence,
                result["notes"]
            ]
            table_data.append(row)
        
        # Print table
        headers = ["", "Text", "Expected", "Predicted", "Confidence", "Notes"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()

def compare_models(test_results):
    """Compare results from different models."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    
    comparison_data = []
    for model_result in test_results:
        model_type = model_result["model_type"]
        accuracy = model_result["accuracy"] * 100
        domain_accuracy = model_result["domain_accuracy"] * 100
        general_accuracy = model_result["general_accuracy"] * 100
        
        row = [
            model_type.upper(),
            f"{accuracy:.1f}%",
            f"{domain_accuracy:.1f}%",
            f"{general_accuracy:.1f}%"
        ]
        comparison_data.append(row)
    
    headers = ["Model", "Overall Accuracy", "Domain-Specific", "General Text"]
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    print()

def main():
    """Main function to run the tests."""
    # Check if API is running
    print(f"Testing API at: {MODEL_API_BASE_URL}")
    if not check_api_health():
        logger.error("Model API is not running. Please start it first.")
        print("Error: Model API is not running. Please start it first.")
        return
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models available from the API.")
        print("Error: No models available from the API.")
        return
    
    logger.info(f"Available models: {', '.join(available_models)}")
    print(f"Testing {len(available_models)} models: {', '.join(available_models)}")
    print(f"Running tests on {len(TEST_EXAMPLES)} examples...")
    
    # Test each model
    test_results = []
    for model_type in available_models:
        logger.info(f"Testing model: {model_type}")
        print(f"Testing {model_type} model...")
        result = test_model(model_type, TEST_EXAMPLES)
        test_results.append(result)
    
    # Print detailed results
    print_results(test_results)
    
    # Compare models
    compare_models(test_results)
    
    print("Test completed.")

if __name__ == "__main__":
    main() 