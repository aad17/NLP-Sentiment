# Sentiment Analysis Model

This directory contains the sentiment analysis model for the fire brigade incident reports.

## Model Overview

The sentiment analysis model classifies incident reports into three categories:

- **Positive** (1): Reports describing successful outcomes, rescues, or effective operations
- **Neutral** (0): Reports of routine activities, false alarms, or standard procedures
- **Negative** (-1): Reports describing injuries, equipment failures, or other issues

## Model Architecture

The model uses a simple but effective approach:

1. **Text Preprocessing**:
   - Lowercase conversion
   - Special character removal
   - Lemmatization
   - Stopword removal

2. **Feature Extraction**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Limited to 5,000 features
   - Min document frequency: 2
   - Max document frequency: 85%

3. **Classification**:
   - Linear Support Vector Classification (LinearSVC)
   - Multi-class classification (3 classes)

## Training the Model

To train the model:

```bash
python app/models/train_model.py
```

The training process:
1. Generates balanced training data from sample reports
2. Preprocesses text data
3. Splits data into training (80%) and testing (20%) sets
4. Trains the TF-IDF vectorizer and SVM classifier
5. Evaluates model performance
6. Saves the trained model and vectorizer to disk

## Making Predictions

To make predictions with the trained model:

```python
from app.models.predict import predict_sentiment

text = "Successfully rescued family from burning building with no injuries."
sentiment_value, sentiment_label, confidence = predict_sentiment(text)
```

## Model API

The model is served through a Flask API:

- **Predict endpoint**: `/predict` (POST)
  - Input: JSON with `text` field
  - Output: Sentiment prediction with confidence score

- **Batch predict endpoint**: `/batch_predict` (POST)
  - Input: JSON with `texts` list
  - Output: List of sentiment predictions

- **Sample endpoint**: `/predict_sample` (GET)
  - Output: Prediction for a sample text

## Model Performance

The model typically achieves:

- Accuracy: ~90-95%
- High precision and recall for positive and negative classes
- Slightly lower performance on neutral class (more ambiguous)

## Limitations

- Simple model architecture prioritizing speed and efficiency
- Limited training data and vocabulary
- May struggle with nuanced or ambiguous reports 