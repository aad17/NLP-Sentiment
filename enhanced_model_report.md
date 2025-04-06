# Enhanced Sentiment Analysis Model Report

## Model Overview

This report documents the training and performance of an enhanced sentiment analysis model for fire brigade incident reports. The model was trained on a significantly larger dataset compared to the original model, with improvements to the feature extraction and classification process.

## Training Dataset

The enhanced model was trained using:
- **900 total samples** (300 samples per sentiment class)
- **Balanced class distribution** across positive, neutral, and negative sentiments
- **Greater text variation** using modifiers for time, location, and result
- **Expanded vocabulary** with 90 unique base templates enhanced with modifiers
- **TF-IDF vectorization** with a larger feature set (10,000 features)

## Training Results

The model achieved perfect accuracy on the validation set (20% of training data):

**Validation Metrics:**
- **Accuracy: 100.0%**
- **Precision (all classes): 100.0%**
- **Recall (all classes): 100.0%**
- **F1-score (all classes): 100.0%**

**Confusion Matrix:**
```
[[64  0  0]
 [ 0 65  0]
 [ 0  0 51]]
```

## Test Results

The model was evaluated on 10 fresh test samples, achieving perfect accuracy:

**Test Accuracy: 100.0% (10/10 correct predictions)**

### Detailed Test Results

| # | Text | Expected | Predicted | Confidence | Result |
|---|------|----------|-----------|------------|--------|
| 1 | Attended cross-departmental coordination meeting, which was documented for training purposes. | neutral (0) | neutral (0) | 0.4298 | ✓ |
| 2 | Yesterday, updated maps of district emergency water sources in the apartment complex, triggering protocol updates. | neutral (0) | neutral (0) | 0.3886 | ✓ |
| 3 | Volunteer firefighter suffered smoke inhalation requiring hospitalization in the downtown area. | negative (-1) | negative (-1) | 0.4643 | ✓ |
| 4 | Communication system failed during critical emergency coordination. | negative (-1) | negative (-1) | 0.5119 | ✓ |
| 5 | During evening shift, successfully rescued family from burning building with no injuries in the north district. | positive (1) | positive (1) | 0.4250 | ✓ |
| 6 | Quick-thinking firefighter prevented gas explosion at apartment complex. | positive (1) | positive (1) | 0.4763 | ✓ |
| 7 | At 3 AM, responded to fire alarm which was determined to be a false alarm, causing service disruption. | neutral (0) | neutral (0) | 0.4363 | ✓ |
| 8 | Multiple injuries reported due to building collapse during firefighting operation in the north district. | negative (-1) | negative (-1) | 0.5011 | ✓ |
| 9 | Electrical failure disabled critical pumping equipment during major fire at the industrial park. | negative (-1) | negative (-1) | 0.5134 | ✓ |
| 10 | Last Thursday, training exercise completed with excellent team coordination at the industrial park, causing service disruption. | positive (1) | positive (1) | 0.3728 | ✓ |

## Real-World Custom Examples Test

To better assess the model's performance on completely novel text patterns, we tested it on 10 custom examples that don't follow our training templates.

**Custom Test Accuracy: 60.0% (6/10 correct predictions)**

### Performance by Sentiment Type:
- **Positive: 33.3%** (1/3 correct)
- **Neutral: 50.0%** (2/4 correct)
- **Negative: 100.0%** (3/3 correct)

### Detailed Custom Test Results

| # | Text | Expected | Predicted | Confidence | Result |
|---|------|----------|-----------|------------|--------|
| 1 | Fire team encountered significant water pressure problems while fighting a blaze at 123 Main Street. | negative | negative | 0.2181 | ✓ |
| 2 | Monthly equipment checks revealed no issues with breathing apparatus units. | neutral | neutral | 0.2729 | ✓ |
| 3 | Chief Johnson commended the team for their exceptional response to the hospital emergency. | positive | negative | 0.2110 | ✗ |
| 4 | Due to budget cuts, equipment replacement has been delayed by 6 months. | negative | negative | 0.5348 | ✓ |
| 5 | The aerial ladder truck was serviced on Thursday as part of regular maintenance. | neutral | neutral | 0.3617 | ✓ |
| 6 | Two firefighters were honored with medals of bravery for the rescue at Johnson Apartments. | positive | negative | 0.4692 | ✗ |
| 7 | Smoke was visible from ten miles away as crews arrived at the warehouse fire. | neutral | negative | 0.1826 | ✗ |
| 8 | Heat exhaustion affected three firefighters during the response to the factory fire. | negative | negative | 0.4608 | ✓ |
| 9 | Cooperation between police and fire departments facilitated rapid evacuation of senior center. | positive | positive | 0.2872 | ✓ |
| 10 | The search and rescue training program has been expanded to include drone operations. | neutral | positive | 0.2797 | ✗ |

### Key Observations:

1. **Strong at Identifying Negative Sentiment**: The model performed perfectly on negative sentiment examples, even with novel phrasing.

2. **Struggles with Positive Recognition**: The model had difficulty identifying positive sentiment in novel contexts, particularly with commendations and honors.

3. **Medium Performance on Neutral Content**: The model achieved 50% accuracy on neutral examples, with a tendency to misclassify neutral statements as either positive or negative.

4. **Lower Confidence Scores**: The confidence scores on custom examples (avg: 0.3278) were generally lower than on template-based examples (avg: 0.4519), indicating the model's reduced certainty when facing unfamiliar text patterns.

## Analysis of Model Confidence

The model's confidence scores averaged around 0.45 (on a scale of 0-1), which indicates:
- Reasonable certainty in predictions
- Higher confidence for negative sentiment predictions (avg: 0.4977)
- Moderate confidence for neutral sentiment predictions (avg: 0.4182)
- Moderate confidence for positive sentiment predictions (avg: 0.4247)

Negative sentiment reports generally received the highest confidence scores, suggesting the model finds negative language patterns most distinctive.

## Improvements Over Original Model

The enhanced model demonstrates several improvements over the original model:

1. **Training Data Quantity:** 900 samples vs. 150 samples (6x more data)
2. **Text Variety:** Greater linguistic variation through modifiers and an expanded template set
3. **Feature Engineering:** Increased feature set (10,000 from 5,000) for better text representation
4. **Perfect Accuracy:** 100% accuracy on both validation and test sets

## Potential Limitations

Despite the perfect accuracy on template-based tests, the model shows important limitations:

1. **Synthetic Data Bias:** The model's performance drops significantly (from 100% to 60%) when tested on completely novel examples
2. **Template Dependence:** The model shows strong bias toward recognizing patterns similar to its training data
3. **Limited Positive Sentiment Recognition:** The model particularly struggles with identifying positive sentiment in novel contexts
4. **Vocabulary Limitations:** The model has difficulty with unique terms not present in training data (e.g., "commended," "honored," "medals")

## Recommendations for Further Improvement

Based on the custom test results, several improvements could enhance model performance:

1. **Real-World Training Data:** Incorporate actual fire brigade reports rather than synthetic data
2. **Expanded Positive Vocabulary:** Add more positive sentiment terms and phrases, especially around commendation and recognition
3. **Fine-Tuning:** Use the current model as a base and fine-tune with a smaller set of real-world examples
4. **Sentiment Lexicon:** Incorporate a domain-specific sentiment lexicon for fire brigade terminology

## Conclusion

The enhanced sentiment analysis model demonstrates excellent performance for template-based classification but shows significant limitations with novel text patterns. While the model achieves 100% accuracy on test samples derived from its training templates, it drops to 60% accuracy on completely new examples.

This performance gap highlights the challenges of deploying models trained on synthetic data to real-world applications. For production use, it's recommended to collect actual fire brigade incident reports for training or fine-tuning to better capture the linguistic nuances of real-world data. 