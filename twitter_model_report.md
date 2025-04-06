# Twitter Sentiment Analysis Model Report

## Dataset Overview

The model was trained on the Twitter dataset with the following characteristics:

- **Original Dataset**: 162,980 Twitter entries
- **After Cleaning**: 162,969 entries (removed 11 with missing values)
- **Distribution**:
  - Negative: 35,509 samples (21.8%)
  - Neutral: 55,211 samples (33.9%) 
  - Positive: 72,249 samples (44.3%)
- **Training Sample**: 900 balanced samples (300 per sentiment class)

## Model Performance

### Validation Performance

The model achieved **60.56%** accuracy on the validation set (20% of training data):

**Classification Report:**
```
              precision    recall  f1-score   support

    negative       0.62      0.65      0.64        63
     neutral       0.57      0.56      0.57        55
    positive       0.62      0.60      0.61        62

    accuracy                           0.61       180
   macro avg       0.60      0.60      0.60       180
weighted avg       0.61      0.61      0.61       180
```

**Confusion Matrix:**
```
[[41 13  9]
 [10 31 14]
 [15 10 37]]
```

### Initial Custom Test Examples Performance

The model was evaluated on 10 domain-specific custom examples and achieved **70%** accuracy:

| # | Text | Expected | Predicted | Confidence | Result |
|---|------|----------|-----------|------------|--------|
| 1 | I love the new fire engine, it's so efficient! | positive | positive | 0.4892 | ✓ |
| 2 | Today's shift was exhausting with multiple emergency calls. | negative | neutral | 0.1670 | ✗ |
| 3 | The fire station just received new equipment for training. | neutral | positive | 0.2608 | ✗ |
| 4 | Terrible response time from the emergency services today. | negative | neutral | 0.1656 | ✗ |
| 5 | Grateful for the quick action of firefighters during today's incident. | positive | positive | 0.3385 | ✓ |
| 6 | The annual inspection of fire extinguishers is scheduled for next week. | neutral | neutral | 0.1668 | ✓ |
| 7 | Disappointed with the lack of resources provided to our department. | negative | negative | 0.2114 | ✓ |
| 8 | Happy to announce we've recruited five new team members this month! | positive | positive | 0.3657 | ✓ |
| 9 | The maintenance schedule for trucks has been updated. | neutral | neutral | 0.1677 | ✓ |
| 10 | Frustrated with the outdated protocols we're still using. | negative | negative | 0.1653 | ✓ |

**Performance by Sentiment Type:**
- Positive: 3/3 correct (100.0%)
- Neutral: 2/3 correct (66.7%)
- Negative: 2/4 correct (50.0%)

### Additional Domain-Specific Test Performance

A second set of 10 more specific fire brigade examples was tested, and the model achieved **40%** accuracy:

| # | Text | Expected | Predicted | Confidence | Result |
|---|------|----------|-----------|------------|--------|
| 1 | Several firefighters were injured due to roof collapse during yesterday's operation. | negative | negative | 0.2325 | ✓ |
| 2 | Our new thermal imaging cameras are making search and rescue much more effective. | positive | negative | 0.2760 | ✗ |
| 3 | Training session on fire suppression techniques will be held next Tuesday. | neutral | positive | 0.1625 | ✗ |
| 4 | Proud of our team for saving all occupants from the apartment fire last night. | positive | positive | 0.4312 | ✓ |
| 5 | Equipment malfunction prevented timely rescue operation at the factory incident. | negative | neutral | 0.1677 | ✗ |
| 6 | Department budget for next fiscal year remains unchanged from current allocation. | neutral | negative | 0.1833 | ✗ |
| 7 | Communication breakdown between units led to confusion during the emergency response. | negative | neutral | 0.2918 | ✗ |
| 8 | All stations will participate in the upcoming county-wide drill next month. | neutral | neutral | 0.1692 | ✓ |
| 9 | Impressive response time led to successful containment of the chemical spill. | positive | positive | 0.1911 | ✓ |
| 10 | Staffing shortages have left us unable to respond to multiple simultaneous calls. | negative | neutral | 0.1751 | ✗ |

**Performance by Sentiment Type on Additional Examples:**
- Positive: 2/3 correct (66.7%)
- Neutral: 1/3 correct (33.3%)
- Negative: 1/4 correct (25.0%)

**Comparison with Synthetic Model on Additional Examples:**
- Twitter model: 40.0% accuracy (4/10 correct)
- Synthetic model: 90.0% accuracy (9/10 correct)

## Analysis

### Strengths

1. **Good Performance on Positive Sentiment**: The model correctly identified most positive examples across both test sets (83.3% overall).
   
2. **Reasonable Initial Accuracy**: The model achieved 70% accuracy on the first set of domain-specific examples.

3. **Balance Across Classes in Validation**: The model maintained reasonable performance across all three sentiment categories in validation.

### Limitations

1. **Domain Adaptation Issues**: Significant drop in performance (40%) when applied to more specific fire brigade examples.

2. **Lower Confidence on Most Predictions**: The confidence scores were generally low, especially on the additional test set (avg: ~0.22).

3. **Poor Performance on Negative Sentiment**: In the additional domain-specific examples, the model only correctly identified 25% of negative sentiments.

4. **Confusion Between Neutral and Negative**: The model frequently misclassified negative statements as neutral in the fire brigade context.

5. **Technical Terminology Challenges**: The model struggled with domain-specific terminology related to fire brigade equipment and operations.

## Comparison with Synthetic Model

| Aspect | Twitter Model | Synthetic Model |
|--------|--------------|-----------------|
| Training Data | Real Twitter data | Synthetic templates |
| Validation Accuracy | 60.56% | 100% |
| Initial Test Accuracy | 70% | 60% |
| Additional Test Accuracy | 40% | 90% |
| Positive Sentiment Accuracy (Overall) | 83.3% | Not fully comparable |
| Neutral Sentiment Accuracy (Overall) | 50% | Not fully comparable |
| Negative Sentiment Accuracy (Overall) | 37.5% | Not fully comparable |

### Key Differences:

1. **Real vs. Synthetic Data**: The Twitter model was trained on real social media data rather than synthetic templates.

2. **Performance Consistency**: The synthetic model shows much more consistent performance on domain-specific examples.

3. **Domain Knowledge**: The synthetic model's 90% accuracy on the additional test set indicates stronger domain-specific knowledge.

4. **Generalization Gap**: While the Twitter model performed well on initial examples, it showed a significant performance drop on the more technically specific examples.

## Recommendations

1. **Domain-Specific Fine-Tuning**: The model should be fine-tuned on fire brigade specific examples to bridge the domain gap.

2. **Ensemble Approach**: Combining predictions from both the Twitter-based and synthetic models could leverage strengths of both.

3. **Expanded Training Data**: Collecting a domain-specific dataset for fire brigade incident reports is critical for significant improvements.

4. **Adjust Threshold for Neutrality**: Consider widening the neutral category threshold to improve classification, especially for borderline cases.

5. **Contextual Embeddings**: Implementing more advanced contextual embeddings like BERT or RoBERTa could improve understanding of domain-specific text.

6. **Active Learning Pipeline**: Implement a feedback loop where incorrect predictions are reviewed and added back to training data.

## Conclusion

The Twitter-trained sentiment analysis model provides a foundation for sentiment analysis of fire brigade reports, but shows significant limitations when faced with more technical domain-specific content. The stark difference in performance between the initial test set (70%) and the additional test set (40%) highlights the challenge of domain adaptation.

The synthetic model's superior performance (90%) on the additional test examples demonstrates the value of domain knowledge. For production use in the fire brigade application, we recommend either using the synthetic model which is already well-adapted to the domain or implementing a hybrid approach that combines the Twitter model's general sentiment understanding with domain-specific knowledge. 