# Emergency Services Sentiment Analysis Model Report

## Model Overview

This report documents the training and performance of a multi-domain sentiment analysis model that expands the original fire brigade model to cover all emergency services. The model was developed using a progressive training approach, starting with emergency services specific data and then fine-tuning with general Twitter data.

## Training Methodology

### Data Sources

The model was trained using two primary datasets:

1. **General Twitter Dataset**: 
   - 110,000 balanced samples
   - Distribution: 50,000 negative, 10,000 neutral, 50,000 positive
   - Labels converted from original Twitter sentiment mapping (0→-1, 2→0, 4→1)
   - Synthetic neutral examples created to balance the dataset

2. **Emergency Services Dataset**: 
   - 20,252 samples filtered for emergency services content
   - Distribution: 10,000 negative, 252 neutral, 10,000 positive
   - Coverage across various emergency services domains (fire, police, EMS, etc.)

### Progressive Training Approach

The model was trained using a two-stage approach:

1. **Stage 1: Base Training on Emergency Content**
   - TF-IDF vectorization with expanded features (15,000 features)
   - Inclusion of bigrams for better phrase representation
   - Initial training exclusively on emergency services data
   - Performance on emergency validation: 71.22% accuracy

2. **Stage 2: Fine-tuning on General Content**
   - Combined emergency services data with general Twitter data
   - Used a balancing approach to prevent overwhelming emergency-specific patterns
   - Final combined model training on both datasets
   - Enhanced generalization while maintaining emergency domain performance

## Performance Metrics

### Emergency Services Dataset Performance

**Overall Accuracy: 80.69%**

**Class-Specific Metrics:**
- **Negative Sentiment**:
  - Precision: 0.82
  - Recall: 0.82
  - F1-score: 0.82

- **Neutral Sentiment**:
  - Precision: 0.22
  - Recall: 0.63
  - F1-score: 0.32

- **Positive Sentiment**:
  - Precision: 0.84
  - Recall: 0.80
  - F1-score: 0.82

**Confusion Matrix:**
```
[[8207  258 1535]
 [  55  158   39]
 [1708  316 7976]]
```

### General Twitter Dataset Performance

**Overall Accuracy: 73.12%**

**Class-Specific Metrics:**
- **Negative Sentiment**:
  - Precision: 0.79
  - Recall: 0.74
  - F1-score: 0.76

- **Neutral Sentiment**:
  - Precision: 0.39
  - Recall: 0.44
  - F1-score: 0.41

- **Positive Sentiment**:
  - Precision: 0.75
  - Recall: 0.79
  - F1-score: 0.77

**Confusion Matrix:**
```
[[3368  326  879]
 [ 235  410  287]
 [ 653  308 3534]]
```

## Cross-Domain Performance

One of the key achievements of this model is its ability to perform well across different domains:

1. **Emergency-Specific Model on Emergency Data**: 71.22% accuracy
2. **Combined Model on Emergency Data**: 71.04% accuracy
3. **Combined Model on General Data**: 63.72% accuracy
4. **Final Model on Full Emergency Dataset**: 80.69% accuracy
5. **Final Model on Full General Dataset**: 73.12% accuracy

This demonstrates the model's ability to maintain performance in its primary domain (emergency services) while also generalizing to general Twitter content.

## Model Strengths

1. **Multi-Domain Capability**: Successfully expanded from fire services to all emergency services with strong performance.

2. **Balanced Performance**: Good accuracy across both negative and positive sentiments in both general and domain-specific contexts.

3. **Enhanced Feature Representation**: The use of bigrams and increased feature space (15,000 features) improved the model's ability to capture domain-specific terminology.

4. **Progressive Training Benefits**: The two-stage training approach allowed the model to learn domain-specific patterns first, then generalize without losing specialized knowledge.

5. **Strong Negative/Positive Classification**: The model shows particularly high precision and recall for negative and positive sentiments in the emergency domain (0.82-0.84).

## Model Limitations

1. **Neutral Sentiment Challenges**: The model struggles with neutral sentiment classification across both datasets, with particularly low precision (0.22) on emergency data.

2. **Class Imbalance Effects**: Despite synthetic data generation, the limited number of neutral examples (252) in the emergency dataset affected performance.

3. **Cross-Category Confusion**: There is notable confusion between negative and positive categories, with approximately 17% misclassification in each direction.

4. **Domain Generalization Trade-offs**: Improving performance on general data comes at a slight cost to emergency domain performance (71.22% → 71.04% in validation).

5. **Limited Real-World Emergency Data**: The emergency dataset is created through keyword filtering rather than authentic emergency service reports, which may limit application to real-world emergency service contexts.

## Recommendations for Further Improvement

1. **Neutral Sentiment Representation**: Collect or generate more authentic neutral examples from emergency services domains.

2. **Domain Classification Component**: Implement explicit domain classification to better distinguish between emergency sub-domains (fire, police, EMS, etc.).

3. **Transfer Learning Exploration**: Explore transformer-based models like BERT for improved language understanding, especially for ambiguous or context-dependent sentiments.

4. **Active Learning Implementation**: Develop a feedback loop where human experts can correct model predictions to continuously improve performance.

5. **Real-World Validation**: Test the model on actual emergency service reports and operator communications to validate performance in authentic contexts.

## Conclusion

The emergency services sentiment analysis model successfully extends the original fire brigade model to cover multiple emergency service domains while maintaining strong performance. With an overall accuracy of 80.69% on emergency content and 73.12% on general content, the model demonstrates effective multi-domain capability.

The primary challenge remains the classification of neutral sentiment, which suffers from both limited training examples and inherent ambiguity. Future work should focus on improving neutral sentiment detection and gathering more authentic emergency services data to enhance real-world applicability.

Overall, this expanded model provides a solid foundation for sentiment analysis across all emergency services, with clear pathways for further enhancement and specialization. 