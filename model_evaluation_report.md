# Emergency Services Sentiment Analysis Model Evaluation

*Report generated on: 2025-04-05 11:40:40*

## Test Data Overview

- **Total samples**: 40
- **Class distribution**:
  - Neutral: 16 samples (40.0%)
  - Positive: 12 samples (30.0%)
  - Negative: 12 samples (30.0%)

## Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | Execution Time | Samples/sec |
|-------|----------|-----------|--------|----------|----------------|--------------|
| Default Model | 0.4000 | 0.1600 | 0.4000 | 0.2286 | 0.43s | 92.14 |
| Domain-Aware Model | 0.5500 | 0.6100 | 0.5500 | 0.5437 | 1.63s | 24.47 |

## Detailed Model Analysis

### Default Model

#### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|--------|
| Negative | 0.0000 | 0.0000 | 0.0000 | 12 |
| Neutral | 0.4000 | 1.0000 | 0.5714 | 16 |
| Positive | 0.0000 | 0.0000 | 0.0000 | 12 |

#### Confusion Matrix

```
              Predicted
              Negative  Neutral  Positive
Actual Negative        0       12        0
       Neutral         0       16        0
       Positive        0       12        0
```

### Domain-Aware Model

#### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|--------|
| Negative | 0.4500 | 0.7500 | 0.5625 | 12 |
| Neutral | 0.7500 | 0.3750 | 0.5000 | 16 |
| Positive | 0.5833 | 0.5833 | 0.5833 | 12 |

#### Confusion Matrix

```
              Predicted
              Negative  Neutral  Positive
Actual Negative        9        1        2
       Neutral         7        6        3
       Positive        4        1        7
```

#### Domain Distribution

| Domain | Count | Percentage |
|--------|-------|------------|
| fire | 11 | 23.4% |
| general | 10 | 21.3% |
| ems | 8 | 17.0% |
| police | 6 | 12.8% |
| disaster_response | 6 | 12.8% |
| coast_guard | 6 | 12.8% |

## Error Analysis

This section highlights examples where models disagreed with the ground truth.

### Example 1

**Text**: The police evacuated the area in time, preventing further damage.

**Ground Truth**: Positive (Rationale: Preventative success.)

**Model Predictions**:

- Domain-Aware Model: Neutral (Confidence: 0.61) - ✗

### Example 2

**Text**: Thanks to the 911 operator who stayed on the line and kept me calm!

**Ground Truth**: Positive (Rationale: Gratitude and successful service.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.62) - ✗

### Example 3

**Text**: Community volunteers and fire crews worked together to clear the flood-hit area.

**Ground Truth**: Positive (Rationale: Collaboration and positive outcome.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.51) - ✗

### Example 4

**Text**: Police managed to de-escalate the situation peacefully.

**Ground Truth**: Positive (Rationale: Positive outcome from intervention.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.89) - ✗

### Example 5

**Text**: Emergency services distributed food and water within 24 hours of the flood.

**Ground Truth**: Positive (Rationale: Rapid relief action.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.41) - ✗

### Example 6

**Text**: Several injured after the bus collided with a police van on the highway.

**Ground Truth**: Negative (Rationale: Injury and accident highlight negativity.)

**Model Predictions**:

- Domain-Aware Model: Neutral (Confidence: 0.62) - ✗

### Example 7

**Text**: A routine fire safety drill was conducted at the community center today.

**Ground Truth**: Neutral (Rationale: Factual, informative.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.76) - ✗

### Example 8

**Text**: Reports of smoke turned out to be a false alarm. All units are returning.

**Ground Truth**: Neutral (Rationale: Just an update, no emotional tone.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.47) - ✗

### Example 9

**Text**: Emergency numbers are updated on the city’s official website.

**Ground Truth**: Neutral (Rationale: Administrative update.)

**Model Predictions**:

- Domain-Aware Model: Negative (Confidence: 0.41) - ✗

### Example 10

**Text**: Paramedics participated in a joint training session with the fire department.

**Ground Truth**: Neutral (Rationale: Non-emotional news.)

**Model Predictions**:

- Domain-Aware Model: Positive (Confidence: 0.45) - ✗

## Conclusion

Based on the evaluation, the **Domain-Aware Model** demonstrates the best overall performance with an F1 score of 0.5437 and accuracy of 0.5500.

### Key Strengths

- Strong performance on positive sentiment detection (F1: 0.5833)
- Effective domain identification with fire being the most detected domain
- Efficient processing speed (24.47 samples/sec)

### Areas for Improvement

- Further improvements could be made for neutral sentiment detection (F1: 0.5000)

### Recommendations

1. Continue to enhance training data for the underperforming sentiment classes
2. Consider ensemble approaches that leverage the strengths of multiple models
3. Focus on improving domain detection for more accurate context-aware sentiment analysis
