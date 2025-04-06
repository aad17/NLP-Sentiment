# Expanded Sentiment Model Test Results

## Custom Examples Results

**Overall Accuracy**: 0.60

### Domain-specific Accuracy

| Domain | Accuracy |
|--------|----------|
| Emergency | 0.00 |
| Tech | 0.60 |
| Restaurant | 0.60 |
| Travel | 0.80 |
| Social Media | 1.00 |

### Sample Predictions

| Domain | Text | Expected | Predicted | Correct |
|--------|------|----------|-----------|--------|
| Emergency | The ambulance arrived quickly and saved my life... | positive | neutral | ✗ |
| Emergency | Had to wait 30 minutes for the ambulance to arrive... | negative | neutral | ✗ |
| Emergency | Fire department response was professional... | positive | neutral | ✗ |
| Emergency | The 911 dispatcher was rude and unhelpful... | negative | positive | ✗ |
| Emergency | Police officer helped me find my lost child... | positive | neutral | ✗ |
| Tech | This new smartphone has amazing battery life... | positive | positive | ✓ |
| Tech | The laptop keeps crashing whenever I open multiple... | negative | negative | ✓ |
| Tech | I bought a standard mid-range TV for my living roo... | neutral | negative | ✗ |
| Tech | This software update broke all my previously worki... | negative | negative | ✓ |
| Tech | The camera quality on this phone exceeds expectati... | positive | neutral | ✗ |
| Restaurant | The food was delicious and the service was excelle... | positive | positive | ✓ |
| Restaurant | Our waiter was very attentive and friendly... | positive | positive | ✓ |
| Restaurant | I ordered a hamburger with fries for lunch today... | neutral | positive | ✗ |
| Restaurant | The restaurant was dirty and the food was cold... | negative | negative | ✓ |
| Restaurant | We waited over an hour for our food to arrive... | negative | neutral | ✗ |
| Travel | The hotel room was spacious and clean with a beaut... | positive | positive | ✓ |
| Travel | My flight was delayed by three hours with no expla... | negative | negative | ✓ |
| Travel | I took the train from New York to Washington DC... | neutral | negative | ✗ |
| Travel | The beach was crowded but the weather was perfect... | positive | positive | ✓ |
| Travel | The taxi driver took a longer route to increase th... | negative | negative | ✓ |
| Social Media | Can't wait for the weekend to start!... | positive | positive | ✓ |
| Social Media | Just found out my exam was postponed again... | negative | negative | ✓ |
| Social Media | Going to the grocery store to pick up some milk... | neutral | neutral | ✓ |
| Social Media | So upset about the election results... | negative | negative | ✓ |
| Social Media | My new puppy is the cutest thing ever!... | positive | positive | ✓ |

## Emergency Services Data Results

**Accuracy**: 0.4750

### Classification Report

```
              precision    recall  f1-score   support

    negative       0.47      0.58      0.52        12
     neutral       0.38      0.38      0.38        16
    positive       0.67      0.50      0.57        12

    accuracy                           0.48        40
   macro avg       0.50      0.49      0.49        40
weighted avg       0.49      0.47      0.48        40
```

### Confusion Matrix

```
[[7 5 0]
 [7 6 3]
 [1 5 6]]```
