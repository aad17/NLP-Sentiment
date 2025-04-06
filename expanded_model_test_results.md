# Expanded Sentiment Analysis Model Test Results

## Model Information
- **Model Type**: Linear Support Vector Classifier (LinearSVC)
- **Vectorizer**: TF-IDF (max_features=15000)
- **Training Data**: Combined multiple Twitter datasets

## Performance Metrics
- **Accuracy**: 0.7353
- **F1 Score (weighted)**: 0.7359

## Classification Report
```
              precision    recall  f1-score   support

    negative       0.73      0.71      0.72     21547
     neutral       0.66      0.80      0.73     14614
    positive       0.79      0.72      0.75     28657

    accuracy                           0.74     64818
   macro avg       0.73      0.74      0.73     64818
weighted avg       0.74      0.74      0.74     64818

```

## Confusion Matrix
```
[[15338  2383  3826]
 [ 1266 11756  1592]
 [ 4430  3660 20567]]
```

![Confusion Matrix](plots/expanded_model_confusion_matrix.png)

## Sample Predictions

| True Sentiment | Predicted Sentiment | Text |
|----------------|---------------------|------|
| Negative | Negative | modi said one favourite time pas routine work rally alliance sprldbsp sarab dangerous health forget ... |
| Negative | Negative | guess bank service like nirav modi sorry shudnt complaint... |
| Negative | Negative | kid choose computer bouncy castle... |
| Negative | Negative | nahi enough money bank given poor taken rich corrupt like modi malaya... |
| Negative | Negative | factual data clearly highlight modi govt poor care taker indian national security modi ji propaganda... |
| Negative | Neutral | der ppl bwood taken risk modi antimodi env tht ind modi def must back dem way... |
| Negative | Positive | country safe hand mahamilavat government modi track live update... |
| Negative | Negative | school... |
| Negative | Negative | shuck missed training... |
| Negative | Negative | modi busy touring foreign country hugging leader nothing youth india street searching job india suff... |
| Neutral | Neutral | modi chachaaa east rise kewya sun nehru west side rise krwa rha tha west sunrise hota tha... |
| Neutral | Neutral | fight... |
| Neutral | Neutral | modi must given sitting chai stall... |
| Neutral | Neutral | look like beijing conceding election modi... |
| Neutral | Neutral | see truth... |
| Neutral | Neutral | omg saying stare modi like... |
| Neutral | Neutral | wonder modi go around hugging everybody... |
| Neutral | Neutral | going call modi anti national saying remember called sidhu said thing... |
| Neutral | Neutral | modi addressed country without permission asks bahujan samajwadi party chief mayawati... |
| Neutral | Neutral | emergency declared india indira gandhi bhakts said india indira indira india thereafter indira gandh... |
| Positive | Neutral | sleeping year per claim setup modi switch year wow... |
| Positive | Positive | oh sweet... |
| Positive | Positive | job farmer wage fighting communal fascist utmost importance let le bother rahul modi fight lose btw ... |
| Positive | Positive | mmitchelldaviss love lt... |
| Positive | Positive | even today modis popularity significantly higher bjps popularity hindi heartland even karnatakabenga... |
| Positive | Positive | talking fap chop getting ready watch sexy sveta... |
| Positive | Positive | tejaswi surya clip sooo full irony apart dynasty bit also ironic call opposition shameless joker run... |
| Positive | Positive | cinnamonflower spread cheer today good day... |
| Positive | Negative | ere eat went t lol known freak uhhhhh year ago summer school cimmaron damnnnn thats hot tat hel... |
| Positive | Positive | rightexecute thanks love let know time celebrate miss yo as... |
