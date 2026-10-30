# Model comparison (held-out test set)


Test set: 300 records. Best model by mean 5-fold cross-validated ROC-AUC on the training set: **Logistic Regression**.

| Model | Type | CV ROC-AUC (mean) | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | False positive rate | False negative rate | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Trained ML model | 0.863 | 0.817 | 0.855 | 0.785 | 0.852 | 0.818 | 0.837 | 0.148 | 0.215 | 124 | 121 | 21 | 34 |
| Decision Tree | Trained ML model | 0.837 | 0.797 | 0.809 | 0.804 | 0.789 | 0.806 | 0.842 | 0.211 | 0.196 | 127 | 112 | 30 | 31 |
| Random Forest | Trained ML model | 0.847 | 0.823 | 0.862 | 0.791 | 0.859 | 0.825 | 0.838 | 0.141 | 0.209 | 125 | 122 | 20 | 33 |
| Baseline (most frequent) | Simple baseline | n/a | 0.527 | 0.527 | 1.000 | 0.000 | 0.690 | 0.500 | 1.000 | 0.000 | 158 | 0 | 142 | 0 |

ROC-AUC uses predicted probabilities. The baseline predicts the same class for everyone, so its ROC-AUC is 0.5 by construction and its specificity is 0.
