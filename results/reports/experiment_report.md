# Experiment report: RLS identification with machine learning

Generated 2026-10-30T13:41:10 · Python 3.14.3 · scikit-learn 1.9.1 · random seed 42 · runtime 45.4 s

## 1. Experimental setup

The pipeline follows five modules: Data Collection → Data Preprocessing → Feature Selection → Machine Learning Prediction → Performance Evaluation. The data were split once into a stratified training set (1200 records) and test set (300 records). All imputation, scaling and feature selection were fitted on the training set only, inside scikit-learn pipelines. Hyperparameters were chosen by 5-fold stratified cross-validation on the training set, and each tuned model was evaluated on the test set once.

## 2. Dataset description

The file `rls_dataset.csv` has 1500 records and 24 columns: 790 labelled RLS (52.7%) and 710 labelled No RLS. There are 20 eligible input features covering demographics, sleep, wellbeing, a ferritin value, family history, five core RLS symptoms, a leg-discomfort score and symptom frequency. Missing values: Ferritin_Level_ng_mL (60), Sleep_Quality (37), Sleep_Duration_Hours (30), BMI (30), Anxiety_Level (30), Fatigue_Level (30). Duplicate rows: 0.

Excluded from the inputs: `Patient_ID` (identifier), `RLS_Probability` and `RLS_Severity` (both derived from the outcome). `RLS_Severity` is 0 for every No RLS record; a Random Forest that is allowed to use it reaches 100.0% test accuracy, which shows why it must be excluded.

## 3. Preprocessing steps

- Numerical features: median imputation, then standardisation (Logistic Regression only; trees are unaffected by scale).
- Binary symptom and history flags: most-frequent imputation, kept as 0/1.
- Gender: most-frequent imputation, one-hot encoding with unknown categories ignored.
- Feature selection: `SelectKBest` with mutual information inside each pipeline; the number of features (all 20 or the top 10) was tuned with the model.
- Logistic Regression uses no class weighting because the training split is close to balanced (52.7% RLS).

Feature ranking on the training set (top 7 by mutual information): Symptom Frequency Days Week, Unpleasant Leg Sensation, Urge to Move Legs, Symptoms While Resting, Leg Discomfort Score, Symptoms Improve With Movement, Symptoms Worse Evening Night. Features with almost no association (MI < 0.005): Fatigue Level, Ferritin Level ng mL, Family History RLS, BMI, Sleep Duration Hours, Physical Activity Level, Anxiety Level, Gender Male.

| rank | feature | mutual_information | rf_importance | correlation_with_target |
|---|---|---|---|---|
| 1 | Symptom_Frequency_Days_Week | 0.1996 | 0.1455 | 0.5327 |
| 2 | Unpleasant_Leg_Sensation | 0.1834 | 0.0933 | 0.5666 |
| 3 | Urge_to_Move_Legs | 0.1513 | 0.0774 | 0.5136 |
| 4 | Symptoms_While_Resting | 0.1194 | 0.0416 | 0.4521 |
| 5 | Leg_Discomfort_Score | 0.1183 | 0.0572 | 0.4501 |
| 6 | Symptoms_Improve_With_Movement | 0.0836 | 0.0263 | 0.4112 |
| 7 | Symptoms_Worse_Evening_Night | 0.0815 | 0.0348 | 0.4340 |
| 8 | Sleep_Quality | 0.0139 | 0.0521 | 0.0416 |
| 9 | Daytime_Sleepiness | 0.0131 | 0.0437 | -0.0189 |
| 10 | Sleep_Disturbance | 0.0125 | 0.0317 | 0.2121 |
| 11 | Age | 0.0093 | 0.0473 | -0.0095 |
| 12 | Stress_Level | 0.0093 | 0.0472 | -0.0426 |

## 4. Models evaluated

- **Baseline:** `DummyClassifier(strategy='most_frequent')`, which predicts the majority class for everyone.
- **Logistic Regression:** linear model of the log-odds of RLS (lbfgs solver, up to 1,000 iterations).
- **Decision Tree:** splits chosen by Gini impurity, with limits on depth, split size and leaf size.
- **Random Forest:** an ensemble of decision trees on bootstrap samples, majority vote.

Search space used for tuning:
- **Logistic Regression:** `feature_selection_k` in ['all', 10]; `C` in [0.01, 0.1, 1, 10]
- **Decision Tree:** `feature_selection_k` in ['all', 10]; `max_depth` in [3, 5, 8]; `min_samples_split` in [2, 10, 20]; `min_samples_leaf` in [5, 20, 40]
- **Random Forest:** `feature_selection_k` in ['all', 10]; `n_estimators` in [100, 200]; `max_depth` in [6, 10, None]; `min_samples_leaf` in [2, 5]; `max_features` in ['sqrt', 0.5]

## 5. Performance results (test set)

Selection criterion: highest mean cross-validated ROC-AUC on the training set. Selected model: **Logistic Regression** (CV ROC-AUC 0.863 ± 0.017). On the test set it reached ROC-AUC 0.837, accuracy 81.7%, precision 85.5%, recall 78.5%, specificity 85.2% and F1-score 0.818.

| Model | Type | CV ROC-AUC (mean) | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | False positive rate | False negative rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Trained ML model | 0.863 | 0.817 | 0.855 | 0.785 | 0.852 | 0.818 | 0.837 | 0.148 | 0.215 |
| Decision Tree | Trained ML model | 0.837 | 0.797 | 0.809 | 0.804 | 0.789 | 0.806 | 0.842 | 0.211 | 0.196 |
| Random Forest | Trained ML model | 0.847 | 0.823 | 0.862 | 0.791 | 0.859 | 0.825 | 0.838 | 0.141 | 0.209 |
| Baseline (most frequent) | Simple baseline | n/a | 0.527 | 0.527 | 1.000 | 0.000 | 0.690 | 0.500 | 1.000 | 0.000 |

Cross-validation scores of the tuned models on the training folds (mean and standard deviation):

| Model | accuracy mean | accuracy std | f1 mean | f1 std | roc_auc mean | roc_auc std |
|---|---|---|---|---|---|---|
| Decision Tree | 0.803 | 0.016 | 0.807 | 0.018 | 0.837 | 0.024 |
| Logistic Regression | 0.817 | 0.017 | 0.823 | 0.019 | 0.863 | 0.017 |
| Random Forest | 0.817 | 0.019 | 0.823 | 0.020 | 0.847 | 0.023 |

Graphs: `results/graphs/10-12_confusion_matrix_*.png`, `13_roc_curves.png`, `14_accuracy_comparison.png`, `15_precision_recall_f1.png`, `17_cross_validation_comparison.png`, `18_precision_recall_curves.png`, `19_calibration_curves.png`.

## 6. Baseline comparison

The baseline predicts RLS for every record, so its accuracy (52.7%) equals the share of that class in the test set and its ROC-AUC is 0.5. Every trained model beats it clearly; the selected model improves accuracy by 29.0 percentage points.

| Model | Type | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | Trained ML model | 0.817 | 0.855 | 0.785 | 0.818 | 0.837 |
| Decision Tree | Trained ML model | 0.797 | 0.809 | 0.804 | 0.806 | 0.842 |
| Random Forest | Trained ML model | 0.823 | 0.862 | 0.791 | 0.825 | 0.838 |
| Baseline (most frequent) | Simple baseline | 0.527 | 0.527 | 1.000 | 0.690 | 0.500 |

## 7. Ablation study

A Random Forest with default settings was trained on each feature group using the same split and preprocessing. These results describe how the groups behave in this dataset, not their clinical importance.

| Feature group | Number of features | CV ROC-AUC (mean) | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| A. Demographic and general | 9 | 0.497 | 0.493 | 0.515 | 0.658 | 0.578 | 0.503 |
| B. Sleep-related | 4 | 0.581 | 0.527 | 0.544 | 0.627 | 0.582 | 0.571 |
| C. RLS symptom-related | 7 | 0.839 | 0.807 | 0.857 | 0.759 | 0.805 | 0.846 |
| D. All eligible features | 20 | 0.841 | 0.823 | 0.857 | 0.797 | 0.826 | 0.849 |

Of the single groups, **C. RLS symptom-related** was strongest (test ROC-AUC 0.846). All features combined reached 0.849, symptom-related features alone 0.846, and demographic and general variables 0.503, which is close to chance (0.5). This suggests that the labels depend mainly on the symptoms: almost all the signal sits in the symptom variables.

## 8. Hyperparameter tuning

| model | C | feature_selection_k | cv_roc_auc_mean | cv_roc_auc_std | configurations_tried | max_depth | min_samples_leaf | min_samples_split | max_features | n_estimators |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 10 | 10 | 0.863 | 0.017 | 8 | n/a | n/a | n/a | n/a | n/a |
| Decision Tree | n/a | 10 | 0.837 | 0.024 | 54 | 8 | 20 | 2 | n/a | n/a |
| Random Forest | n/a | all | 0.847 | 0.023 | 48 | 6 | 5 | n/a | sqrt | 100 |

Default versus tuned settings (test ROC-AUC):

| Model | Default | Tuned | Change |
|---|---|---|---|
| Decision Tree | 0.832 | 0.842 | 0.010 |
| Logistic Regression | 0.859 | 0.837 | -0.022 |
| Random Forest | 0.849 | 0.838 | -0.011 |

Tuning changed test ROC-AUC by between -0.022 and +0.010. Changes this small are within the variation expected from a single 300-record test set, so tuning should be read as confirming reasonable settings rather than as a clear improvement.

Full search results: `results/metrics/hyperparameter_search_summary.csv`. Graphs: `20_hyperparameter_tuning.png`, `21_before_after_tuning.png`.

## 9. Discussion of findings

Logistic Regression was selected, with a cross-validated ROC-AUC 0.016 above Random Forest. On the test set the three models were close (ROC-AUC 0.837 to 0.842): Random Forest had the highest test accuracy and F1-score, and Decision Tree the highest test ROC-AUC. The selected model is therefore not the best on every test metric. The choice was made on cross-validated ROC-AUC so that the test set did not influence it, and test differences of this size are within the noise of a 300-record test set. 

Learning curve: trained on full cross-validation folds (960 records each), the selected model scores ROC-AUC 0.866 on the training folds and 0.863 on the validation folds. A gap this small means the model is not overfitting, and more data of the same kind is unlikely to change it much (`25_learning_curve.png`). Permutation importance on the test set ranks Symptoms Improve With Movement (0.009), Symptom Frequency Days Week (0.007), Symptoms Worse Evening Night (0.007) as the most influential inputs (`26_permutation_importance.png`). `27_threshold_analysis.png` shows how moving the 0.5 decision threshold trades recall against specificity.

A label-consistency check explains part of the remaining error: 111 records labelled RLS report none of the five core symptoms, and 83 records labelled No RLS report three or more. That is 194 records (12.9% of the data) whose labels contradict their symptoms, so a model using these features will usually misclassify them. See `24_symptom_count_vs_label.png`.

| core_symptoms_reported | No RLS | RLS |
|---|---|---|
| 0 | 480 | 111 |
| 1 | 114 | 44 |
| 2 | 33 | 66 |
| 3 | 49 | 214 |
| 4 | 25 | 249 |
| 5 | 9 | 106 |

Because the labels depend mainly on the symptom variables, the relationships between features and labels may be cleaner or simpler than in other clinical data. Performance on a new population may therefore be lower, or differently distributed.

## 10. Limitations and conclusion

- The model was evaluated on a single dataset; it needs validation on independent clinical data.
- One train/test split of 300 test records was used; scores would vary somewhat with a different split.
- The labels contain records that contradict the symptoms, which limits achievable accuracy.
- Real-world validation would need an ethically sourced clinical dataset with diagnoses confirmed by specialists, and evaluation across different sites and populations.
- The system is a teaching prototype and must not replace professional medical assessment.

**Conclusion.** On this dataset, Logistic Regression gave the best cross-validated discrimination (ROC-AUC 0.863) and a test ROC-AUC of 0.837, well above the most-frequent baseline. Symptom-related features carried most of the signal. These findings describe model behaviour on this dataset only.
