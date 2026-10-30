# Restless Legs Syndrome Identification Using Machine Learning-Based Predictive Analysis


## Project overview

The project classifies records as **RLS** or **No RLS** from demographic, sleep and symptom features. It
follows five modules:

**Data Collection → Data Preprocessing → Feature Selection → Machine Learning Prediction → Performance Evaluation**

It trains and compares **Logistic Regression**, **Decision Tree** and **Random Forest** against a simple
baseline, runs an ablation study and hyperparameter tuning, writes all graphs, tables and an experiment
report, and includes a Streamlit dashboard with a prediction demonstration.

## Objectives

1. Load and validate the dataset, and report its structure, missing values and class balance.
2. Preprocess the data without leakage (all learned steps fitted on training data only).
3. Rank and select the most informative features.
4. Train and tune three classifiers, and compare them with a most-frequent baseline.
5. Evaluate them with accuracy, precision, recall, specificity, F1-score, ROC-AUC and error rates.
6. Present the results in graphs, a report and an interactive dashboard.

## Problem statement

RLS is diagnosed mainly from patient-reported symptoms, which makes consistent screening difficult. This
project explores whether standard machine learning models can separate RLS and No RLS records from structured
symptom and lifestyle data. It is a research prototype, not a clinical tool.

## Dataset

| Item | Value |
|---|---|
| File | `data/rls_dataset.csv` (kept unchanged) |
| Records / columns | 1,500 / 24 |
| Target | `RLS_Diagnosis` (0 = No RLS, 1 = RLS): 710 / 790 |
| Input features | 20 (13 numeric, 6 binary, 1 categorical) |
| Missing values | Ferritin 60, sleep quality 37; BMI, sleep duration, fatigue, anxiety 30 each |
| Excluded from inputs | `Patient_ID` (identifier), `RLS_Probability`, `RLS_Severity` (derived from the outcome) |

`data/data_dictionary_generated.csv` describes every column. If you have the original
`rls_data_dictionary.csv`, put it in `data/`; the dashboard uses it in preference.

### Dataset limitations

* The labels appear to depend mainly on the symptom variables. Feature-label relationships may be cleaner,
  or different, from those in other clinical data, which can inflate scores.
* `RLS_Severity` is 0 for every No RLS record. A model allowed to use it scores 100% accuracy, so it is excluded.
* Some records contradict their label (for example, RLS records with no core symptoms). The report quantifies this.

## Technology stack

Python 3.11+ · pandas · NumPy · scikit-learn · Matplotlib · Joblib · Streamlit · pytest.
No Docker, cloud services, paid APIs or internet access are needed after installation.

Seaborn was not used: every chart is drawn with Matplotlib so the project has one fewer dependency and the
charts were all verified when the project was built.

## Project structure

```text
RLS_ML_Project/
├── app.py                     Streamlit dashboard
├── .streamlit/config.toml     Dashboard theme
├── main.py                    Runs the whole experiment
├── requirements.txt
├── README.md
├── data/
│   ├── rls_dataset.csv
│   └── data_dictionary_generated.csv
├── src/
│   ├── config.py              Paths, column roles, feature groups, random seed
│   ├── data_loader.py         Module 1: loading and validation
│   ├── preprocessing.py       Module 2: imputation, scaling, encoding, split
│   ├── feature_selection.py   Module 4: mutual information, Random Forest ranking
│   ├── train_models.py        Module 5: pipelines, tuning, baseline, cross-validation
│   ├── evaluation.py          Metrics
│   ├── visualization.py       All graphs (EDA and results)
│   ├── prediction.py          Reusable, validated prediction function
│   ├── ui.py                  Dashboard styles and display components
│   └── report.py              Writes the Markdown reports from results
├── models/                    Saved pipelines (.joblib)
├── results/
│   ├── graphs/                27 PNG graphs
│   ├── tables/                Dataset overview, missing values, statistics, feature ranking
│   ├── metrics/               Model comparison, baseline, ablation, tuning, CV scores, test predictions
│   └── reports/               experiment_report.md, model_comparison_report.md
└── tests/test_pipeline.py     37 automated tests
```

## Installation (Windows)

Open **Command Prompt** in the project folder (in File Explorer, type `cmd` in the address bar and press Enter).

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not found, use `python -m venv .venv` with any installed Python 3.11 or newer.

**PowerShell:** activate with `.venv\Scripts\Activate.ps1`. If scripts are blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**Anaconda:** `conda create -n rls python=3.11 -y`, then `conda activate rls`, then `pip install -r requirements.txt`.

## How to run training

```bat
python main.py
```

It takes about 2–3 minutes. It prints each module's progress and recreates everything in `models/` and
`results/`. Run it once after installing, because saved models only load with the scikit-learn version that
created them.

## How to run the dashboard

```bat
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The sections are:

| Section | What it does |
|---|---|
| Overview | Best model, headline metrics and a model comparison chart |
| Methodology | The five modules, the three models, metric definitions and leakage control |
| Dataset | Filter and inspect records, column types, missing values, data dictionary |
| Exploratory analysis | Compare any feature between classes; all EDA figures |
| Model performance | Metrics, ROC curves, confusion matrices, features, baseline, ablation, tuning, CV, diagnostics |
| Threshold explorer | Move the decision threshold and watch precision, recall and the confusion matrix change |
| Try the model | Enter a record, pick any of the three models, see the probability and what drove it |
| Batch prediction | Upload a CSV of records, score them all and download the predictions |
| Downloads | Report, tables, graphs and a zip of all results | The dashboard reads saved results; if they are
missing, it tells you to run `python main.py`. Stop it with `Ctrl+C` in the terminal.

## How to run the tests

```bat
python -m pytest -q
```

## Models

* **Baseline (DummyClassifier, most frequent):** predicts the majority class for everyone. A model is only
  useful if it clearly beats this.
* **Logistic Regression:** models the log-odds of RLS as a weighted sum of the features. Numeric features
  are standardised. No class weighting, because the training split is close to balanced (52.7% RLS).
* **Decision Tree:** splits records with if-then rules. Depth, split size and leaf size are limited to stop
  it memorising the training data.
* **Random Forest:** averages many decision trees trained on bootstrap samples with random feature subsets.

Each model is a pipeline: preprocessing → `SelectKBest` (mutual information) → classifier. Grid search with
stratified 5-fold cross-validation on the training set tunes the model and the number of selected features.
The **best model is the one with the highest mean cross-validated ROC-AUC**, so the test set plays no part in
the choice. Each tuned model is then evaluated once on the held-out 20% test set.

## Evaluation metrics

| Metric | Meaning |
|---|---|
| Accuracy | Share of records classified correctly: (TP + TN) / all |
| Precision | Of records predicted RLS, the share that are RLS: TP / (TP + FP) |
| Recall (sensitivity) | Of RLS records, the share found: TP / (TP + FN) |
| Specificity | Of No RLS records, the share correctly identified: TN / (TN + FP) |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | How well predicted probabilities rank RLS above No RLS (0.5 = chance, 1 = perfect) |
| False positive rate | FP / (FP + TN) = 1 − specificity |
| False negative rate | FN / (FN + TP) = 1 − recall |

## Where to find the results

* **Experiment report:** `results/reports/experiment_report.md`, with all numbers, the discussion and limitations.
* **Model comparison:** `results/metrics/model_comparison.csv` and `results/reports/model_comparison_report.md`.
* **Baseline, ablation, tuning:** `baseline_comparison.csv`, `ablation_results.csv`, `best_parameters.csv`,
  `hyperparameter_search_summary.csv` and `default_vs_tuned.csv` in `results/metrics/`.
* **Dataset summaries:** `results/tables/`.
* **Graphs:** `results/graphs/` (01–09 exploratory analysis, 10–27 results, including learning curve,
  permutation importance and threshold analysis).

## Reproducing the results

The random seed is fixed at 42 for the split, cross-validation folds, mutual-information estimates and all
models. Running `python main.py` again on the same data gives the same results. Small differences can appear
with a different scikit-learn version.

## Limitations and ethical considerations

* The model has been evaluated on one dataset only. It would need validation on independent clinical data.
* The model is not a diagnostic tool and gives no treatment advice. It must not replace assessment by a clinician.
* Real-world use would need an ethically sourced clinical dataset, diagnoses confirmed by specialists,
  external validation at several sites and evaluation across different populations.
* The evaluation uses a single 300-record test set; scores would vary somewhat with a different split.
