"""Module 5 (part) - Performance evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC",
                  "False positive rate", "False negative rate"]


def _ratio(num, den):
    """num / den, or NaN when the denominator is zero (metric undefined)."""
    return float(num) / float(den) if den else float("nan")


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """All requested metrics. ROC-AUC uses probabilities, never predicted labels.

    Precision/recall/F1 use zero_division=0, so a model that never predicts RLS
    gets 0 rather than an error. ROC-AUC is NaN if only one class is present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float("nan")
    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": _ratio(tn, tn + fp),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": auc,
        "False positive rate": _ratio(fp, fp + tn),
        "False negative rate": _ratio(fn, fn + tp),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def evaluate(pipeline, X_test, y_test) -> tuple:
    """Predict the test set; return (metrics dict, predicted labels, positive-class probabilities)."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    return compute_metrics(y_test, y_pred, y_prob), y_pred, y_prob


def metrics_table(rows: dict) -> pd.DataFrame:
    """{model name: metrics dict} -> tidy DataFrame with one row per model."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Model"
    return df.reset_index()


def metrics_at_threshold(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Metrics when a record is called RLS if its probability is at or above `threshold`."""
    y_prob = np.asarray(y_prob, dtype=float)
    return compute_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision, recall, specificity, F1 and accuracy across a range of decision thresholds."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, y_prob, t)
        rows.append({"threshold": float(t), **{k: m[k] for k in ["Accuracy", "Precision", "Recall", "Specificity",
                                                               "F1-score", "TP", "TN", "FP", "FN"]}})
    return pd.DataFrame(rows)
