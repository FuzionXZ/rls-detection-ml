"""Reusable prediction function for the saved model.

Outputs are estimates from a research model. They are not medical assessments
and must not be used to diagnose anyone.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C

DISCLAIMER = ("Research demonstration only. This output comes from a machine learning model "
              "and is not a medical diagnosis or assessment.")


class PredictionInputError(ValueError):
    """Raised when prediction input is missing fields or has invalid values."""


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model can be loaded."""


def load_model(path: Path = C.BEST_MODEL_PATH) -> dict:
    """Load the saved bundle {'pipeline', 'model_name', 'features', ...}."""
    path = Path(path)
    if not path.exists():
        raise ModelNotAvailableError(f"No trained model at '{path}'. Run 'python main.py' first to train it.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # usually a scikit-learn version mismatch
        raise ModelNotAvailableError(
            f"Could not load '{path.name}' ({exc}). It was probably saved with a different scikit-learn "
            "version. Run 'python main.py' to retrain it on this computer.") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelNotAvailableError(f"'{path.name}' is not a model bundle created by main.py.")
    return bundle


def validate_input(data) -> pd.DataFrame:
    """Turn a dict / list of dicts / DataFrame into a checked DataFrame of model features.

    Missing numeric values (None/NaN) are allowed: the pipeline imputes them. Any
    column that is not a model input (ID, target, leakage columns) is ignored.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise PredictionInputError("Input must be a dict, a list of dicts, or a pandas DataFrame.")
    if df.empty:
        raise PredictionInputError("Input contains no records.")

    missing = [c for c in C.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise PredictionInputError(f"Missing required fields: {missing}")

    errors = []
    for col in C.NUMERIC_FEATURES + C.BINARY_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        bad_type = df[col].notna() & values.isna()
        if bad_type.any():
            errors.append(f"{col} must be a number.")
            continue
        df[col] = values
    if errors:  # stop before range checks, which need numbers
        raise PredictionInputError("Invalid input: " + " ".join(errors))
    for col, (lo, hi) in C.VALUE_RANGES.items():
        if col in df and ((df[col] < lo) | (df[col] > hi)).any():
            errors.append(f"{col} must be between {lo} and {hi}.")
    for col in C.BINARY_FEATURES:
        if (df[col].notna() & ~df[col].isin([0, 1])).any():
            errors.append(f"{col} must be 0 or 1.")
    for col, allowed in C.ALLOWED_CATEGORIES.items():
        if (df[col].notna() & ~df[col].isin(allowed)).any():
            errors.append(f"{col} must be one of {allowed}.")
    if errors:
        raise PredictionInputError("Invalid input: " + " ".join(errors))
    return df[C.MODEL_FEATURES]


def predict(data, bundle: dict = None) -> pd.DataFrame:
    """Predict class and probability for one or more records using the saved pipeline."""
    bundle = bundle or load_model()
    X = validate_input(data)
    pipe = bundle["pipeline"]
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else np.full(len(X), np.nan)
    return pd.DataFrame({
        "predicted_label": pred.astype(int),
        "predicted_class": [C.CLASS_LABELS[int(p)] for p in pred],
        "probability_rls": np.round(prob, 4),
        "model": bundle.get("model_name", "unknown"),
        "note": "Research demonstration, not a diagnosis",
    })


def load_all_models() -> dict:
    """Every tuned model that exists on disk, as {name: bundle}. Missing or unreadable files are skipped."""
    bundles = {}
    for name in C.MODEL_NAMES:
        path = C.model_file(name, tuned=True)
        if not path.exists():
            continue
        try:
            bundles[name] = {"pipeline": joblib.load(path), "model_name": name}
        except Exception:  # version mismatch: the dashboard tells the user to retrain
            continue
    return bundles


def template_csv() -> str:
    """Empty CSV with the required columns, for the batch-prediction upload."""
    return ",".join([C.ID_COLUMN] + C.MODEL_FEATURES) + "\n"
