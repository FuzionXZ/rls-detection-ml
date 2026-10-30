"""Module 1 - Data loading and validation.

load_dataset() reads the CSV and refuses to continue if the file is missing, the
target is absent or invalid, or required columns are missing. dataset_report()
builds the summary tables that main.py saves to results/tables/.
"""
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as C


class DataValidationError(Exception):
    """Raised when the dataset cannot be used safely."""


def load_dataset(path: Path = C.DATASET_PATH) -> pd.DataFrame:
    """Load the CSV and run the blocking validation checks."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(
            f"Dataset not found at '{path}'. Place rls_dataset.csv in the data/ folder.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV, wrong encoding, etc.
        raise DataValidationError(f"Could not read '{path.name}' as CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"'{path.name}' contains no rows.")
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    """Raise DataValidationError for problems that make training unsafe.

    Returns a list of non-blocking warnings (unexpected values, duplicates) so the
    caller can report them without stopping.
    """
    missing = [c for c in C.EXPECTED_COLUMNS if c not in df.columns]
    if C.TARGET in missing:
        raise DataValidationError(f"Target column '{C.TARGET}' is missing.")
    if missing:
        raise DataValidationError(f"Expected columns are missing: {missing}")

    target = df[C.TARGET]
    if target.isna().any():
        raise DataValidationError(f"Target '{C.TARGET}' has {int(target.isna().sum())} missing values.")
    bad = sorted(set(target.unique()) - {0, 1})
    if bad:
        raise DataValidationError(f"Target '{C.TARGET}' must contain only 0 and 1; found {bad}.")
    if target.nunique() < 2:
        raise DataValidationError("Target contains only one class, so a classifier cannot be trained.")

    for col in C.NUMERIC_FEATURES + C.BINARY_FEATURES:
        if not is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' should be numeric but has type {df[col].dtype}.")

    return find_warnings(df)


def find_warnings(df: pd.DataFrame) -> list:
    """Non-blocking issues: out-of-range values, unknown categories, duplicates."""
    warnings = []
    for col, (lo, hi) in C.VALUE_RANGES.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            warnings.append(f"{col}: {n} value(s) outside the expected range {lo}-{hi}.")
    for col in C.BINARY_FEATURES:
        n = int((~df[col].isin([0, 1]) & df[col].notna()).sum())
        if n:
            warnings.append(f"{col}: {n} value(s) other than 0/1.")
    for col, allowed in C.ALLOWED_CATEGORIES.items():
        unknown = sorted(set(df[col].dropna().unique()) - set(allowed))
        if unknown:
            warnings.append(f"{col}: unexpected categories {unknown} (handled as unknown by the encoder).")
    dup_rows = int(df.duplicated().sum())
    dup_content = int(df.drop(columns=[C.ID_COLUMN]).duplicated().sum())
    dup_ids = int(df[C.ID_COLUMN].duplicated().sum())
    if dup_rows or dup_content or dup_ids:
        warnings.append(f"Duplicates: {dup_rows} identical rows, {dup_content} rows identical apart from ID, "
                        f"{dup_ids} repeated IDs.")
    return warnings


def dataset_report(df: pd.DataFrame) -> dict:
    """Summary tables for the dataset (overview, missing values, target, statistics)."""
    n = len(df)
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_missing": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "unique_values": df.nunique().values,
        "role": ["target" if c == C.TARGET else "identifier (excluded)" if c == C.ID_COLUMN
                 else "leakage (excluded)" if c in C.LEAKAGE_COLUMNS else "input feature" for c in df.columns],
    })
    missing = (overview[["column", "missing"]].assign(missing_percent=lambda t: (t["missing"] / n * 100).round(2))
               .sort_values("missing", ascending=False).reset_index(drop=True))
    counts = df[C.TARGET].value_counts().sort_index()
    target = pd.DataFrame({
        "label": counts.index, "class": [C.CLASS_LABELS[i] for i in counts.index],
        "count": counts.values, "percent": (counts.values / n * 100).round(2),
    })
    stats = df.drop(columns=[C.ID_COLUMN]).describe(include="all").T
    stats.index.name = "column"
    summary = {
        "rows": n, "columns": df.shape[1], "input_features": len(C.MODEL_FEATURES),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "rls": int(counts.get(1, 0)), "no_rls": int(counts.get(0, 0)),
    }
    return {"overview": overview, "missing": missing, "target": target, "stats": stats, "summary": summary}


def load_data_dictionary() -> pd.DataFrame:
    """Prefer the supplied dictionary; fall back to the one generated for this project."""
    for path in (C.SUPPLIED_DICTIONARY_PATH, C.GENERATED_DICTIONARY_PATH):
        if path.exists():
            d = pd.read_csv(path)
            d.attrs["source"] = path.name
            return d
    return pd.DataFrame()
