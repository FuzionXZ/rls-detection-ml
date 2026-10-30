"""Module 2 - Data preprocessing.

Decisions (also documented in the README and the experiment report):
* Patient_ID, RLS_Probability and RLS_Severity are dropped from the inputs.
  RLS_Severity is 0 for every No RLS record, so it would give away the answer.
* Numeric features: median imputation (robust to skew, e.g. ferritin), then
  standardisation (z-score) when scale=True. Logistic Regression needs scaling;
  trees do not, so tree pipelines skip it.
* Binary 0/1 features: most-frequent imputation, no scaling.
* Gender: most-frequent imputation, then one-hot encoding with
  handle_unknown="ignore" so an unseen category cannot crash prediction.
* Everything sits inside a scikit-learn Pipeline, so the imputation values and
  scaling statistics are learned from the training rows only.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def split_features_target(df: pd.DataFrame, features=None):
    """Return X (eligible inputs only) and y (target). Excluded columns never reach X."""
    features = list(C.MODEL_FEATURES if features is None else features)
    leaked = [f for f in features if f in C.EXCLUDED_COLUMNS]
    if leaked:
        raise ValueError(f"These columns may not be used as inputs: {leaked}")
    return df[features].copy(), df[C.TARGET].astype(int).copy()


def stratified_split(X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE):
    """80/20 split that keeps the RLS / No RLS ratio the same in both parts."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def make_preprocessor(features=None, scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer for whichever subset of features a pipeline uses."""
    features = list(C.MODEL_FEATURES if features is None else features)
    numeric = [f for f in C.NUMERIC_FEATURES if f in features]
    binary = [f for f in C.BINARY_FEATURES if f in features]
    categorical = [f for f in C.CATEGORICAL_FEATURES if f in features]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    blocks = []
    if numeric:
        blocks.append(("num", Pipeline(num_steps), numeric))
    if binary:
        blocks.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]), categorical))
    return ColumnTransformer(blocks, verbose_feature_names_out=False)
