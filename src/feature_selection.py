"""Module 4 - Feature selection.

Two pieces:
* make_selector(k): SelectKBest with mutual information. It is placed inside each
  model pipeline, so features are chosen from the training folds only, and k is
  tuned along with the model's own hyperparameters.
* rank_features(): a ranked table (mutual information + Random Forest importance)
  computed on the training set, used for the importance chart and the report.
"""
from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from . import config as C
from .preprocessing import make_preprocessor

# Fixed random_state so the mutual-information estimate is reproducible.
mi_score = partial(mutual_info_classif, random_state=C.RANDOM_STATE)


def make_selector(k="all") -> SelectKBest:
    return SelectKBest(score_func=mi_score, k=k)


def rank_features(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Score every eligible feature using training data only."""
    pre = make_preprocessor(X_train.columns, scale=True).fit(X_train)
    Xt = pd.DataFrame(pre.transform(X_train), columns=pre.get_feature_names_out(), index=X_train.index)

    mi = mi_score(Xt, y_train)
    corr = Xt.apply(lambda col: col.corr(y_train))
    rf = RandomForestClassifier(n_estimators=300, random_state=C.RANDOM_STATE, n_jobs=-1).fit(Xt, y_train)

    table = pd.DataFrame({
        "feature": Xt.columns,
        "mutual_information": mi,
        "rf_importance": rf.feature_importances_,
        "correlation_with_target": corr.values,
    }).sort_values("mutual_information", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def selected_feature_names(pipeline) -> list:
    """Names of the features a fitted pipeline's SelectKBest step kept."""
    names = pipeline.named_steps["pre"].get_feature_names_out()
    mask = pipeline.named_steps["select"].get_support()
    return [n for n, keep in zip(names, mask) if keep]
