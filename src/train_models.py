"""Module 5 - Model building, tuning, baseline and ablation.

Every model is a Pipeline:  preprocessing -> SelectKBest(mutual information) -> classifier.
That keeps imputation, scaling and feature selection inside the training data at
every step, including inside each cross-validation fold.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .feature_selection import make_selector
from .preprocessing import make_preprocessor

RS = C.RANDOM_STATE

# Logistic Regression: lbfgs solver, max_iter raised so it always converges. No class
# weighting: the training split is 52.7% RLS / 47.3% No RLS, which is close to balanced.
# Decision Tree and Random Forest have depth / leaf limits so they cannot grow without bound.
DEFAULT_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RS),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2,
                                                    random_state=RS, n_jobs=1),
}
NEEDS_SCALING = {"Logistic Regression": True, "Decision Tree": False, "Random Forest": False}

# Small grids that run comfortably on a laptop. select__k tunes feature selection too.
PARAM_GRIDS = {
    "Logistic Regression": {
        "select__k": ["all", 10],
        "clf__C": [0.01, 0.1, 1, 10],
    },
    "Decision Tree": {
        "select__k": ["all", 10],
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [5, 20, 40],
    },
    "Random Forest": {
        "select__k": ["all", 10],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [2, 5],
        "clf__max_features": ["sqrt", 0.5],
    },
}
# The parameter each tuning graph plots on its x-axis.
TUNING_PLOT_PARAM = {"Logistic Regression": "clf__C", "Decision Tree": "clf__max_depth",
                     "Random Forest": "clf__min_samples_leaf"}


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=RS)


def build_pipeline(name: str, features=None, k="all") -> Pipeline:
    """Untrained pipeline for one of the three models with its default settings."""
    if name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(DEFAULT_MODELS)}.")
    return Pipeline([
        ("pre", make_preprocessor(features, scale=NEEDS_SCALING[name])),
        ("select", make_selector(k)),
        ("clf", DEFAULT_MODELS[name]()),
    ])


def build_baseline(features=None) -> Pipeline:
    """Always predicts the most frequent training class. Shows the floor any real model must beat."""
    return Pipeline([("pre", make_preprocessor(features, scale=False)),
                     ("clf", DummyClassifier(strategy="most_frequent"))])


def tune_model(name: str, X_train, y_train) -> GridSearchCV:
    """Grid search with stratified 5-fold CV on the training set only, scored by ROC-AUC."""
    search = GridSearchCV(build_pipeline(name), PARAM_GRIDS[name], scoring=C.SELECTION_METRIC,
                          cv=cv_splitter(), n_jobs=-1, refit=True, return_train_score=False)
    return search.fit(X_train, y_train)


def search_summary(name: str, search: GridSearchCV) -> pd.DataFrame:
    """Tidy table of every configuration tried."""
    res = pd.DataFrame(search.cv_results_)
    params = pd.json_normalize(res["params"]).rename(columns=lambda c: c.replace("clf__", "").replace("select__", "select_"))
    out = pd.concat([params, res[["mean_test_score", "std_test_score", "rank_test_score"]]], axis=1)
    out.insert(0, "model", name)
    return out.rename(columns={"mean_test_score": "cv_roc_auc_mean", "std_test_score": "cv_roc_auc_std",
                               "rank_test_score": "rank"}).sort_values("rank").reset_index(drop=True)


def cross_validation_scores(pipelines: dict, X_train, y_train) -> pd.DataFrame:
    """Per-fold accuracy, F1 and ROC-AUC for each pipeline (training data, same folds)."""
    rows = []
    for name, pipe in pipelines.items():
        cv = cross_validate(pipe, X_train, y_train, cv=cv_splitter(),
                            scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}, n_jobs=-1)
        for fold in range(C.CV_FOLDS):
            rows.append({"model": name, "fold": fold + 1, "accuracy": cv["test_accuracy"][fold],
                         "f1": cv["test_f1"][fold], "roc_auc": cv["test_roc_auc"][fold]})
    return pd.DataFrame(rows)
