"""Run the full experiment end to end.

    python main.py

Data Collection -> Data Preprocessing -> Feature Selection -> Machine Learning
Prediction -> Performance Evaluation. Every number written to results/ is computed
here from the supplied dataset; nothing is hard-coded.
"""
import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from src import config as C
from src import visualization as viz
from src.data_loader import DataValidationError, dataset_report, load_dataset, validate_dataset
from src.evaluation import METRIC_COLUMNS, evaluate, metrics_table, threshold_sweep
from src.feature_selection import rank_features, selected_feature_names
from src.preprocessing import split_features_target, stratified_split
from src.report import write_experiment_report, write_model_comparison_report
from src.train_models import (PARAM_GRIDS, TUNING_PLOT_PARAM, build_baseline, build_pipeline,
                              cross_validation_scores, cv_splitter, search_summary, tune_model)


def step(msg):
    print(f"\n=== {msg}")


def main():
    t0 = time.time()
    C.ensure_output_dirs()
    run = {"started": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
           "scikit_learn": sklearn.__version__, "random_state": C.RANDOM_STATE}

    # ------------------------------------------------ Module 1: data loading and validation
    step("Module 1: Data loading and validation")
    try:
        df = load_dataset()
    except DataValidationError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    warnings = validate_dataset(df)
    rep = dataset_report(df)
    s = rep["summary"]
    print(f"Rows: {s['rows']}  Columns: {s['columns']}  Missing cells: {s['missing_cells']}  "
          f"Duplicate rows: {s['duplicate_rows']}")
    print("Column types:\n" + rep["overview"][["column", "dtype", "missing", "role"]].to_string(index=False))
    print("Target distribution:\n" + rep["target"].to_string(index=False))
    print("Warnings: " + ("; ".join(warnings) if warnings else "none"))
    rep["overview"].to_csv(C.TABLES_DIR / "dataset_overview.csv", index=False)
    rep["missing"].to_csv(C.TABLES_DIR / "missing_values.csv", index=False)
    rep["target"].to_csv(C.TABLES_DIR / "target_distribution.csv", index=False)
    rep["stats"].to_csv(C.TABLES_DIR / "descriptive_statistics.csv")
    run["dataset"] = s
    run["validation_warnings"] = warnings

    # ------------------------------------------------ Module 3: exploratory analysis
    step("Module 3: Exploratory data analysis graphs")
    for fn in (viz.class_distribution, viz.age_distribution, viz.gender_distribution,
               viz.sleep_duration_distribution, viz.sleep_quality_by_class, viz.symptom_frequency_by_class,
               viz.correlation_heatmap, viz.symptom_comparison):
        print("  saved", fn(df))
    print("  saved", viz.missing_values(rep["missing"]))

    # Label-consistency check: how many core symptoms does each record report, per label?
    n_symptoms = df[C.SYMPTOM_COLUMNS].sum(axis=1)
    sym_tab = pd.crosstab(n_symptoms, df[C.TARGET]).reindex(range(6), fill_value=0)
    sym_tab.index.name = "core_symptoms_reported"
    sym_tab.rename(columns=C.CLASS_LABELS).to_csv(C.TABLES_DIR / "symptom_count_vs_label.csv")
    print("  saved", viz.symptom_count_vs_label(sym_tab))
    run["label_consistency"] = {
        "rls_with_no_core_symptoms": int(sym_tab.loc[0, 1]),
        "no_rls_with_3plus_core_symptoms": int(sym_tab.loc[3:, 0].sum()),
    }

    # ------------------------------------------------ Module 2: preprocessing + split
    step("Module 2: Preprocessing and stratified train/test split")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Input features: {X.shape[1]} (excluded: {C.EXCLUDED_COLUMNS})")
    print(f"Train: {len(X_train)} rows ({y_train.mean():.1%} RLS)  Test: {len(X_test)} rows ({y_test.mean():.1%} RLS)")
    run["split"] = {"train": len(X_train), "test": len(X_test),
                    "train_rls_share": round(float(y_train.mean()), 4), "test_rls_share": round(float(y_test.mean()), 4)}

    # ------------------------------------------------ Module 4: feature ranking
    step("Module 4: Feature selection (training data only)")
    ranking = rank_features(X_train, y_train)
    ranking.to_csv(C.TABLES_DIR / "feature_ranking.csv", index=False)
    print(ranking.head(10).round(4).to_string(index=False))
    print("  saved", viz.feature_importance(ranking))

    # ------------------------------------------------ Module 5: models
    step("Module 5a: Baseline and default models")
    rows_default, rows_tuned, curves = {}, {}, {}
    baseline = build_baseline().fit(X_train, y_train)
    joblib.dump(baseline, C.MODELS_DIR / "baseline_most_frequent.joblib")
    m_base, _, p_base = evaluate(baseline, X_test, y_test)
    base_name = "Baseline (most frequent)"
    for name in C.MODEL_NAMES:
        pipe = build_pipeline(name).fit(X_train, y_train)
        joblib.dump(pipe, C.model_file(name, tuned=False))
        rows_default[name], _, _ = evaluate(pipe, X_test, y_test)
        print(f"  {name:20s} default  test ROC-AUC {rows_default[name]['ROC-AUC']:.3f}  "
              f"accuracy {rows_default[name]['Accuracy']:.3f}")

    step("Module 5b: Hyperparameter tuning (GridSearchCV, stratified 5-fold, training data only)")
    searches, summaries, best_params = {}, {}, []
    for name in C.MODEL_NAMES:
        t = time.time()
        search = tune_model(name, X_train, y_train)
        searches[name] = search
        summaries[name] = search_summary(name, search)
        bp = {k.replace("clf__", "").replace("select__", "feature_selection_"): v for k, v in search.best_params_.items()}
        best_params.append({"model": name, **{k: str(v) for k, v in bp.items()},
                            "cv_roc_auc_mean": round(search.best_score_, 4),
                            "cv_roc_auc_std": round(float(summaries[name].loc[0, "cv_roc_auc_std"]), 4),
                            "configurations_tried": len(summaries[name])})
        joblib.dump(search.best_estimator_, C.model_file(name, tuned=True))
        print(f"  {name:20s} best CV ROC-AUC {search.best_score_:.3f}  params {bp}  ({time.time() - t:.0f}s)")
    pd.concat(summaries.values(), ignore_index=True).to_csv(C.METRICS_DIR / "hyperparameter_search_summary.csv", index=False)
    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(C.METRICS_DIR / "best_parameters.csv", index=False)
    print("  saved", viz.tuning_plot(summaries, TUNING_PLOT_PARAM))

    step("Module 5c: Final evaluation of tuned models on the held-out test set (once)")
    for i, name in enumerate(C.MODEL_NAMES):
        pipe = searches[name].best_estimator_
        m, _, prob = evaluate(pipe, X_test, y_test)
        rows_tuned[name] = m
        curves[name] = (y_test.to_numpy(), prob, m["ROC-AUC"])
        print("  saved", viz.confusion_matrix_plot(name, m, 10 + i))
    curves[base_name] = (y_test.to_numpy(), p_base, m_base["ROC-AUC"])
    preds = pd.DataFrame({"y_true": y_test.to_numpy(), **{n: c[1] for n, c in curves.items()}}, index=y_test.index)
    preds.index.name = "row"
    preds.to_csv(C.METRICS_DIR / "test_predictions.csv")  # used by the dashboard's interactive charts

    comparison = metrics_table({**rows_tuned, base_name: m_base})
    comparison.insert(1, "Type", ["Trained ML model"] * 3 + ["Simple baseline"])
    cv_auc = {r["model"]: (r["cv_roc_auc_mean"], r["cv_roc_auc_std"]) for r in best_params}
    comparison.insert(2, "CV ROC-AUC (mean)", [cv_auc.get(m, (float("nan"),))[0] for m in comparison["Model"]])
    comparison.insert(3, "CV ROC-AUC (std)", [cv_auc.get(m, (float("nan"), float("nan")))[1] for m in comparison["Model"]])
    comparison.to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)
    print(comparison[["Model", "CV ROC-AUC (mean)"] + METRIC_COLUMNS].round(3).to_string(index=False))

    # Best model: highest mean cross-validated ROC-AUC (training data). Ties -> higher test F1 is NOT
    # used, so the test set never influences the choice.
    best_name = max(C.MODEL_NAMES, key=lambda n: searches[n].best_score_)
    best_pipe = searches[best_name].best_estimator_
    joblib.dump({"pipeline": best_pipe, "model_name": best_name, "features": C.MODEL_FEATURES,
                 "selected_features": selected_feature_names(best_pipe),
                 "selection_criterion": "highest mean 5-fold cross-validated ROC-AUC on the training set",
                 "test_metrics": rows_tuned[best_name], "scikit_learn_version": sklearn.__version__,
                 "trained_on": "rls_dataset.csv"}, C.BEST_MODEL_PATH)
    print(f"\nBest model: {best_name} (CV ROC-AUC {searches[best_name].best_score_:.3f}); saved to {C.BEST_MODEL_PATH}")

    baseline_table = comparison[["Model", "Type", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    baseline_table.to_csv(C.METRICS_DIR / "baseline_comparison.csv", index=False)

    before_after = pd.concat([
        metrics_table(rows_default).assign(Stage="Default"),
        metrics_table(rows_tuned).assign(Stage="Tuned"),
    ], ignore_index=True)[["Model", "Stage"] + METRIC_COLUMNS]
    before_after.to_csv(C.METRICS_DIR / "default_vs_tuned.csv", index=False)

    step("Graphs from the results")
    model_rows = comparison[comparison["Type"] == "Trained ML model"]
    for path in (viz.roc_curves(curves), viz.accuracy_comparison(comparison), viz.prf_comparison(comparison),
                 viz.precision_recall_curves(curves), viz.calibration_plot(curves),
                 viz.before_after_tuning(before_after), viz.baseline_comparison(comparison)):
        print("  saved", path)

    cv_scores = cross_validation_scores({n: searches[n].best_estimator_ for n in C.MODEL_NAMES}, X_train, y_train)
    cv_scores.to_csv(C.METRICS_DIR / "cross_validation_scores.csv", index=False)
    print("  saved", viz.cv_comparison(cv_scores))

    # ------------------------------------------------ Further analysis of the selected model
    step("Further analysis: learning curve, permutation importance, threshold sweep")
    sizes, tr_sc, va_sc = learning_curve(best_pipe, X_train, y_train, cv=cv_splitter(), scoring="roc_auc",
                                         train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    lc = pd.DataFrame({"train_size": sizes, "train_mean": tr_sc.mean(1), "train_std": tr_sc.std(1),
                       "validation_mean": va_sc.mean(1), "validation_std": va_sc.std(1)})
    lc.to_csv(C.METRICS_DIR / "learning_curve.csv", index=False)
    print("  saved", viz.learning_curve_plot(lc))
    pi = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=30,
                                random_state=C.RANDOM_STATE, n_jobs=-1)
    pi_df = (pd.DataFrame({"feature": X_test.columns, "importance_mean": pi.importances_mean,
                           "importance_std": pi.importances_std})
             .sort_values("importance_mean", ascending=False).reset_index(drop=True))
    pi_df.to_csv(C.METRICS_DIR / "permutation_importance.csv", index=False)
    print("  saved", viz.permutation_importance_plot(pi_df))
    sweep = threshold_sweep(y_test, curves[best_name][1])
    sweep.to_csv(C.METRICS_DIR / "threshold_sweep.csv", index=False)
    print("  saved", viz.threshold_plot(sweep))
    run["learning_curve_final"] = {"train_size": int(sizes[-1]), "train_roc_auc": round(float(tr_sc.mean(1)[-1]), 4),
                                   "validation_roc_auc": round(float(va_sc.mean(1)[-1]), 4)}
    run["permutation_top3"] = pi_df.head(3)[["feature", "importance_mean"]].round(4).to_dict("records")

    # ------------------------------------------------ Ablation study
    step("Ablation study (Random Forest, default settings, same split)")
    ablation_rows = []
    for group, feats in C.FEATURE_GROUPS.items():
        pipe = build_pipeline("Random Forest", features=feats).fit(X_train[feats], y_train)
        m, _, _ = evaluate(pipe, X_test[feats], y_test)
        cv = cross_validation_scores({group: pipe}, X_train[feats], y_train)
        ablation_rows.append({"Feature group": group, "Number of features": len(feats),
                              "Features": ", ".join(feats) if len(feats) < 20 else "all 20 eligible features",
                              "CV ROC-AUC (mean)": cv["roc_auc"].mean(),
                              **{k: m[k] for k in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}})
        print(f"  {group:30s} test ROC-AUC {m['ROC-AUC']:.3f}  accuracy {m['Accuracy']:.3f}")
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(C.METRICS_DIR / "ablation_results.csv", index=False)
    print("  saved", viz.ablation_plot(ablation))

    # ------------------------------------------------ Leakage check (documentation only)
    leak_feats = C.MODEL_FEATURES + ["RLS_Severity"]
    leak_X = df[leak_feats].drop(columns=["Gender"])
    lx_tr, lx_te = leak_X.loc[X_train.index], leak_X.loc[X_test.index]
    leak_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("clf", RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_STATE))])
    leak_acc = float((leak_pipe.fit(lx_tr, y_train).predict(lx_te) == y_test).mean())
    run["leakage_check_accuracy_with_RLS_Severity"] = leak_acc
    print(f"\nLeakage check: accuracy if RLS_Severity were (wrongly) used as an input = {leak_acc:.3f}")

    # ------------------------------------------------ Reports
    step("Reports")
    run.update({"best_model": best_name, "finished": datetime.now().isoformat(timespec="seconds"),
                "runtime_seconds": round(time.time() - t0, 1),
                "best_model_selected_features": selected_feature_names(best_pipe)})
    (C.METRICS_DIR / "run_summary.json").write_text(json.dumps(run, indent=2, default=str))
    print("  saved", write_model_comparison_report(comparison, best_name))
    print("  saved", write_experiment_report(run, rep, ranking, comparison, best_params_df, before_after,
                                             ablation, cv_scores, sym_tab, PARAM_GRIDS))
    print(f"\nDone in {time.time() - t0:.0f}s. Graphs: {C.GRAPHS_DIR}  Tables: {C.TABLES_DIR}  "
          f"Metrics: {C.METRICS_DIR}  Reports: {C.REPORTS_DIR}")


if __name__ == "__main__":  # required on Windows because GridSearchCV uses multiple processes
    main()
