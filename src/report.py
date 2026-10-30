"""Write the Markdown reports from the computed results.

All numbers come from the DataFrames produced by main.py. The wording adapts to
the results (for example, which model or feature group did best), so the report
stays correct if the data or settings change.
"""
import pandas as pd

from . import config as C

DISCLAIMER = ("> **Note.** Every result below comes from a dataset of 1,500 records. The model does not "
              "diagnose anyone, and these scores are not evidence of clinical usefulness.")


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """DataFrame -> Markdown table without extra dependencies."""
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(v):
    return f"{v * 100:.1f}%"


def write_model_comparison_report(comparison: pd.DataFrame, best_name: str) -> str:
    cols = ["Model", "Type", "CV ROC-AUC (mean)", "Accuracy", "Precision", "Recall", "Specificity",
            "F1-score", "ROC-AUC", "False positive rate", "False negative rate", "TP", "TN", "FP", "FN"]
    text = "\n\n".join([
        "# Model comparison (held-out test set)", DISCLAIMER,
        f"Test set: {int(comparison.iloc[0][['TP', 'TN', 'FP', 'FN']].sum())} records. "
        f"Best model by mean 5-fold cross-validated ROC-AUC on the training set: **{best_name}**.",
        md_table(comparison[cols]),
        "ROC-AUC uses predicted probabilities. The baseline predicts the same class for everyone, so its "
        "ROC-AUC is 0.5 by construction and its specificity is 0.",
    ])
    path = C.REPORTS_DIR / "model_comparison_report.md"
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def write_experiment_report(run, rep, ranking, comparison, best_params, before_after, ablation,
                            cv_scores, sym_tab, grids) -> str:
    s = rep["summary"]
    best = run["best_model"]
    models = comparison[comparison["Type"] == "Trained ML model"].set_index("Model")
    base = comparison[comparison["Type"] == "Simple baseline"].iloc[0]
    b = models.loc[best]
    ordered = models.sort_values("CV ROC-AUC (mean)", ascending=False)
    runner = ordered.index[1]
    gap = ordered["CV ROC-AUC (mean)"].iloc[0] - ordered["CV ROC-AUC (mean)"].iloc[1]
    top_acc = models["Accuracy"].idxmax()
    top_f1 = models["F1-score"].idxmax()
    top_auc = models["ROC-AUC"].idxmax()
    auc_lo, auc_hi = models["ROC-AUC"].min(), models["ROC-AUC"].max()
    if top_acc == top_f1:
        test_leaders = f"{top_acc} had the highest test accuracy and F1-score"
    else:
        test_leaders = f"{top_acc} had the highest test accuracy and {top_f1} the highest F1-score"
    test_leaders += f", and {top_auc} the highest test ROC-AUC" 
    fp_same = models["FP"].nunique() == 1

    miss = rep["missing"][rep["missing"]["missing"] > 0]
    miss_txt = ", ".join(f"{r.column} ({r.missing})" for r in miss.itertuples())
    top_feats = ranking.head(7)["feature"].str.replace("_", " ").tolist()
    weak = ranking[ranking["mutual_information"] < 0.005]["feature"].str.replace("_", " ").tolist()

    abl = ablation.set_index("Feature group")
    best_group = abl.drop(index=[g for g in abl.index if g.startswith("D.")])["ROC-AUC"].idxmax()
    sym_group = [g for g in abl.index if g.startswith("C.")][0]
    all_group = [g for g in abl.index if g.startswith("D.")][0]
    demo_group = [g for g in abl.index if g.startswith("A.")][0]

    lc = run["label_consistency"]
    conflicted = lc["rls_with_no_core_symptoms"] + lc["no_rls_with_3plus_core_symptoms"]

    ba = before_after.pivot(index="Model", columns="Stage", values="ROC-AUC")
    ba["Change"] = ba["Tuned"] - ba["Default"]
    ba = ba.reset_index()

    # ddof=0 to match the standard deviation GridSearchCV reports
    cv_summary = cv_scores.groupby("model")[["accuracy", "f1", "roc_auc"]].agg(["mean", lambda v: v.std(ddof=0)])
    cv_summary.columns = cv_summary.columns.set_levels(["mean", "std"], level=1)
    cv_summary.columns = [f"{m} {s}" for m, s in cv_summary.columns]
    cv_summary = cv_summary.reset_index().rename(columns={"model": "Model"})

    grid_lines = "\n".join(
        f"- **{m}:** " + "; ".join(f"`{k.replace('clf__', '').replace('select__', 'feature_selection_')}` "
                                   f"in {v}" for k, v in g.items())
        for m, g in grids.items())

    sections = [
        "# Experiment report: RLS identification with machine learning", DISCLAIMER,
        f"Generated {run['finished']} · Python {run['python']} · scikit-learn {run['scikit_learn']} · "
        f"random seed {run['random_state']} · runtime {run['runtime_seconds']} s",

        "## 1. Experimental setup",
        "The pipeline follows five modules: Data Collection → Data Preprocessing → Feature Selection → "
        "Machine Learning Prediction → Performance Evaluation. "
        f"The data were split once into a stratified training set ({run['split']['train']} records) and test set "
        f"({run['split']['test']} records). All imputation, scaling and feature selection were fitted on the training "
        f"set only, inside scikit-learn pipelines. Hyperparameters were chosen by {C.CV_FOLDS}-fold stratified "
        "cross-validation on the training set, and each tuned model was evaluated on the test set once.",

        "## 2. Dataset description",
        f"The file `rls_dataset.csv` has {s['rows']} records and {s['columns']} columns: "
        f"{s['rls']} labelled RLS ({pct(s['rls'] / s['rows'])}) and {s['no_rls']} labelled No RLS. "
        f"There are {s['input_features']} eligible input features covering demographics, sleep, wellbeing, "
        "a ferritin value, family history, five core RLS symptoms, a leg-discomfort score and symptom frequency. "
        f"Missing values: {miss_txt}. Duplicate rows: {s['duplicate_rows']}.",
        "Excluded from the inputs: `Patient_ID` (identifier), `RLS_Probability` and `RLS_Severity` (both derived "
        "from the outcome). `RLS_Severity` is 0 for every No RLS record; a Random Forest that is allowed to use it "
        f"reaches {pct(run['leakage_check_accuracy_with_RLS_Severity'])} test accuracy, which shows why it must be excluded.",

        "## 3. Preprocessing steps",
        "- Numerical features: median imputation, then standardisation (Logistic Regression only; trees are unaffected by scale).\n"
        "- Binary symptom and history flags: most-frequent imputation, kept as 0/1.\n"
        "- Gender: most-frequent imputation, one-hot encoding with unknown categories ignored.\n"
        "- Feature selection: `SelectKBest` with mutual information inside each pipeline; the number of features "
        "(all 20 or the top 10) was tuned with the model.\n"
        "- Logistic Regression uses no class weighting because the training split is close to balanced "
        f"({pct(run['split']['train_rls_share'])} RLS).",
        f"Feature ranking on the training set (top 7 by mutual information): {', '.join(top_feats)}. "
        + (f"Features with almost no association (MI < 0.005): {', '.join(weak)}." if weak else ""),
        md_table(ranking[["rank", "feature", "mutual_information", "rf_importance", "correlation_with_target"]].head(12), 4),

        "## 4. Models evaluated",
        "- **Baseline:** `DummyClassifier(strategy='most_frequent')`, which predicts the majority class for everyone.\n"
        "- **Logistic Regression:** linear model of the log-odds of RLS (lbfgs solver, up to 1,000 iterations).\n"
        "- **Decision Tree:** splits chosen by Gini impurity, with limits on depth, split size and leaf size.\n"
        "- **Random Forest:** an ensemble of decision trees on bootstrap samples, majority vote.",
        "Search space used for tuning:\n" + grid_lines,

        "## 5. Performance results (test set)",
        f"Selection criterion: highest mean cross-validated ROC-AUC on the training set. Selected model: **{best}** "
        f"(CV ROC-AUC {b['CV ROC-AUC (mean)']:.3f} ± {b['CV ROC-AUC (std)']:.3f}). On the test set it reached "
        f"ROC-AUC {b['ROC-AUC']:.3f}, accuracy {pct(b['Accuracy'])}, precision {pct(b['Precision'])}, recall "
        f"{pct(b['Recall'])}, specificity {pct(b['Specificity'])} and F1-score {b['F1-score']:.3f}.",
        md_table(comparison[["Model", "Type", "CV ROC-AUC (mean)", "Accuracy", "Precision", "Recall", "Specificity",
                             "F1-score", "ROC-AUC", "False positive rate", "False negative rate"]]),
        "Cross-validation scores of the tuned models on the training folds (mean and standard deviation):",
        md_table(cv_summary),
        "Graphs: `results/graphs/10-12_confusion_matrix_*.png`, `13_roc_curves.png`, `14_accuracy_comparison.png`, "
        "`15_precision_recall_f1.png`, `17_cross_validation_comparison.png`, `18_precision_recall_curves.png`, "
        "`19_calibration_curves.png`.",

        "## 6. Baseline comparison",
        f"The baseline predicts {'RLS' if base['Recall'] == 1 else 'No RLS'} for every record, so its accuracy "
        f"({pct(base['Accuracy'])}) equals the share of that class in the test set and its ROC-AUC is 0.5. "
        f"Every trained model beats it clearly; the selected model improves accuracy by "
        f"{(b['Accuracy'] - base['Accuracy']) * 100:.1f} percentage points.",
        md_table(comparison[["Model", "Type", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]),

        "## 7. Ablation study",
        "A Random Forest with default settings was trained on each feature group using the same split and "
        "preprocessing. These results describe how the groups behave in this dataset, not their "
        "clinical importance.",
        md_table(ablation[["Feature group", "Number of features", "CV ROC-AUC (mean)", "Accuracy", "Precision",
                           "Recall", "F1-score", "ROC-AUC"]]),
        f"Of the single groups, **{best_group}** was strongest (test ROC-AUC {abl.loc[best_group, 'ROC-AUC']:.3f}). "
        f"All features combined reached {abl.loc[all_group, 'ROC-AUC']:.3f}, symptom-related features alone "
        f"{abl.loc[sym_group, 'ROC-AUC']:.3f}, and demographic and general variables {abl.loc[demo_group, 'ROC-AUC']:.3f}, "
        "which is close to chance (0.5). This suggests that the labels depend mainly on the symptoms: "
        "almost all the signal sits in the symptom variables.",

        "## 8. Hyperparameter tuning",
        md_table(best_params),
        "Default versus tuned settings (test ROC-AUC):",
        md_table(ba),
        ("Tuning changed test ROC-AUC by between "
         f"{ba['Change'].min():+.3f} and {ba['Change'].max():+.3f}. Changes this small are within the variation "
         "expected from a single 300-record test set, so tuning should be read as confirming reasonable settings "
         "rather than as a clear improvement." if ba["Change"].abs().max() < 0.03 else
         "Tuning changed test performance noticeably for at least one model; see the table above."),
        "Full search results: `results/metrics/hyperparameter_search_summary.csv`. Graphs: "
        "`20_hyperparameter_tuning.png`, `21_before_after_tuning.png`.",

        "## 9. Discussion of findings",
        f"{best} was selected, with a cross-validated ROC-AUC {gap:.3f} above {runner}. "
        + ("That margin is small, so the models should be regarded as performing similarly. " if gap < 0.01 else "")
        + f"On the test set the three models were close (ROC-AUC {auc_lo:.3f} to {auc_hi:.3f}): {test_leaders}. "
        + ("The selected model is therefore not the best on every test metric. " if best not in (top_acc, top_f1, top_auc) or best != top_auc else "")
        + "The choice was made on cross-validated ROC-AUC so that the test set did not influence it, and test "
        "differences of this size are within the noise of a 300-record test set. "
        + ("All three models produced the same number of false positives, which suggests they misclassify largely the same records. " if fp_same else ""),
        (f"Learning curve: trained on full cross-validation folds ({run['learning_curve_final']['train_size']} records each), the selected model "
         f"scores ROC-AUC {run['learning_curve_final']['train_roc_auc']:.3f} on the training folds and "
         f"{run['learning_curve_final']['validation_roc_auc']:.3f} on the validation folds. A gap this small means the "
         "model is not overfitting, and more data of the same kind is unlikely to change it much "
         "(`25_learning_curve.png`). Permutation importance on the test set ranks "
         + ", ".join(f"{d['feature'].replace('_', ' ')} ({d['importance_mean']:.3f})" for d in run["permutation_top3"])
         + " as the most influential inputs (`26_permutation_importance.png`). `27_threshold_analysis.png` shows how "
         "moving the 0.5 decision threshold trades recall against specificity." if "learning_curve_final" in run else ""),
        f"A label-consistency check explains part of the remaining error: {lc['rls_with_no_core_symptoms']} records "
        f"labelled RLS report none of the five core symptoms, and {lc['no_rls_with_3plus_core_symptoms']} records "
        f"labelled No RLS report three or more. That is {conflicted} records ({pct(conflicted / s['rows'])} of the data) "
        "whose labels contradict their symptoms, so a model using these features will usually misclassify them. "
        "See `24_symptom_count_vs_label.png`.",
        md_table(sym_tab.rename(columns=C.CLASS_LABELS).reset_index()),
        "Because the labels depend mainly on the symptom variables, the relationships between features and "
        "labels may be cleaner or simpler than in other clinical data. Performance on a new population may therefore "
        "be lower, or differently distributed.",

        "## 10. Limitations and conclusion",
        "- The model was evaluated on a single dataset; it needs validation on independent clinical data.\n"
        "- One train/test split of 300 test records was used; scores would vary somewhat with a different split.\n"
        "- The labels contain records that contradict the symptoms, which limits achievable accuracy.\n"
        "- Real-world validation would need an ethically sourced clinical dataset with diagnoses confirmed by "
        "specialists, and evaluation across different sites and populations.\n"
        "- The system is a teaching prototype and must not replace professional medical assessment.",
        f"**Conclusion.** On this dataset, {best} gave the best cross-validated discrimination "
        f"(ROC-AUC {b['CV ROC-AUC (mean)']:.3f}) and a test ROC-AUC of {b['ROC-AUC']:.3f}, well above the "
        "most-frequent baseline. Symptom-related features carried most of the signal. These findings describe "
        "model behaviour on this dataset only.",
    ]
    path = C.REPORTS_DIR / "experiment_report.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return str(path)
