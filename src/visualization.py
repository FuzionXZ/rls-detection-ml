"""Graphs for EDA (Module 3) and results (Module 5).

Every function draws from real data or computed model outputs passed in by
main.py and saves a PNG to results/graphs/. Matplotlib only, so the charts
render the same on any machine.
"""
import matplotlib

matplotlib.use("Agg")  # no display needed; safe on Windows and servers
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used for axis number formats)
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C

CLASS_COLORS = {0: "#5B7A99", 1: "#C0703A"}  # No RLS, RLS
MODEL_COLORS = {"Logistic Regression": "#2F4B7C", "Decision Tree": "#B5651D",
                "Random Forest": "#3C8D5A", "Baseline (most frequent)": "#9A9A9A"}
INK, GRID = "#222222", "#E3E3E3"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "axes.edgecolor": "#888888",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name: str) -> str:
    path = C.GRAPHS_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _label_bars(ax, bars, fmt="{:.0f}", inside=False):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center",
                    va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


# =============================================================== EDA
def class_distribution(df):
    counts = df[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar([C.CLASS_LABELS[i] for i in counts.index], counts.values,
                  color=[CLASS_COLORS[i] for i in counts.index], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v}  ({v / counts.sum():.1%})", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set(title="Target class distribution", ylabel="Number of records", ylim=(0, counts.max() * 1.15))
    return _save(fig, "01_class_distribution.png")


def _hist_by_class(df, col, title, xlabel, fname, bins):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cls in (0, 1):
        ax.hist(df.loc[df[C.TARGET] == cls, col].dropna(), bins=bins, alpha=0.6,
                color=CLASS_COLORS[cls], label=C.CLASS_LABELS[cls], edgecolor="white")
    ax.set(title=title, xlabel=xlabel, ylabel="Number of records")
    ax.legend()
    return _save(fig, fname)


def age_distribution(df):
    return _hist_by_class(df, "Age", "Age distribution by class", "Age (years)", "02_age_distribution.png",
                          bins=range(15, 90, 5))


def gender_distribution(df):
    tab = pd.crosstab(df["Gender"], df[C.TARGET])
    x = np.arange(len(tab.index))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, cls in enumerate((0, 1)):
        bars = ax.bar(x + (i - 0.5) * 0.36, tab[cls], 0.36, color=CLASS_COLORS[cls], label=C.CLASS_LABELS[cls])
        _label_bars(ax, bars)
    ax.set_xticks(x, tab.index)
    ax.set(title="Gender distribution by class", ylabel="Number of records", ylim=(0, tab.values.max() * 1.15))
    ax.legend()
    return _save(fig, "03_gender_distribution.png")


def sleep_duration_distribution(df):
    return _hist_by_class(df, "Sleep_Duration_Hours", "Sleep duration distribution by class",
                          "Sleep duration (hours per night)", "04_sleep_duration_distribution.png",
                          bins=np.arange(3.5, 10.75, 0.5))


def sleep_quality_by_class(df):
    data = [df.loc[df[C.TARGET] == c, "Sleep_Quality"].dropna() for c in (0, 1)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK})
    for patch, c in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.75)
    ax.set_xticks([1, 2], [C.CLASS_LABELS[0], C.CLASS_LABELS[1]])
    ax.set(title="Sleep quality by class", ylabel="Sleep quality (1-10)")
    return _save(fig, "05_sleep_quality_by_class.png")


def symptom_frequency_by_class(df):
    tab = pd.crosstab(df["Symptom_Frequency_Days_Week"], df[C.TARGET]).reindex(range(8), fill_value=0)
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cls in enumerate((0, 1)):
        ax.bar(x + (i - 0.5) * 0.4, tab[cls], 0.4, color=CLASS_COLORS[cls], label=C.CLASS_LABELS[cls])
    ax.set_xticks(x, [str(v) for v in x])
    ax.set(title="Symptom frequency by class", xlabel="Days per week with symptoms", ylabel="Number of records")
    ax.legend()
    return _save(fig, "06_symptom_frequency_by_class.png")


def missing_values(missing_table):
    t = missing_table[missing_table["missing"] > 0].sort_values("missing")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(t["column"], t["missing"], color="#7A6C8F")
    for b, pct in zip(bars, t["missing_percent"]):
        ax.annotate(f"{int(b.get_width())} ({pct:.1f}%)", (b.get_width(), b.get_y() + b.get_height() / 2),
                    va="center", xytext=(3, 0), textcoords="offset points", fontsize=8)
    ax.set(title="Missing values by feature", xlabel="Missing values (of 1,500 records)",
           xlim=(0, t["missing"].max() * 1.3))
    ax.grid(axis="y", visible=False)
    return _save(fig, "07_missing_values.png")


def correlation_heatmap(df):
    cols = C.NUMERIC_FEATURES + C.BINARY_FEATURES + [C.TARGET]
    corr = df[cols].corr()
    labels = [c.replace("_", " ") for c in cols]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.5 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson correlation")
    ax.set_title("Correlation between numerical features (including the target)")
    return _save(fig, "08_correlation_heatmap.png")


def symptom_comparison(df):
    rates = df.groupby(C.TARGET)[C.SYMPTOM_COLUMNS + ["Family_History_RLS"]].mean() * 100
    names = [c.replace("_", " ") for c in rates.columns]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, cls in enumerate((0, 1)):
        bars = ax.barh(y + (0.5 - i) * 0.38, rates.loc[cls], 0.38, color=CLASS_COLORS[cls], label=C.CLASS_LABELS[cls])
        for b in bars:
            ax.annotate(f"{b.get_width():.0f}%", (b.get_width(), b.get_y() + b.get_height() / 2), va="center",
                        xytext=(3, 0), textcoords="offset points", fontsize=8)
    ax.set_yticks(y, names)
    ax.set(title="Share of records reporting each symptom, by class", xlabel="Records with the symptom (%)",
           xlim=(0, 100))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, "09_symptom_comparison.png")


# =============================================================== RESULTS
def confusion_matrix_plot(name, m, index):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap="Blues", vmin=0)
    total = cm.sum()
    tags = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{tags[i][j]} ({cm[i, j] / total:.1%})", ha="center", va="center",
                    fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["No RLS", "RLS"]); ax.set_yticks([0, 1], ["No RLS", "RLS"])
    ax.set(xlabel="Predicted class", ylabel="Actual class", title=f"Confusion matrix: {name}")
    ax.grid(False)
    slug = name.lower().replace(" ", "_")
    return _save(fig, f"{index:02d}_confusion_matrix_{slug}.png")


def roc_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ls = "--" if name.startswith("Baseline") else "-"
        ax.plot(fpr, tpr, ls, color=MODEL_COLORS[name], lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#BBBBBB", lw=1)
    ax.set(title="ROC curves on the test set", xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "13_roc_curves.png")


def accuracy_comparison(table):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(table["Model"], table["Accuracy"], color=[MODEL_COLORS[m] for m in table["Model"]], width=0.55)
    _label_bars(ax, bars, "{:.3f}")
    ax.set(title="Test-set accuracy", ylabel="Accuracy", ylim=(0, 1.05))
    ax.tick_params(axis="x", labelsize=9)
    return _save(fig, "14_accuracy_comparison.png")


def prf_comparison(table):
    metrics = ["Precision", "Recall", "F1-score"]
    shades = ["#2F4B7C", "#6C8EBF", "#B9CBE3"]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (m, col) in enumerate(zip(metrics, shades)):
        bars = ax.bar(x + (i - 1) * 0.26, table[m], 0.26, color=col, label=m)
        _label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(x, table["Model"], fontsize=9)
    ax.set(title="Precision, recall and F1-score on the test set", ylabel="Score", ylim=(0, 1.12))
    ax.legend(ncol=3, loc="upper left")
    return _save(fig, "15_precision_recall_f1.png")


def feature_importance(ranking):
    t = ranking.sort_values("mutual_information")
    names = [f.replace("_", " ") for f in t["feature"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    axes[0].barh(names, t["mutual_information"], color="#2F4B7C")
    axes[0].set(title="Mutual information with diagnosis", xlabel="Mutual information")
    axes[1].barh(names, t["rf_importance"], color="#3C8D5A")
    axes[1].set(title="Random Forest importance", xlabel="Mean decrease in impurity")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Feature ranking (training data only)", fontweight="bold")
    return _save(fig, "16_feature_importance.png")


def cv_comparison(cv_scores):
    models = list(dict.fromkeys(cv_scores["model"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    for ax, metric, label in zip(axes, ["accuracy", "f1", "roc_auc"], ["Accuracy", "F1-score", "ROC-AUC"]):
        data = [cv_scores.loc[cv_scores["model"] == m, metric] for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK})
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(MODEL_COLORS[m]); patch.set_alpha(0.7)
        for k, d in enumerate(data, start=1):
            ax.scatter(np.full(len(d), k) + np.linspace(-0.08, 0.08, len(d)), d, s=12, color=INK, zorder=3)
        ax.set_xticks(range(1, len(models) + 1), [m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_title(label)
    fig.suptitle(f"{C.CV_FOLDS}-fold cross-validation on the training set (tuned models)", fontweight="bold")
    return _save(fig, "17_cross_validation_comparison.png")


def precision_recall_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, (y, p, _) in curves.items():
        if name.startswith("Baseline"):
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2, label=name)
    base = np.mean(next(iter(curves.values()))[0])
    ax.axhline(base, ls=":", color="#999999", lw=1, label=f"Share of RLS in test set ({base:.2f})")
    ax.set(title="Precision-recall curves on the test set", xlabel="Recall", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, "18_precision_recall_curves.png")


def calibration_plot(curves: dict):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, (y, p, _) in curves.items():
        if name.startswith("Baseline"):
            continue
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=MODEL_COLORS[name], lw=1.8, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=1, label="Perfect calibration")
    ax.set(title="Calibration on the test set", xlabel="Mean predicted probability of RLS",
           ylabel="Observed share of RLS", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "19_calibration_curves.png")


def tuning_plot(summaries: dict, plot_params: dict):
    fig, axes = plt.subplots(1, len(summaries), figsize=(12, 3.8))
    for ax, (name, s) in zip(np.atleast_1d(axes), summaries.items()):
        p = plot_params[name].replace("clf__", "")
        best = s.groupby(s[p].astype(str), sort=False)["cv_roc_auc_mean"].max()
        order = sorted(best.index, key=lambda v: (v == "None", float(v) if v not in ("None",) else 0))
        best = best.reindex(order)
        ax.plot(range(len(best)), best.values, "o-", color=MODEL_COLORS[name], lw=2)
        ax.set_xticks(range(len(best)), best.index)
        ax.set(title=name, xlabel=p.replace("_", " "), ylabel="Best CV ROC-AUC")
        mid = float(best.mean())
        ax.set_ylim(mid - 0.02, mid + 0.02)  # same 0.04-wide window on every panel, so slopes are comparable
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    fig.suptitle("Hyperparameter tuning (best cross-validated ROC-AUC at each value)", fontweight="bold")
    return _save(fig, "20_hyperparameter_tuning.png")


def before_after_tuning(table):
    models = C.MODEL_NAMES
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (stage, col) in enumerate([("Default", "#B9CBE3"), ("Tuned", "#2F4B7C")]):
        vals = [table.loc[(table["Model"] == m) & (table["Stage"] == stage), "ROC-AUC"].iloc[0] for m in models]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=col, label=stage)
        _label_bars(ax, bars, "{:.3f}")
    ax.set_xticks(x, models)
    ax.set(title="Test ROC-AUC before and after tuning", ylabel="ROC-AUC", ylim=(0, 1.08))
    ax.legend(ncol=2, loc="upper left")
    return _save(fig, "21_before_after_tuning.png")


def ablation_plot(table):
    metrics = ["Accuracy", "F1-score", "ROC-AUC"]
    shades = ["#2F4B7C", "#6C8EBF", "#B9CBE3"]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (m, col) in enumerate(zip(metrics, shades)):
        bars = ax.barh(y + (i - 1) * 0.26, table[m], 0.26, color=col, label=m)
        for b in bars:
            ax.annotate(f"{b.get_width():.3f}", (b.get_width(), b.get_y() + b.get_height() / 2), va="center",
                        xytext=(3, 0), textcoords="offset points", fontsize=8)
    ax.set_yticks(y, table["Feature group"])
    ax.invert_yaxis()
    ax.set(title="Ablation study: Random Forest trained on each feature group", xlabel="Test-set score",
           xlim=(0, 1.1))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("Ablation study: Random Forest trained on each feature group", pad=26)
    return _save(fig, "22_ablation_study.png")


def baseline_comparison(table):
    """Accuracy, F1 and ROC-AUC for the baseline next to the three tuned models."""
    metrics = ["Accuracy", "F1-score", "ROC-AUC"]
    x = np.arange(len(metrics))
    n = len(table)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (_, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, [row[m] for m in metrics], w, color=MODEL_COLORS[row["Model"]],
                      label=row["Model"], hatch="//" if row["Model"].startswith("Baseline") else None,
                      edgecolor="white")
        _label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(x, metrics)
    ax.set(title="Trained models versus the most-frequent baseline", ylabel="Test-set score", ylim=(0, 1.12))
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return _save(fig, "23_baseline_comparison.png")


def symptom_count_vs_label(crosstab):
    x = np.arange(len(crosstab.index))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, cls in enumerate((0, 1)):
        bars = ax.bar(x + (i - 0.5) * 0.4, crosstab[cls], 0.4, color=CLASS_COLORS[cls], label=C.CLASS_LABELS[cls])
        _label_bars(ax, bars)
    ax.set_xticks(x, [str(v) for v in crosstab.index])
    ax.set(title="Core symptoms reported versus recorded label", xlabel="Number of the 5 core symptoms reported",
           ylabel="Number of records", ylim=(0, crosstab.values.max() * 1.15))
    ax.legend()
    return _save(fig, "24_symptom_count_vs_label.png")


def learning_curve_plot(lc):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, label, color in [("train", "Training folds", "#9DB3D6"), ("validation", "Validation folds", "#2F4B7C")]:
        ax.plot(lc["train_size"], lc[f"{col}_mean"], "o-", color=color, lw=2, label=label)
        ax.fill_between(lc["train_size"], lc[f"{col}_mean"] - lc[f"{col}_std"], lc[f"{col}_mean"] + lc[f"{col}_std"],
                        color=color, alpha=0.18, lw=0)
    ax.set(title="Learning curve (selected model)", xlabel="Training records used", ylabel="ROC-AUC", ylim=(0.5, 1.0))
    ax.legend(loc="lower right")
    return _save(fig, "25_learning_curve.png")


def permutation_importance_plot(pi):
    t = pi.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([f.replace("_", " ") for f in t["feature"]], t["importance_mean"], xerr=t["importance_std"],
            color="#3C8D5A", error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#555555"})
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set(title="Permutation importance on the test set (selected model)",
           xlabel="Drop in ROC-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    return _save(fig, "26_permutation_importance.png")


def threshold_plot(sweep, chosen=0.5):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, color in [("Precision", "#2F4B7C"), ("Recall", "#C0703A"), ("Specificity", "#3C8D5A"), ("F1-score", "#7A6C8F")]:
        ax.plot(sweep["threshold"], sweep[m], "-", color=color, lw=2, label=m)
    ax.axvline(chosen, ls="--", color="#888888", lw=1)
    ax.annotate("default 0.5", (chosen, 1.0), xytext=(4, -10), textcoords="offset points", fontsize=8, color="#555555")
    ax.set(title="Effect of the decision threshold (selected model, test set)", xlabel="Decision threshold",
           ylabel="Score", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8)
    return _save(fig, "27_threshold_analysis.png")
