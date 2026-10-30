"""Streamlit dashboard for the RLS identification project.

    streamlit run app.py

The dashboard only reads what main.py saved (models, tables, graphs); it never
retrains. If those files are missing it explains how to create them.
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from src import config as C
from src import ui
from src.data_loader import DataValidationError, load_data_dictionary, load_dataset
from src.feature_selection import selected_feature_names
from src.evaluation import metrics_at_threshold, threshold_sweep
from src.prediction import (ModelNotAvailableError, PredictionInputError, load_all_models, load_model, predict,
                            template_csv)

st.set_page_config(page_title="RLS Identification", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS + ui.CHART_CSS + ui.STEPS_CSS, unsafe_allow_html=True)

PRETTY = {c: c.replace("_", " ").replace("ng mL", "(ng/mL)").replace("Days Week", "(days/week)") for c in C.MODEL_FEATURES}
PRETTY.update({"Gender_Male": "Gender (male)", "Symptom_Frequency_Days_Week": "Symptom days per week",
               "Leg_Discomfort_Score": "Leg discomfort score", "Sleep_Duration_Hours": "Sleep duration (h)"})
MODEL_ORDER = C.MODEL_NAMES + ["Baseline (most frequent)"]


# ------------------------------------------------------------------ cached loaders
@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_data
def get_table(folder, name, index_col=None):
    path = folder / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


@st.cache_data
def get_run_summary():
    path = C.METRICS_DIR / "run_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_all_models():
    return load_all_models()


@st.cache_data
def zip_folder(folder_names) -> bytes:
    """Zip one or more results sub-folders in memory for a download button."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in folder_names:
            folder = C.RESULTS_DIR / name
            if folder.exists():
                for f in sorted(folder.iterdir()):
                    if f.is_file():
                        z.write(f, f"{name}/{f.name}")
    return buf.getvalue()


def graph(name, caption=None):
    path = C.GRAPHS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Graph `{name}` not found. Run `python main.py` to create it.")


def need_results():
    st.warning("No saved results yet. Open a terminal in the project folder, run `python main.py` "
               "(about 2-3 minutes), then refresh this page.")
    st.stop()


def metric_columns(cols):
    """Column formats: scores as small bars between 0 and 1."""
    cfg = {}
    for c in cols:
        if c in ("Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"):
            cfg[c] = st.column_config.ProgressColumn(c, format="%.3f", min_value=0.0, max_value=1.0)
        elif c in ("CV ROC-AUC (mean)", "CV ROC-AUC (std)", "False positive rate", "False negative rate",
                   "mutual_information", "rf_importance", "correlation_with_target", "CV ROC-AUC"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.3f")
    return cfg


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<div class='brand'><div class='mark'>RLS</div><div class='name'>Restless Legs Syndrome "
                "Identification</div><div class='sub'>Machine learning predictive analysis</div></div>",
                unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview", "Methodology", "Dataset", "Exploratory analysis", "Model performance",
                                   "Threshold explorer", "Try the model", "Batch prediction", "Downloads"],
                    label_visibility="collapsed")
    st.markdown("<div style='height:24px'></div><div class='pipeline'><b>Pipeline</b><br>1 · Data collection<br>"
                "2 · Preprocessing<br>3 · Feature selection<br>4 · ML prediction<br>5 · Evaluation</div>",
                unsafe_allow_html=True)

try:
    df = get_dataset()
except DataValidationError as exc:
    st.error(f"The dataset could not be loaded: {exc}")
    st.stop()

run = get_run_summary()
comparison = get_table(C.METRICS_DIR, "model_comparison.csv")
preds = get_table(C.METRICS_DIR, "test_predictions.csv", index_col=0)

# ================================================================== Overview
if page == "Overview":
    st.markdown(ui.page_header("Overview", "Restless Legs Syndrome Identification",
                               "Logistic Regression, Decision Tree and Random Forest classifiers compared on "
                               "1,500 records with demographic, sleep and symptom features."), unsafe_allow_html=True)
    if comparison is None or run is None:
        need_results()
    best = run["best_model"]
    b = comparison.set_index("Model").loc[best]
    counts = df[C.TARGET].value_counts()
    st.markdown(ui.kpi_cards([
        ("Best model", best, "highest cross-validated ROC-AUC", True),
        ("Test ROC-AUC", f"{b['ROC-AUC']:.3f}", f"cross-validated {b['CV ROC-AUC (mean)']:.3f}", False),
        ("Accuracy", f"{b['Accuracy']:.1%}", f"{int(b['TP'] + b['TN'])} of {int(b[['TP', 'TN', 'FP', 'FN']].sum())} test records", False),
        ("Recall", f"{b['Recall']:.1%}", "RLS records found", False),
        ("Specificity", f"{b['Specificity']:.1%}", "No RLS records found", False),
    ]), unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(ui.section("Model comparison", "Test-set scores for the three models and the baseline"),
                        unsafe_allow_html=True)
            cmp = comparison.set_index("Model")
            metrics = ["Accuracy", "F1-score", "ROC-AUC"]
            html(ui.grouped_bars(metrics, [(m, [float(cmp.loc[m, k]) for k in metrics], ui.MODEL_COLORS[m])
                                           for m in MODEL_ORDER], ymax=1.0, height=320, y_label="Score"))
    with right:
        with st.container(border=True):
            st.markdown(ui.section("Dataset", f"{len(df):,} records · {len(C.MODEL_FEATURES)} input features"),
                        unsafe_allow_html=True)
            html(ui.grouped_bars(["No RLS", "RLS"], [("Records", [int(counts.get(0, 0)), int(counts.get(1, 0))], ui.ACCENT)],
                                 fmt="{:,.0f}", height=300, y_label="Records",
                                 bar_colors=[ui.NO_RLS_COLOR, ui.RLS_COLOR], width=400))
            st.markdown(f"<div class='muted'>{counts.get(1, 0)} RLS ({counts.get(1, 0) / len(df):.1%}) · "
                        f"{counts.get(0, 0)} No RLS ({counts.get(0, 0) / len(df):.1%})</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(ui.section("All results", "Best model chosen on cross-validated ROC-AUC (training data), "
                                              "then scored once on the 300-record test set"), unsafe_allow_html=True)
        cols = ["Model", "CV ROC-AUC (mean)", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]
        st.dataframe(comparison[cols], hide_index=True, use_container_width=True, column_config=metric_columns(cols))

# ================================================================== Dataset
elif page == "Dataset":
    st.markdown(ui.page_header("Dataset", "Explore the records",
                               "Filter the 1,500 records, inspect individual rows and check column types and "
                               "missing values."), unsafe_allow_html=True)
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)
        cls = f1.multiselect("Class", ["No RLS", "RLS"], default=["No RLS", "RLS"])
        gender = f2.multiselect("Gender", sorted(df["Gender"].dropna().unique()),
                                default=sorted(df["Gender"].dropna().unique()))
        age = f3.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
        freq = f4.slider("Symptom days per week", 0, 7, (0, 7))
    labels = {v: k for k, v in C.CLASS_LABELS.items()}
    view = df[df[C.TARGET].isin([labels[c] for c in cls]) & df["Gender"].isin(gender)
              & df["Age"].between(*age) & df["Symptom_Frequency_Days_Week"].between(*freq)]
    st.markdown(f"<div class='muted' style='margin:6px 0 8px 0'>{len(view):,} of {len(df):,} records match</div>",
                unsafe_allow_html=True)
    st.dataframe(view, use_container_width=True, height=380, hide_index=True)

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(ui.section("Inspect a record"), unsafe_allow_html=True)
            ids = view[C.ID_COLUMN].tolist()
            if ids:
                rid = st.selectbox("Record ID", ids)
                rec = view[view[C.ID_COLUMN] == rid].iloc[0].drop(C.ID_COLUMN)
                st.dataframe(rec.rename("Value").to_frame().astype(str), use_container_width=True, height=420)
            else:
                st.info("No records match the filters.")
    with right:
        with st.container(border=True):
            st.markdown(ui.section("Columns"), unsafe_allow_html=True)
            t1, t2 = st.tabs(["Types and missing values", "Data dictionary"])
            with t1:
                ov = get_table(C.TABLES_DIR, "dataset_overview.csv")
                if ov is None:
                    ov = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str),
                                       "missing": df.isna().sum().values})
                show = ov[[c for c in ["column", "dtype", "missing", "unique_values", "role"] if c in ov.columns]]
                st.dataframe(show, hide_index=True, use_container_width=True, height=420)
            with t2:
                dd = load_data_dictionary()
                if dd.empty:
                    st.info("No data dictionary found in data/.")
                else:
                    st.dataframe(dd, hide_index=True, use_container_width=True, height=420)

# ================================================================== Exploratory analysis
elif page == "Exploratory analysis":
    st.markdown(ui.page_header("Exploratory analysis", "How the features differ between classes",
                               "Pick any feature to compare RLS and No RLS records, or browse the saved figures."),
                unsafe_allow_html=True)
    with st.container(border=True):
        options = C.NUMERIC_FEATURES + C.BINARY_FEATURES
        feat = st.selectbox("Feature", options, index=options.index("Symptom_Frequency_Days_Week"),
                            format_func=lambda c: PRETTY.get(c, c))
        a, b = st.columns([1.6, 1], gap="large")
        with a:
            if feat in C.BINARY_FEATURES:
                tab = pd.crosstab(df[feat], df[C.TARGET])
                tab.index = ["No" if v == 0 else "Yes" for v in tab.index]
            elif df[feat].nunique() <= 11:
                tab = pd.crosstab(df[feat], df[C.TARGET])  # numeric index -> ordered axis
            else:
                bins = pd.cut(df[feat], bins=10)
                tab = pd.crosstab(bins, df[C.TARGET])
                tab.index = [round((iv.left + iv.right) / 2, 1) for iv in tab.index]  # bin midpoints
            tab = tab.reindex(columns=[0, 1], fill_value=0)
            cats = [str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer()
                    else str(v) for v in tab.index]
            html(ui.grouped_bars(cats, [("No RLS", tab[0].astype(int).tolist(), ui.NO_RLS_COLOR),
                                        ("RLS", tab[1].astype(int).tolist(), ui.RLS_COLOR)],
                                 fmt="{:,.0f}", height=320, y_label="Records"))
            st.caption(f"x-axis: {PRETTY.get(feat, feat)}" + (" (bin midpoints)" if df[feat].nunique() > 11 and feat not in C.BINARY_FEATURES else ""))
        with b:
            desc = df.groupby(C.TARGET)[feat].describe().rename(index=C.CLASS_LABELS).T.round(2)
            st.dataframe(desc, use_container_width=True, height=320)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    eda = [("02_age_distribution.png", "Age"), ("03_gender_distribution.png", "Gender"),
           ("04_sleep_duration_distribution.png", "Sleep duration"), ("05_sleep_quality_by_class.png", "Sleep quality"),
           ("06_symptom_frequency_by_class.png", "Symptom frequency"), ("09_symptom_comparison.png", "Core symptoms"),
           ("07_missing_values.png", "Missing values"), ("24_symptom_count_vs_label.png", "Symptoms vs. label")]
    tabs = st.tabs([cap for _, cap in eda] + ["Correlations"])
    for tab, (f, cap) in zip(tabs, eda):
        with tab:
            _, mid, _ = st.columns([1, 3, 1])
            with mid:
                graph(f)
    with tabs[-1]:
        graph("08_correlation_heatmap.png")

# ================================================================== Model performance
elif page == "Model performance":
    st.markdown(ui.page_header("Model performance", "How the models compare",
                               "Test-set results for the tuned models against a most-frequent baseline, with "
                               "feature importance, ablation and tuning experiments."), unsafe_allow_html=True)
    if comparison is None:
        need_results()
    tabs = st.tabs(["Overview", "ROC and errors", "Features", "Baseline", "Ablation", "Tuning", "Cross-validation",
                    "Diagnostics"])

    with tabs[0]:
        cols = ["Model", "Type", "CV ROC-AUC (mean)", "Accuracy", "Precision", "Recall", "Specificity", "F1-score",
                "ROC-AUC", "False positive rate", "False negative rate"]
        st.dataframe(comparison[cols], hide_index=True, use_container_width=True, column_config=metric_columns(cols))
        with st.container(border=True):
            st.markdown(ui.section("Precision, recall and F1-score"), unsafe_allow_html=True)
            cmp = comparison.set_index("Model")
            metrics = ["Precision", "Recall", "F1-score", "Accuracy"]
            html(ui.grouped_bars(metrics, [(m, [float(cmp.loc[m, k]) for k in metrics], ui.MODEL_COLORS[m])
                                           for m in MODEL_ORDER], ymax=1.0, height=330, y_label="Score",
                                 show_values=False))

    with tabs[1]:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown(ui.section("ROC curves", "True positive rate against false positive rate"),
                            unsafe_allow_html=True)
                if preds is not None:
                    names = [n for n in MODEL_ORDER if n in preds.columns]
                    lines = []
                    for n in names:
                        fpr, tpr, _ = roc_curve(preds["y_true"], preds[n])
                        lines.append((n, fpr, tpr, ui.MODEL_COLORS[n], n.startswith("Baseline")))
                    html(ui.line_chart(lines, x_label="False positive rate (1 − specificity)",
                                       y_label="True positive rate (recall)", diagonal=False))
                    aucs = comparison.set_index("Model")["ROC-AUC"]
                    st.markdown("<div class='muted'>" + " · ".join(f"{n}: AUC {aucs[n]:.3f}" for n in names) + "</div>",
                                unsafe_allow_html=True)
                else:
                    graph("13_roc_curves.png")
        with right:
            with st.container(border=True):
                st.markdown(ui.section("Confusion matrices", "Test set, 300 records"), unsafe_allow_html=True)
                which = st.radio("Model", C.MODEL_NAMES, horizontal=True, label_visibility="collapsed")
                r = comparison.set_index("Model").loc[which]
                st.markdown(ui.confusion_matrix(int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])),
                            unsafe_allow_html=True)
                st.markdown(f"<div class='muted' style='margin-top:12px'>False positive rate {r['False positive rate']:.1%} · "
                            f"false negative rate {r['False negative rate']:.1%}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            graph("18_precision_recall_curves.png", "Precision-recall curves")
        with c2:
            graph("19_calibration_curves.png", "Calibration")

    with tabs[2]:
        fr = get_table(C.TABLES_DIR, "feature_ranking.csv")
        if fr is not None:
            with st.container(border=True):
                st.markdown(ui.section("Feature ranking", "Mutual information with the diagnosis, training data only"),
                            unsafe_allow_html=True)
                rows = [(PRETTY.get(c, c.replace("_", " ")), [float(v)]) for c, v in zip(fr["feature"], fr["mutual_information"])]
                html(ui.hbars(rows, [("Mutual information", ui.ACCENT)]))
            cols = ["rank", "feature", "mutual_information", "rf_importance", "correlation_with_target"]
            st.dataframe(fr[cols], hide_index=True, use_container_width=True, column_config=metric_columns(cols))
        if run and run.get("best_model_selected_features"):
            st.markdown(f"<div class='muted'>Features kept by the selected {run['best_model']} pipeline: "
                        + ", ".join(PRETTY.get(f, f) for f in run["best_model_selected_features"]) + "</div>",
                        unsafe_allow_html=True)

    with tabs[3]:
        bt = get_table(C.METRICS_DIR, "baseline_comparison.csv")
        if bt is not None:
            st.dataframe(bt, hide_index=True, use_container_width=True, column_config=metric_columns(bt.columns))
            st.markdown("<div class='muted'>The baseline predicts the most common class (RLS) for every record. "
                        "Any useful model has to beat it clearly.</div>", unsafe_allow_html=True)
            _, mid, _ = st.columns([1, 3, 1])
            with mid:
                graph("23_baseline_comparison.png")

    with tabs[4]:
        ab = get_table(C.METRICS_DIR, "ablation_results.csv")
        if ab is not None:
            with st.container(border=True):
                st.markdown(ui.section("Random Forest trained on each feature group", "Test-set ROC-AUC and accuracy"),
                            unsafe_allow_html=True)
                rows = [(g, [float(a), float(c)]) for g, a, c in zip(ab["Feature group"], ab["ROC-AUC"], ab["Accuracy"])]
                html(ui.hbars(rows, [("ROC-AUC", ui.ACCENT), ("Accuracy", "#9DB3D6")], xmax=1.0))
            cols = ["Feature group", "Number of features", "CV ROC-AUC (mean)", "Accuracy", "Precision", "Recall",
                    "F1-score", "ROC-AUC"]
            st.dataframe(ab[cols], hide_index=True, use_container_width=True, column_config=metric_columns(cols))

    with tabs[5]:
        bp = get_table(C.METRICS_DIR, "best_parameters.csv")
        dv = get_table(C.METRICS_DIR, "default_vs_tuned.csv")
        if bp is not None:
            st.markdown(ui.section("Best settings found", "Grid search, stratified 5-fold cross-validation"),
                        unsafe_allow_html=True)
            st.dataframe(bp.fillna("–"), hide_index=True, use_container_width=True)
        if dv is not None:
            with st.container(border=True):
                st.markdown(ui.section("Test ROC-AUC before and after tuning"), unsafe_allow_html=True)
                ba = dv.pivot(index="Model", columns="Stage", values="ROC-AUC").reindex(C.MODEL_NAMES)
                html(ui.grouped_bars(C.MODEL_NAMES, [("Default", ba["Default"].tolist(), "#9DB3D6"),
                                                     ("Tuned", ba["Tuned"].tolist(), ui.ACCENT)],
                                     ymax=1.0, fmt="{:.3f}", height=300, y_label="ROC-AUC"))
        graph("20_hyperparameter_tuning.png")

    with tabs[6]:
        cv = get_table(C.METRICS_DIR, "cross_validation_scores.csv")
        if cv is not None:
            with st.container(border=True):
                st.markdown(ui.section("ROC-AUC on each training fold"), unsafe_allow_html=True)
                wide = cv.pivot(index="fold", columns="model", values="roc_auc").sort_index()
                names = [n for n in C.MODEL_NAMES if n in wide.columns]
                html(ui.grouped_bars([f"Fold {f}" for f in wide.index],
                                     [(n, wide[n].tolist(), ui.MODEL_COLORS[n]) for n in names],
                                     ymax=1.0, fmt="{:.3f}", height=310, y_label="ROC-AUC", show_values=False))
            summary = cv.groupby("model")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"]).round(3)
            summary.columns = [f"{m} {s}" for m, s in summary.columns]
            st.dataframe(summary, use_container_width=True)

    with tabs[7]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            graph("25_learning_curve.png", "Learning curve: training and validation ROC-AUC as training data grows")
        with c2:
            graph("26_permutation_importance.png", "Permutation importance of each input on the test set")
        lc = get_table(C.METRICS_DIR, "learning_curve.csv")
        if lc is not None:
            last = lc.iloc[-1]
            st.markdown(f"<div class='muted'>With {int(last['train_size'])} training records the selected model scores "
                        f"{last['train_mean']:.3f} on training folds and {last['validation_mean']:.3f} on validation folds. "
                        "The small gap means the model is not overfitting.</div>", unsafe_allow_html=True)

# ================================================================== Try the model
elif page == "Try the model":
    st.markdown(ui.page_header("Prediction", "Try the model",
                               "Enter a record's details to see the selected model's estimate. Start from typical "
                               "values or load an example record from the dataset."), unsafe_allow_html=True)
    try:
        bundle = get_model()
    except ModelNotAvailableError as exc:
        st.warning(str(exc))
        st.stop()

    all_models = get_all_models()
    form_col, result_col = st.columns([1.35, 1], gap="large")
    with form_col:
        if all_models:
            names = list(all_models)
            chosen = st.radio("Model", names, index=names.index(bundle["model_name"]) if bundle["model_name"] in names else 0,
                              horizontal=True, help="The selected model (best by cross-validation) is chosen by default.")
            bundle = all_models[chosen]
        start = st.selectbox("Start from", ["Typical values (dataset medians)"] +
                             [f"Example record {pid}" for pid in df[C.ID_COLUMN].head(30)])
        if start.startswith("Example"):
            base = df[df[C.ID_COLUMN] == start.split()[-1]].iloc[0]
        else:
            base = df[C.MODEL_FEATURES].median(numeric_only=True)
            base["Gender"] = df["Gender"].mode()[0]

        def num(col, lo, hi, step, label, where):
            v = base[col]
            v = float(df[col].median()) if pd.isna(v) else float(v)
            return where.number_input(label, min_value=float(lo), max_value=float(hi), value=min(max(v, lo), hi),
                                      step=float(step), key=f"{start}-{col}")

        with st.form("predict", border=True):
            st.markdown(ui.section("Core symptoms"), unsafe_allow_html=True)
            sym_labels = {"Unpleasant_Leg_Sensation": "Unpleasant leg sensations",
                          "Urge_to_Move_Legs": "Urge to move the legs",
                          "Symptoms_While_Resting": "Starts or worsens at rest",
                          "Symptoms_Improve_With_Movement": "Eased by movement",
                          "Symptoms_Worse_Evening_Night": "Worse in the evening or night"}
            values = {}
            s1, s2 = st.columns(2)
            for i, (col, label) in enumerate(sym_labels.items()):
                where = s1 if i % 2 == 0 else s2
                values[col] = 1 if where.toggle(label, value=bool(base[col]), key=f"{start}-{col}") else 0
            values["Family_History_RLS"] = 1 if s2.toggle("Family history of RLS", value=bool(base["Family_History_RLS"]),
                                                          key=f"{start}-fh") else 0

            st.markdown(ui.section("Symptom burden and sleep"), unsafe_allow_html=True)
            r1 = st.columns(3)
            values["Symptom_Frequency_Days_Week"] = num("Symptom_Frequency_Days_Week", 0, 7, 1, "Symptom days per week", r1[0])
            values["Leg_Discomfort_Score"] = num("Leg_Discomfort_Score", 0, 10, 1, "Leg discomfort (0–10)", r1[1])
            values["Sleep_Disturbance"] = num("Sleep_Disturbance", 0, 10, 1, "Sleep disturbance (0–10)", r1[2])
            r2 = st.columns(3)
            values["Sleep_Duration_Hours"] = num("Sleep_Duration_Hours", 0, 24, 0.1, "Sleep duration (hours)", r2[0])
            values["Sleep_Quality"] = num("Sleep_Quality", 0, 10, 0.1, "Sleep quality (0–10)", r2[1])
            values["Daytime_Sleepiness"] = num("Daytime_Sleepiness", 0, 10, 0.1, "Daytime sleepiness (0–10)", r2[2])

            st.markdown(ui.section("General"), unsafe_allow_html=True)
            r3 = st.columns(3)
            values["Age"] = num("Age", 0, 120, 1, "Age (years)", r3[0])
            g_opts = C.ALLOWED_CATEGORIES["Gender"]
            values["Gender"] = r3[1].selectbox("Gender", g_opts,
                                               index=g_opts.index(base["Gender"]) if base["Gender"] in g_opts else 0,
                                               key=f"{start}-Gender")
            values["BMI"] = num("BMI", 10, 70, 0.1, "BMI (kg/m²)", r3[2])
            r4 = st.columns(3)
            values["Ferritin_Level_ng_mL"] = num("Ferritin_Level_ng_mL", 0, 1000, 1, "Ferritin (ng/mL)", r4[0])
            values["Fatigue_Level"] = num("Fatigue_Level", 0, 10, 0.1, "Fatigue (0–10)", r4[1])
            values["Stress_Level"] = num("Stress_Level", 0, 10, 0.1, "Stress (0–10)", r4[2])
            r5 = st.columns(3)
            values["Anxiety_Level"] = num("Anxiety_Level", 0, 10, 0.1, "Anxiety (0–10)", r5[0])
            values["Physical_Activity_Level"] = num("Physical_Activity_Level", 0, 10, 0.1, "Physical activity (0–10)", r5[1])
            submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

    with result_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if not submitted:
            st.markdown("<div class='result'><div class='section-title'>Result</div><div class='muted'>Fill in the "
                        "form and select <b>Run prediction</b>. The estimate and the features that drove it appear "
                        "here.</div></div>", unsafe_allow_html=True)
        else:
            try:
                res = predict(values, bundle).iloc[0]
            except PredictionInputError as exc:
                st.error(str(exc))
            else:
                recorded = C.CLASS_LABELS[int(base[C.TARGET])] if start.startswith("Example") else ""
                st.markdown(ui.prediction_card(res["predicted_class"], float(res["probability_rls"]),
                                               bundle["model_name"], recorded), unsafe_allow_html=True)
                pipe = bundle["pipeline"]
                clf = pipe.named_steps.get("clf")
                if hasattr(clf, "coef_"):
                    # Contribution of each selected feature in log-odds, relative to an average record.
                    front = pipe[:-1]
                    x = front.transform(pd.DataFrame([values])[C.MODEL_FEATURES])[0]
                    avg = front.transform(df[C.MODEL_FEATURES]).mean(axis=0)
                    contrib = clf.coef_[0] * (x - avg)
                    names = selected_feature_names(pipe)
                    rows = sorted(zip([PRETTY.get(n, n.replace("_", " ")) for n in names], contrib),
                                  key=lambda t: -abs(t[1]))[:8]
                    st.markdown("<div class='result' style='margin-top:14px'><div class='section-title'>What drove "
                                "this estimate</div><div class='muted'>Push toward RLS (right) or away from it (left), "
                                "compared with an average record</div>"
                                + ui.contributions(rows, scale=max(1.0, max(abs(v) for _, v in rows)))
                                + "</div>", unsafe_allow_html=True)
                if len(all_models) > 1:
                    probs = [(n, float(predict(values, b).iloc[0]["probability_rls"])) for n, b in all_models.items()]
                    cmp_rows = "".join(
                        f"<span>{n}</span><div class='bar'><span style='left:0;width:{p * 100:.1f}%;background:"
                        f"{ui.MODEL_COLORS[n]}'></span><em></em></div><span class='v'>{p:.0%}</span>" for n, p in probs)
                    st.markdown("<div class='result' style='margin-top:14px'><div class='section-title'>All three models"
                                "</div><div class='muted'>Estimated probability of RLS for the same inputs; the line marks "
                                f"the 50% threshold</div><div class='contrib'>{cmp_rows}</div></div>", unsafe_allow_html=True)
                st.caption("A screening demonstration only, not a medical diagnosis.")

# ================================================================== Methodology
elif page == "Methodology":
    st.markdown(ui.page_header("Methodology", "How the system works",
                               "Five modules take the raw records to an evaluated, explainable prediction. Every "
                               "learned step is fitted on the training data only."), unsafe_allow_html=True)
    sel = run.get("best_model_selected_features", []) if run else []
    html(ui.steps([
        ("Data collection", f"Load and validate {len(df):,} records and {df.shape[1]} columns. Check the target, "
                            "expected columns, value ranges and duplicates. Drop the ID and two outcome-derived "
                            "columns that would leak the answer."),
        ("Preprocessing", "Median imputation for numeric gaps, most-frequent imputation for flags and gender, one-hot "
                          "encoding, and z-score scaling for Logistic Regression. Stratified 80/20 split."),
        ("Feature selection", "Mutual information ranks every feature. SelectKBest sits inside each pipeline, and the "
                              f"number kept is tuned. The selected model keeps {len(sel) or 'the top'} features."),
        ("ML prediction", "Logistic Regression, Decision Tree and Random Forest, each tuned by grid search with "
                          "stratified 5-fold cross-validation. Best model chosen on cross-validated ROC-AUC."),
        ("Evaluation", "Accuracy, precision, recall, specificity, F1-score, ROC-AUC and error rates on a held-out "
                       "test set, plus baseline, ablation, tuning, learning-curve and threshold analyses."),
    ]))
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            html(ui.section("The three models"))
            html("<div class='muted' style='line-height:1.7'><b>Logistic Regression</b> estimates the log-odds of RLS "
                 "as a weighted sum of the inputs, <span class='formula'>p = 1 / (1 + e<sup>−z</sup>)</span>.<br>"
                 "<b>Decision Tree</b> splits records with if-then rules chosen by Gini impurity, with limits on depth "
                 "and leaf size.<br><b>Random Forest</b> averages many trees trained on bootstrap samples with random "
                 "feature subsets.<br><b>Baseline</b> always predicts the most common class, as a floor to beat.</div>")
    with right:
        with st.container(border=True):
            html(ui.section("Evaluation metrics"))
            html("<div class='muted' style='line-height:1.9'>"
                 "Accuracy <span class='formula'>(TP + TN) / N</span><br>"
                 "Precision <span class='formula'>TP / (TP + FP)</span><br>"
                 "Recall <span class='formula'>TP / (TP + FN)</span><br>"
                 "Specificity <span class='formula'>TN / (TN + FP)</span><br>"
                 "F1-score <span class='formula'>2 · P · R / (P + R)</span><br>"
                 "ROC-AUC: area under the true-positive vs false-positive curve</div>")
    with st.container(border=True):
        html(ui.section("Leakage control", "Why two columns were removed"))
        leak = run.get("leakage_check_accuracy_with_RLS_Severity") if run else None
        html("<div class='muted'>RLS Severity is 0 for every No RLS record, and RLS Probability is derived from the "
             "outcome. Both reveal the label, so they are never used as inputs."
             + (f" A model allowed to use RLS Severity reaches {leak:.0%} test accuracy, which is meaningless." if leak else "")
             + "</div>")

# ================================================================== Threshold explorer
elif page == "Threshold explorer":
    st.markdown(ui.page_header("Threshold explorer", "Move the decision threshold",
                               "A record is called RLS when its predicted probability is at or above the threshold. "
                               "Lower it to catch more RLS cases; raise it to avoid false alarms."), unsafe_allow_html=True)
    if preds is None or comparison is None:
        need_results()
    models = [m for m in C.MODEL_NAMES if m in preds.columns]
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        which = st.radio("Model", models, index=models.index(run["best_model"]) if run and run["best_model"] in models else 0,
                         horizontal=True)
    with c2:
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    y, p = preds["y_true"].to_numpy(), preds[which].to_numpy()
    m = metrics_at_threshold(y, p, thr)
    d = metrics_at_threshold(y, p, 0.5)
    delta = lambda k: f"{(m[k] - d[k]) * 100:+.1f} pts vs 0.50"
    html(ui.kpi_cards([("Accuracy", f"{m['Accuracy']:.1%}", delta("Accuracy"), False),
                       ("Precision", f"{m['Precision']:.1%}", delta("Precision"), False),
                       ("Recall", f"{m['Recall']:.1%}", delta("Recall"), True),
                       ("Specificity", f"{m['Specificity']:.1%}", delta("Specificity"), False),
                       ("F1-score", f"{m['F1-score']:.3f}", f"{m['F1-score'] - d['F1-score']:+.3f} vs 0.50", False)]))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            html(ui.section("Metrics across all thresholds", f"{which}, 300 test records"))
            sw = threshold_sweep(y, p, np.round(np.arange(0.05, 0.951, 0.01), 2))
            cols = [("Precision", "#2F4B7C"), ("Recall", "#C0703A"), ("Specificity", "#3C8D5A"), ("F1-score", "#7A6C8F")]
            html(ui.line_chart([(k, sw["threshold"], sw[k], c, False) for k, c in cols], x_label="Decision threshold",
                               y_label="Score", diagonal=False, height=380, vline=thr))
    with right:
        with st.container(border=True):
            html(ui.section("Confusion matrix at this threshold"))
            html(ui.confusion_matrix(m["TN"], m["FP"], m["FN"], m["TP"]))
            html(f"<div class='muted' style='margin-top:12px'>{m['FN']} RLS records missed and {m['FP']} false alarms "
                 f"out of {len(y)}. At 0.50 these were {d['FN']} and {d['FP']}.</div>")

# ================================================================== Batch prediction
elif page == "Batch prediction":
    st.markdown(ui.page_header("Batch prediction", "Score many records at once",
                               "Upload a CSV with the 20 input columns. Each row gets a predicted class and probability, "
                               "and you can download the results."), unsafe_allow_html=True)
    try:
        bundle = get_model()
    except ModelNotAvailableError as exc:
        st.warning(str(exc))
        st.stop()
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        upload = st.file_uploader("CSV file", type=["csv"])
    with c2:
        st.download_button("Download a blank template", template_csv(), file_name="rls_batch_template.csv",
                           mime="text/csv", use_container_width=True)
        sample = st.toggle("Use 25 sample records from the dataset instead", value=upload is None)
    data = None
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read the file as CSV: {exc}")
    elif sample:
        data = df.sample(25, random_state=7).drop(columns=[C.TARGET] + C.LEAKAGE_COLUMNS)
    if data is not None:
        try:
            out = predict(data, bundle)
        except PredictionInputError as exc:
            st.error(str(exc))
        else:
            result = pd.concat([data.reset_index(drop=True)[[C.ID_COLUMN]] if C.ID_COLUMN in data else
                                pd.DataFrame({"row": range(1, len(data) + 1)}),
                                out[["predicted_class", "probability_rls"]]], axis=1)
            n_rls = int((out["predicted_label"] == 1).sum())
            html(ui.kpi_cards([("Records scored", f"{len(out):,}", f"model: {bundle['model_name']}", True),
                               ("Predicted RLS", f"{n_rls}", f"{n_rls / len(out):.0%} of records", False),
                               ("Predicted No RLS", f"{len(out) - n_rls}", f"{1 - n_rls / len(out):.0%} of records", False),
                               ("Mean probability", f"{out['probability_rls'].mean():.0%}", "estimated chance of RLS", False)]))
            st.dataframe(result, hide_index=True, use_container_width=True, height=420, column_config={
                "probability_rls": st.column_config.ProgressColumn("Probability of RLS", format="%.2f",
                                                                   min_value=0.0, max_value=1.0)})
            st.download_button("Download predictions (CSV)", result.to_csv(index=False), file_name="rls_predictions.csv",
                               mime="text/csv", type="primary")
    st.caption("A screening demonstration only, not a medical diagnosis.")

# ================================================================== Downloads
elif page == "Downloads":
    st.markdown(ui.page_header("Downloads", "Results and report",
                               "Everything produced by the training run, ready to attach to a report or presentation."),
                unsafe_allow_html=True)
    rows = [
        ("Experiment report", "Full write-up of the experiment, generated from the results (Markdown).",
         C.REPORTS_DIR / "experiment_report.md", "text/markdown"),
        ("Model comparison table", "Test-set metrics for every model and the baseline (CSV).",
         C.METRICS_DIR / "model_comparison.csv", "text/csv"),
        ("Feature ranking", "Mutual information, Random Forest importance and correlation per feature (CSV).",
         C.TABLES_DIR / "feature_ranking.csv", "text/csv"),
    ]
    for title, desc, path, mime in rows:
        with st.container(border=True):
            a, b = st.columns([3, 1], vertical_alignment="center")
            a.markdown(ui.section(title, desc), unsafe_allow_html=True)
            if path.exists():
                b.download_button("Download", path.read_bytes(), file_name=path.name, mime=mime,
                                  use_container_width=True, key=f"dl-{path.name}")
            else:
                b.caption("Run main.py first")
    bundles = [("All graphs", "Every figure as a high-resolution PNG (zip).", ["graphs"], "rls_graphs.zip"),
               ("All tables and metrics", "Dataset summaries, metrics, tuning, ablation and CV results (zip).",
                ["tables", "metrics"], "rls_tables_metrics.zip"),
               ("Everything", "Graphs, tables, metrics and reports in one file (zip).",
                ["graphs", "tables", "metrics", "reports"], "rls_all_results.zip")]
    for title, desc, folders, fname in bundles:
        with st.container(border=True):
            a, b = st.columns([3, 1], vertical_alignment="center")
            a.markdown(ui.section(title, desc), unsafe_allow_html=True)
            b.download_button("Download", zip_folder(tuple(folders)), file_name=fname, mime="application/zip",
                              use_container_width=True, key=f"dl-{fname}")
