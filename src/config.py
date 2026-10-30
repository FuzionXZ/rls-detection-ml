"""Central settings: paths, column roles, feature groups and the random seed.

Every other module imports from here, so a change (for example a new file name)
only has to be made once.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "rls_dataset.csv"
SUPPLIED_DICTIONARY_PATH = DATA_DIR / "rls_data_dictionary.csv"
GENERATED_DICTIONARY_PATH = DATA_DIR / "data_dictionary_generated.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
GRAPHS_DIR = RESULTS_DIR / "graphs"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
OUTPUT_DIRS = [MODELS_DIR, METRICS_DIR, GRAPHS_DIR, TABLES_DIR, REPORTS_DIR]

# ---------------------------------------------------------------- experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"  # used to pick the best model (cross-validated, training data only)

# ---------------------------------------------------------------- column roles
TARGET = "RLS_Diagnosis"
ID_COLUMN = "Patient_ID"
# Derived from the outcome, so using them as inputs would leak the answer.
LEAKAGE_COLUMNS = ["RLS_Probability", "RLS_Severity"]
EXCLUDED_COLUMNS = [ID_COLUMN, TARGET] + LEAKAGE_COLUMNS

CATEGORICAL_FEATURES = ["Gender"]
BINARY_FEATURES = [
    "Family_History_RLS",
    "Unpleasant_Leg_Sensation",
    "Urge_to_Move_Legs",
    "Symptoms_Improve_With_Movement",
    "Symptoms_Worse_Evening_Night",
    "Symptoms_While_Resting",
]
NUMERIC_FEATURES = [
    "Age",
    "BMI",
    "Sleep_Duration_Hours",
    "Sleep_Quality",
    "Fatigue_Level",
    "Stress_Level",
    "Daytime_Sleepiness",
    "Physical_Activity_Level",
    "Anxiety_Level",
    "Ferritin_Level_ng_mL",
    "Leg_Discomfort_Score",
    "Symptom_Frequency_Days_Week",
    "Sleep_Disturbance",
]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = [ID_COLUMN] + MODEL_FEATURES + LEAKAGE_COLUMNS + [TARGET]

CLASS_LABELS = {0: "No RLS", 1: "RLS"}

# Allowed value ranges, used to flag unexpected values during validation.
VALUE_RANGES = {
    "Age": (0, 120), "BMI": (10, 70), "Sleep_Duration_Hours": (0, 24),
    "Sleep_Quality": (0, 10), "Fatigue_Level": (0, 10), "Stress_Level": (0, 10),
    "Daytime_Sleepiness": (0, 10), "Physical_Activity_Level": (0, 10), "Anxiety_Level": (0, 10),
    "Ferritin_Level_ng_mL": (0, 1000), "Leg_Discomfort_Score": (0, 10),
    "Symptom_Frequency_Days_Week": (0, 7), "Sleep_Disturbance": (0, 10),
}
ALLOWED_CATEGORIES = {"Gender": ["Female", "Male"]}

# ---------------------------------------------------------------- ablation feature groups
SYMPTOM_COLUMNS = [
    "Unpleasant_Leg_Sensation", "Urge_to_Move_Legs", "Symptoms_Improve_With_Movement",
    "Symptoms_Worse_Evening_Night", "Symptoms_While_Resting",
]
FEATURE_GROUPS = {
    "A. Demographic and general": [
        "Age", "Gender", "BMI", "Fatigue_Level", "Stress_Level", "Physical_Activity_Level",
        "Anxiety_Level", "Ferritin_Level_ng_mL", "Family_History_RLS",
    ],
    "B. Sleep-related": [
        "Sleep_Duration_Hours", "Sleep_Quality", "Daytime_Sleepiness", "Sleep_Disturbance",
    ],
    "C. RLS symptom-related": SYMPTOM_COLUMNS + ["Leg_Discomfort_Score", "Symptom_Frequency_Days_Week"],
    "D. All eligible features": MODEL_FEATURES,
}

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]


def model_file(name: str, tuned: bool = True) -> Path:
    """File name for a saved pipeline, e.g. models/random_forest_tuned.joblib."""
    slug = name.lower().replace(" ", "_")
    return MODELS_DIR / f"{slug}_{'tuned' if tuned else 'default'}.joblib"


BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def ensure_output_dirs() -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
