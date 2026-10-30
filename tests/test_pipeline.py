"""Automated tests.  Run with:  python -m pytest -q   (or: python -m unittest discover tests)

They use the real dataset but small, fast models, and never depend on files
created by main.py, so they pass on a fresh checkout.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data_loader import DataValidationError, load_dataset, validate_dataset  # noqa: E402
from src.evaluation import compute_metrics, metrics_at_threshold, threshold_sweep  # noqa: E402
from src.prediction import PredictionInputError, predict, template_csv, validate_input  # noqa: E402
from src.preprocessing import make_preprocessor, split_features_target, stratified_split  # noqa: E402
from src.train_models import build_baseline, build_pipeline  # noqa: E402


def sample_record(df):
    """One valid input record taken from the dataset (model features only)."""
    return df[C.MODEL_FEATURES].dropna().iloc[0].to_dict()


class TestDatasetLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_shape_and_columns(self):
        self.assertEqual(self.df.shape, (1500, 24))
        self.assertTrue(set(C.EXPECTED_COLUMNS).issubset(self.df.columns))

    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(DataValidationError) as ctx:
            load_dataset(Path("does_not_exist.csv"))
        self.assertIn("not found", str(ctx.exception))

    def test_unreadable_file_gives_clear_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("")
        with self.assertRaises(DataValidationError):
            load_dataset(Path(f.name))

    def test_missing_expected_column(self):
        with self.assertRaises(DataValidationError):
            validate_dataset(self.df.drop(columns=["Age"]))


class TestTargetValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_target_is_binary(self):
        self.assertEqual(set(self.df[C.TARGET].unique()), {0, 1})

    def test_invalid_label_rejected(self):
        bad = self.df.copy()
        bad.loc[0, C.TARGET] = 2
        with self.assertRaises(DataValidationError):
            validate_dataset(bad)

    def test_missing_target_column_rejected(self):
        with self.assertRaises(DataValidationError):
            validate_dataset(self.df.drop(columns=[C.TARGET]))

    def test_missing_target_values_rejected(self):
        bad = self.df.copy()
        bad[C.TARGET] = bad[C.TARGET].astype(float)
        bad.loc[0, C.TARGET] = np.nan
        with self.assertRaises(DataValidationError):
            validate_dataset(bad)

    def test_single_class_rejected(self):
        bad = self.df.copy()
        bad[C.TARGET] = 1
        with self.assertRaises(DataValidationError):
            validate_dataset(bad)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.X, cls.y = split_features_target(cls.df)

    def test_excluded_columns_not_in_features(self):
        for col in C.EXCLUDED_COLUMNS:
            self.assertNotIn(col, self.X.columns)
        self.assertEqual(self.X.shape[1], 20)

    def test_leakage_column_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            split_features_target(self.df, features=C.MODEL_FEATURES + ["RLS_Severity"])

    def test_no_missing_values_after_transform(self):
        X_tr, X_te, _, _ = stratified_split(self.X, self.y)
        pre = make_preprocessor().fit(X_tr)
        self.assertFalse(np.isnan(pre.transform(X_te)).any())

    def test_imputation_uses_training_medians_only(self):
        X_tr, _, _, _ = stratified_split(self.X, self.y)
        pre = make_preprocessor(scale=False).fit(X_tr)
        learned = pre.named_transformers_["num"].named_steps["impute"].statistics_
        expected = X_tr[C.NUMERIC_FEATURES].median().to_numpy()
        np.testing.assert_allclose(learned, expected)

    def test_unknown_gender_does_not_crash(self):
        X_tr, X_te, _, _ = stratified_split(self.X, self.y)
        pre = make_preprocessor().fit(X_tr)
        odd = X_te.head(3).copy()
        odd["Gender"] = "Unknown"
        self.assertEqual(pre.transform(odd).shape[0], 3)


class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = split_features_target(load_dataset())

    def test_sizes(self):
        X_tr, X_te, y_tr, y_te = stratified_split(self.X, self.y)
        self.assertEqual((len(X_tr), len(X_te)), (1200, 300))

    def test_stratified(self):
        _, _, y_tr, y_te = stratified_split(self.X, self.y)
        self.assertAlmostEqual(y_tr.mean(), self.y.mean(), delta=0.01)
        self.assertAlmostEqual(y_te.mean(), self.y.mean(), delta=0.01)

    def test_reproducible_and_disjoint(self):
        a = stratified_split(self.X, self.y)
        b = stratified_split(self.X, self.y)
        self.assertListEqual(list(a[1].index), list(b[1].index))
        self.assertEqual(len(set(a[0].index) & set(a[1].index)), 0)


class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = split_features_target(load_dataset())
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te = stratified_split(X, y)

    def test_each_model_trains_and_predicts(self):
        for name in C.MODEL_NAMES:
            with self.subTest(model=name):
                pipe = build_pipeline(name).fit(self.X_tr, self.y_tr)
                prob = pipe.predict_proba(self.X_te)[:, 1]
                self.assertEqual(prob.shape, (300,))
                self.assertTrue(((prob >= 0) & (prob <= 1)).all())
                acc = (pipe.predict(self.X_te) == self.y_te).mean()
                self.assertGreater(acc, 0.6, f"{name} should beat chance on this data")

    def test_baseline_predicts_majority_class(self):
        pipe = build_baseline().fit(self.X_tr, self.y_tr)
        self.assertEqual(set(pipe.predict(self.X_te)), {int(self.y_tr.mode()[0])})

    def test_feature_selection_k(self):
        pipe = build_pipeline("Logistic Regression", k=5).fit(self.X_tr, self.y_tr)
        self.assertEqual(pipe.named_steps["select"].get_support().sum(), 5)

    def test_unknown_model_name(self):
        with self.assertRaises(ValueError):
            build_pipeline("SVM")


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        p = [1, 1, 0, 0, 0, 1, 0, 1]
        prob = [0.9, 0.8, 0.4, 0.2, 0.1, 0.6, 0.3, 0.7]
        m = compute_metrics(y, p, prob)
        self.assertEqual((m["TP"], m["TN"], m["FP"], m["FN"]), (3, 3, 1, 1))
        self.assertAlmostEqual(m["Accuracy"], 0.75)
        self.assertAlmostEqual(m["Precision"], 0.75)
        self.assertAlmostEqual(m["Recall"], 0.75)
        self.assertAlmostEqual(m["Specificity"], 0.75)
        self.assertAlmostEqual(m["F1-score"], 0.75)
        self.assertAlmostEqual(m["False positive rate"], 0.25)
        self.assertAlmostEqual(m["False negative rate"], 0.25)
        self.assertAlmostEqual(m["ROC-AUC"], 0.9375)

    def test_zero_division_handled(self):
        m = compute_metrics([0, 1, 0, 1], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(m["Precision"], 0.0)
        self.assertEqual(m["Recall"], 0.0)

    def test_threshold_changes_predictions(self):
        y, prob = [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]
        self.assertEqual(metrics_at_threshold(y, prob, 0.5)["TP"], 1)
        self.assertEqual(metrics_at_threshold(y, prob, 0.3)["TP"], 2)
        self.assertEqual(metrics_at_threshold(y, prob, 0.95)["TP"], 0)

    def test_threshold_sweep_is_monotonic_in_recall(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        prob = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
        recall = threshold_sweep(y, prob)["Recall"].to_numpy()
        self.assertTrue((np.diff(recall) <= 1e-12).all(), "recall can only fall as the threshold rises")

    def test_auc_undefined_with_one_class(self):
        m = compute_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.4])
        self.assertTrue(np.isnan(m["ROC-AUC"]))


class TestPredictionValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        X, y = split_features_target(cls.df)
        pipe = build_pipeline("Logistic Regression").fit(X, y)
        cls.bundle = {"pipeline": pipe, "model_name": "Logistic Regression"}
        cls.record = sample_record(cls.df)

    def test_valid_record_predicts(self):
        out = predict(self.record, self.bundle)
        self.assertIn(out.loc[0, "predicted_class"], ("RLS", "No RLS"))
        self.assertTrue(0 <= out.loc[0, "probability_rls"] <= 1)

    def test_missing_numeric_value_is_imputed(self):
        rec = dict(self.record, Ferritin_Level_ng_mL=None)
        self.assertEqual(len(predict(rec, self.bundle)), 1)

    def test_leakage_and_target_fields_ignored(self):
        rec = dict(self.record, RLS_Severity=99, RLS_Probability=1.0, RLS_Diagnosis=1)
        self.assertListEqual(list(validate_input(rec).columns), C.MODEL_FEATURES)

    def test_missing_field_rejected(self):
        rec = dict(self.record)
        rec.pop("Age")
        with self.assertRaises(PredictionInputError):
            validate_input(rec)

    def test_invalid_binary_rejected(self):
        with self.assertRaises(PredictionInputError):
            validate_input(dict(self.record, Urge_to_Move_Legs=3))

    def test_out_of_range_rejected(self):
        with self.assertRaises(PredictionInputError):
            validate_input(dict(self.record, Symptom_Frequency_Days_Week=9))

    def test_non_numeric_rejected(self):
        with self.assertRaises(PredictionInputError):
            validate_input(dict(self.record, Age="forty"))

    def test_unknown_gender_rejected(self):
        with self.assertRaises(PredictionInputError):
            validate_input(dict(self.record, Gender="X"))

    def test_wrong_input_type_rejected(self):
        with self.assertRaises(PredictionInputError):
            validate_input("not a record")

    def test_template_has_every_required_column(self):
        cols = template_csv().strip().split(",")
        self.assertTrue(set(C.MODEL_FEATURES).issubset(cols))

    def test_dataframe_input(self):
        out = predict(self.df.head(5), self.bundle)
        self.assertEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main()
