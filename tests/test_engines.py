import unittest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_ulb_dataset, preprocess_data, PCA_FEATURE_DESCRIPTIONS, detect_local_dataset, load_custom_csv
from src.ml_engine import FraudMLPipeline
from src.sorites_engine import (
    compute_elkan_threshold,
    evaluate_threshold_cost,
    analyze_contested_zone,
    find_sorites_twin_transactions,
    run_cost_ratio_sweep,
    run_threshold_cost_curve
)

class TestThresholdLabEngines(unittest.TestCase):
    def test_elkan_formula(self):
        # If C(FP) = 50 and C(FN) = 950, tau* should be 50 / 1000 = 0.05
        tau = compute_elkan_threshold(50.0, 950.0)
        self.assertAlmostEqual(tau, 0.05)

        # If C(FP) = C(FN), tau* = 0.5
        tau_equal = compute_elkan_threshold(100.0, 100.0)
        self.assertAlmostEqual(tau_equal, 0.5)

    def test_dataset_generation(self):
        df = generate_synthetic_ulb_dataset(n_samples=1000, fraud_rate=0.01, random_state=42)
        self.assertEqual(len(df), 1000)
        self.assertIn('Class', df.columns)
        self.assertIn('Time', df.columns)
        self.assertIn('Amount', df.columns)
        self.assertIn('V1', df.columns)
        self.assertIn('V28', df.columns)
        self.assertEqual(len(PCA_FEATURE_DESCRIPTIONS), 28)

    def test_preprocessing(self):
        df = generate_synthetic_ulb_dataset(n_samples=500, fraud_rate=0.02, random_state=42)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train) + len(X_test), len(df))
        self.assertIsNotNone(scaler)

    def test_ml_pipeline_and_calibration(self):
        df = generate_synthetic_ulb_dataset(n_samples=600, fraud_rate=0.05, random_state=42)
        X_train, X_test, y_train, y_test, _ = preprocess_data(df, test_size=0.2, random_state=42)
        
        for m_type in ['logistic', 'ensemble']:
            pipeline = FraudMLPipeline(model_type=m_type)
            pipeline.fit(X_train, y_train)
            
            probas = pipeline.predict_probabilities(X_test)
            self.assertIn('raw', probas)
            self.assertIn('platt', probas)
            self.assertIn('isotonic', probas)
            
            metrics = pipeline.evaluate_calibration(y_test, probas)
            for mode in ['raw', 'platt', 'isotonic']:
                self.assertIn('brier_score', metrics[mode])
                self.assertIn('log_loss', metrics[mode])
                self.assertIn('pr_auc', metrics[mode])
                self.assertGreaterEqual(metrics[mode]['brier_score'], 0.0)

    def test_contested_zone(self):
        df = generate_synthetic_ulb_dataset(n_samples=500, fraud_rate=0.05, random_state=42)
        y_probas = np.random.uniform(0, 0.2, len(df))
        
        res = analyze_contested_zone(df, y_probas, c_fp_range=(10.0, 100.0), c_fn=1000.0)
        self.assertTrue(res['tau_min'] < res['tau_max'])
        self.assertGreaterEqual(res['contested_count'], 0)

    def test_threshold_cost_curve(self):
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        y_probas = np.array([0.1, 0.2, 0.05, 0.8, 0.9, 0.4, 0.7, 0.15])
        curve_df = run_threshold_cost_curve(y_true, y_probas, c_fp=25.0, c_fn=1000.0, n_thresholds=20)
        self.assertEqual(len(curve_df), 20)
        self.assertIn('Total_Cost', curve_df.columns)
        self.assertIn('Threshold', curve_df.columns)

    def test_twin_transactions(self):
        df = generate_synthetic_ulb_dataset(n_samples=100, fraud_rate=0.1, random_state=42)
        y_probas = np.linspace(0.01, 0.99, len(df))
        res = find_sorites_twin_transactions(df, y_probas, tau_star=0.50)
        self.assertIn('tx_a', res)
        self.assertIn('tx_b', res)
        self.assertIn('diff', res)

class TestLoadCustomCsv(unittest.TestCase):
    """Covers the data-processing step applied to uploaded/local CSV files."""

    def _required_columns_row(self, time_v=1, amount_v=10.0, cls=0, v28_override=None):
        vals = [f"{i/10:.1f}" for i in range(1, 29)]
        if v28_override is not None:
            vals[-1] = str(v28_override)
        return f"{time_v},{amount_v},{','.join(vals)},{cls}"

    def _header(self, lowercase=False, extra_col=False):
        cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["Class"]
        if lowercase:
            cols = [c.lower() for c in cols]
        if extra_col:
            cols = ["Unnamed: 0"] + cols
        return ",".join(cols)

    def test_valid_csv_round_trips_cleanly(self):
        import io
        csv_text = self._header() + "\n" + self._required_columns_row(1, 10.0, 0) + "\n" + self._required_columns_row(2, 20.0, 1) + "\n"
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertEqual(len(df), 2)
        self.assertEqual(report['rows_read'], 2)
        self.assertEqual(report['rows_final'], 2)
        self.assertEqual(sorted(df['Class'].unique().tolist()), [0, 1])

    def test_lowercase_headers_are_renamed(self):
        import io
        csv_text = self._header(lowercase=True) + "\n" + self._required_columns_row(1, 10.0, 0) + "\n" + self._required_columns_row(2, 20.0, 1) + "\n"
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertListEqual(list(df.columns), ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)])
        self.assertGreater(len(report['renamed_columns']), 0)

    def test_extra_column_is_dropped(self):
        import io
        header = "Unnamed: 0," + self._header()
        csv_text = header + "\n0," + self._required_columns_row(1, 10.0, 0) + "\n1," + self._required_columns_row(2, 20.0, 1) + "\n"
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertNotIn('Unnamed: 0', df.columns)
        self.assertIn('Unnamed: 0', report['dropped_extra_columns'])

    def test_duplicate_rows_are_removed(self):
        import io
        row = self._required_columns_row(1, 10.0, 1)
        csv_text = self._header() + "\n" + row + "\n" + self._required_columns_row(2, 20.0, 0) + "\n" + row + "\n"
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertEqual(report['duplicates_removed'], 1)
        self.assertEqual(len(df), 2)

    def test_invalid_class_value_is_dropped(self):
        import io
        csv_text = (
            self._header() + "\n"
            + self._required_columns_row(1, 10.0, 0) + "\n"
            + self._required_columns_row(2, 20.0, 7) + "\n"   # invalid Class
            + self._required_columns_row(3, 30.0, 1) + "\n"
        )
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertEqual(report['invalid_rows_removed'], 1)
        self.assertEqual(len(df), 2)

    def test_missing_amount_value_is_dropped(self):
        import io
        csv_text = (
            self._header() + "\n"
            + self._required_columns_row(1, 10.0, 1) + "\n"
            + self._required_columns_row(2, "", 1) + "\n"    # missing Amount
        )
        df, report = load_custom_csv(io.StringIO(csv_text))
        self.assertEqual(report['invalid_rows_removed'], 1)
        self.assertEqual(len(df), 1)

    def test_missing_required_column_raises(self):
        import io
        csv_text = "Time,Amount,Class\n1,10,0\n"
        with self.assertRaises(ValueError):
            load_custom_csv(io.StringIO(csv_text))

    def test_no_fraud_rows_raises(self):
        import io
        rows = "\n".join(self._required_columns_row(i, 10.0, 0) for i in range(5))
        csv_text = self._header() + "\n" + rows + "\n"
        with self.assertRaises(ValueError):
            load_custom_csv(io.StringIO(csv_text))

    def test_class_dtype_is_int(self):
        import io
        csv_text = self._header() + "\n" + self._required_columns_row(1, 10.0, 0) + "\n" + self._required_columns_row(2, 20.0, 1) + "\n"
        df, _ = load_custom_csv(io.StringIO(csv_text))
        self.assertTrue(pd.api.types.is_integer_dtype(df['Class']))


if __name__ == '__main__':
    unittest.main()
