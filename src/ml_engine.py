import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss, log_loss

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

class FraudMLPipeline:
    def __init__(self, model_type: str = 'logistic', imbalance_strategy: str = 'class_weight'):
        """
        Initializes the ML pipeline with the chosen classifier type ('logistic', 'ensemble', or 'xgboost')
        and imbalance strategy ('class_weight', 'smote', or 'none').
        """
        self.model_type = model_type
        self.imbalance_strategy = imbalance_strategy
        self.raw_model = None
        self.platt_model = None
        self.isotonic_model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains baseline model and fits Platt-scaled and Isotonic-calibrated models.
        """
        X_tr = X_train
        y_tr = y_train

        # Handle SMOTE if requested
        if self.imbalance_strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_tr, y_tr = smote.fit_resample(X_train, y_train)
            except Exception:
                pass

        use_weights = (self.imbalance_strategy == 'class_weight')

        # 1. Initialize Base Classifier
        if self.model_type == 'logistic':
            class_weight = 'balanced' if use_weights else None
            base_clf = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        elif self.model_type == 'xgboost' and XGB_AVAILABLE:
            scale_pos_weight = ((len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))) if use_weights else 1.0
            base_clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'
            )
        elif self.model_type in ['ensemble', 'xgboost']:
            class_weight = 'balanced_subsample' if use_weights else None
            base_clf = RandomForestClassifier(
                n_estimators=100, max_depth=8, class_weight=class_weight, random_state=42, n_jobs=-1
            )
        else:
            class_weight = 'balanced' if use_weights else None
            base_clf = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)

        # 2. Fit Base Raw Classifier
        base_clf.fit(X_tr, y_tr)
        self.raw_model = base_clf

        # 3. Fit Platt-Scaled Classifier (Sigmoid calibration)
        platt_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
        platt_clf.fit(X_tr, y_tr)
        self.platt_model = platt_clf

        # 4. Fit Isotonic-Calibrated Classifier (Piecewise-monotone non-parametric calibration)
        isotonic_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=3)
        isotonic_clf.fit(X_tr, y_tr)
        self.isotonic_model = isotonic_clf

    def predict_probabilities(self, X_test: pd.DataFrame) -> dict:
        """
        Returns estimated fraud probabilities p(x) for all 3 calibration variants:
        - raw: Uncalibrated probabilities from base model
        - platt: Platt-scaled probabilities
        - isotonic: Isotonic-calibrated probabilities
        """
        if self.raw_model is None:
            raise ValueError("Model has not been fitted yet!")

        p_raw = self.raw_model.predict_proba(X_test)[:, 1]
        p_platt = self.platt_model.predict_proba(X_test)[:, 1]
        p_isotonic = self.isotonic_model.predict_proba(X_test)[:, 1]

        return {
            'raw': p_raw,
            'platt': p_platt,
            'isotonic': p_isotonic
        }

    def evaluate_calibration(self, y_true: pd.Series, prob_dict: dict) -> dict:
        """
        Computes Brier Score Loss, Log-Loss (Cross-Entropy), Average Precision (PR-AUC),
        and reliability curve points for each calibration variant.
        """
        metrics = {}
        eps = 1e-15
        for mode, probas in prob_dict.items():
            probas_clipped = np.clip(probas, eps, 1 - eps)
            brier = brier_score_loss(y_true, probas)
            ll = log_loss(y_true, probas_clipped)
            pr_auc = average_precision_score(y_true, probas)
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, probas, n_bins=10, strategy='quantile'
            )
            precision, recall, thresholds = precision_recall_curve(y_true, probas)

            metrics[mode] = {
                'brier_score': brier,
                'log_loss': ll,
                'pr_auc': pr_auc,
                'reliability_true': fraction_of_positives,
                'reliability_pred': mean_predicted_value,
                'precision': precision,
                'recall': recall,
                'pr_thresholds': thresholds
            }

        return metrics

