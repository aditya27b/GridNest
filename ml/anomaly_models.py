"""
Anomaly Models Suite: Isolation Forest and Transparent Statistical Detector.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

class StatisticalAnomalyDetector:
    """
    Transparent, rule-based baseline detector using dynamic Z-Score and IQR.
    Provides immediate interpretable outlier bounds without black-box fitting.
    """
    def __init__(self, z_thresh: float = 2.5):
        self.z_thresh = z_thresh

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes anomaly score in range [0, 100].
        """
        z_scores = np.abs(X["z_score_30"].values)
        neg_dev = X["negative_deviation_pct"].values
        consec_low = X["consecutive_low_days"].values

        # Outlier scoring
        z_part = np.clip(z_scores / 3.5, 0.0, 1.0)
        dev_part = np.clip(neg_dev / 0.8, 0.0, 1.0)
        persist_part = np.clip(consec_low / 14.0, 0.0, 1.0)

        score = (0.40 * z_part + 0.35 * dev_part + 0.25 * persist_part) * 100.0
        return score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= 60.0, -1, 1)

class IsolationForestWrapper:
    """
    Wrapper for Scikit-Learn IsolationForest with decision function calibration.
    """
    def __init__(self, n_estimators: int = 200, contamination: float = 0.10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame, y=None):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        # decision_function gives negative values for anomalies
        raw_scores = self.model.decision_function(X)
        # Calibrate to [0, 100] where 100 = extreme anomaly
        scores = np.clip((0.20 - raw_scores) / 0.40, 0.0, 1.0) * 100.0
        return scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)
