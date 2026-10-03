"""
Supervised Classifiers Suite for Electricity Theft Detection.
Handles severe class imbalance via scale_pos_weight and class weighting.
Includes automatic fallback to HistGradientBoosting if XGBoost libomp is missing.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

class SupervisedClassifierWrapper:
    def __init__(self, model_type: str = "xgboost", **params):
        self.model_type = model_type.lower()
        self.params = params
        self.model = None
        self.actual_model_name = self.model_type

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        scale_pos = max(1.0, float(neg_count) / max(1, pos_count))

        if self.model_type in ["xgboost", "xgb"]:
            try:
                from xgboost import XGBClassifier
                n_est = self.params.get("n_estimators", 200)
                lr = self.params.get("learning_rate", 0.05)
                max_depth = self.params.get("max_depth", 5)
                self.model = XGBClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=max_depth,
                    scale_pos_weight=scale_pos,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    eval_metric="logloss"
                )
                self.model.fit(X, y)
                self.actual_model_name = "XGBoost"
                return self
            except Exception as e:
                print(f"[!] Note on XGBoost: {e}")
                print("[*] Automatically falling back to Scikit-Learn HistGradientBoostingClassifier (native gradient boosted trees with class balancing).")
                self.model_type = "hist_gradient_boosting"

        if self.model_type in ["hist_gradient_boosting", "hist_gb", "gradient_boosting"]:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=self.params.get("n_estimators", 150),
                class_weight="balanced",
                random_state=42
            )
            self.actual_model_name = "HistGradientBoosting (Balanced Trees)"

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 150),
                max_depth=self.params.get("max_depth", 10),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            self.actual_model_name = "RandomForest (Balanced)"

        elif self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
            self.actual_model_name = "LogisticRegression"
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
