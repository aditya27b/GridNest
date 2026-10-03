"""
Model Factory for Anomaly and Supervised Classifiers.
"""
from typing import Optional, Any
from ml.config import CONFIG
from ml.anomaly_models import IsolationForestWrapper, StatisticalAnomalyDetector
from ml.supervised_models import SupervisedClassifierWrapper

def get_anomaly_model(name: Optional[str] = None, **kwargs) -> Any:
    cfg = CONFIG.get("models", {}).get("anomaly_model", {})
    model_name = (name or cfg.get("name", "isolation_forest")).lower()

    if model_name in ["isolation_forest", "iforest"]:
        n_est = kwargs.get("n_estimators", cfg.get("n_estimators", 200))
        contam = kwargs.get("contamination", cfg.get("contamination", 0.10))
        return IsolationForestWrapper(n_estimators=n_est, contamination=contam)
    elif model_name in ["statistical", "rule_based", "iqr"]:
        return StatisticalAnomalyDetector()
    else:
        raise ValueError(f"Unknown anomaly model: {model_name}")

def get_classifier(name: Optional[str] = None, **kwargs) -> Any:
    cfg = CONFIG.get("models", {}).get("classifier", {})
    clf_name = (name or cfg.get("name", "xgboost")).lower()
    
    params = {
        "n_estimators": kwargs.get("n_estimators", cfg.get("n_estimators", 250)),
        "max_depth": kwargs.get("max_depth", cfg.get("max_depth", 5)),
        "learning_rate": kwargs.get("learning_rate", cfg.get("learning_rate", 0.05)),
    }
    return SupervisedClassifierWrapper(model_type=clf_name, **params)
