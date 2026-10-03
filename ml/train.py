"""
Main Model Training & Evaluation Pipeline.
Implements strict time-aware temporal splitting (Train -> Val -> Out-of-Time Test)
and benchmarks Isolation Forest & Supervised Classifiers under severe class imbalance.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ml.config import ROOT, FEATURE_COLUMNS, load_config
from ml.dataset_adapter import load_clean_timeseries
from ml.feature_engineering import extract_features_at_timestep, generate_training_feature_dataset
from ml.model_factory import get_anomaly_model, get_classifier
from ml.evaluation import evaluate_predictions, print_evaluation_summary

def run_training_pipeline(
    data_path: str = "data/Electricity_Theft_Data.csv",
    classifier_name: str = "xgboost",
    anomaly_name: str = "isolation_forest",
    max_consumers: int = None
):
    print("\n=======================================================")
    print(" INITIATING ENERGY INTELLIGENCE TRAINING PIPELINE")
    print(f" Dataset:    {data_path}")
    print(f" Classifier: {classifier_name.upper()}")
    print(f" Anomaly:    {anomaly_name.upper()}")
    print("=======================================================")

    # 1. Ingest clean chronologically sorted time-series
    cids, dates, matrix, labels, meta = load_clean_timeseries(data_path, max_consumers=max_consumers)
    num_consumers, num_days = matrix.shape
    print(f"[*] Loaded {num_consumers:,} consumers over {num_days} consecutive days.")
    print(f"[*] Date Range: {meta['start_date']} to {meta['end_date']}")
    print(f"[*] Target Distribution: {meta['negative_labels']} Normal, {meta['positive_labels']} Theft")

    # 2. Strict Time-Aware Splitting
    # Example: 365 days
    # Training checkpoints: days 60, 110, 160, 210
    # Validation checkpoint: day 270 (early autumn)
    # Out-of-Time Test checkpoint: day 340 (unseen future winter)
    train_days = [int(num_days * f) for f in [0.20, 0.35, 0.50, 0.65]]
    val_day = int(num_days * 0.80)
    test_day = int(num_days * 0.95)

    print(f"\n[*] Time-Aware Temporal Checkpoints:")
    print(f"    - Training Snapshots (Days): {train_days}")
    print(f"    - Validation Snapshot (Day): {val_day} ({dates[val_day].strftime('%Y-%m-%d')})")
    print(f"    - Out-of-Time Test Snapshot (Day): {test_day} ({dates[test_day].strftime('%Y-%m-%d')})")

    # Build Training Matrix
    print(f"\n[*] Generating Training Feature Matrix across checkpoints...")
    X_train, y_train, train_cids = generate_training_feature_dataset(
        cids, dates, matrix, labels, timesteps=train_days
    )

    # Build Validation Matrix
    print(f"[*] Generating Validation Feature Matrix (Day {val_day})...")
    X_val = extract_features_at_timestep(matrix, val_day, dates)
    y_val = labels

    # Build Out-of-Time Test Matrix
    print(f"[*] Generating Out-of-Time Test Feature Matrix (Day {test_day})...")
    X_test = extract_features_at_timestep(matrix, test_day, dates)
    y_test = labels

    # 3. Train Unsupervised Anomaly Model (Isolation Forest)
    print(f"\n[*] Training Unsupervised Anomaly Model ({anomaly_name.upper()})...")
    anomaly_model = get_anomaly_model(anomaly_name)
    # Fit on training data features
    anomaly_model.fit(X_train)
    test_anomaly_scores = anomaly_model.score(X_test)
    test_anomaly_probs = test_anomaly_scores / 100.0

    print("[+] Anomaly model trained successfully.")

    # 4. Train Supervised Classifier (XGBoost / LightGBM / RF)
    print(f"\n[*] Training Supervised Classifier ({classifier_name.upper()})...")
    clf = get_classifier(classifier_name)
    clf.fit(X_train, y_train)

    # Predict on Validation to tune threshold
    val_probs = clf.predict_proba(X_val)
    best_thresh = 0.50
    best_f1 = 0.0
    for th in np.linspace(0.20, 0.80, 25):
        m = evaluate_predictions(y_val, val_probs, threshold=th)
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_thresh = round(float(th), 2)

    print(f"[+] Optimal Operating Threshold from Validation: {best_thresh} (Val F1: {best_f1:.4f})")

    # 5. Out-of-Time Final Evaluation
    print(f"\n[*] Evaluating Supervised Classifier on Unseen Out-of-Time Test Data...")
    test_probs = clf.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_thresh)
    print_evaluation_summary(classifier_name, test_metrics)

    print(f"[*] Evaluating Unsupervised Anomaly Model on Test Data...")
    anomaly_metrics = evaluate_predictions(y_test, test_anomaly_probs, threshold=0.50)
    print_evaluation_summary(f"{anomaly_name} (unsupervised)", anomaly_metrics)

    # 6. Save Model Artifacts
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(anomaly_model, models_dir / "isolation_forest.joblib")
    joblib.dump({"model": clf, "threshold": best_thresh}, models_dir / "cause_classifier.joblib")
    (models_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    metadata = {
        "dataset": data_path,
        "classifier": classifier_name,
        "anomaly_model": anomaly_name,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "best_threshold": best_thresh,
        "test_metrics": test_metrics,
        "anomaly_metrics": anomaly_metrics,
        "features": FEATURE_COLUMNS
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    # 7. Save Comparison Report
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_rows = [
        {"model": classifier_name, **test_metrics},
        {"model": f"{anomaly_name}_unsupervised", **anomaly_metrics}
    ]
    pd.DataFrame(report_rows).to_csv(reports_dir / "model_comparison.csv", index=False)
    print(f"[+] Artifacts and reports saved to 'models/' and 'reports/model_comparison.csv'.")
    print("=======================================================\n")
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train energy intelligence models on real datasets.")
    parser.add_argument("--data", type=str, default="data/Electricity_Theft_Data.csv", help="Path to raw dataset")
    parser.add_argument("--classifier", type=str, default="xgboost", help="Classifier type (xgboost, random_forest, hist_gradient_boosting, logistic_regression)")
    parser.add_argument("--anomaly", type=str, default="isolation_forest", help="Anomaly model (isolation_forest, statistical)")
    parser.add_argument("--max-consumers", type=int, default=None, help="Limit number of consumers for fast iteration")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        classifier_name=args.classifier,
        anomaly_name=args.anomaly,
        max_consumers=args.max_consumers
    )
