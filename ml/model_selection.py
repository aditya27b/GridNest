"""
Model Selection & Multi-Model Benchmark Suite.
Compares Logistic Regression, Random Forest, Gradient Boosted Trees, and Isolation Forest
under strict time-aware evaluation.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from ml.config import ROOT
from ml.dataset_adapter import load_clean_timeseries
from ml.feature_engineering import extract_features_at_timestep, generate_training_feature_dataset
from ml.supervised_models import SupervisedClassifierWrapper
from ml.anomaly_models import IsolationForestWrapper, StatisticalAnomalyDetector
from ml.evaluation import evaluate_predictions

def compare_models(data_path: str = "data/Electricity_Theft_Data.csv", max_consumers: int = None):
    print("\n=======================================================")
    print(" EXECUTING MULTI-MODEL BENCHMARK SUITE")
    print(f" Dataset: {data_path}")
    print("=======================================================")

    cids, dates, matrix, labels, meta = load_clean_timeseries(data_path, max_consumers=max_consumers)
    num_days = matrix.shape[1]

    train_days = [int(num_days * f) for f in [0.25, 0.45, 0.65]]
    val_day = int(num_days * 0.80)
    test_day = int(num_days * 0.95)

    print(f"[*] Extracting time-aware train features across checkpoints: {train_days}")
    X_train, y_train, _ = generate_training_feature_dataset(cids, dates, matrix, labels, timesteps=train_days)

    print(f"[*] Extracting validation features (Day {val_day}) and test features (Day {test_day})...")
    X_val = extract_features_at_timestep(matrix, val_day, dates)
    X_test = extract_features_at_timestep(matrix, test_day, dates)
    y_test = labels

    candidate_models = [
        ("Logistic Regression", SupervisedClassifierWrapper("logistic_regression")),
        ("Random Forest", SupervisedClassifierWrapper("random_forest", n_estimators=100)),
        ("Gradient Boosted Trees", SupervisedClassifierWrapper("hist_gradient_boosting", n_estimators=100)),
        ("Isolation Forest (Unsupervised)", IsolationForestWrapper(n_estimators=100)),
        ("Statistical Detector (Unsupervised)", StatisticalAnomalyDetector())
    ]

    results = []
    print("\n[*] Training and benchmarking candidates...\n")

    for name, model in candidate_models:
        print(f"[*] Evaluating: {name}...")
        if hasattr(model, "score"):
            # Anomaly model
            model.fit(X_train)
            test_scores = model.score(X_test)
            test_probs = test_scores / 100.0
            metrics = evaluate_predictions(y_test, test_probs, threshold=0.50)
        else:
            # Supervised classifier
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)
            # Tune threshold on validation
            best_th = 0.50
            best_f1 = 0.0
            for th in np.linspace(0.20, 0.80, 15):
                m = evaluate_predictions(labels, val_probs, threshold=th)
                if m["f1_score"] > best_f1:
                    best_f1 = m["f1_score"]
                    best_th = float(th)

            test_probs = model.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, test_probs, threshold=best_th)

        results.append({
            "Model": name,
            "PR-AUC": metrics["pr_auc"],
            "ROC-AUC": metrics["roc_auc"],
            "Precision @ Thresh": f"{metrics['precision']*100:.1f}%",
            "Recall": f"{metrics['recall']*100:.1f}%",
            "F1-Score": metrics["f1_score"],
            "False Alarm Rate": f"{metrics['false_alarm_rate']*100:.2f}%",
            "Precision @ Top 5%": f"{metrics['precision_top_5pct']*100:.1f}%"
        })

    df_results = pd.DataFrame(results)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(reports_dir / "model_comparison.csv", index=False)

    print("\n=======================================================")
    print(" MODEL COMPARISON LEADERBOARD (Out-of-Time Test Set)")
    print("=======================================================")
    print(df_results.to_string(index=False))
    print(f"\n[+] Full results saved to: {reports_dir / 'model_comparison.csv'}")
    print("=======================================================\n")
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multiple models on electricity dataset.")
    parser.add_argument("--data", type=str, default="data/Electricity_Theft_Data.csv", help="Dataset path")
    parser.add_argument("--max-consumers", type=int, default=2000, help="Consumer limit for fast comparison")
    args = parser.parse_args()

    compare_models(args.data, args.max_consumers)
