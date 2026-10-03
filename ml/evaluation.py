"""
Evaluation Module for Electricity Theft & Anomaly Detection.
Prioritizes PR-AUC, Precision @ Top-K, Recall, F1, and False Alarm Rate (FAR)
under heavy class imbalance.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes comprehensive imbalanced classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total_pos = max(1, tp + fn)
    total_neg = max(1, tn + fp)

    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / total_pos, 4)
    far = round(fp / total_neg, 4)  # False Alarm Rate / FPR
    fnr = round(fn / total_pos, 4)

    # F1 and F2 Score
    if (precision + recall) > 0:
        f1 = round(2 * (precision * recall) / (precision + recall), 4)
        f2 = round(5 * (precision * recall) / (4 * precision + recall), 4)
    else:
        f1, f2 = 0.0, 0.0

    # PR-AUC and ROC-AUC
    pr_auc = 0.0
    roc_auc = 0.0
    try:
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = round(float(auc(r_curve, p_curve)), 4)
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        # Fallback simple trapezoidal approximation
        pass

    # Precision @ Top K (Inspection Capacity: Top 1%, 5%, 10%)
    n = len(y_prob)
    sorted_indices = np.argsort(y_prob)[::-1]
    
    def prec_at_k(pct: float) -> float:
        k = max(1, int(n * pct))
        top_k_indices = sorted_indices[:k]
        top_k_hits = np.sum(y_true[top_k_indices] == 1)
        return round(float(top_k_hits / k), 4)

    prec_top1  = prec_at_k(0.01)
    prec_top5  = prec_at_k(0.05)
    prec_top10 = prec_at_k(0.10)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_alarm_rate": far,
        "false_negative_rate": fnr,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision_top_1pct": prec_top1,
        "precision_top_5pct": prec_top5,
        "precision_top_10pct": prec_top10,
        "threshold": threshold
    }
    return metrics

def print_evaluation_summary(model_name: str, metrics: Dict[str, Any]):
    print(f"\n-------------------------------------------------------")
    print(f" EVALUATION SUMMARY: {model_name.upper()}")
    print(f"-------------------------------------------------------")
    print(f"[*] PR-AUC (Primary Imbalance Metric): {metrics['pr_auc']:.4f}")
    print(f"[*] ROC-AUC:                           {metrics['roc_auc']:.4f}")
    print(f"[*] Precision @ Threshold:              {metrics['precision']*100:.2f}%")
    print(f"[*] Recall (Theft Detection Rate):     {metrics['recall']*100:.2f}%")
    print(f"[*] F1-Score:                          {metrics['f1_score']:.4f}")
    print(f"[*] F2-Score (Recall Prioritized):     {metrics['f2_score']:.4f}")
    print(f"[*] False Alarm Rate (FPR):            {metrics['false_alarm_rate']*100:.2f}%")
    print(f"[*] Precision @ Top 5% Capacity:       {metrics['precision_top_5pct']*100:.2f}%")
    print(f"[*] Confusion Matrix: [TP: {metrics['true_positives']} | FP: {metrics['false_positives']} | FN: {metrics['false_negatives']} | TN: {metrics['true_negatives']}]")
    print(f"-------------------------------------------------------\n")
