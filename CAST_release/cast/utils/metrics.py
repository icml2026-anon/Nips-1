import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def find_optimal_threshold(y_true, y_pred_proba):
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (y_pred_proba > thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def compute_binary_metrics(y_true, y_pred_proba, threshold=None):
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_pred_proba)
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "threshold": round(threshold, 2),
        "support": len(y_true)
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
    else:
        metrics["auc"] = np.nan
        metrics["auprc"] = np.nan
    return metrics


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics.get('auc', np.nan):.4f}")
    print(f"  AUPRC:     {metrics.get('auprc', np.nan):.4f}")
    print(f"  Threshold: {metrics.get('threshold', 0.5)}")
    print(f"  Support:   {metrics['support']}")
