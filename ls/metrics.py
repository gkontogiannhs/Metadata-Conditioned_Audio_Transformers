import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from typing import List, Tuple


def get_icbhi_scores(hits: List[int], counts: List[int], pflag: bool = True) -> Tuple[float, float, float]:
    """
    Compute ICBHI specificity, sensitivity, and average score.

    hits[0] = correct predictions for normal class
    counts[0] = total samples for normal class
    hits[1:] = correct predictions for abnormal classes
    counts[1:] = total abnormal samples
    """
    sp = hits[0] / (counts[0] + 1e-10) * 100 # normal
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100 # crackles, wheezes, both
    sc = (sp + se) / 2.0
    if pflag:
        print(f"[ICBHI] S_p={sp:.2f} | S_e={se:.2f} | Score={sc:.2f}")
    return sp, se, sc


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, n_classes: int = None):
    """
    Compute a comprehensive set of metrics for imbalanced medical datasets.
    - Accuracy
    - Balanced Accuracy
    - F1 (macro, weighted)
    - AUROC (macro, weighted)
    - Specificity & Sensitivity (ICBHI-style)

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
        y_prob: prediction probabilities (N, C) or None
        n_classes: number of classes
    Returns:
        dict of metrics
    """
    metrics = {}
    metrics["accuracy"] = (y_true == y_pred).mean() * 100
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred) * 100

    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro") * 100
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted") * 100

    if y_prob is not None and n_classes and n_classes > 2:
        try:
            print("[DEBUG] n_classes:", n_classes)
            print("[DEBUG] y_prob.shape:", y_prob.shape)
            print("[DEBUG] y_true unique:", np.unique(y_true))
            print("[DEBUG] any NaN in y_prob:", np.isnan(y_prob).any())
            print("[DEBUG] min/max prob:", np.min(y_prob), np.max(y_prob))
            print("Row sum mean:", np.mean(y_prob.sum(axis=1)))
            print("Row sum range:", np.min(y_prob.sum(axis=1)), np.max(y_prob.sum(axis=1)))

            print("y_prob dtype:", y_prob.dtype)
            print("y_true dtype:", y_true.dtype)
            print("unique y_true:", np.unique(y_true))
            print("y_prob shape:", y_prob.shape)
            print("sum(y_prob[0]):", y_prob[0].sum())
            print("np.isnan(y_prob).any():", np.isnan(y_prob).any())
            print("np.isinf(y_prob).any():", np.isinf(y_prob).any())
            print("y_prob row sums:", y_prob.sum(axis=1)[:5])
            print("classes vs probs columns:", len(np.unique(y_true)), y_prob.shape[1])
            y_prob = np.asarray(y_prob, dtype=np.float32)
            metrics["auroc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100
            metrics["auroc_weighted"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted") * 100
        except Exception as e:
            print(f"[WARN] AUROC computation failed: {e}")
            metrics["auroc_macro"] = -1
            metrics["auroc_weighted"] = -1


    # ---- ICBHI metric ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    hits = cm.diagonal().tolist()
    print(cm)
    
    counts = cm.sum(axis=1).tolist()
    sp, se, sc = get_icbhi_scores(hits, counts, pflag=False)
    metrics["specificity"] = sp
    metrics["sensitivity"] = se
    metrics["icbhi_score"] = sc

    return metrics