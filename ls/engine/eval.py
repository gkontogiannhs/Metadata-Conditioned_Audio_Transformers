import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def _forward_model(model, inputs, batch, device):
    """Smart forward for both vanilla AST and FiLM-AST."""
    is_film_model = hasattr(model, 'film_generators')
    
    if is_film_model and all(k in batch for k in ["device_id", "site_id", "m_rest"]):
        return model(
            inputs,
            batch["device_id"].to(device),
            batch["site_id"].to(device),
            batch["m_rest"].to(device)
        )
    else:
        return model(inputs)
    
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    multi_label=True,
    thresholds=(0.5, 0.5),
    tune_thresholds=True,
    verbose=True
):
    """
    Evaluate model on a dataset.
    
    Supports both multi-label and multi-class evaluation with optional
    threshold tuning for multi-label mode.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on
        multi_label: If True, use multi-label evaluation; else multi-class
        thresholds: Tuple of (crackle_threshold, wheeze_threshold) for multi-label
        tune_thresholds: If True, find optimal thresholds on this data (validation only!)
        verbose: Print detailed metrics
    
    Returns:
        Tuple of (avg_loss, global_metrics, group_metrics)
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    all_logits = []
    all_labels = []
    
    # For subgroup analysis
    group_logits = defaultdict(list)
    group_labels = defaultdict(list)
    sample_metadata = []  # Store device/site for each sample

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]", leave=False):
            inputs = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            
            # Get metadata (may be missing for vanilla AST)
            devices = batch.get("device", ["Unknown"] * len(labels))
            sites = batch.get("site", ["Unknown"] * len(labels))

            with torch.amp.autocast(device.type):
                logits = _forward_model(model, inputs, batch, device)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_logits.extend(logits_np)
            all_labels.extend(labels_np)

            # Collect subgroup data
            for d, s, y_true, y_logit in zip(devices, sites, labels_np, logits_np):
                group_labels[f"device_{d}"].append(y_true)
                group_logits[f"device_{d}"].append(y_logit)
                group_labels[f"site_{s}"].append(y_true)
                group_logits[f"site_{s}"].append(y_logit)
                sample_metadata.append({'device': d, 'site': s})

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / n_samples

    # ============================================================
    # MULTI-LABEL EVALUATION
    # ============================================================
    if multi_label:
        all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid
        
        # Threshold tuning (only on validation!)
        if tune_thresholds:
            best = find_best_thresholds_icbhi(all_labels, all_probs)
            thresholds = (best["tC"], best["tW"])
            if verbose:
                print(f"→ Optimized thresholds: tC={thresholds[0]:.3f}, tW={thresholds[1]:.3f} "
                      f"(ICBHI={best['hs']*100:.2f}%)")

        # Apply thresholds
        all_preds = np.stack([
            (all_probs[:, 0] >= thresholds[0]).astype(int),
            (all_probs[:, 1] >= thresholds[1]).astype(int)
        ], axis=1)

        # Compute metrics
        global_metrics = compute_multilabel_metrics(
            all_labels, all_preds, all_probs, verbose=verbose
        )
        
        # Store thresholds in metrics
        global_metrics["threshold_crackle"] = thresholds[0]
        global_metrics["threshold_wheeze"] = thresholds[1]

        # Subgroup metrics
        group_metrics = {}
        for group, y_true_list in group_labels.items():
            if len(y_true_list) == 0:
                continue

            y_true = np.array(y_true_list)
            y_logits = np.array(group_logits[group])
            y_probs = 1 / (1 + np.exp(-y_logits))

            y_pred = np.stack([
                (y_probs[:, 0] >= thresholds[0]).astype(int),
                (y_probs[:, 1] >= thresholds[1]).astype(int)
            ], axis=1)

            group_metrics[group] = compute_multilabel_metrics(
                y_true, y_pred, y_probs, verbose=False
            )

    # ============================================================
    # MULTI-CLASS EVALUATION
    # ============================================================
    else:
        all_probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)  # Softmax
        all_preds = np.argmax(all_logits, axis=1)
        
        # Ensure labels are integers
        if all_labels.ndim > 1:
            all_labels = all_labels.squeeze()
        all_labels = all_labels.astype(int)

        global_metrics = compute_multiclass_metrics(
            all_labels, all_preds, all_probs, verbose=verbose
        )

        # Subgroup metrics
        group_metrics = {}
        for group, y_true_list in group_labels.items():
            if len(y_true_list) == 0:
                continue

            y_true = np.array(y_true_list).astype(int)
            y_logits = np.array(group_logits[group])
            y_preds = np.argmax(y_logits, axis=1)
            y_probs = np.exp(y_logits) / np.exp(y_logits).sum(axis=1, keepdims=True)

            group_metrics[group] = compute_multiclass_metrics(
                y_true, y_preds, y_probs, verbose=False
            )

    return avg_loss, global_metrics, group_metrics


def compute_multiclass_metrics(all_labels, all_preds, all_probs=None, verbose=True):
    """
    Compute metrics for multi-class (single-label) classification.
    
    Args:
        all_labels: Ground truth labels (N,) integers
        all_preds: Predicted labels (N,) integers
        all_probs: Predicted probabilities (N, C) optional
        verbose: Print summary
    
    Returns:
        Dict of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    
    metrics = {}
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    metrics['accuracy'] = accuracy
    metrics['per_class_precision'] = precision.tolist()
    metrics['per_class_recall'] = recall.tolist()
    metrics['per_class_f1'] = f1.tolist()
    metrics['per_class_support'] = support.tolist()
    
    # Macro averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    
    # Weighted averages
    total_support = support.sum()
    if total_support > 0:
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ICBHI-style metrics (if 4-class: Normal, Crackle, Wheeze, Both)
    n_classes = len(np.unique(all_labels))
    if n_classes == 4:        
        # Sensitivity = correctly classified abnormal / total abnormal
        abnormal_mask = all_labels > 0
        if abnormal_mask.sum() > 0:
            sensitivity = (all_preds[abnormal_mask] == all_labels[abnormal_mask]).sum() / abnormal_mask.sum()
        else:
            sensitivity = 0.0
        
        normal_mask = all_labels == 0
        if normal_mask.sum() > 0:
            specificity = (all_preds[normal_mask] == 0).sum() / normal_mask.sum()
        else:
            specificity = 0.0
        
        metrics['specificity'] = specificity
        metrics['sensitivity'] = sensitivity
        metrics['icbhi_score'] = 0.5 * (specificity + sensitivity)
    
    if verbose:
        print(f"\n[Multi-class Metrics]")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']*100:.2f}%")
        if n_classes == 4:
            print(f"  ICBHI Score: {metrics['icbhi_score']*100:.2f}% "
                  f"(Sp={metrics['specificity']*100:.2f}%, Se={metrics['sensitivity']*100:.2f}%)")
        print(f"\n  Per-class F1: {[f'{x*100:.1f}%' for x in f1]}")
    
    return metrics


def compute_multilabel_metrics(all_labels, all_preds, all_probs=None, verbose=True):
    """
    Compute detailed multi-label metrics + official ICBHI metrics.
    Fixed: Added safety checks for empty classes and proper metric computation.
    """
    metrics = {}

    # === Composite 4-class masks ===
    is_n = (all_labels[:,0]==0) & (all_labels[:,1]==0)
    is_c = (all_labels[:,0]==1) & (all_labels[:,1]==0)
    is_w = (all_labels[:,0]==0) & (all_labels[:,1]==1)
    is_b = (all_labels[:,0]==1) & (all_labels[:,1]==1)

    pr_n = (all_preds[:,0]==0) & (all_preds[:,1]==0)
    pr_c = (all_preds[:,0]==1) & (all_preds[:,1]==0)
    pr_w = (all_preds[:,0]==0) & (all_preds[:,1]==1)
    pr_b = (all_preds[:,0]==1) & (all_preds[:,1]==1)

    # === Class-wise totals ===
    Nn, Nc, Nw, Nb = is_n.sum(), is_c.sum(), is_w.sum(), is_b.sum()
    Pn = np.sum(is_n & pr_n)
    Pc = np.sum(is_c & pr_c)
    Pw = np.sum(is_w & pr_w)
    Pb = np.sum(is_b & pr_b)

    # === Official ICBHI metrics ===
    sp = Pn / (Nn + 1e-12)
    se = (Pc + Pw + Pb) / (Nc + Nw + Nb + 1e-12)
    hs = 0.5 * (sp + se)
    metrics.update({'specificity': sp, 'sensitivity': se, 'icbhi_score': hs})

    if verbose:
        print(f"[ICBHI 4-class] SPE={sp*100:.2f}% | SEN={se*100:.2f}% | HS={hs*100:.2f}%")

    # === Per-class metrics ===
    pattern_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    is_true = [is_n, is_c, is_w, is_b]
    is_pred = [pr_n, pr_c, pr_w, pr_b]
    Ps = [Pn, Pc, Pw, Pb]
    Ns = [Nn, Nc, Nw, Nb]

    total_samples = len(all_labels)
    per_class = []

    for name, tmask, pmask, P, N in zip(pattern_names, is_true, is_pred, Ps, Ns):
        TP = P
        FN = N - P
        FP = np.sum(~tmask & pmask)
        TN = total_samples - TP - FP - FN

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        specificity = TN / (TN + FP + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        per_class.append({
            'name': name,
            'precision': precision,
            'sensitivity': recall,
            'specificity': specificity,
            'f1': f1,
            'support': N
        })

        metrics.update({
            f'{name}_precision': precision,
            f'{name}_sensitivity': recall,
            f'{name}_specificity': specificity,
            f'{name}_f1': f1,
        })

    # === Macro averages ===
    precisions = [c['precision'] for c in per_class]
    sensitivities = [c['sensitivity'] for c in per_class]
    specificities = [c['specificity'] for c in per_class]
    f1s = [c['f1'] for c in per_class]
    supports = [c['support'] for c in per_class]

    metrics.update({
        'macro_precision': np.mean(precisions),
        'macro_sensitivity': np.mean(sensitivities),
        'macro_specificity': np.mean(specificities),
        'macro_f1': np.mean(f1s),
        'weighted_precision': np.average(precisions, weights=supports),
        'weighted_sensitivity': np.average(sensitivities, weights=supports),
        'weighted_specificity': np.average(specificities, weights=supports),
        'weighted_f1': np.average(f1s, weights=supports),
    })

    if verbose:
        print(f"[Macro] P={metrics['macro_precision']*100:.2f}% | "
              f"R={metrics['macro_sensitivity']*100:.2f}% | "
              f"F1={metrics['macro_f1']*100:.2f}%")

    # === Binary Normal vs Abnormal ===
    is_abn_true = ~is_n
    is_abn_pred = ~pr_n

    TP = np.sum(is_abn_true & is_abn_pred)
    TN = np.sum(is_n & pr_n)
    FP = np.sum(~is_abn_true & is_abn_pred)
    FN = np.sum(is_abn_true & ~is_abn_pred)

    binary_spe = TN / (TN + FP + 1e-12)
    binary_sen = TP / (TP + FN + 1e-12)
    binary_hs = 0.5 * (binary_spe + binary_sen)

    metrics.update({
        'binary_specificity': binary_spe,
        'binary_sensitivity': binary_sen,
        'binary_icbhi_score': binary_hs
    })

    return metrics


def _icbhi_from_bits(all_labels, all_preds):
    """Compute official ICBHI metrics from predictions."""
    is_n = (all_labels[:,0]==0) & (all_labels[:,1]==0)
    is_c = (all_labels[:,0]==1) & (all_labels[:,1]==0)
    is_w = (all_labels[:,0]==0) & (all_labels[:,1]==1)
    is_b = (all_labels[:,0]==1) & (all_labels[:,1]==1)

    pr_n = (all_preds[:,0]==0) & (all_preds[:,1]==0)
    pr_c = (all_preds[:,0]==1) & (all_preds[:,1]==0)
    pr_w = (all_preds[:,0]==0) & (all_preds[:,1]==1)
    pr_b = (all_preds[:,0]==1) & (all_preds[:,1]==1)

    Nn, Nc, Nw, Nb = is_n.sum(), is_c.sum(), is_w.sum(), is_b.sum()
    Pn = np.sum(is_n & pr_n)
    Pc = np.sum(is_c & pr_c)
    Pw = np.sum(is_w & pr_w)
    Pb = np.sum(is_b & pr_b)

    sp = Pn / (Nn + 1e-12)
    se = (Pc + Pw + Pb) / (Nc + Nw + Nb + 1e-12)
    hs = 0.5 * (sp + se)
    return sp, se, hs


def find_best_thresholds_icbhi(val_labels_ml, val_probs_ml, grid=np.linspace(0.05, 0.95, 19)):
    """Sweep thresholds to maximize ICBHI score."""
    best = {"tC": 0.5, "tW": 0.5, "sp": 0.0, "se": 0.0, "hs": 0.0}
    y_true = val_labels_ml.astype(int)

    for tC in grid:
        for tW in grid:
            y_pred = np.stack([
                (val_probs_ml[:,0] >= tC).astype(int),
                (val_probs_ml[:,1] >= tW).astype(int)
            ], axis=1)
            sp, se, hs = _icbhi_from_bits(y_true, y_pred)
            if hs > best["hs"]:
                best.update({"tC": float(tC), "tW": float(tW), "sp": sp, "se": se, "hs": hs})
    return best