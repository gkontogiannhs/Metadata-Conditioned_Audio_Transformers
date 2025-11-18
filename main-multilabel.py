import os
import mlflow
from ls.config.loader import load_config
# from ls.engine.train import train_loop
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import mlflow
from tqdm import tqdm
from ls.engine.utils import get_device
from ls.config.dataclasses import TrainingConfig
from ls.engine.scheduler import build_scheduler
import numpy as np
from collections import defaultdict


def set_visible_gpus(gpus: str, verbose: bool = True):
    """
    Restrict which GPUs PyTorch can see by setting CUDA_VISIBLE_DEVICES.

    Args:
        gpus (str): Comma-separated GPU indices, e.g., "0,1,2,3".
        verbose (bool): If True, print the selection info.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if verbose:
        print(f"[CUDA] Visible devices set to: {gpus}")

    # Optional sanity check after setting
    torch.cuda.device_count()  # forces CUDA to reinitialize


def compute_multilabel_metrics(all_labels, all_preds, all_probs=None, verbose=True):
    """
    Compute detailed multi-label metrics + official ICBHI metrics +
    macro-averaged metrics + binary Normal-vs-Abnormal ICBHI score.

    Works for multilabel setup with two binary labels [crackle, wheeze],
    forming 4 composite classes (Normal, Crackle, Wheeze, Both).
    """

    metrics = {}

    # === Composite 4-class masks (true/pred) ===
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

    # === Official 4-class ICBHI metrics ===
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
        recall = TP / (TP + FN + 1e-12)         # sensitivity
        specificity = TN / (TN + FP + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        # accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)

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

        # if verbose:
        #     print(f"[{name}] P={precision*100:.2f}% | R={recall*100:.2f}% | "
        #           f"Sp={specificity*100:.2f}% | F1={f1*100:.2f}%")

    # === Macro and Weighted averages ===
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
        print("\n[Macro averages]")
        print(f"P={metrics['macro_precision']*100:.2f}% | R={metrics['macro_sensitivity']*100:.2f}% | "
              f"Sp={metrics['macro_specificity']*100:.2f}% | F1={metrics['macro_f1']*100:.2f}%")

    # === Binary Normal vs Abnormal (ICBHI style) ===
    is_abn_true = ~is_n
    is_abn_pred = ~pr_n

    TP = np.sum(is_abn_true & is_abn_pred)  # correctly abnormal
    TN = np.sum(is_n & pr_n)                # correctly normal
    FP = np.sum(~is_abn_true & is_abn_pred) # predicted abnormal but actually normal
    FN = np.sum(is_abn_true & ~is_abn_pred) # predicted normal but actually abnormal

    binary_spe = TN / (TN + FP + 1e-12)
    binary_sen = TP / (TP + FN + 1e-12)
    binary_hs = 0.5 * (binary_spe + binary_sen)

    metrics.update({
        'binary_specificity': binary_spe,
        'binary_sensitivity': binary_sen,
        'binary_icbhi_score': binary_hs
    })

    if verbose:
        print("\n[Binary Normal-vs-Abnormal ICBHI]")
        print(f"SPE={binary_spe*100:.2f}% | SEN={binary_sen*100:.2f}% | HS={binary_hs*100:.2f}%")

    return metrics


# def compute_multilabel_metrics(all_labels, all_preds, all_probs, verbose=True):
#     """
#     Compute detailed multi-label metrics + official ICBHI metrics.
#     Now includes Sensitivity/Specificity for Normal and Both composite classes.
#     """
#     metrics = {}

#     # === Composite 4-class masks ===
#     is_n = (all_labels[:,0]==0) & (all_labels[:,1]==0)
#     is_c = (all_labels[:,0]==1) & (all_labels[:,1]==0)
#     is_w = (all_labels[:,0]==0) & (all_labels[:,1]==1)
#     is_b = (all_labels[:,0]==1) & (all_labels[:,1]==1)

#     pr_n = (all_preds[:,0]==0) & (all_preds[:,1]==0)
#     pr_c = (all_preds[:,0]==1) & (all_preds[:,1]==0)
#     pr_w = (all_preds[:,0]==0) & (all_preds[:,1]==1)
#     pr_b = (all_preds[:,0]==1) & (all_preds[:,1]==1)

#     # === Counts ===
#     Nn, Nc, Nw, Nb = is_n.sum(), is_c.sum(), is_w.sum(), is_b.sum()
#     Pn = np.sum(is_n & pr_n)
#     Pc = np.sum(is_c & pr_c)
#     Pw = np.sum(is_w & pr_w)
#     Pb = np.sum(is_b & pr_b)

#     # === Official ICBHI metrics ===
#     sp = Pn / (Nn + 1e-12)
#     se = (Pc + Pw + Pb) / (Nc + Nw + Nb + 1e-12)
#     hs = 0.5 * (sp + se)
#     if verbose:
#         print(f"[ICBHI official] SPE={sp*100:.2f}% | SEN={se*100:.2f}% | SC={hs*100:.2f}%")
#     metrics.update({'specificity': sp, 'sensitivity': se, 'icbhi_score': hs})

#     # === Per-pattern metrics (Normal, Crackle, Wheeze, Both) ===
#     pattern_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
#     is_true = [is_n, is_c, is_w, is_b]
#     is_pred = [pr_n, pr_c, pr_w, pr_b]

#     for name, tmask, pmask in zip(pattern_names, is_true, is_pred):
#         tp = np.sum(tmask & pmask)
#         fp = np.sum(~tmask & pmask)
#         fn = np.sum(tmask & ~pmask)
#         tn = np.sum(~tmask & ~pmask)

#         precision = tp / (tp + fp + 1e-12)
#         recall = tp / (tp + fn + 1e-12)
#         f1 = 2 * precision * recall / (precision + recall + 1e-12)
#         # acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
#         # sensitivity_c = recall  # TP / (TP+FN)
#         specificity_c = tn / (tn + fp + 1e-12)

#         metrics[f'{name}_precision'] = precision
#         # metrics[f'{name}_recall'] = recall
#         metrics[f'{name}_f1'] = f1
#         # metrics[f'{name}_accuracy'] = acc
#         metrics[f'{name}_sensitivity'] = recall
#         metrics[f'{name}_specificity'] = specificity_c

#     # === Compute as 2-class (normal vs abnormal) ===
#     tp_abn = Pc + Pw + Pb
#     fn_abn = Nc + Nw + Nb - tp_abn
#     tn_abn = Nn
#     fp_abn = tp_abn - fn_abn

#     sp_abn = tn_abn / (tn_abn + fp_abn + 1e-12)
#     se_abn = tp_abn / (tp_abn + fn_abn + 1e-12)
#     hs_abn = 0.5 * (sp_abn + se_abn)
#     metrics.update({'abnormal_specificity': sp_abn, 'abnormal_sensitivity': se_abn, 'abnormal_icbhi_score': hs_abn})

    # === Macro summaries ===
    # metrics['f1_macro'] = np.mean([metrics['Crackle_f1'], metrics['Wheeze_f1']])
    # metrics['auc_macro'] = np.mean([metrics['Crackle_auc'], metrics['Wheeze_auc']])
    # metrics['accuracy_macro'] = np.mean([metrics['Crackle_accuracy'], metrics['Wheeze_accuracy']])

    # return metrics


# def convert_4class_to_multilabel(labels_4class):
#     """
#     Convert 4-class labels to 2-label binary format.
    
#     0 (Normal)   → [0, 0]
#     1 (Crackles) → [1, 0]
#     2 (Wheezes)  → [0, 1]
#     3 (Both)     → [1, 1]
#     """
#     batch_size = labels_4class.shape[0]
#     labels_multilabel = np.zeros((batch_size, 2), dtype=np.float32)
    
#     for i, label in enumerate(labels_4class):
#         if label == 0:  # Normal
#             labels_multilabel[i] = [0, 0]
#         elif label == 1:  # Crackles
#             labels_multilabel[i] = [1, 0]
#         elif label == 2:  # Wheezes
#             labels_multilabel[i] = [0, 1]
#         elif label == 3:  # Both
#             labels_multilabel[i] = [1, 1]
    
#     return labels_multilabel


def _icbhi_from_bits(all_labels, all_preds):
    """Compute official ICBHI specificity, sensitivity, score from exact class matches."""
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
    """
    Sweep thresholds on validation set to maximize official ICBHI score.
    Returns dict with best_tC, best_tW, sp, se, hs.
    """
    best = {"tC": 0.5, "tW": 0.5, "specificity": 0.0, "sensitivity": 0.0, "icbhi_score": 0.0}
    y_true = val_labels_ml.astype(int)

    for tC in grid:
        for tW in grid:
            y_pred = np.stack([
                (val_probs_ml[:,0] >= tC).astype(int),
                (val_probs_ml[:,1] >= tW).astype(int)
            ], axis=1)
            sp, se, hs = _icbhi_from_bits(y_true, y_pred)
            if hs > best["icbhi_score"]:
                best.update({"tC": float(tC), "tW": float(tW),
                             "specificity": float(sp), "sensitivity": float(se), "icbhi_score": float(hs)})
    return best

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, grdscaler, epoch):
    """
    Train model for one epoch with multi-label binary classification.
    Handles both 4-class labels (auto-converts) and multi-label labels.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    group_preds = defaultdict(list)
    group_labels = defaultdict(list)
    group_probs = defaultdict(list)

    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)
        devices, sites = batch["device"], batch["site"]

        # stats per batch
        # normals = ((labels[:,0]==0) & (labels[:,1]==0)).sum().item()
        # crackles = ((labels[:,0]==1) & (labels[:,1]==0)).sum().item()
        # wheezes = ((labels[:,0]==0) & (labels[:,1]==1)).sum().item()
        # both = ((labels[:,0]==1) & (labels[:,1]==1)).sum().item()
        # print(f"Batch stats - Normals: {normals}, Crackles: {crackles}, Wheezes: {wheezes}, Both: {both}")

        # unique, counts = np.unique(devices, return_counts=True)
        # print("Type of device per batch:", dict(zip(unique, counts)))
        # print("Type of site per batch:", dict(zip(*np.unique(sites, return_counts=True))))
        # Convert 4-class labels to multi-label if needed
        # labels_multilabel = convert_4class_to_multilabel(labels.cpu().numpy())
        # labels = torch.from_numpy(labels_multilabel).to(device)
        # print(batch["labels"])
        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = model(inputs)  # (B, 2)
            loss = criterion(logits, labels)  # BCEWithLogitsLoss

        grdscaler.scale(loss).backward()
        grdscaler.step(optimizer)
        grdscaler.update()

        total_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

        # subgroup metrics
        for d, s, y_true, y_pred, y_prob in zip(devices, sites, labels_np, preds, probs):
            group_preds[f"device::{d}"].append(y_pred)
            group_labels[f"device::{d}"].append(y_true)
            group_probs[f"device::{d}"].append(y_prob)

            group_preds[f"site::{s}"].append(y_pred)
            group_labels[f"site::{s}"].append(y_true)
            group_probs[f"site::{s}"].append(y_prob)

    # ----- Global metrics -----
    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    # print(f"[Epoch {epoch}] Global | HS: {global_metrics['icbhi_score']*100:.2f}% | "
    #       f"Sp: {global_metrics['specificity']*100:.2f}% | Se: {global_metrics['sensitivity']*100:.2f}%")

    # ----- Group-level metrics -----
    group_metrics = {}
    for group in group_labels.keys():
        group_metrics[group] = compute_multilabel_metrics(
            np.array(group_labels[group]),
            np.array(group_preds[group]),
            np.array(group_probs[group]),
            verbose=False
        )
        print(f"[Epoch {epoch}] Group: {group} | HS: {group_metrics[group]['icbhi_score']*100:.2f}% | "
              f"Sp: {group_metrics[group]['specificity']*100:.2f}% | Se: {group_metrics[group]['sensitivity']*100:.2f}%")
    return avg_loss, global_metrics, group_metrics


def evaluate(model, dataloader, criterion, device,
             thresholds=(0.5, 0.5), tune_thresholds=True, verbose=True):
    """
    Evaluate model on validation/test set with multi-label metrics.
    Computes both global and subgroup (device/site) metrics.
    """
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels = [], []

    # subgroup accumulators
    group_probs = defaultdict(list)
    group_labels = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]", leave=False):
            inputs = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            devices = batch.get("device", ["Unknown"] * len(labels))
            sites = batch.get("site", ["Unknown"] * len(labels))

            with torch.amp.autocast(device.type):
                logits = model(inputs)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels_np)

            # Collect subgroup predictions
            for d, s, y_true, y_prob in zip(devices, sites, labels_np, probs):
                group_labels[f"device::{d}"].append(y_true)
                group_probs[f"device::{d}"].append(y_prob)
                group_labels[f"site::{s}"].append(y_true)
                group_probs[f"site::{s}"].append(y_prob)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Tune thresholds (if validation set)
    if tune_thresholds:
        best = find_best_thresholds_icbhi(all_labels, all_probs)
        thresholds = (best["tC"], best["tW"])
        # if verbose:
        #     print(f"→ Best thresholds: tC={thresholds[0]:.3f}, tW={thresholds[1]:.3f} "
        #           f"(HS={best['hs']*100:.2f}%, Sp={best['sp']*100:.2f}%, Se={best['se']*100:.2f}%)")

    # Apply thresholds
    all_preds = np.stack([
        (all_probs[:, 0] >= thresholds[0]).astype(int),
        (all_probs[:, 1] >= thresholds[1]).astype(int)
    ], axis=1)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(all_labels, all_preds, all_probs)

    # --- Subgroup metrics ---
    group_metrics = {}
    for group, y_true_list in group_labels.items():
        y_true = np.array(y_true_list)
        y_prob = np.array(group_probs[group])

        # Handle empty group (if no samples)
        if len(y_true) == 0:
            print(f"[Eval] Group: {group} has no samples, skipping metrics.")
            continue

        y_pred = np.stack([
            (y_prob[:, 0] >= thresholds[0]).astype(int),
            (y_prob[:, 1] >= thresholds[1]).astype(int)
        ], axis=1)

        group_metrics[group] = compute_multilabel_metrics(y_true, y_pred, y_prob, verbose=False)
        print(f"[Eval] Group: {group} | HS: {group_metrics[group]['icbhi_score']*100:.2f}% | "
              f"Sp: {group_metrics[group]['specificity']*100:.2f}% | "
              f"Se: {group_metrics[group]['sensitivity']*100:.2f}%")
    # Include thresholds in main metrics for logging
    global_metrics["threshold_crackle"] = thresholds[0]
    global_metrics["threshold_wheeze"] = thresholds[1]

    return avg_loss, global_metrics, group_metrics


# def evaluate(model, dataloader, criterion, device,
#              thresholds=(0.5, 0.5), tune_thresholds=True, verbose=True):
#     """
#     Evaluate model on validation/test set with multi-label metrics.
#     If tune_thresholds=True, sweeps thresholds on this set to maximize ICBHI score.
#     """
#     model.eval()
#     total_loss, n_samples = 0.0, 0
#     all_probs, all_labels = [], []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="[Eval]", leave=False):
#             inputs, labels = batch["input_values"].to(device), batch["label"].to(device)

#             # Convert 4-class to multi-label if needed
#             # labels_multilabel = convert_4class_to_multilabel(labels.cpu().numpy())
#             # labels = torch.from_numpy(labels_multilabel).to(device)

#             with torch.amp.autocast(device.type):
#                 logits = model(inputs)
#                 loss = criterion(logits, labels)
#                 # loss = multilabel_icbhi_loss(logits, labels, lambda_joint=0.3, gamma=1.5)

#             total_loss += loss.item() * inputs.size(0)
#             n_samples += inputs.size(0)

#             probs = torch.sigmoid(logits).detach().cpu().numpy()
#             all_probs.extend(probs)
#             all_labels.extend(labels.cpu().numpy())

#     all_probs = np.array(all_probs)
#     all_labels = np.array(all_labels)

#     # === Tune thresholds (if validation set) ===
#     if tune_thresholds:
#         best = find_best_thresholds_icbhi(all_labels, all_probs)
#         thresholds = (best["tC"], best["tW"])
#         if verbose:
#             print(f"→ Best thresholds: tC={thresholds[0]:.3f}, tW={thresholds[1]:.3f} "
#                   f"(HS={best['hs']*100:.2f}%, Sp={best['sp']*100:.2f}%, Se={best['se']*100:.2f}%)")

#     # Apply thresholds to probabilities
#     all_preds = np.stack([
#         (all_probs[:,0] >= thresholds[0]).astype(int),
#         (all_probs[:,1] >= thresholds[1]).astype(int)
#     ], axis=1)

#     avg_loss = total_loss / n_samples
#     metrics = compute_multilabel_metrics(all_labels, all_preds, all_probs)

#     # Store thresholds and HS from val sweep (if available)
#     metrics["threshold_crackle"] = thresholds[0]
#     metrics["threshold_wheeze"] = thresholds[1]
#     return avg_loss, metrics

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """
    Adapted training loop for multi-label binary classification.
    """

    # Hardware config
    hw_cfg = cfg.hardware
    if "visible_gpus" in hw_cfg:
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    model = model.to(device)
    print(f"Model moved to {device}")

    # ============================================================
    # LOSS FUNCTION (Multi-Label Binary)
    # ============================================================
    criterion = nn.BCEWithLogitsLoss()
    epochs = cfg.epochs
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(cfg.optimizer.final_weight_decay)
    lr = float(cfg.optimizer.lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=initial_wd)
    scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    print(f"Using Scheduler: {cfg.scheduler.type}")
    print(f"Using Loss: BCEWithLogitsLoss (multi-label binary)")

    def _get_cosine_weight_decay(epoch):
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(1, epochs + 1):
        # --- TRAINING ---
        train_loss, train_metrics, train_metrics_groups = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        # Log per-device/site training metrics
        for group, metrics_dict in train_metrics_groups.items():
            for mk, mv in metrics_dict.items():
                mlflow.log_metric(f"{prefix}_{group}_{mk}", mv, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] "
              f"Loss={train_loss:.4f} | "
              f"Normal(Se/Sp)={train_metrics['Normal_sensitivity']:.2f}/{train_metrics['Normal_specificity']:.2f} | "
              f"Crackles(Se/Sp)={train_metrics['Crackle_sensitivity']:.2f}/{train_metrics['Crackle_specificity']:.2f} | "
              f"Wheezes(Se/Sp)={train_metrics['Wheeze_sensitivity']:.2f}/{train_metrics['Wheeze_specificity']:.2f} | "
              f"Both(Se/Sp)={train_metrics['Both_sensitivity']:.2f}/{train_metrics['Both_specificity']:.2f} | "
              f"Sensitivity={train_metrics['sensitivity']:.2f}/Specificity={train_metrics['specificity']:.2f} | "
              f"ICBHI={train_metrics['icbhi_score']:.2f}")

        # --- VALIDATION ---
        if val_loader:
            val_loss, val_metrics, val_metrics_groups = evaluate(model, val_loader, criterion, device, tune_thresholds=True)

            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            # Log per-device/site validation metrics
            for group, metrics_dict in val_metrics_groups.items():
                for mk, mv in metrics_dict.items():
                    mlflow.log_metric(f"{prefix}_{group}_{mk}", mv, step=epoch)

            print(f"[{prefix}][Epoch {epoch}] "
                  f"Loss={val_loss:.4f} | "
                  f"Normal(Se/Sp)={val_metrics['Normal_sensitivity']:.2f}/{val_metrics['Normal_specificity']:.2f} | "
                  f"Crackles(Se/Sp)={val_metrics['Crackle_sensitivity']:.2f}/{val_metrics['Crackle_specificity']:.2f} | "
                  f"Wheezes(Se/Sp)={val_metrics['Wheeze_sensitivity']:.2f}/{val_metrics['Wheeze_specificity']:.2f} | "
                  f"Both(Se/Sp)={val_metrics['Both_sensitivity']:.2f}/{val_metrics['Both_specificity']:.2f} | "
                  f"Sensitivity={val_metrics['sensitivity']:.2f}/Specificity={val_metrics['specificity']:.2f} | "
                  f"ICBHI={val_metrics['icbhi_score']:.2f}")

            icbhi = val_metrics["icbhi_score"]
            if icbhi > best_icbhi:
                best_icbhi, best_state_dict, best_epoch = icbhi, model.state_dict(), epoch
                ckpt_path = (
                    f"checkpoints/{epoch}_"
                    # f"Crack_Se={val_metrics['Crackle_sensitivity']:.2f}_"
                    # f"Crack_Sp={val_metrics['Crackle_specificity']:.2f}_"
                    # f"Whz_Se={val_metrics['Wheeze_sensitivity']:.2f}_"
                    # f"Whz_Sp={val_metrics['Wheeze_specificity']:.2f}_"
                    f"Sp={val_metrics['specificity']:.2f}_"
                    f"Se={val_metrics['sensitivity']:.2f}_"
                    f"ICBHI={icbhi:.2f}_"
                    f"fold{fold_idx or 0}_best.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state_dict, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                print(f"New best model saved (Epoch {epoch}, ICBHI={icbhi*100:.2f})")

        # --- SCHEDULER & WEIGHT DECAY ---
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score")
                val_metric = val_metrics[metric_name] if cfg.scheduler.reduce_mode == "max" else val_loss
                scheduler.step(val_metric)
            else:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {lr}")
            mlflow.log_metric("lr", lr, step=epoch)

        if cfg.optimizer.cosine_weight_decay:
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # --- LOAD BEST MODEL FOR TESTING ---
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from Epoch {best_epoch} (ICBHI={best_icbhi:.2f})")
    else:
        print("No validation set provided — using last epoch weights.")
        best_state_dict = model.state_dict()

    # --- FINAL TEST EVALUATION ---
    if test_loader:
        test_loss, test_metrics, test_metrics_groups = evaluate(model, test_loader, criterion, device)

        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v)

        # Log per-device/site test metrics
        for group, metrics_dict in test_metrics_groups.items():
            for mk, mv in metrics_dict.items():
                mlflow.log_metric(f"{prefix}_{group}_{mk}", mv)

        print(f"[{prefix}] Final | "
              f"Loss={test_loss:.4f} | "
              f"Normal(Se/Sp)={test_metrics['Normal_sensitivity']:.2f}/{test_metrics['Normal_specificity']:.2f} | "
              f"Crackles(Se/Sp)={test_metrics['Crackle_sensitivity']:.2f}/{test_metrics['Crackle_specificity']:.2f} | "
              f"Wheezes(Se/Sp)={test_metrics['Wheeze_sensitivity']:.2f}/{test_metrics['Wheeze_specificity']:.2f} | "
              f"Both(Se/Sp)={test_metrics['Both_sensitivity']:.2f}/{test_metrics['Both_specificity']:.2f} | "
              f"Sensitivity={test_metrics['sensitivity']:.2f}/Specificity={test_metrics['specificity']:.2f} | "
              f"ICBHI={test_metrics['icbhi_score']:.2f}")

        # === SUMMARY TABLE (PER DEVICE / SITE) ===
        try:
            import pandas as pd
            key_metrics = ["f1_macro", "sensitivity", "specificity", "icbhi_score"]
            df_summary = pd.DataFrame(test_metrics_groups).T.fillna("-")
            cols = [c for c in key_metrics if c in df_summary.columns]
            df_summary = df_summary[cols]

            print("\n" + "=" * 65)
            print("PER-DEVICE / PER-SITE PERFORMANCE SUMMARY (Test Set)")
            print("=" * 65)
            print(df_summary.to_string(float_format=lambda x: f"{x:.3f}"))
            print("=" * 65 + "\n")

            # Save to CSV and Markdown
            os.makedirs("summaries", exist_ok=True)
            csv_path = f"summaries/group_performance_summary_fold{fold_idx or 0}.csv"
            md_path = f"summaries/group_performance_summary_fold{fold_idx or 0}.md"
            df_summary.to_csv(csv_path)
            df_summary.to_markdown(md_path)

            # Log artifacts to MLflow
            mlflow.log_artifact(csv_path, artifact_path="summaries")
            mlflow.log_artifact(md_path, artifact_path="summaries")

        except Exception as e:
            print(f"[WARN] Could not print or export per-device/site summary: {e}")


    return model, criterion



# def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
#     """
#     Adapted training loop for multi-label binary classification.
    
#     Key changes from 4-class version:
#     - Uses BCEWithLogitsLoss instead of CrossEntropyLoss
#     - Computes multi-label specific metrics
#     - Prints per-label sensitivity/specificity
#     """
    
#     # Hardware config
#     hw_cfg = cfg.hardware
    
#     if "visible_gpus" in hw_cfg:
#         set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    
#     device_id = getattr(hw_cfg, "device_id", 0)
#     device = get_device(device_id=device_id, verbose=True)
    
#     model = model.to(device)
#     print(f"Model moved to {device}")
    
#     # ============================================================
#     # LOSS FUNCTION (Multi-Label Binary)
#     # ============================================================
#     # For multi-label, we use BCEWithLogitsLoss (combines sigmoid + BCE)
#     # No class weights needed (handled per-label)
#     criterion = nn.BCEWithLogitsLoss()
    
#     # Setup optimizer
#     epochs = cfg.epochs
#     initial_wd = float(cfg.optimizer.weight_decay)
#     final_wd = float(cfg.optimizer.final_weight_decay)
#     lr = float(cfg.optimizer.lr)
    
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=lr,
#         weight_decay=initial_wd,
#     )
    
#     scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
#     scaler = torch.amp.GradScaler(device.type)
    
#     print(f"Using Scheduler: {cfg.scheduler.type}")
#     print(f"Using Loss: BCEWithLogitsLoss (multi-label binary)")
    
#     def _get_cosine_weight_decay(epoch):
#         return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))
    
#     best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0
    
#     # ============================================================
#     # TRAINING LOOP
#     # ============================================================
#     for epoch in range(1, epochs + 1):
        
#         # --- TRAINING ---
#         train_loss, train_metrics, train_metrics_groups = train_one_epoch(
#             model, train_loader, criterion, optimizer, device, scaler, epoch
#         )
        
#         prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
#         mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
#         for k, v in train_metrics.items():
#             mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)
        
#         print(f"[{prefix}][Epoch {epoch}] "
#               f"Loss={train_loss:.4f} | "
#               f"Normal(Se/Sp)={train_metrics['Normal_sensitivity']:.2f}/{train_metrics['Normal_specificity']:.2f} | "
#               f"Crackles(Se/Sp)={train_metrics['Crackle_sensitivity']:.2f}/{train_metrics['Crackle_specificity']:.2f} | "
#               f"Wheezes(Se/Sp)={train_metrics['Wheeze_sensitivity']:.2f}/{train_metrics['Wheeze_specificity']:.2f} | "
#               f"Both(Se/Sp)={train_metrics['Both_sensitivity']:.2f}/{train_metrics['Both_specificity']:.2f} | "
#               f"Sensitivity={train_metrics['sensitivity']:.2f}/Specificity={train_metrics['specificity']:.2f} | "
#               f"ICBHI={train_metrics['icbhi_score']:.2f}")
        
#         # --- VALIDATION ---
#         if val_loader:
#             val_loss, val_metrics, val_metrics_groups = evaluate(model, val_loader, criterion, device)
            
#             prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
#             mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
#             for k, v in val_metrics.items():
#                 mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)
#             # overide val-icbhi score
#             mlflow.log_metric(f"{prefix}_icbhi_score", val_metrics['icbhi_score'])
            
#             print(f"[{prefix}][Epoch {epoch}] "
#                   f"Loss={val_loss:.4f} | "
#                   f"Normal(Se/Sp)={val_metrics['Normal_sensitivity']:.2f}/{val_metrics['Normal_specificity']:.2f} | "
#                   f"Crackles(Se/Sp)={val_metrics['Crackle_sensitivity']:.2f}/{val_metrics['Crackle_specificity']:.2f} | "
#                   f"Wheezes(Se/Sp)={val_metrics['Wheeze_sensitivity']:.2f}/{val_metrics['Wheeze_specificity']:.2f} | "
#                   f"Both(Se/Sp)={val_metrics['Both_sensitivity']:.2f}/{val_metrics['Both_specificity']:.2f} | "
#                   f"Sensitivity={val_metrics['sensitivity']:.2f}/Specificity={val_metrics['specificity']:.2f} | "
#                   f"ICBHI={val_metrics['icbhi_score']:.2f}")
            
#             icbhi = val_metrics["icbhi_score"]
            
#             # Save best model by ICBHI score
#             if icbhi > best_icbhi:
#                 best_icbhi = icbhi
#                 best_state_dict = model.state_dict()
#                 best_epoch = epoch
                
#                 ckpt_path = (
#                     f"checkpoints/{epoch}_"
#                     f"Crack_Se={val_metrics['Crackle_sensitivity']:.2f}_"
#                     f"Crack_Sp={val_metrics['Crackle_specificity']:.2f}_"
#                     f"Whz_Se={val_metrics['Wheeze_sensitivity']:.2f}_"
#                     f"Whz_Sp={val_metrics['Wheeze_specificity']:.2f}_"
#                     f"Sp={val_metrics['specificity']:.2f}_"
#                     f"Se={val_metrics['sensitivity']:.2f}_"
#                     f"ICBHI={icbhi:.2f}_"
#                     f"fold{fold_idx or 0}_best.pt"
#                 )
#                 os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
#                 torch.save(best_state_dict, ckpt_path)
#                 mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
#                 print(f"New best model saved (Epoch {epoch}, ICBHI={icbhi*100:.2f})")
        
#         # --- SCHEDULER & WEIGHT DECAY ---
#         if scheduler:
#             if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
#                 metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score")
#                 val_metric = val_metrics[metric_name] if cfg.scheduler.reduce_mode == "max" else val_loss
#                 scheduler.step(val_metric)
#             else:
#                 scheduler.step()
            
#             lr = optimizer.param_groups[0]["lr"]
#             print(f"Learning rate: {lr}")
#             mlflow.log_metric("lr", lr, step=epoch)
        
#         if cfg.optimizer.cosine_weight_decay:
#             new_wd = _get_cosine_weight_decay(epoch)
#             for g in optimizer.param_groups:
#                 g["weight_decay"] = new_wd
#             mlflow.log_metric("weight_decay", new_wd, step=epoch)
    
#     # --- LOAD BEST MODEL FOR TESTING ---
#     if best_state_dict is not None:
#         model.load_state_dict(best_state_dict)
#         print(f"Loaded best model from Epoch {best_epoch} (ICBHI={best_icbhi:.2f})")
#     else:
#         print("No validation set provided — using last epoch weights.")
#         best_state_dict = model.state_dict()
    
#     # --- FINAL TEST EVALUATION ---
#     if test_loader:
#         test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        
#         prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
#         mlflow.log_metric(f"{prefix}_loss", test_loss)
#         for k, v in test_metrics.items():
#             mlflow.log_metric(f"{prefix}_{k}", v)
        
#         print(f"[{prefix}] Final | "
#               f"Loss={test_loss:.4f} | "
#               f"Normal(Se/Sp)={test_metrics['Normal_sensitivity']:.2f}/{test_metrics['Normal_specificity']:.2f} | "
#               f"Crackles(Se/Sp)={test_metrics['Crackle_sensitivity']:.2f}/{test_metrics['Crackle_specificity']:.2f} | "
#               f"Wheezes(Se/Sp)={test_metrics['Wheeze_sensitivity']:.2f}/{test_metrics['Wheeze_specificity']:.2f} | "
#               f"Both(Se/Sp)={test_metrics['Both_sensitivity']:.2f}/{test_metrics['Both_specificity']:.2f} | "
#               f"Sensitivity={test_metrics['sensitivity']:.2f}/Specificity={test_metrics['specificity']:.2f} | "
#               f"ICBHI={test_metrics['icbhi_score']:.2f}")
    
#     return model, criterion


def main_single():
    cfg = load_config("configs/config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")
    MODEL_KEY = "ast"  # Options: "cnn6", "ast", "simplerespcnn"
    
    print(f"Using model: {MODEL_KEY}")

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Build Dataset
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    # Build Model
    model = build_model(cfg.models, model_key=MODEL_KEY)
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Authenticate with MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    # Start MLFlow experiment
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))
    run_name = f"{MODEL_KEY}_{cfg.training.epochs}ep-BCE-tune-thresh"

    with mlflow.start_run(run_name=run_name):

        # Log configuration parameterss
        log_all_params(cfg)

        _, _ = train_loop(cfg.training, model, train_loader, val_loader=test_loader, test_loader=test_loader, fold_idx=0)
        
        mlflow.end_run()


if __name__ == "__main__":
    main_single()