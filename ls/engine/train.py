
"""
Fixed training code compatible with all experiment types.
Handles vanilla AST, FiLM-AST, and all ablation configurations.
"""

import os
import mlflow
from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed, get_device
from ls.engine.logging_utils import get_or_create_experiment, log_all_params
from ls.engine.scheduler import build_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict


def set_visible_gpus(gpus: str, verbose: bool = True):
    """Restrict which GPUs PyTorch can see."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if verbose:
        print(f"[CUDA] Visible devices set to: {gpus}")
    torch.cuda.device_count()


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


def forward_model(model, inputs, batch, device):
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

def train_one_epoch(model, dataloader, criterion, optimizer, device, grdscaler, epoch):
    """
    Train for one epoch.
    Fixed: Better handling of metadata for both vanilla and FiLM models.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    group_preds = defaultdict(list)
    group_labels = defaultdict(list)
    group_probs = defaultdict(list)

    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        inputs = batch["input_values"].to(device)
        labels = batch["label"].to(device)
        
        # Get metadata (may be missing for vanilla AST)
        devices = batch.get("device", ["Unknown"] * len(labels))
        sites = batch.get("site", ["Unknown"] * len(labels))

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = forward_model(model, inputs, batch, device)
            loss = criterion(logits, labels)

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

        # Subgroup metrics
        for d, s, y_true, y_pred, y_prob in zip(devices, sites, labels_np, preds, probs):
            group_preds[f"device_{d}"].append(y_pred)
            group_labels[f"device_{d}"].append(y_true)
            group_probs[f"device_{d}"].append(y_prob)

            group_preds[f"site_{s}"].append(y_pred)
            group_labels[f"site_{s}"].append(y_true)
            group_probs[f"site_{s}"].append(y_prob)

    # Global metrics
    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
    )

    # Group-level metrics
    group_metrics = {}
    for group in group_labels.keys():
        if len(group_labels[group]) > 0:  # Skip empty groups
            group_metrics[group] = compute_multilabel_metrics(
                np.array(group_labels[group]),
                np.array(group_preds[group]),
                np.array(group_probs[group]),
                verbose=False
            )

    return avg_loss, global_metrics, group_metrics


def evaluate(model, dataloader, criterion, device,
             thresholds=(0.5, 0.5), tune_thresholds=True, verbose=True):
    """
    Evaluate model.
    Fixed: Better metadata handling and threshold tuning.
    """
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels = [], []
    group_probs = defaultdict(list)
    group_labels = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]", leave=False):
            inputs = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            devices = batch.get("device", ["Unknown"] * len(labels))
            sites = batch.get("site", ["Unknown"] * len(labels))

            with torch.amp.autocast(device.type):
                logits = forward_model(model, inputs, batch, device)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels_np)

            # Collect subgroup data
            for d, s, y_true, y_prob in zip(devices, sites, labels_np, probs):
                group_labels[f"device_{d}"].append(y_true)
                group_probs[f"device_{d}"].append(y_prob)
                group_labels[f"site_{s}"].append(y_true)
                group_probs[f"site_{s}"].append(y_prob)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Tune thresholds
    if tune_thresholds:
        best = find_best_thresholds_icbhi(all_labels, all_probs)
        thresholds = (best["tC"], best["tW"])
        if verbose:
            print(f"→ Best thresholds: tC={thresholds[0]:.3f}, tW={thresholds[1]:.3f} "
                  f"(HS={best['hs']*100:.2f}%)")

    # Apply thresholds
    all_preds = np.stack([
        (all_probs[:, 0] >= thresholds[0]).astype(int),
        (all_probs[:, 1] >= thresholds[1]).astype(int)
    ], axis=1)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(all_labels, all_preds, all_probs, verbose=False)

    # Subgroup metrics
    group_metrics = {}
    for group, y_true_list in group_labels.items():
        if len(y_true_list) == 0:
            continue

        y_true = np.array(y_true_list)
        y_prob = np.array(group_probs[group])

        y_pred = np.stack([
            (y_prob[:, 0] >= thresholds[0]).astype(int),
            (y_prob[:, 1] >= thresholds[1]).astype(int)
        ], axis=1)

        group_metrics[group] = compute_multilabel_metrics(y_true, y_pred, y_prob, verbose=False)

    # Store thresholds
    global_metrics["threshold_crackle"] = thresholds[0]
    global_metrics["threshold_wheeze"] = thresholds[1]

    return avg_loss, global_metrics, group_metrics


def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """
    Main training loop.
    Fixed: Better compatibility with all model types and experiment configurations.
    """
    # Hardware setup
    hw_cfg = cfg.hardware
    if hasattr(hw_cfg, 'visible_gpus'):
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    model = model.to(device)
    print(f"Model moved to {device}")

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    epochs = cfg.epochs
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(getattr(cfg.optimizer, 'final_weight_decay', initial_wd))
    lr = float(cfg.optimizer.lr)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=initial_wd)
    scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    
    print(f"Using Scheduler: {cfg.scheduler.type}")
    print(f"Using Loss: BCEWithLogitsLoss")
    print(f"Epochs: {epochs}, LR: {lr}, WD: {initial_wd}")

    def _get_cosine_weight_decay(epoch):
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0

    # Training loop
    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_metrics, train_metrics_groups = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        
        for k, v in train_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        print(f"[{prefix}][Epoch {epoch}/{epochs}] "
              f"Loss={train_loss:.4f} | "
              f"ICBHI={train_metrics['icbhi_score']:.4f} | "
              f"Se={train_metrics['sensitivity']:.4f} | "
              f"Sp={train_metrics['specificity']:.4f}")

        # Validation
        if val_loader:
            val_loss, val_metrics, val_metrics_groups = evaluate(
                model, val_loader, criterion, device, tune_thresholds=True, verbose=False
            )

            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            
            for k, v in val_metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            print(f"[{prefix}][Epoch {epoch}/{epochs}] "
                  f"Loss={val_loss:.4f} | "
                  f"ICBHI={val_metrics['icbhi_score']:.4f} | "
                  f"Se={val_metrics['sensitivity']:.4f} | "
                  f"Sp={val_metrics['specificity']:.4f} | "
                  f"Thresholds=({val_metrics['threshold_crackle']:.2f}, {val_metrics['threshold_wheeze']:.2f})")

            # Save best model
            icbhi = val_metrics["icbhi_score"]
            if icbhi > best_icbhi:
                best_icbhi, best_state_dict, best_epoch = icbhi, model.state_dict(), epoch
                ckpt_path = (
                    f"checkpoints/epoch{epoch}_"
                    f"ICBHI{icbhi:.4f}_"
                    f"Se{val_metrics['sensitivity']:.4f}_"
                    f"Sp{val_metrics['specificity']:.4f}_"
                    f"fold{fold_idx or 0}.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'thresholds': (val_metrics['threshold_crackle'], val_metrics['threshold_wheeze'])
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                os.remove(ckpt_path)  # Clean up local file after logging
                print(f"  → New best model saved! (ICBHI={icbhi*100:.2f}%)")

        # Scheduler step
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score")
                val_metric = val_metrics.get(metric_name, val_loss)
                scheduler.step(val_metric)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", current_lr, step=epoch)

        # Cosine weight decay
        if getattr(cfg.optimizer, 'cosine_weight_decay', False):
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # Load best model for testing
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n✓ Loaded best model from Epoch {best_epoch} (ICBHI={best_icbhi:.4f})")
    else:
        print("\n⚠ No validation - using final epoch weights")
        best_state_dict = model.state_dict()

    # Final test evaluation
    if test_loader:
        test_loss, test_metrics, test_metrics_groups = evaluate(
            model, test_loader, criterion, device, tune_thresholds=False, verbose=True
        )

        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v)

        print("\n" + "="*70)
        print(f"FINAL TEST RESULTS")
        print("="*70)
        print(f"Loss: {test_loss:.4f}")
        print(f"ICBHI Score: {test_metrics['icbhi_score']:.4f}")
        print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
        print("="*70 + "\n")

        # Per-class breakdown
        print("Per-Class Performance:")
        for cls in ['Normal', 'Crackle', 'Wheeze', 'Both']:
            print(f"  {cls:8s}: Se={test_metrics[f'{cls}_sensitivity']:.4f} | "
                  f"Sp={test_metrics[f'{cls}_specificity']:.4f} | "
                  f"F1={test_metrics[f'{cls}_f1']:.4f}")

        # Group-level summary
        if test_metrics_groups:
            try:
                import pandas as pd
                key_metrics = ["icbhi_score", "sensitivity", "specificity", "macro_f1"]
                df_summary = pd.DataFrame(test_metrics_groups).T
                cols = [c for c in key_metrics if c in df_summary.columns]
                if cols:
                    df_summary = df_summary[cols]

                    print("\n" + "="*70)
                    print("PER-DEVICE / PER-SITE PERFORMANCE")
                    print("="*70)
                    print(df_summary.to_string(float_format=lambda x: f"{x:.4f}"))
                    print("="*70 + "\n")

                    # Save summary
                    os.makedirs("summaries", exist_ok=True)
                    csv_path = f"summaries/group_summary_fold{fold_idx or 0}.csv"
                    md_path = f"summaries/group_summary_fold{fold_idx or 0}.md"
                    df_summary.to_csv(csv_path)
                    with open(md_path, 'w') as f:
                        f.write(df_summary.to_markdown())

                    mlflow.log_artifact(csv_path, artifact_path="summaries")
                    mlflow.log_artifact(md_path, artifact_path="summaries")
            except Exception as e:
                print(f"[WARN] Could not generate summary table: {e}")

    return model, criterion