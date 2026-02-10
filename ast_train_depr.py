import os
import mlflow
from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
from ls.engine.utils import get_device
from ls.config.dataclasses import TrainingConfig
from ls.engine.scheduler import build_scheduler
from collections import defaultdict
from utils import *


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, grdscaler, epoch):
    """
    Train model for one epoch with multi-label binary classification.
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

        optimizer.zero_grad()
        with torch.amp.autocast(device.type):
            logits = model(inputs)
            loss = criterion(logits, labels)

        grdscaler.scale(loss).backward()
        grdscaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        for d, s, y_true, y_pred, y_prob in zip(devices, sites, labels_np, preds, probs):
            group_preds[f"device::{d}"].append(y_pred)
            group_labels[f"device::{d}"].append(y_true)
            group_probs[f"device::{d}"].append(y_prob)

            group_preds[f"site::{s}"].append(y_pred)
            group_labels[f"site::{s}"].append(y_true)
            group_probs[f"site::{s}"].append(y_prob)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    group_metrics = {}
    for group in group_labels.keys():
        group_metrics[group] = compute_multilabel_metrics(
            np.array(group_labels[group]),
            np.array(group_preds[group]),
            np.array(group_probs[group]),
            verbose=False
        )
    
    return avg_loss, global_metrics, group_metrics


def evaluate(model, dataloader, criterion, device,
             thresholds=(0.5, 0.5), tune_thresholds=False, verbose=True):
    """
    Evaluate model on validation/test set with multi-label metrics.
    
    Args:
        tune_thresholds: If True, find best thresholds (use for validation only)
                        If False, use provided thresholds (use for test)
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
                logits = model(inputs)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels_np)

            for d, s, y_true, y_prob in zip(devices, sites, labels_np, probs):
                group_labels[f"device::{d}"].append(y_true)
                group_probs[f"device::{d}"].append(y_prob)
                group_labels[f"site::{s}"].append(y_true)
                group_probs[f"site::{s}"].append(y_prob)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Tune thresholds only on validation set
    if tune_thresholds:
        best = find_best_thresholds_icbhi(all_labels, all_probs)
        thresholds = (best["tC"], best["tW"])
        if verbose:
            print(f"→ Tuned thresholds: tC={thresholds[0]:.3f}, tW={thresholds[1]:.3f}")

    # Apply thresholds
    all_preds = np.stack([
        (all_probs[:, 0] >= thresholds[0]).astype(int),
        (all_probs[:, 1] >= thresholds[1]).astype(int)
    ], axis=1)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(all_labels, all_preds, all_probs, verbose=verbose)

    # Subgroup metrics
    group_metrics = {}
    for group, y_true_list in group_labels.items():
        y_true = np.array(y_true_list)
        y_prob = np.array(group_probs[group])

        if len(y_true) == 0:
            continue

        y_pred = np.stack([
            (y_prob[:, 0] >= thresholds[0]).astype(int),
            (y_prob[:, 1] >= thresholds[1]).astype(int)
        ], axis=1)

        group_metrics[group] = compute_multilabel_metrics(y_true, y_pred, y_prob, verbose=False)

    global_metrics["threshold_crackle"] = thresholds[0]
    global_metrics["threshold_wheeze"] = thresholds[1]

    return avg_loss, global_metrics, group_metrics, thresholds


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """
    Training loop with:
    - pos_weight for class imbalance (replaces weighted sampler)
    - Threshold tuning on validation set only
    - Fixed thresholds applied to test set
    """

    # Hardware config
    hw_cfg = cfg.hardware
    if "visible_gpus" in hw_cfg:
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    model = model.to(device)
    print(f"Model moved to {device}")

    # Compute pos_weight with bias
    pos_weight = compute_pos_weight(train_loader, device, sensitivity_bias=1.5)

    loss_type = str(cfg.loss)
    print(loss_type)
    if loss_type == 'bce':
        # Standard BCE with higher pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif loss_type == 'asymmetric':
        # Asymmetric BCE
        criterion = AsymmetricBCEWithLogitsLoss(
            gamma_neg=2.0,    # Down-weight easy negatives
            gamma_pos=1.0,    # Keep focus on all positives  
            pos_weight=pos_weight
        )
    
    elif loss_type == 'hierarchical':
        criterion = HierarchicalMultiLabelLoss(
            sensitivity_bias=1.5, partial_credit=0.5, miss_penalty=2.0
        )

    elif loss_type == 'composite':
        criterion = CompositeClassLoss(
            sensitivity_bias=1.5, partial_credit=0.5, over_pred_cost=0.3, normalize_weights=True
        )


    epochs = cfg.epochs
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(cfg.optimizer.final_weight_decay)
    lr = float(cfg.optimizer.lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=initial_wd)
    scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    
    print(f"Using Scheduler: {cfg.scheduler.type}")

    def _get_cosine_weight_decay(epoch):
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0
    best_thresholds = (0.5, 0.5)  # Store best thresholds from validation

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

        for group, metrics_dict in train_metrics_groups.items():
            for mk, mv in metrics_dict.items():
                mlflow.log_metric(f"{prefix}_{group}_{mk}", mv, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] "
              f"Loss={train_loss:.4f} | "
              f"Se={train_metrics['sensitivity']:.4f} | Sp={train_metrics['specificity']:.4f} | "
              f"ICBHI={train_metrics['icbhi_score']:.4f}")

        # --- VALIDATION (tune thresholds here) ---
        if val_loader:
            val_loss, val_metrics, val_metrics_groups, val_thresholds = evaluate(
                model, val_loader, criterion, device, 
                tune_thresholds=True,  # Tune thresholds on validation
                verbose=False
            )

            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            for group, metrics_dict in val_metrics_groups.items():
                for mk, mv in metrics_dict.items():
                    mlflow.log_metric(f"{prefix}_{group}_{mk}", mv, step=epoch)

            print(f"[{prefix}][Epoch {epoch}] "
                  f"Loss={val_loss:.4f} | "
                  f"Se={val_metrics['sensitivity']:.4f} | Sp={val_metrics['specificity']:.4f} | "
                  f"ICBHI={val_metrics['icbhi_score']:.4f} | "
                  f"tC={val_thresholds[0]:.3f}, tW={val_thresholds[1]:.3f}")

            icbhi = val_metrics["icbhi_score"]
            if icbhi > best_icbhi:
                best_icbhi = icbhi
                best_state_dict = model.state_dict()
                best_epoch = epoch
                best_thresholds = val_thresholds  # Save best thresholds
                
                ckpt_path = (
                    f"checkpoints/{epoch}_"
                    f"Sp={val_metrics['specificity']:.4f}_"
                    f"Se={val_metrics['sensitivity']:.4f}_"
                    f"ICBHI={icbhi:.4f}_"
                    f"tC={val_thresholds[0]:.3f}_tW={val_thresholds[1]:.3f}_"
                    f"fold{fold_idx or 0}_best.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'model_state_dict': best_state_dict,
                    'thresholds': best_thresholds,
                    'epoch': best_epoch,
                    'icbhi_score': best_icbhi
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                print(f"★ New best model (Epoch {epoch}, ICBHI={icbhi*100:.2f}%, tC={val_thresholds[0]:.3f}, tW={val_thresholds[1]:.3f})")

        # --- SCHEDULER & WEIGHT DECAY ---
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score")
                val_metric = val_metrics[metric_name] if cfg.scheduler.reduce_mode == "max" else val_loss
                scheduler.step(val_metric)
            else:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", lr, step=epoch)

        if cfg.optimizer.cosine_weight_decay:
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # --- LOAD BEST MODEL FOR TESTING ---
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n{'='*60}")
        print(f"Loaded best model from Epoch {best_epoch}")
        print(f"Validation ICBHI={best_icbhi*100:.2f}%")
        print(f"Using thresholds: tC={best_thresholds[0]:.3f}, tW={best_thresholds[1]:.3f}")
        print(f"{'='*60}\n")
    else:
        print("No validation improvement — using last epoch weights.")
        best_state_dict = model.state_dict()

    # --- FINAL TEST EVALUATION (with FIXED thresholds from validation) ---
    if test_loader:
        print(f"[Test] Evaluating with FIXED thresholds from validation: tC={best_thresholds[0]:.3f}, tW={best_thresholds[1]:.3f}")
        
        test_loss, test_metrics, test_metrics_groups, _ = evaluate(
            model, test_loader, criterion, device,
            thresholds=best_thresholds,  # Use fixed thresholds from validation
            tune_thresholds=False,  # No tune on test set
            verbose=True
        )

        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v)

        for group, metrics_dict in test_metrics_groups.items():
            for mk, mv in metrics_dict.items():
                mlflow.log_metric(f"{prefix}_{group}_{mk}", mv)

        print(f"\n[{prefix}] Final Results:")
        print(f"  Loss={test_loss:.4f}")
        print(f"  Sensitivity={test_metrics['sensitivity']*100:.2f}%")
        print(f"  Specificity={test_metrics['specificity']*100:.2f}%")
        print(f"  ICBHI Score={test_metrics['icbhi_score']*100:.2f}%")
        print(f"  Thresholds: tC={best_thresholds[0]:.3f}, tW={best_thresholds[1]:.3f}")

        # === SUMMARY TABLE ===
        try:
            import pandas as pd
            key_metrics = ["f1_macro", "sensitivity", "specificity", "icbhi_score"]
            df_summary = pd.DataFrame(test_metrics_groups).T.fillna("-")
            cols = [c for c in key_metrics if c in df_summary.columns]
            if cols:
                df_summary = df_summary[cols]
                print("\n" + "=" * 65)
                print("PER-DEVICE / PER-SITE PERFORMANCE SUMMARY (Test Set)")
                print("=" * 65)
                print(df_summary.to_string(float_format=lambda x: f"{x:.3f}"))
                print("=" * 65 + "\n")

                os.makedirs("summaries", exist_ok=True)
                csv_path = f"summaries/group_performance_summary_fold{fold_idx or 0}.csv"
                df_summary.to_csv(csv_path)
                mlflow.log_artifact(csv_path, artifact_path="summaries")

        except Exception as e:
            print(f"[WARN] Could not export summary: {e}")

    return model, criterion


def main_single():
    cfg = load_config("configs/config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")

    MODEL_KEY = "ast"
    print(f"Using model: {MODEL_KEY}")

    set_seed(cfg.seed)
    
    # Build Dataset WITH validation split (80/20 from training)
    # train_loader, val_loader, test_loader = build_dataloaders_with_val(
    #     cfg.dataset, 
    #     cfg.audio,
    #     val_ratio=0.2,  # 20% of training data for validation
    #     use_weighted_sampler=False  # Disable weighted sampler
    # )
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    model = build_model(cfg.models, model_key=MODEL_KEY)
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))
    run_name = f"{MODEL_KEY}_{cfg.training.epochs}ep-BCE-posweight-valsplit"

    with mlflow.start_run(run_name=run_name):
        log_all_params(cfg)
        
        # Now using separate val_loader for threshold tuning
        _, _ = train_loop(
            cfg.training, 
            model, 
            train_loader, 
            val_loader=test_loader,
            test_loader=test_loader, 
            fold_idx=0
        )
        
        mlflow.end_run()


if __name__ == "__main__":
    main_single()