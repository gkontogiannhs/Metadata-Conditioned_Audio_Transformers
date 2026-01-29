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

from ls.engine.utils import build_criterion, set_visible_gpus
from ls.engine.eval import evaluate, compute_multilabel_metrics, compute_multiclass_metrics


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
    

def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """
    Main training loop with proper threshold handling and multi-mode support.
    
    Supports:
        - Vanilla AST and FiLM-AST models
        - Multi-label and multi-class classification
        - Multiple loss functions (BCE, Focal, Asymmetric, CE)
        - Threshold optimization on validation, applied to test
        - Cosine weight decay scheduling
        - MLflow logging
    
    Args:
        cfg: Configuration object
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        test_loader: Test DataLoader (optional)
        fold_idx: Fold index for cross-validation (optional)
    
    Returns:
        Tuple[model, criterion]: Trained model and loss function
    """
    # ============================================================
    # HARDWARE SETUP
    # ============================================================
    hw_cfg = cfg.hardware
    if hasattr(hw_cfg, 'visible_gpus'):
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    model = model.to(device)
    print(f"[Hardware] Model moved to {device}")

    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    print(cfg)
    criterion = build_criterion(cfg, device=device)
    multi_label = True # cfg.dataset.multi_label

    # ============================================================
    # OPTIMIZER SETUP
    # ============================================================
    epochs = cfg.epochs
    lr = float(cfg.optimizer.lr)
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(getattr(cfg.optimizer, 'final_weight_decay', initial_wd))
    use_cosine_wd = getattr(cfg.optimizer, 'cosine_weight_decay', False)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=initial_wd)
    scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    
    print(f"\n{'='*70}")
    print(f"[Training Configuration]")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {initial_wd}" + (f" → {final_wd} (cosine)" if use_cosine_wd else ""))
    print(f"  Scheduler: {cfg.scheduler.type}")
    print(f"  Multi-label: {multi_label}")
    print(f"  Loss: {type(criterion).__name__}")
    print(f"{'='*70}\n")

    def _get_cosine_weight_decay(epoch):
        """Compute cosine-annealed weight decay."""
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    # ============================================================
    # TRACKING VARIABLES
    # ============================================================
    best_metric = -np.inf
    best_state_dict = None
    best_epoch = 0
    best_thresholds = (0.5, 0.5) if multi_label else None  # Only for multi-label
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    # model.freeze_backbone(until=10)
    for epoch in range(1, epochs + 1):
        
        # ----------------------------------------------------------
        # TRAINING PHASE
        # ----------------------------------------------------------
        if multi_label:
            train_loss, train_metrics, train_metrics_groups = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch, multi_label=True
            )
        else:
            train_loss, train_metrics, train_metrics_groups = train_one_epoch_multiclass(
                model, train_loader, criterion, optimizer, device, scaler, epoch
            )

        # if epoch == 10:
        #     model.unfreeze_all()
        
        # Log training metrics
        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        # Print training summary
        if multi_label:
            print(f"[{prefix}][Epoch {epoch}/{epochs}] "
                  f"Loss={train_loss:.4f} | "
                  f"ICBHI={train_metrics['icbhi_score']:.4f} | "
                  f"Se={train_metrics['sensitivity']:.4f} | "
                  f"Sp={train_metrics['specificity']:.4f}")
        else:
            print(f"[{prefix}][Epoch {epoch}/{epochs}] "
                  f"Loss={train_loss:.4f} | "
                  f"Acc={train_metrics.get('accuracy', 0):.4f} | "
                  f"MacroF1={train_metrics.get('macro_f1', 0):.4f}")

        # ----------------------------------------------------------
        # VALIDATION PHASE
        # ----------------------------------------------------------
        if val_loader:
            if multi_label:
                val_loss, val_metrics, val_metrics_groups = evaluate(
                    model, val_loader, criterion, device,
                    multi_label=True,
                    thresholds=(0.5, 0.5),
                    tune_thresholds=True,
                    verbose=False
                )
                current_metric = val_metrics["icbhi_score"]
                val_thresholds = (val_metrics['threshold_crackle'], val_metrics['threshold_wheeze'])
            else:
                val_loss, val_metrics, val_metrics_groups = evaluate(
                    model, val_loader, criterion, device,
                    multi_label=False,
                    tune_thresholds=False,
                    verbose=False
                )
                current_metric = val_metrics.get("macro_f1", val_metrics.get("accuracy", 0))
                val_thresholds = None

            # Log validation metrics
            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            # Print validation summary
            if multi_label:
                print(f"[{prefix}][Epoch {epoch}/{epochs}] "
                      f"Loss={val_loss:.4f} | "
                      f"ICBHI={val_metrics['icbhi_score']:.4f} | "
                      f"Se={val_metrics['sensitivity']:.4f} | "
                      f"Sp={val_metrics['specificity']:.4f} | "
                      f"Thresh=({val_thresholds[0]:.2f}, {val_thresholds[1]:.2f})")
            else:
                print(f"[{prefix}][Epoch {epoch}/{epochs}] "
                      f"Loss={val_loss:.4f} | "
                      f"Acc={val_metrics.get('accuracy', 0):.4f} | "
                      f"MacroF1={val_metrics.get('macro_f1', 0):.4f}")

            # ----------------------------------------------------------
            # SAVE BEST MODEL
            # ----------------------------------------------------------
            if current_metric > best_metric:
                best_metric = current_metric
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                best_thresholds = val_thresholds  # Save thresholds with best model

                # Save checkpoint
                ckpt_info = {
                    'epoch': epoch,
                    'model_state_dict': best_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'multi_label': multi_label,
                }
                if multi_label:
                    ckpt_info['thresholds'] = best_thresholds

                if multi_label:
                    ckpt_path = (
                        f"checkpoints/epoch{epoch}_"
                        f"ICBHI{current_metric:.4f}_"
                        f"Se{val_metrics['sensitivity']:.4f}_"
                        f"Sp{val_metrics['specificity']:.4f}_"
                        f"fold{fold_idx or 0}.pt"
                    )
                else:
                    ckpt_path = (
                        f"checkpoints/epoch{epoch}_"
                        f"F1{current_metric:.4f}_"
                        f"Acc{val_metrics.get('accuracy', 0):.4f}_"
                        f"fold{fold_idx or 0}.pt"
                    )

                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(ckpt_info, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                os.remove(ckpt_path)
                
                metric_name = "ICBHI" if multi_label else "F1"
                print(f"  → New best model saved! ({metric_name}={current_metric*100:.2f}%)")

        # ----------------------------------------------------------
        # SCHEDULER STEP
        # ----------------------------------------------------------
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score" if multi_label else "macro_f1")
                scheduler_metric = val_metrics.get(metric_name, val_loss)
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", current_lr, step=epoch)

        # ----------------------------------------------------------
        # COSINE WEIGHT DECAY
        # ----------------------------------------------------------
        if use_cosine_wd:
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # ============================================================
    # LOAD BEST MODEL
    # ============================================================
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        metric_name = "ICBHI" if multi_label else "F1"
        print(f"\n{'='*70}")
        print(f"✓ Loaded best model from Epoch {best_epoch} ({metric_name}={best_metric:.4f})")
        if multi_label and best_thresholds:
            print(f"✓ Using optimized thresholds: Crackle={best_thresholds[0]:.3f}, Wheeze={best_thresholds[1]:.3f}")
        print(f"{'='*70}\n")
    else:
        print("\n⚠ No validation performed - using final epoch weights")
        best_state_dict = model.state_dict()
        best_thresholds = (0.5, 0.5) if multi_label else None

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================
    if test_loader:
        print(f"\n{'='*70}")
        print(f"FINAL TEST EVALUATION")
        print(f"{'='*70}")
        
        if multi_label:
            test_loss, test_metrics, test_metrics_groups = evaluate(
                model, test_loader, criterion, device,
                multi_label=True,
                thresholds=best_thresholds,  # Use thresholds from validation
                tune_thresholds=False,       # Do NOT re-tune on test
                verbose=True
            )
        else:
            test_loss, test_metrics, test_metrics_groups = evaluate(
                model, test_loader, criterion, device,
                multi_label=False,
                tune_thresholds=False,
                verbose=True
            )

        # Log test metrics
        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v)

        # Print test summary
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"  Loss: {test_loss:.4f}")
        
        if multi_label:
            print(f"  ICBHI Score: {test_metrics['icbhi_score']:.4f}")
            print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {test_metrics['specificity']:.4f}")
            print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"  Thresholds: Crackle={best_thresholds[0]:.3f}, Wheeze={best_thresholds[1]:.3f}")
            
            print(f"\n  Per-Class Performance:")
            for cls in ['Normal', 'Crackle', 'Wheeze', 'Both']:
                print(f"    {cls:8s}: Se={test_metrics[f'{cls}_sensitivity']:.4f} | "
                      f"Sp={test_metrics[f'{cls}_specificity']:.4f} | "
                      f"F1={test_metrics[f'{cls}_f1']:.4f}")
        else:
            print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"  Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
            
            if 'per_class_f1' in test_metrics:
                print(f"\n  Per-Class F1:")
                for i, f1 in enumerate(test_metrics['per_class_f1']):
                    print(f"    Class {i}: {f1:.4f}")
        
        print(f"{'='*70}\n")

        # ----------------------------------------------------------
        # GROUP-LEVEL SUMMARY
        # ----------------------------------------------------------
        if test_metrics_groups:
            try:
                import pandas as pd
                
                if multi_label:
                    key_metrics = ["icbhi_score", "sensitivity", "specificity", "macro_f1"]
                else:
                    key_metrics = ["accuracy", "macro_f1"]
                
                df_summary = pd.DataFrame(test_metrics_groups).T
                cols = [c for c in key_metrics if c in df_summary.columns]
                
                if cols:
                    df_summary = df_summary[cols]

                    print(f"\n{'='*70}")
                    print("PER-DEVICE / PER-SITE PERFORMANCE")
                    print(f"{'='*70}")
                    print(df_summary.to_string(float_format=lambda x: f"{x:.4f}"))
                    print(f"{'='*70}\n")

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, multi_label=True):
    """
    Train for one epoch (multi-label mode).
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
        
        devices = batch.get("device", ["Unknown"] * len(labels))
        sites = batch.get("site", ["Unknown"] * len(labels))

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = _forward_model(model, inputs, batch, device)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

        # Subgroup tracking
        for d, s, y_true, y_pred, y_prob in zip(devices, sites, labels_np, preds, probs):
            group_preds[f"device_{d}"].append(y_pred)
            group_labels[f"device_{d}"].append(y_true)
            group_probs[f"device_{d}"].append(y_prob)
            group_preds[f"site_{s}"].append(y_pred)
            group_labels[f"site_{s}"].append(y_true)
            group_probs[f"site_{s}"].append(y_prob)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multilabel_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
    )

    group_metrics = {}
    for group in group_labels.keys():
        if len(group_labels[group]) > 0:
            group_metrics[group] = compute_multilabel_metrics(
                np.array(group_labels[group]),
                np.array(group_preds[group]),
                np.array(group_probs[group]),
                verbose=False
            )

    return avg_loss, global_metrics, group_metrics


def train_one_epoch_multiclass(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """
    Train for one epoch (multi-class mode).
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
        
        devices = batch.get("device", ["Unknown"] * len(labels))
        sites = batch.get("site", ["Unknown"] * len(labels))

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = _forward_model(model, inputs, batch, device)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

        # Subgroup tracking
        for d, s, y_true, y_pred, y_prob in zip(devices, sites, labels_np, preds, probs):
            group_preds[f"device_{d}"].append(y_pred)
            group_labels[f"device_{d}"].append(y_true)
            group_probs[f"device_{d}"].append(y_prob)
            group_preds[f"site_{s}"].append(y_pred)
            group_labels[f"site_{s}"].append(y_true)
            group_probs[f"site_{s}"].append(y_prob)

    avg_loss = total_loss / n_samples
    global_metrics = compute_multiclass_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
    )

    group_metrics = {}
    for group in group_labels.keys():
        if len(group_labels[group]) > 0:
            group_metrics[group] = compute_multiclass_metrics(
                np.array(group_labels[group]),
                np.array(group_preds[group]),
                np.array(group_probs[group]),
                verbose=False
            )

    return avg_loss, global_metrics, group_metrics