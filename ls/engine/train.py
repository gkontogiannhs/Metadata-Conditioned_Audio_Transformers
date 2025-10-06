import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import mlflow
import math
from sklearn.utils.class_weight import compute_class_weight

from ls.engine.eval import evaluate
from ls.metrics import compute_classification_metrics
# from ls.engine.logging_utils import init_mlflow
from ls.engine.scheduler import build_scheduler
from ls.engine.utils import get_device
from ls.config.dataclasses import TrainingConfig


def get_loss(cfg: TrainingConfig, device: torch.device, class_weights: torch.Tensor = None):
    """
    Return loss function based on config and optional per-fold class weights.

    Args:
        cfg: Configuration object (Box or dataclass-like)
        device: torch.device
        class_weights: torch.Tensor or None (computed dynamically per fold)

    Returns:
        torch.nn.Module: Configured loss function
    """
    loss_type = getattr(cfg, "loss", "cross_entropy")

    # -------------------------------
    # 1) Standard Cross Entropy
    # -------------------------------
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss().to(device)

    # -------------------------------
    # 2) Weighted Cross Entropy
    # -------------------------------
    elif loss_type == "weighted_ce":
        # Priority: dynamically computed weights > static config weights
        if class_weights is not None:
            weights = class_weights.to(device)
            print(f"[INFO] Using dynamically computed class weights: {weights.cpu().numpy().round(3).tolist()}")
        # elif hasattr(cfg, "loss") and hasattr(cfg.loss, "class_weights"):
        #     weights = torch.tensor(cfg.loss.class_weights, dtype=torch.float32).to(device)
        #     print(f"[INFO] Using static class weights from config: {weights.cpu().numpy().round(3).tolist()}")
        else:
            raise ValueError(
                "Weighted CE loss selected but no class weights provided. "
                "Either compute them dynamically or specify cfg.loss.class_weights."
            )
        return nn.CrossEntropyLoss(weight=weights).to(device)

    # -------------------------------
    # 3) Focal Loss
    # -------------------------------
    elif loss_type == "focal":

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, weight=None):
                super().__init__()
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(weight=weight)

            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                return ((1 - pt) ** self.gamma * ce_loss).mean()

        gamma = getattr(cfg.loss, "gamma", 2.0)
        weights = class_weights.to(device) if class_weights is not None else None
        return FocalLoss(gamma=gamma, weight=weights).to(device)

    # -------------------------------
    # 4) Unknown loss
    # -------------------------------
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ------------------------------------------------------
# Training / Evaluation steps
# ------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, grdscaler, epoch):
    """
    Train model for one epoch and compute full set of classification metrics.
    No logging or printing here — pure computation.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type):
            logits = model(inputs)
            loss = criterion(logits, labels)

        grdscaler.scale(loss).backward()
        grdscaler.step(optimizer)
        grdscaler.update()

        total_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / n_samples
    n_classes = logits.shape[1]
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), n_classes=n_classes
    )

    return avg_loss, metrics


# ------------------------------------------------------
# Main training loops
# ------------------------------------------------------

def train_loop(cfg: TrainingConfig, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """Train model with optional validation, final test evaluation."""

    device = get_device()
    model = model.to(device)

    # Setup training components
    if getattr(cfg, "use_class_weights", False):
        labels = np.array([s['label'] for s  in train_loader.dataset.samples])
        n_classes = len(train_loader.dataset.class_counts)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=labels
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
    else:
        class_weights = None
    criterion = get_loss(cfg, device, class_weights)

    epochs = cfg.epochs
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(cfg.scheduler.final_weight_decay)
    lr = float(cfg.optimizer.lr)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=initial_wd,
    )
    scheduler = build_scheduler(cfg.scheduler, cfg.epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    print(f"Using Scheduler: {cfg.scheduler.type}")
    
    def _get_cosine_weight_decay(epoch):
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        # ------------------------
        # TRAIN
        # ------------------------
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] "
              f"Loss={train_loss:.4f} | Acc={train_metrics['accuracy']:.2f} | "
              f"BalAcc={train_metrics['balanced_acc']:.2f} | F1={train_metrics['f1_macro']:.2f} | "
              f"S_p={train_metrics['specificity']:.2f} | S_e={train_metrics['sensitivity']:.2f} | "
              f"ICBHI={train_metrics['icbhi_score']:.2f}")

        # ------------------------
        # VALIDATION
        # ------------------------
        if val_loader:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            print(f"[{prefix}][Epoch {epoch}] "
                  f"Loss={val_loss:.4f} | Acc={val_metrics['accuracy']:.2f} | "
                  f"BalAcc={val_metrics['balanced_acc']:.2f} | F1={val_metrics['f1_macro']:.2f} | "
                  f"S_p={val_metrics['specificity']:.2f} | S_e={val_metrics['sensitivity']:.2f} | "
                  f"ICBHI={val_metrics['icbhi_score']:.2f}")
            
            icbhi = val_metrics["icbhi_score"]
            # --- Save best model by ICBHI score ---
            if icbhi > best_icbhi:
                best_icbhi = icbhi
                best_state_dict = model.state_dict()
                ckpt_path = f"checkpoints/{cfg.model.name}_fold{fold_idx or 0}_best.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state_dict, ckpt_path)
                mlflow.log_artifact(ckpt_path)
                print(f"New best model saved (Epoch {epoch}, ICBHI={icbhi:.2f})")
        # ------------------------
        # END OF EPOCH — Update LR & WD
        # ------------------------
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                # Step based on validation metric
                metric_name = getattr(cfg.scheduler, "reduce_metric", "icbhi_score")
                val_metric = val_metrics[metric_name] if cfg.scheduler.reduce_mode == "max" else val_loss
                scheduler.step(val_metric)
            else:
                scheduler.step()
            
            lr = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {lr}")
            mlflow.log_metric("lr", lr, step=epoch)

        if cfg.scheduler.cosine_weight_decay:
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # ------------------------
    # LOAD BEST MODEL for final testing
    # ------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, map_location=device)
        print(f"Loaded best model from Epoch {best_epoch} (ICBHI={best_icbhi:.2f})")
    else:
        print("No validation set provided — using last epoch weights as best model.")
        best_state_dict = model.state_dict()

    # ------------------------
    # FINAL TEST EVALUATION
    # ------------------------
    if test_loader:
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"{prefix}_{k}", v)
        print(f"[{prefix}] Final | Loss={test_loss:.4f} | ICBHI={test_metrics['icbhi_score']:.2f}")

    return model, criterion