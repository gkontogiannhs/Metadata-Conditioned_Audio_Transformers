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
from ls.engine.utils import get_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, grdscaler, epoch):
    """
    Train model for one epoch and compute full set of classification metrics.
    No logging or printing here — pure computation.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)

        optimizer.zero_grad(set_to_none=False)
        with torch.amp.autocast(device.type):
            logits = model(inputs)
            loss = criterion(logits, labels)

        grdscaler.scale(loss).backward()
        grdscaler.step(optimizer)
        grdscaler.update()

        total_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

        # probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
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

# ------------------------------------------------------
# Main training loops
# ------------------------------------------------------


def train_loop(cfg: TrainingConfig, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """Train model with optional validation, final test evaluation."""

    # Hardware config
    hw_cfg = cfg.hardware

    # Restrict visible GPUs if specified
    if "visible_gpus" in hw_cfg:
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)

    # Select device
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)

    # Handle DataParallel logic
    use_dp = getattr(hw_cfg, "use_dataparallel", False)

    # if use_dp and torch.cuda.device_count() > 1:
    #     print(f"[Model] Using {torch.cuda.device_count()} GPUs via DataParallel")
    #     model = nn.DataParallel(model)
    # elif use_dp and torch.cuda.device_count() <= 1:
    #     print("[Model] Skipping DataParallel: only one GPU visible")
    # else:
    #     print(f"[Model] Using single device: {device}")

    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Setup training components
    if getattr(cfg, "use_class_weights", False):
        class_counts = train_loader.dataset.class_counts
        print(f"Class counts: {class_counts}")
        total = sum(class_counts)
        print(f"Total samples: {total}")
        class_weights = torch.tensor([total/count for count in class_counts])
        print(f"Initial alpha (inverse freq): {class_weights}")
        # Normalize to 1 and scale by the factor of length
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        print(f"Normalized alpha (sum to num classes): {class_weights}")
    else:
        class_weights = None
    criterion = get_loss(cfg, device, class_weights)

    epochs = cfg.epochs
    initial_wd = float(cfg.optimizer.weight_decay)
    final_wd = float(cfg.optimizer.final_weight_decay)
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

        # if epoch < 10:
        #     try:
        #         model.module.freeze_backbone()                  # train only classifier
        #     except AttributeError:
        #         model.freeze_backbone()
        # elif epoch == 10:
        #     try:
        #         model.module.unfreeze_all()
        #         model.module.freeze_backbone(until_block=9)     # unfreeze last 3–4 blocks
        #     except AttributeError:
        #         model.unfreeze_all()
        #         model.freeze_backbone(until_block=9)
        # elif epoch == 40:
        #     try:
        #         model.module.unfreeze_all()
        #     except AttributeError:
        #         model.unfreeze_all()

        # adjust_learning_rate(optimizer, epoch, cfg)

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
                ckpt_path = f"checkpoints/{epoch}_Sp={val_metrics['specificity']:.2f}_S_e={val_metrics['sensitivity']:.2f}_icbhiScore={icbhi:.2f}_fold{fold_idx or 0}_best.pt"
                # ckpt_path = f"checkpoints/{cfg.model_name}-{epoch}_fold{fold_idx or 0}_best.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state_dict, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                print(f"New best model saved (Epoch {epoch}, ICBHI={icbhi:.2f})")
        # -----------------------------
        # END OF EPOCH — Update LR & WD
        # -----------------------------
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

        if cfg.optimizer.cosine_weight_decay:
            new_wd = _get_cosine_weight_decay(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            mlflow.log_metric("weight_decay", new_wd, step=epoch)

    # ------------------------
    # LOAD BEST MODEL for final testing
    # ------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
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