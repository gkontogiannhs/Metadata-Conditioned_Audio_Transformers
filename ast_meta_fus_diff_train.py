import os
import mlflow
from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
from ls.models.builder import build_model
from ls.engine.utils import set_seed
from ls.engine.logging_utils import get_or_create_experiment, log_all_params
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def get_meta_proj_parameter_groups(model, base_lr, cfg):
    """
    Differential learning rates for ASTMetaProj.
    """
    backbone_lr_scale = getattr(cfg.optimizer, 'backbone_lr_scale', 0.01)
    embedding_lr_scale = getattr(cfg.optimizer, 'embedding_lr_scale', 0.5)
    proj_lr_scale = getattr(cfg.optimizer, 'proj_lr_scale', 1.0)
    classifier_lr_scale = getattr(cfg.optimizer, 'classifier_lr_scale', 2.0)
    
    ast_backbone_params = []
    embedding_params = []
    proj_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'classifier' in name:
            classifier_params.append(param)
        elif 'metadata_proj' in name or 'gate' in name:
            proj_params.append(param)
        elif 'dev_emb' in name or 'site_emb' in name:
            embedding_params.append(param)
        elif 'ast' in name:
            ast_backbone_params.append(param)
        else:
            proj_params.append(param)
    
    groups = []
    
    if ast_backbone_params:
        groups.append({
            'params': ast_backbone_params,
            'lr': base_lr * backbone_lr_scale,
            'name': 'ast_backbone'
        })
    
    if embedding_params:
        groups.append({
            'params': embedding_params,
            'lr': base_lr * embedding_lr_scale,
            'name': 'embeddings'
        })
    
    if proj_params:
        groups.append({
            'params': proj_params,
            'lr': base_lr * proj_lr_scale,
            'name': 'metadata_proj'
        })
    
    if classifier_params:
        groups.append({
            'params': classifier_params,
            'lr': base_lr * classifier_lr_scale,
            'name': 'classifier'
        })
    
    print("\n[Optimizer] Differential learning rates:")
    for g in groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  {g['name']}: {n_params:,} params @ lr={g['lr']:.2e}")
    
    return groups


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, binary_mode=False):
    """Train model for one epoch with metadata."""
    
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    gate_values = []

    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        # Move inputs to device
        inputs = batch["input_values"].to(device)
        labels = batch["label"].to(device)
        device_id = batch["device_id"].to(device)
        site_id = batch["site_id"].to(device)
        m_rest = batch["m_rest"].to(device)
        
        # Handle label shape
        if binary_mode:
            labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
        else:
            labels = labels.float()

        optimizer.zero_grad()
        
        with torch.amp.autocast(device.type):
            logits = model(
                inputs, 
                device_id=device_id, 
                site_id=site_id, 
                m_rest=m_rest
            )
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # Track gate value
        if hasattr(model, 'get_gate_value'):
            gate_values.append(model.get_gate_value())

    avg_loss = total_loss / n_samples
    avg_gate = np.mean(gate_values) if gate_values else 0.0
    
    if binary_mode:
        metrics = compute_binary_metrics(all_labels, all_preds, all_probs, verbose=False)
    else:
        metrics = compute_multilabel_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
        )
    
    metrics['gate'] = avg_gate
    
    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device, thresholds=None, 
             tune_thresholds=False, verbose=True, binary_mode=False):
    """Evaluate model with metadata."""
    
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]", leave=False):
            inputs = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            device_id = batch["device_id"].to(device)
            site_id = batch["site_id"].to(device)
            m_rest = batch["m_rest"].to(device)
            
            if binary_mode:
                labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
            else:
                labels = labels.float()

            with torch.amp.autocast(device.type):
                logits = model(inputs, device_id=device_id, site_id=site_id, m_rest=m_rest)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Default thresholds
    if thresholds is None:
        thresholds = 0.5 if binary_mode else (0.5, 0.5)

    # Tune thresholds
    if tune_thresholds:
        if binary_mode:
            best = find_best_threshold_binary(all_labels, all_probs)
            thresholds = best["threshold"]
        else:
            best = find_best_thresholds_icbhi(all_labels, all_probs)
            thresholds = (best["tC"], best["tW"])

    # Apply thresholds
    if binary_mode:
        all_preds = (all_probs >= thresholds).astype(int)
    else:
        all_preds = np.stack([
            (all_probs[:, 0] >= thresholds[0]).astype(int),
            (all_probs[:, 1] >= thresholds[1]).astype(int)
        ], axis=1)

    avg_loss = total_loss / n_samples
    
    if binary_mode:
        metrics = compute_binary_metrics(all_labels, all_preds, all_probs, verbose=verbose)
        metrics["threshold"] = thresholds
    else:
        metrics = compute_multilabel_metrics(all_labels, all_preds, all_probs, verbose=verbose)
        metrics["threshold_crackle"] = thresholds[0]
        metrics["threshold_wheeze"] = thresholds[1]

    return avg_loss, metrics, thresholds


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """Training loop for ASTMetaProj."""

    # Hardware
    hw_cfg = cfg.hardware
    if hasattr(hw_cfg, "visible_gpus"):
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    
    # Freezing
    freeze_cfg = getattr(cfg, 'freeze', None)
    # if freeze_cfg:
    #     configure_ast_freezing(model, freeze_cfg)
    
    model = model.to(device)
    print(f"Model moved to {device}")

    # Mode detection
    binary_mode = cfg.n_cls == 2 or getattr(cfg, 'binary_mode', False)
    print(f"\n[Mode] {'Binary (Normal vs Abnormal)' if binary_mode else 'Multi-label (Crackle, Wheeze)'}")

    # Loss function
    sensitivity_bias = getattr(cfg, 'sensitivity_bias', 1.5)
    loss_type = str(cfg.loss)
    print(f"[Loss] Type: {loss_type}, Sensitivity bias: {sensitivity_bias}")
    
    if binary_mode:
        pos_weight = compute_pos_weight(train_loader, device, sensitivity_bias, binary_mode=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        class_counts = compute_class_counts(train_loader, binary_mode=False)
        print(f"[Loss] Class counts: N={class_counts[0]}, C={class_counts[1]}, W={class_counts[2]}, B={class_counts[3]}")
        
        if loss_type == 'bce':
            pos_weight = compute_pos_weight(train_loader, device, sensitivity_bias, binary_mode=False)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == 'hierarchical':
            criterion = HierarchicalMultiLabelLoss(
                class_counts=class_counts,
                sensitivity_bias=sensitivity_bias,
                partial_credit=getattr(cfg, 'partial_credit', 0.5),
                miss_penalty=getattr(cfg, 'miss_penalty', 2.0),
            )
        elif loss_type == 'composite':
            criterion = CompositeClassLoss(
                class_counts=class_counts,
                sensitivity_bias=sensitivity_bias,
                partial_credit=getattr(cfg, 'partial_credit', 0.5),
                over_pred_cost=getattr(cfg, 'over_pred_cost', 0.3),
            )
        else:
            pos_weight = compute_pos_weight(train_loader, device, sensitivity_bias, binary_mode=False)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    criterion = criterion.to(device)

    # Optimizer
    epochs = cfg.epochs
    lr = float(cfg.optimizer.lr)
    wd = float(cfg.optimizer.weight_decay)
    final_wd = float(getattr(cfg.optimizer, 'final_weight_decay', wd))
    
    # Get parameter groups with differential LRs
    use_differential_lr = getattr(cfg.optimizer, 'use_differential_lr', True)
    
    if use_differential_lr:
        param_groups = get_meta_proj_parameter_groups(model, lr, cfg)
        optimizer = optim.AdamW(param_groups, weight_decay=wd)
    else:
        # Fallback to standard optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        print(f"[Optimizer] Standard AdamW, lr={lr}, wd={wd}")
    
    scheduler = build_scheduler(cfg.scheduler, epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    
    print(f"[Optimizer] AdamW, lr={lr}, wd={wd}")

    def _cosine_wd(epoch):
        return final_wd + 0.5 * (wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0
    best_thresholds = 0.5 if binary_mode else (0.5, 0.5)

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, binary_mode
        )

        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        mlflow.log_metric(f"{prefix}_gate", train_metrics.get('gate', 0), step=epoch)
        
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] Loss={train_loss:.4f} | "
              f"Se={train_metrics['sensitivity']:.4f} | Sp={train_metrics['specificity']:.4f} | "
              f"ICBHI={train_metrics['icbhi_score']:.4f} | Gate={train_metrics.get('gate', 0):.4f}")

        # Validation
        if val_loader:
            val_loss, val_metrics, val_thresholds = evaluate(
                model, val_loader, criterion, device,
                tune_thresholds=True, verbose=False, binary_mode=binary_mode
            )

            prefix = "Val" if fold_idx is None else f"Val_Fold{fold_idx}"
            mlflow.log_metric(f"{prefix}_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

            if binary_mode:
                thresh_str = f"t={val_thresholds:.3f}"
            else:
                thresh_str = f"tC={val_thresholds[0]:.3f}, tW={val_thresholds[1]:.3f}"

            print(f"[{prefix}][Epoch {epoch}] Loss={val_loss:.4f} | "
                  f"Se={val_metrics['sensitivity']:.4f} | Sp={val_metrics['specificity']:.4f} | "
                  f"ICBHI={val_metrics['icbhi_score']:.4f} | {thresh_str}")

            icbhi = val_metrics["icbhi_score"]
            if icbhi > best_icbhi:
                best_icbhi = icbhi
                best_state_dict = model.state_dict()
                best_epoch = epoch
                best_thresholds = val_thresholds

                ckpt_path = f"checkpoints/ast_meta_{epoch}_ICBHI={icbhi:.4f}_fold{fold_idx or 0}.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'model_state_dict': best_state_dict,
                    'thresholds': best_thresholds,
                    'epoch': best_epoch,
                    'icbhi_score': best_icbhi,
                    'gate': model.get_gate_value(),
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                print(f"★ New best (Epoch {epoch}, ICBHI={icbhi*100:.2f}%, Gate={model.get_gate_value():.4f})")

        # Scheduler
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                scheduler.step(val_metrics.get("icbhi_score", val_loss))
            else:
                scheduler.step()
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

        # Weight decay schedule
        if getattr(cfg.optimizer, 'cosine_weight_decay', False):
            new_wd = _cosine_wd(epoch)
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd

    # Load best model
    if best_state_dict:
        model.load_state_dict(best_state_dict)
        print(f"\n{'='*60}")
        print(f"Loaded best model from Epoch {best_epoch}, ICBHI={best_icbhi*100:.2f}%")
        print(f"{'='*60}\n")

    # Final test
    if test_loader:
        print(f"[Test] Evaluating with thresholds from validation...")
        test_loss, test_metrics, _ = evaluate(
            model, test_loader, criterion, device,
            thresholds=best_thresholds, tune_thresholds=False,
            verbose=True, binary_mode=binary_mode
        )

        prefix = "Test" if fold_idx is None else f"Test_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", test_loss)
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v)

        print(f"\n[{prefix}] Final: Se={test_metrics['sensitivity']*100:.2f}% | "
              f"Sp={test_metrics['specificity']*100:.2f}% | "
              f"ICBHI={test_metrics['icbhi_score']*100:.2f}%")

    return model, criterion


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = load_config("configs/ast_fus_config.yaml")
    mlflow_cfg = load_config("configs/mlflow.yaml")

    MODEL_KEY = "ast_meta_proj"
    print(f"\n{'='*70}")
    print(f"Training: {MODEL_KEY}")
    print(f"{'='*70}")

    set_seed(cfg.seed)

    # Build dataloaders
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    # Build model
    # model_cfg = cfg.models.get(MODEL_KEY, cfg.models.ast_meta_proj)
    model = build_model(cfg.models, MODEL_KEY, num_devices=4, num_sites=7, rest_dim=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total: {total_params:,} | Trainable: {trainable_params:,}")

    # MLflow setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))

    # Build run name from ablation config
    model_cfg = cfg[MODEL_KEY]
    ablation_str = ""
    if model_cfg.get("use_device", True):
        ablation_str += "D"
    if model_cfg.get("use_site", True):
        ablation_str += "S"
    if model_cfg.get("use_continuous", True):
        ablation_str += "C"
    if not ablation_str:
        ablation_str = "none"
    
    binary_str = "bin" if cfg.training.n_cls == 2 else "ml"
    run_name = f"{MODEL_KEY}_{cfg.training.epochs}ep_{cfg.training.loss}_{binary_str}_meta{ablation_str}"

    with mlflow.start_run(run_name=run_name):
        # Log config
        log_all_params(cfg)
        mlflow.log_param("metadata_use_device", model_cfg.get("use_device", True))
        mlflow.log_param("metadata_use_site", model_cfg.get("use_site", True))
        mlflow.log_param("metadata_use_continuous", model_cfg.get("use_continuous", True))
        mlflow.log_param("metadata_use_missing_flags", model_cfg.get("use_missing_flags", False))

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
    main()
