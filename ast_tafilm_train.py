import os
import mlflow
from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
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
from ls.engine.scheduler import build_scheduler
from collections import defaultdict
from ls.models.builder import build_model

# Import shared utilities
from utils import *
import argparse


def configure_ast_freezing(model, freeze_cfg):
    """Configure AST layer freezing based on config."""
    
    if freeze_cfg is None:
        print("[Freeze] No freeze config - training all parameters")
        return
    
    strategy = freeze_cfg.get('strategy', 'none')
    
    # Handle different model types
    if hasattr(model, 'ast'):
        # ASTMetaProj or ASTFiLM: AST is wrapped inside model.ast
        ast_backbone = model.ast
    else:
        # Vanilla AST
        ast_backbone = model
    
    if strategy == 'none':
        model.unfreeze_all()
        print("[Freeze] Strategy: none - all parameters trainable")
        
    elif strategy == 'all':
        freeze_film = freeze_cfg.get('freeze_film', False)
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone(until=None, freeze_film=freeze_film)
        else:
            model.freeze_backbone(until_block=None)
        print(f"[Freeze] Strategy: all - backbone frozen, FiLM frozen: {freeze_film}")
        
    elif strategy == 'until_block':
        until_block = freeze_cfg.get('until_block', 9)
        freeze_film = freeze_cfg.get('freeze_film', False)
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone(until=until_block, freeze_film=freeze_film)
        else:
            model.freeze_backbone(until_block=until_block)
        print(f"[Freeze] Strategy: until_block - blocks 0-{until_block} frozen")
        
    elif strategy == 'trainable_blocks':
        num_blocks = len(ast_backbone.v.blocks)
        trainable_blocks = freeze_cfg.get('trainable_blocks', 2)
        until_block = num_blocks - trainable_blocks - 1
        freeze_film = freeze_cfg.get('freeze_film', False)
        
        if until_block >= 0:
            if hasattr(model, 'freeze_backbone'):
                model.freeze_backbone(until=until_block, freeze_film=freeze_film)
            else:
                model.freeze_backbone(until_block=until_block)
            print(f"[Freeze] Strategy: trainable_blocks={trainable_blocks} - "
                  f"blocks 0-{until_block} frozen, {until_block+1}-{num_blocks-1} trainable")
        else:
            model.unfreeze_all()
            print(f"[Freeze] Strategy: trainable_blocks={trainable_blocks} - all blocks trainable")
    else:
        print(f"[Freeze] Unknown strategy '{strategy}' - training all")
        model.unfreeze_all()
    
    # Print parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"[Freeze] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%) | Frozen: {frozen:,}")
    
    # Print FiLM-specific parameter counts
    if hasattr(model, 'film_generators'):
        film_params = sum(p.numel() for p in model.film_generators.parameters())
        film_trainable = sum(p.numel() for p in model.film_generators.parameters() if p.requires_grad)
        print(f"[Freeze] FiLM generators: {film_params:,} total, {film_trainable:,} trainable")


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, binary_mode=False):
    """Train model for one epoch with metadata (FiLM conditioning)."""
    
    model.train()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    # Track FiLM statistics
    gamma_stats = defaultdict(list)
    beta_stats = defaultdict(list)

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

    avg_loss = total_loss / n_samples
    
    if binary_mode:
        metrics = compute_binary_metrics(all_labels, all_preds, all_probs, verbose=False)
    else:
        metrics = compute_multilabel_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
        )
    
    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device, thresholds=None,
             tune_thresholds=False, verbose=True, binary_mode=False):
    """Evaluate model with FiLM conditioning."""
    
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels = [], []
    
    # For FiLM analysis
    film_gamma_norms = defaultdict(list)
    film_beta_norms = defaultdict(list)

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


def analyze_film_parameters(model, dataloader, device, num_batches=10):
    """
    Analyze FiLM parameter distributions for interpretability.
    
    Returns statistics about γ and β across different conditions.
    """
    model.eval()
    
    # Collect FiLM parameters grouped by device and site
    film_by_device = defaultdict(lambda: defaultdict(list))
    film_by_site = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = batch["input_values"].to(device)
            device_ids = batch["device_id"].to(device)
            site_ids = batch["site_id"].to(device)
            m_rest = batch["m_rest"].to(device)
            devices = batch["device"]
            sites = batch["site"]
            
            # Get FiLM info
            _, film_info = model.forward_with_film_info(
                inputs, device_id=device_ids, site_id=site_ids, m_rest=m_rest
            )
            
            # Collect per-sample statistics
            for b_idx in range(len(devices)):
                dev = devices[b_idx]
                site = sites[b_idx]
                
                for layer_idx in film_info['gamma'].keys():
                    gamma_b = film_info['gamma'][layer_idx][b_idx].cpu().numpy()
                    beta_b = film_info['beta'][layer_idx][b_idx].cpu().numpy()
                    
                    film_by_device[dev][f'gamma_L{layer_idx}'].append(gamma_b.mean())
                    film_by_device[dev][f'beta_L{layer_idx}'].append(beta_b.mean())
                    film_by_site[site][f'gamma_L{layer_idx}'].append(gamma_b.mean())
                    film_by_site[site][f'beta_L{layer_idx}'].append(beta_b.mean())
    
    # Compute statistics
    stats = {'by_device': {}, 'by_site': {}}
    
    for dev, params in film_by_device.items():
        stats['by_device'][dev] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in params.items()
        }
    
    for site, params in film_by_site.items():
        stats['by_site'][site] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in params.items()
        }
    
    return stats


def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """Training loop for ASTFiLM."""

    # Hardware
    hw_cfg = cfg.hardware
    if hasattr(hw_cfg, "visible_gpus"):
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)
    
    # Freezing
    freeze_cfg = getattr(cfg, 'freeze', None)
    if freeze_cfg:
        configure_ast_freezing(model, freeze_cfg)
    
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
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    scheduler = build_scheduler(cfg.scheduler, epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)
    
    print(f"[Optimizer] AdamW, lr={lr}, wd={wd}")
    print(f"[Scheduler] {cfg.scheduler.type}")

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
        
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] Loss={train_loss:.4f} | "
              f"Se={train_metrics['sensitivity']:.4f} | Sp={train_metrics['specificity']:.4f} | "
              f"ICBHI={train_metrics['icbhi_score']:.4f}")

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

                ckpt_path = f"checkpoints/ast_film_{epoch}_ICBHI={icbhi:.4f}_fold{fold_idx or 0}.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'model_state_dict': best_state_dict,
                    'thresholds': best_thresholds,
                    'epoch': best_epoch,
                    'icbhi_score': best_icbhi,
                    'conditioned_layers': list(model.conditioned_layers),
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                print(f"New best (Epoch {epoch}, ICBHI={icbhi*100:.2f}%)")

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
        
        # Analyze FiLM parameters
        if hasattr(model, 'forward_with_film_info'):
            print("\n[FiLM Analysis] Analyzing conditioning parameters...")
            try:
                film_stats = analyze_film_parameters(model, test_loader, device, num_batches=10)
                
                print("\n  By Device:")
                for dev, params in film_stats['by_device'].items():
                    gamma_mean = np.mean([v['mean'] for k, v in params.items() if 'gamma' in k])
                    beta_mean = np.mean([v['mean'] for k, v in params.items() if 'beta' in k])
                    print(f"    {dev}: γ_mean={gamma_mean:.4f}, β_mean={beta_mean:.4f}")
                
                print("\n  By Site:")
                for site, params in film_stats['by_site'].items():
                    gamma_mean = np.mean([v['mean'] for k, v in params.items() if 'gamma' in k])
                    beta_mean = np.mean([v['mean'] for k, v in params.items() if 'beta' in k])
                    print(f"    {site}: γ_mean={gamma_mean:.4f}, β_mean={beta_mean:.4f}")
            except Exception as e:
                print(f"  [WARN] FiLM analysis failed: {e}")

    return model, criterion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--mlflow-config", type=str, required=True, help="Path to MLflow config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow_cfg = load_config(args.mlflow_config)


    MODEL_KEY = "tafilm"
    print(f"\n{'='*70}")
    print(f"Training: {MODEL_KEY} (TAFiLM-conditioned AST)")
    print(f"{'='*70}")

    set_seed(cfg.seed)

    # Build dataloaders
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    # Build model
    model_cfg = cfg.models # .get(MODEL_KEY, cfg.models.ast_film)
    model = build_model(model_cfg, MODEL_KEY, num_devices=4, num_sites=7, rest_dim=3)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Print conditioned layers
    print(f"[FiLM] Conditioned layers: {model.conditioned_layers}")

    # MLflow setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))

    # Build run name from config
    ablation_str = ""
    if model_cfg.get("use_device", True):
        ablation_str += "D"
    if model_cfg.get("use_site", True):
        ablation_str += "S"
    if model_cfg.get("use_continuous", True):
        ablation_str += "C"
    if not ablation_str:
        ablation_str = "none"
    
    layers_str = "-".join(map(str, model.conditioned_layers))
    binary_str = "bin" if cfg.training.n_cls == 2 else "ml"
    run_name = f"{MODEL_KEY}_{cfg.training.epochs}ep_{cfg.training.loss}_{binary_str}_L{layers_str}_meta{ablation_str}"

    with mlflow.start_run(run_name=run_name):
        # Log config
        log_all_params(cfg)
        mlflow.log_param("conditioned_layers", str(model.conditioned_layers))
        mlflow.log_param("metadata_use_device", model_cfg.get("use_device", True))
        mlflow.log_param("metadata_use_site", model_cfg.get("use_site", True))
        mlflow.log_param("metadata_use_continuous", model_cfg.get("use_continuous", True))

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