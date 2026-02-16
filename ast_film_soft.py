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

# Import shared utilities
from utils import (
    compute_multilabel_metrics,
    find_best_thresholds_icbhi,
    compute_pos_weight,
    compute_class_counts,
    compute_binary_metrics,
    find_best_threshold_binary,
    set_visible_gpus,
    CompositeClassLoss,
    HierarchicalMultiLabelLoss,
)
from ls.models.builder import build_model

# Import temperature scheduler from model module
from ls.models.ast_filmpp_soft import MaskTemperatureScheduler
import argparse


def add_freeze_methods(model):
    """Add freeze_backbone and unfreeze_all methods if not present."""

    if hasattr(model, 'freeze_backbone'):
        return

    def freeze_backbone(self, until=None, freeze_film=False):
        """Freeze AST backbone layers."""
        # Freeze patch embedding
        for p in self.ast.v.patch_embed.parameters():
            p.requires_grad = False

        # Freeze positional embeddings
        self.ast.v.pos_embed.requires_grad = False
        self.ast.v.cls_token.requires_grad = False
        self.ast.v.dist_token.requires_grad = False

        # Freeze transformer blocks
        for i, blk in enumerate(self.ast.v.blocks):
            if until is None or i <= until:
                for p in blk.parameters():
                    p.requires_grad = False

        # Optionally freeze FiLM++ components
        if freeze_film:
            for p in self.dev_emb.parameters():
                p.requires_grad = False
            for p in self.site_emb.parameters():
                p.requires_grad = False
            for p in self.dev_encoder.parameters():
                p.requires_grad = False
            for p in self.site_encoder.parameters():
                p.requires_grad = False
            for p in self.rest_encoder.parameters():
                p.requires_grad = False
            for p in self.dev_generators.parameters():
                p.requires_grad = False
            for p in self.site_generators.parameters():
                p.requires_grad = False
            for p in self.rest_generators.parameters():
                p.requires_grad = False

            # Freeze masks
            if self.per_layer_masks:
                for l in self.conditioned_layers:
                    self.mask_dev[str(l)].requires_grad = False
                    self.mask_site[str(l)].requires_grad = False
                    self.mask_rest[str(l)].requires_grad = False
            else:
                self.mask_dev.requires_grad = False
                self.mask_site.requires_grad = False
                self.mask_rest.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    import types
    model.freeze_backbone = types.MethodType(freeze_backbone, model)
    model.unfreeze_all = types.MethodType(unfreeze_all, model)


def configure_ast_freezing(model, freeze_cfg):
    """Configure AST layer freezing."""

    if freeze_cfg is None:
        print("[Freeze] No freeze config - training all parameters")
        return

    add_freeze_methods(model)

    strategy = freeze_cfg.get('strategy', 'none')

    if hasattr(model, 'ast'):
        ast_backbone = model.ast
    else:
        ast_backbone = model

    if strategy == 'none':
        model.unfreeze_all()
        print("[Freeze] Strategy: none - all parameters trainable")

    elif strategy == 'all':
        freeze_film = freeze_cfg.get('freeze_film', False)
        model.freeze_backbone(until=None, freeze_film=freeze_film)
        print(f"[Freeze] Strategy: all - backbone frozen, FiLM++ frozen: {freeze_film}")

    elif strategy == 'until_block':
        until_block = freeze_cfg.get('until_block', 9)
        freeze_film = freeze_cfg.get('freeze_film', False)
        model.freeze_backbone(until=until_block, freeze_film=freeze_film)
        print(f"[Freeze] Strategy: until_block={until_block}")

    elif strategy == 'trainable_blocks':
        num_blocks = len(ast_backbone.v.blocks)
        trainable_blocks = freeze_cfg.get('trainable_blocks', 2)
        until_block = num_blocks - trainable_blocks - 1
        freeze_film = freeze_cfg.get('freeze_film', False)

        if until_block >= 0:
            model.freeze_backbone(until=until_block, freeze_film=freeze_film)
            print(f"[Freeze] Strategy: trainable_blocks={trainable_blocks} - "
                  f"blocks 0-{until_block} frozen, {until_block+1}-{num_blocks-1} trainable")
        else:
            model.unfreeze_all()
            print(f"[Freeze] Strategy: trainable_blocks={trainable_blocks} - all blocks trainable")
    else:
        print(f"[Freeze] Unknown strategy '{strategy}' - training all")
        model.unfreeze_all()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"[Freeze] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%) | Frozen: {frozen:,}")


def get_filmpp_parameter_groups(model, base_lr, cfg):
    """
    Create parameter groups with differential learning rates for FiLM++Soft.

    v2 change: mask_lr_scale default is 10.0 (was 1.5) — masks need much
    higher LR to escape sigmoid saturation from initialization.
    """
    backbone_lr_scale = getattr(cfg.optimizer, 'backbone_lr_scale', 0.01)
    embedding_lr_scale = getattr(cfg.optimizer, 'embedding_lr_scale', 0.5)
    encoder_lr_scale = getattr(cfg.optimizer, 'encoder_lr_scale', 1.0)
    film_lr_scale = getattr(cfg.optimizer, 'film_lr_scale', 1.0)
    mask_lr_scale = getattr(cfg.optimizer, 'mask_lr_scale', 10.0)
    classifier_lr_scale = getattr(cfg.optimizer, 'classifier_lr_scale', 2.0)

    ast_backbone_params = []
    embedding_params = []
    encoder_params = []
    film_params = []
    mask_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'classifier' in name:
            classifier_params.append(param)
        elif 'mask_' in name:
            mask_params.append(param)
        elif '_generators' in name:
            film_params.append(param)
        elif '_encoder' in name:
            encoder_params.append(param)
        elif 'dev_emb' in name or 'site_emb' in name:
            embedding_params.append(param)
        elif 'ast' in name:
            ast_backbone_params.append(param)
        else:
            encoder_params.append(param)

    groups = []

    if ast_backbone_params:
        groups.append({
            'params': ast_backbone_params,
            'lr': base_lr * backbone_lr_scale,
            'name': 'ast_backbone',
        })

    if embedding_params:
        groups.append({
            'params': embedding_params,
            'lr': base_lr * embedding_lr_scale,
            'name': 'embeddings',
        })

    if encoder_params:
        groups.append({
            'params': encoder_params,
            'lr': base_lr * encoder_lr_scale,
            'name': 'encoders',
        })

    if film_params:
        groups.append({
            'params': film_params,
            'lr': base_lr * film_lr_scale,
            'name': 'film_generators',
        })

    if mask_params:
        groups.append({
            'params': mask_params,
            'lr': base_lr * mask_lr_scale,
            'weight_decay': 0.0,  
            'name': 'masks',
        })

    if classifier_params:
        groups.append({
            'params': classifier_params,
            'lr': base_lr * classifier_lr_scale,
            'name': 'classifier',
        })

    print("\n[Optimizer] Differential learning rates:")
    for g in groups:
        n_params = sum(p.numel() for p in g['params'])
        wd_str = f", wd={g['weight_decay']}" if 'weight_decay' in g else ""
        print(f"  {g['name']}: {n_params:,} params @ lr={g['lr']:.2e}{wd_str}")

    return groups


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch,
                    binary_mode=False):
    """
    Train FiLM++Soft for one epoch.

    v2 change: uses model.mask_regularization_loss() which combines overlap +
    coverage losses with internally stored lambdas, instead of manual
    mask_lambda * mask_overlap_loss().
    """
    model.train()
    total_loss, total_task_loss, total_mask_loss = 0.0, 0.0, 0.0
    n_samples = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc=f"[Train][Epoch {epoch}]", leave=False):
        inputs = batch["input_values"].to(device)
        labels = batch["label"].to(device)
        device_id = batch["device_id"].to(device)
        site_id = batch["site_id"].to(device)
        m_rest = batch["m_rest"].to(device)

        if binary_mode:
            labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
        else:
            labels = labels.float()

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = model(inputs, device_id, site_id, m_rest)
            task_loss = criterion(logits, labels)

            # v2: combined mask regularization (overlap + coverage)
            # lambdas are stored inside the model (mask_sparsity_lambda, mask_coverage_lambda)
            mask_loss = model.mask_regularization_loss()
            loss = task_loss + mask_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_task_loss += task_loss.item() * batch_size
        total_mask_loss += mask_loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / n_samples
    avg_task_loss = total_task_loss / n_samples
    avg_mask_loss = total_mask_loss / n_samples

    if binary_mode:
        metrics = compute_binary_metrics(all_labels, all_preds, all_probs, verbose=False)
    else:
        metrics = compute_multilabel_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
        )

    metrics['task_loss'] = avg_task_loss
    metrics['mask_loss'] = avg_mask_loss

    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device, thresholds=None,
             tune_thresholds=False, verbose=True, binary_mode=False):
    """Evaluate FiLM++Soft model."""

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
                logits = model(inputs, device_id, site_id, m_rest)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if thresholds is None:
        thresholds = 0.5 if binary_mode else (0.5, 0.5)

    if tune_thresholds:
        if binary_mode:
            best = find_best_threshold_binary(all_labels, all_probs)
            thresholds = best["threshold"]
        else:
            best = find_best_thresholds_icbhi(all_labels, all_probs)
            thresholds = (best["tC"], best["tW"])

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


def log_mask_stats(model, epoch):
    """Log mask statistics to MLflow."""

    if hasattr(model, 'get_mask_stats'):
        stats = model.get_mask_stats()

        for key, layer_stats in stats.items():
            prefix = f"mask_{key}"
            for stat_name, value in layer_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}_{stat_name}", value, step=epoch)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_loop(cfg, model, train_loader, val_loader=None, test_loader=None, fold_idx=None):
    """
    Training loop for ASTFiLMPlusPlusSoft v2.
    """

    # ---- Hardware ----
    hw_cfg = cfg.hardware
    if hasattr(hw_cfg, "visible_gpus"):
        set_visible_gpus(hw_cfg.visible_gpus, verbose=True)
    device_id = getattr(hw_cfg, "device_id", 0)
    device = get_device(device_id=device_id, verbose=True)

    # ---- Freezing ----
    freeze_cfg = getattr(cfg, 'freeze', None)
    if freeze_cfg:
        configure_ast_freezing(model, freeze_cfg)

    model = model.to(device)
    print(f"Model moved to {device}")

    # ---- Mode ----
    binary_mode = cfg.n_cls == 2 or getattr(cfg, 'binary_mode', False)
    print(f"\n[Mode] {'Binary (Normal vs Abnormal)' if binary_mode else 'Multi-label (Crackle, Wheeze)'}")

    # ---- Loss ----
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

    # ---- Print mask config ----
    print(f"\n[FiLM++ v2] Mask config:")
    print(f"  mask_sparsity_lambda (overlap):  {model.mask_sparsity_lambda}")
    print(f"  mask_coverage_lambda (coverage): {model.mask_coverage_lambda}")
    print(f"  mask_temperature (initial):      {model.mask_temperature.item():.2f}")
    print(f"  per_layer_masks:                 {model.per_layer_masks}")

    # ---- Optimizer with differential LRs ----
    epochs = cfg.epochs
    lr = float(cfg.optimizer.lr)
    wd = float(cfg.optimizer.weight_decay)
    final_wd = float(getattr(cfg.optimizer, 'final_weight_decay', wd))

    use_differential_lr = getattr(cfg.optimizer, 'use_differential_lr', True)

    if use_differential_lr:
        param_groups = get_filmpp_parameter_groups(model, lr, cfg)
        optimizer = optim.AdamW(param_groups, weight_decay=wd)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        print(f"[Optimizer] Standard AdamW, lr={lr}, wd={wd}")

    scheduler = build_scheduler(cfg.scheduler, epochs, optimizer)
    scaler = torch.amp.GradScaler(device.type)

    print(f"[Scheduler] {cfg.scheduler.type}")

    # Temperature annealing scheduler
    tau_start = getattr(cfg, 'mask_temperature', 1.0)
    tau_end = getattr(cfg, 'mask_temperature_end', 0.2)
    tau_warmup = getattr(cfg, 'mask_temperature_warmup', 5)

    temp_scheduler = MaskTemperatureScheduler(
        model,
        tau_start=tau_start,
        tau_end=tau_end,
        total_epochs=epochs,
        warmup_epochs=tau_warmup,
    )
    print(f"[Temperature] τ: {tau_start} → {tau_end} (warmup={tau_warmup} epochs, cosine annealing)")

    def _cosine_wd(epoch):
        return final_wd + 0.5 * (wd - final_wd) * (1 + math.cos(math.pi * epoch / epochs))

    best_icbhi, best_state_dict, best_epoch = -np.inf, None, 0
    best_thresholds = 0.5 if binary_mode else (0.5, 0.5)

    # Mask history tracking
    mask_history = []
    mask_snapshot_interval =getattr(cfg, 'mask_snapshot_interval', 5)
    figures_dir = getattr(cfg, 'figures_dir', 'figures')
    # os.makedirs(figures_dir, exist_ok=True)

    # ==================================================================
    # TRAINING LOOP
    # ==================================================================
    for epoch in range(1, epochs + 1):

        # Step temperature scheduler
        tau = temp_scheduler.step(epoch - 1)  # 0-indexed internally

        # ---- Train ----
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch,
            binary_mode=binary_mode,
        )

        prefix = "Train" if fold_idx is None else f"Train_Fold{fold_idx}"
        mlflow.log_metric(f"{prefix}_loss", train_loss, step=epoch)
        mlflow.log_metric(f"{prefix}_task_loss", train_metrics['task_loss'], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_loss", train_metrics['mask_loss'], step=epoch)
        mlflow.log_metric("mask_temperature", tau, step=epoch)

        for k, v in train_metrics.items():
            if isinstance(v, (int, float)) and k not in ['task_loss', 'mask_loss']:
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

        print(f"[{prefix}][Epoch {epoch}] Loss={train_loss:.4f} "
              f"(task={train_metrics['task_loss']:.4f}, mask={train_metrics['mask_loss']:.4f}) | "
              f"Se={train_metrics['sensitivity']:.4f} | Sp={train_metrics['specificity']:.4f} | "
              f"ICBHI={train_metrics['icbhi_score']:.4f}")

        # ---- Log mask stats ----
        log_mask_stats(model, epoch)

        # Print mask summary
        mask_stats = model.get_mask_stats()
        for key, s in mask_stats.items():
            print(f"  [Masks][{key}] τ={s['temperature']:.3f} | "
                  f"active: dev={s['dev_active']} site={s['site_active']} rest={s['rest_active']} | "
                  f"dominant: dev={s['dev_dominant']} site={s['site_dominant']} rest={s['rest_dominant']} | "
                  f"overlap: d-s={s['overlap_dev_site']:.4f} d-r={s['overlap_dev_rest']:.4f} s-r={s['overlap_site_rest']:.4f}")

        # ---- Validation ----
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

                ckpt_path = f"checkpoints/ast_filmpp_soft_{epoch}_ICBHI={icbhi:.4f}_fold{fold_idx or 0}.pt"
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

                current_mask_stats = model.get_mask_stats() if hasattr(model, 'get_mask_stats') else {}

                torch.save({
                    'model_state_dict': best_state_dict,
                    'thresholds': best_thresholds,
                    'epoch': best_epoch,
                    'icbhi_score': best_icbhi,
                    'conditioned_layers': list(model.conditioned_layers),
                    'mask_stats': current_mask_stats,
                    'mask_temperature': tau,
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                print(f"New best (Epoch {epoch}, ICBHI={icbhi*100:.2f}%)")

        # ---- LR Scheduler ----
        if scheduler:
            if cfg.scheduler.type == "reduce_on_plateau" and val_loader:
                scheduler.step(val_metrics.get("icbhi_score", val_loss))
            else:
                scheduler.step()

            for i, g in enumerate(optimizer.param_groups):
                group_name = g.get('name', f'group_{i}')
                mlflow.log_metric(f"lr_{group_name}", g['lr'], step=epoch)

        # ---- Weight decay schedule ----
        if getattr(cfg.optimizer, 'cosine_weight_decay', False):
            new_wd = _cosine_wd(epoch)
            for g in optimizer.param_groups:
                if g.get('name') != 'masks':  # Don't apply WD to masks
                    g["weight_decay"] = new_wd

    if best_state_dict:
        model.load_state_dict(best_state_dict)
        print(f"\n{'='*60}")
        print(f"Loaded best model from Epoch {best_epoch}, ICBHI={best_icbhi*100:.2f}%")
        print(f"{'='*60}\n")

    # ---- Test evaluation ----
    if test_loader:
        print(f"\n[Test] Evaluating with thresholds from validation...")
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

        # Print final mask stats
        if hasattr(model, 'get_mask_stats'):
            print("\n[FiLM++ v2] Final mask statistics:")
            final_stats = model.get_mask_stats()
            for key, stats in final_stats.items():
                print(f"  {key}:")
                print(f"    Active dims  - dev: {stats['dev_active']}, site: {stats['site_active']}, rest: {stats['rest_active']}")
                print(f"    Dominant dims- dev: {stats['dev_dominant']}, site: {stats['site_dominant']}, rest: {stats['rest_dominant']}")
                print(f"    Overlap      - dev-site: {stats['overlap_dev_site']:.4f}, "
                      f"dev-rest: {stats['overlap_dev_rest']:.4f}, site-rest: {stats['overlap_site_rest']:.4f}")

    return model, criterion


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--mlflow-config", type=str, required=True, help="Path to MLflow config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow_cfg = load_config(args.mlflow_config)

    MODEL_KEY = "ast_film_soft"
    print(f"\n{'='*70}")
    print(f"Training: {MODEL_KEY} (FiLM++ with Soft Learned Factorization v2)")
    print(f"{'='*70}")

    set_seed(cfg.seed)

    # Build dataloaders
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    # Build model
    model_cfg = cfg.models
    model = build_model(model_cfg, MODEL_KEY, num_devices=4, num_sites=7, rest_dim=3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"[FiLM++ v2] Conditioned layers: {model.conditioned_layers}")
    print(f"[FiLM++ v2] Per-layer masks:    {model.per_layer_masks}")
    print(f"[FiLM++ v2] Sparsity λ:         {model.mask_sparsity_lambda}")
    print(f"[FiLM++ v2] Coverage λ:         {model.mask_coverage_lambda}")
    print(f"[FiLM++ v2] Temperature:        {model.mask_temperature.item():.2f}")
    print(f"[FiLM++ v2] Film init gain:     {model.film_init_gain}")

    # MLflow setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg.tracking_password
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(mlflow_cfg.experiment_name))

    # Build run name
    layers_str = "-".join(map(str, model.conditioned_layers))
    binary_str = "bin" if cfg.training.n_cls == 2 else "ml"
    per_layer_str = "perL" if model.per_layer_masks else "shared"
    run_name = f"{MODEL_KEY}_v2_{cfg.training.epochs}ep_{cfg.training.loss}_{binary_str}_L{layers_str}_{per_layer_str}"

    with mlflow.start_run(run_name=run_name):
        # Log config
        log_all_params(cfg)
        mlflow.log_param("conditioned_layers", str(model.conditioned_layers))
        mlflow.log_param("per_layer_masks", model.per_layer_masks)
        mlflow.log_param("mask_sparsity_lambda", model.mask_sparsity_lambda)
        mlflow.log_param("mask_coverage_lambda", model.mask_coverage_lambda)
        mlflow.log_param("mask_temperature_start", getattr(cfg.training, 'mask_temperature', 1.0))
        mlflow.log_param("mask_temperature_end", getattr(cfg.training, 'mask_temperature_end', 0.2))
        mlflow.log_param("mask_temperature_warmup", getattr(cfg.training, 'mask_temperature_warmup', 5))
        mlflow.log_param("film_init_gain", model.film_init_gain)
        mlflow.log_param("mask_lr_scale", getattr(cfg.training.optimizer, 'mask_lr_scale', 10.0))

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