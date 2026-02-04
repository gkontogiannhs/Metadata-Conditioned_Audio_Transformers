"""
filmpp_train.py

Dedicated training script for ASTFiLMPlusPlusSoft model.
"""

import os
import argparse
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ls.models.ast_filmpp_soft import ASTFiLMPlusPlusSoft
from ls.data.dataloaders import build_dataloaders
from ls.config.dataclasses import DatasetConfig, AudioConfig, AugmentationConfig
from ls.engine.eval import compute_multilabel_metrics


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train ASTFiLMPlusPlusSoft")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weighted_sampler", action="store_true")
    
    # Model
    parser.add_argument("--ast_pretrained", type=str, default=None)
    parser.add_argument("--conditioned_layers", type=int, nargs="+", default=[10, 11])
    parser.add_argument("--dev_emb_dim", type=int, default=4)
    parser.add_argument("--site_emb_dim", type=int, default=7)
    parser.add_argument("--metadata_hidden_dim", type=int, default=32)
    parser.add_argument("--film_hidden_dim", type=int, default=32)
    parser.add_argument("--mask_init_scale", type=float, default=2.0)
    parser.add_argument("--mask_sparsity_lambda", type=float, default=0.01)
    parser.add_argument("--per_layer_masks", action="store_true")
    
    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_until_block", type=int, default=None)
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="outputs/filmpp")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--debug_film", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


# ============================================================================
# Config Building
# ============================================================================

def build_configs(args):
    """Build dataset and audio configs from args."""
    
    metadata_path = args.metadata_csv or os.path.join(args.data_root, "icbhi_metadata.csv")
    
    dataset_cfg = DatasetConfig(
        name="icbhi",
        data_folder=args.data_root,
        cycle_metadata_path=metadata_path,
        class_split="lungsound",
        split_strategy="official",
        test_fold=0,
        multi_label=True,
        n_cls=4,
        weighted_sampler=args.weighted_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        h=128,
        w=1024,
    )
    
    audio_cfg = AudioConfig(
        sample_rate=16000,
        desired_length=10.0,
        remove_dc=True,
        normalize=False,
        pad_type="repeat",
        use_fade=True,
        fade_samples_ratio=64,
        n_mels=128,
        frame_length=40,
        frame_shift=10,
        low_freq=100,
        high_freq=8000,
        window_type="hanning",
        use_energy=False,
        dither=0.0,
        mel_norm="mit",
        resz=1.0,
        raw_augment=0,
        # wave_aug=[
        #     AugmentationConfig(type="Crop", p=0.0, sampling_rate=16000, zone=[0.0, 1.0], coverage=1.0),
        #     AugmentationConfig(type="Noise", p=0.1, color="white"),
        #     AugmentationConfig(type="Speed", p=0.1, factor=[0.9, 1.1]),
        #     AugmentationConfig(type="Loudness", p=0.1, factor=[0.5, 2.0]),
        #     AugmentationConfig(type="VTLP", p=0.1, sampling_rate=16000, zone=[0.0, 1.0], fhi=4800, factor=[0.9, 1.1]),
        #     AugmentationConfig(type="Pitch", p=0.0, sampling_rate=16000, factor=[-1, 3]),
        # ],
        # spec_aug=[
        #     AugmentationConfig(type="SpecAugment", p=0.3, policy="icbhi_ast_sup", mask="zero"),
        # ],
    )
    
    return dataset_cfg, audio_cfg


# ============================================================================
# Model Building
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, train_dataset):
    """Construct ASTFiLMPlusPlusSoft model."""
    
    num_devices = train_dataset.num_devices
    num_sites = train_dataset.num_sites
    rest_dim = train_dataset.rest_dim
    
    print(f"\n[Model] Metadata dimensions:")
    print(f"  Devices: {num_devices}")
    print(f"  Sites: {num_sites}")
    print(f"  Continuous features: {rest_dim}")
    
    ast_kwargs = {
        "fstride": 10,
        "tstride": 10,
        "input_fdim": 128,
        "input_tdim": 1024,
        "imagenet_pretrain": True,
        "audioset_pretrain": True,
        "model_size": "base384",
    }
    
    if args.ast_pretrained:
        ast_kwargs["audioset_ckpt_path"] = args.ast_pretrained
    
    model = ASTFiLMPlusPlusSoft(
        ast_kwargs=ast_kwargs,
        num_devices=num_devices,
        num_sites=num_sites,
        rest_dim=rest_dim,
        conditioned_layers=tuple(args.conditioned_layers),
        dev_emb_dim=args.dev_emb_dim,
        site_emb_dim=args.site_emb_dim,
        metadata_hidden_dim=args.metadata_hidden_dim,
        film_hidden_dim=args.film_hidden_dim,
        dropout_p=args.dropout,
        mask_init_scale=args.mask_init_scale,
        mask_sparsity_lambda=args.mask_sparsity_lambda,
        per_layer_masks=args.per_layer_masks,
        num_labels=2,
        debug_film=args.debug_film,
    )
    
    if args.freeze_backbone:
        freeze_ast_backbone(model, until_block=args.freeze_until_block)
    
    return model


def freeze_ast_backbone(model, until_block=None):
    """Freeze AST backbone, keep FiLM components trainable."""
    for p in model.ast.v.patch_embed.parameters():
        p.requires_grad = False
    
    model.ast.v.pos_embed.requires_grad = False
    model.ast.v.cls_token.requires_grad = False
    model.ast.v.dist_token.requires_grad = False
    
    for i, blk in enumerate(model.ast.v.blocks):
        if until_block is None or i <= until_block:
            for p in blk.parameters():
                p.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ============================================================================
# Training Functions
# ============================================================================

def forward_batch(model, batch, device):
    """Unpack batch and forward through model."""
    inputs = batch["input_values"].to(device)
    device_id = batch["device_id"].to(device)
    site_id = batch["site_id"].to(device)
    m_rest = batch["m_rest"].to(device)
    
    logits = model(inputs, device_id, site_id, m_rest)
    return logits


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    stats = {"loss": 0.0, "ce_loss": 0.0, "overlap_loss": 0.0, "n_samples": 0}
    all_preds, all_labels, all_probs = [], [], []
    group_preds = defaultdict(list)
    group_labels = defaultdict(list)
    group_probs = defaultdict(list)

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input_values"].to(device)
        labels = batch["label"].to(device)
        device_names = batch.get("device_name", ["unk"] * len(labels))
        site_names = batch.get("site_name", ["unk"] * len(labels))

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = forward_batch(model, batch, device)
            ce_loss = criterion(logits, labels)
            
            if model.mask_sparsity_lambda > 0:
                overlap_loss = model.mask_overlap_loss()
                loss = ce_loss + model.mask_sparsity_lambda * overlap_loss
            else:
                overlap_loss = torch.tensor(0.0, device=device)
                loss = ce_loss

        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        bs = inputs.size(0)
        stats["loss"] += loss.item() * bs
        stats["ce_loss"] += ce_loss.item() * bs
        stats["overlap_loss"] += overlap_loss.item() * bs
        stats["n_samples"] += bs

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

        for d, s, y_true, y_pred, y_prob in zip(device_names, site_names, labels_np, preds, probs):
            group_preds[f"dev_{d}"].append(y_pred)
            group_labels[f"dev_{d}"].append(y_true)
            group_probs[f"dev_{d}"].append(y_prob)
            group_preds[f"site_{s}"].append(y_pred)
            group_labels[f"site_{s}"].append(y_true)
            group_probs[f"site_{s}"].append(y_prob)

        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "ovlp": f"{overlap_loss.item():.4f}",
            })

    n = stats["n_samples"]
    result = {
        "loss": stats["loss"] / n,
        "ce_loss": stats["ce_loss"] / n,
        "overlap_loss": stats["overlap_loss"] / n,
        "metrics": compute_multilabel_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
        ),
        "group_metrics": {},
        "mask_stats": model.get_mask_stats(),
    }

    for group in group_labels:
        if len(group_labels[group]) >= 10:
            result["group_metrics"][group] = compute_multilabel_metrics(
                np.array(group_labels[group]),
                np.array(group_preds[group]),
                np.array(group_probs[group]),
                verbose=False,
            )

    return result


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    
    stats = {"loss": 0.0, "ce_loss": 0.0, "n_samples": 0}
    all_preds, all_labels, all_probs = [], [], []
    group_preds = defaultdict(list)
    group_labels = defaultdict(list)
    group_probs = defaultdict(list)

    for batch in tqdm(dataloader, desc=f"[Val] Epoch {epoch}", leave=False):
        inputs = batch["input_values"].to(device)
        labels = batch["label"].to(device)
        device_names = batch.get("device_name", ["unk"] * len(labels))
        site_names = batch.get("site_name", ["unk"] * len(labels))

        with torch.amp.autocast(device.type):
            logits = forward_batch(model, batch, device)
            ce_loss = criterion(logits, labels)
            
            if model.mask_sparsity_lambda > 0:
                overlap_loss = model.mask_overlap_loss()
                loss = ce_loss + model.mask_sparsity_lambda * overlap_loss
            else:
                loss = ce_loss

        bs = inputs.size(0)
        stats["loss"] += loss.item() * bs
        stats["ce_loss"] += ce_loss.item() * bs
        stats["n_samples"] += bs

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

        for d, s, y_true, y_pred, y_prob in zip(device_names, site_names, labels_np, preds, probs):
            group_preds[f"dev_{d}"].append(y_pred)
            group_labels[f"dev_{d}"].append(y_true)
            group_probs[f"dev_{d}"].append(y_prob)
            group_preds[f"site_{s}"].append(y_pred)
            group_labels[f"site_{s}"].append(y_true)
            group_probs[f"site_{s}"].append(y_prob)

    n = stats["n_samples"]
    result = {
        "loss": stats["loss"] / n,
        "ce_loss": stats["ce_loss"] / n,
        "metrics": compute_multilabel_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs), verbose=False
        ),
        "group_metrics": {},
        "mask_stats": model.get_mask_stats(),
    }

    for group in group_labels:
        if len(group_labels[group]) >= 10:
            result["group_metrics"][group] = compute_multilabel_metrics(
                np.array(group_labels[group]),
                np.array(group_preds[group]),
                np.array(group_probs[group]),
                verbose=False,
            )

    return result


# ============================================================================
# Logging & Checkpointing
# ============================================================================

def log_epoch(epoch, train_result, val_result, lr):
    """Print epoch summary."""
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}")
    print(f"{'='*70}")
    
    tm = train_result['metrics']
    vm = val_result['metrics']
    
    print(f"  LR: {lr:.2e}")
    print(f"  Train | Loss: {train_result['loss']:.4f} "
          f"(CE: {train_result['ce_loss']:.4f}, Overlap: {train_result['overlap_loss']:.4f})")
    print(f"        | ICBHI: Se={tm['sensitivity']*100:.2f}%, Sp={tm['specificity']*100:.2f}%, "
          f"Score={tm['icbhi_score']*100:.2f}%")
    print(f"        | Macro: F1={tm['macro_f1']*100:.2f}%, P={tm['macro_precision']*100:.2f}%, "
          f"R={tm['macro_sensitivity']*100:.2f}%")
    
    print(f"  Val   | Loss: {val_result['loss']:.4f} (CE: {val_result['ce_loss']:.4f})")
    print(f"        | ICBHI: Se={vm['sensitivity']*100:.2f}%, Sp={vm['specificity']*100:.2f}%, "
          f"Score={vm['icbhi_score']*100:.2f}%")
    print(f"        | Macro: F1={vm['macro_f1']*100:.2f}%, P={vm['macro_precision']*100:.2f}%, "
          f"R={vm['macro_sensitivity']*100:.2f}%")
    print(f"        | Binary: Se={vm['binary_sensitivity']*100:.2f}%, Sp={vm['binary_specificity']*100:.2f}%, "
          f"Score={vm['binary_icbhi_score']*100:.2f}%")
    
    if train_result["mask_stats"]:
        print(f"  Masks |", end="")
        for layer, stats in train_result["mask_stats"].items():
            overlap_avg = (stats['overlap_dev_site'] + stats['overlap_dev_rest'] + 
                          stats['overlap_site_rest']) / 3
            print(f" {layer}: D={stats['dev_active']}, S={stats['site_active']}, "
                  f"R={stats['rest_active']}, ovlp={overlap_avg:.3f}", end="")
        print()


def save_checkpoint(model, optimizer, scheduler, epoch, val_result, path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_metrics": val_result["metrics"],
        "mask_stats": val_result["mask_stats"],
    }, path)
    print(f"  [Checkpoint] Saved: {path}")


def save_mask_visualization(model, output_dir, epoch):
    """Save mask visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    with torch.no_grad():
        if model.per_layer_masks:
            for l in model.conditioned_layers:
                w_dev, w_site, w_rest = model._get_masks(l)
                _plot_masks(w_dev, w_site, w_rest, 
                           f"Layer {l} - Epoch {epoch}",
                           os.path.join(output_dir, f"masks_layer{l}_epoch{epoch}.png"))
        else:
            w_dev, w_site, w_rest = model._get_masks(model.conditioned_layers[0])
            _plot_masks(w_dev, w_site, w_rest,
                       f"Shared Masks - Epoch {epoch}",
                       os.path.join(output_dir, f"masks_shared_epoch{epoch}.png"))


def _plot_masks(w_dev, w_site, w_rest, title, save_path):
    """Helper to plot masks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    axes[0].plot(w_dev.cpu().numpy(), label="Device", alpha=0.8, linewidth=0.8)
    axes[0].plot(w_site.cpu().numpy(), label="Site", alpha=0.8, linewidth=0.8)
    axes[0].plot(w_rest.cpu().numpy(), label="Rest", alpha=0.8, linewidth=0.8)
    axes[0].set_xlabel("Hidden Dimension")
    axes[0].set_ylabel("Mask Weight")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)
    
    D = len(w_dev)
    dominant = torch.stack([w_dev, w_site, w_rest]).argmax(dim=0).cpu().numpy()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(D):
        axes[1].bar(i, 1, color=colors[dominant[i]], width=1.0)
    axes[1].set_xlabel("Hidden Dimension")
    axes[1].set_ylabel("Dominant Source")
    axes[1].set_yticks([])
    
    legend_elements = [Patch(facecolor=colors[0], label='Device'),
                       Patch(facecolor=colors[1], label='Site'),
                       Patch(facecolor=colors[2], label='Rest')]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Scheduler with Warmup
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-8):
    """Cosine schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'], 0.5 * (1 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Main
# ============================================================================

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


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup output directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build configs and dataloaders
    dataset_cfg, audio_cfg = build_configs(args)
    train_loader, val_loader = build_dataloaders(
        dataset_cfg, 
        audio_cfg,
        use_weighted_sampler=args.weighted_sampler,
    )
    
    # Build model
    model = build_model(args, train_loader.dataset)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss, optimizer, scheduler, scaler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        total_epochs=args.epochs,
        min_lr=1e-8,
    )
    scaler = torch.amp.GradScaler(device.type)
    
    # Training loop
    best_val_icbhi = 0.0
    SAVE_THRESHOLD = 0.61  # Only save if ICBHI score > 62%
    history = []
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        val_result = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        log_epoch(epoch, train_result, val_result, current_lr)
        
        # Save history
        tm = train_result['metrics']
        vm = val_result['metrics']
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_result["loss"],
            "train_ce_loss": train_result["ce_loss"],
            "train_overlap_loss": train_result["overlap_loss"],
            "train_icbhi_score": tm["icbhi_score"],
            "train_sensitivity": tm["sensitivity"],
            "train_specificity": tm["specificity"],
            "train_macro_f1": tm["macro_f1"],
            "val_loss": val_result["loss"],
            "val_icbhi_score": vm["icbhi_score"],
            "val_sensitivity": vm["sensitivity"],
            "val_specificity": vm["specificity"],
            "val_macro_f1": vm["macro_f1"],
            "val_binary_icbhi": vm["binary_icbhi_score"],
            "mask_stats": train_result["mask_stats"],
        })
        
        # Save best model (based on ICBHI score)
        val_icbhi = val_result["metrics"]["icbhi_score"]
        if val_icbhi > best_val_icbhi:
            best_val_icbhi = val_icbhi
            if args.save_best and val_icbhi > SAVE_THRESHOLD:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_result,
                    os.path.join(output_dir, "best_model.pt")
                )
                print(f"  [NEW BEST] ICBHI Score: {val_icbhi*100:.2f}% (saved)")
            elif val_icbhi > best_val_icbhi:
                print(f"  [NEW BEST] ICBHI Score: {val_icbhi*100:.2f}% (not saved, < {SAVE_THRESHOLD*100:.0f}%)")
        
        # Save mask visualization periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            save_mask_visualization(model, output_dir, epoch)
    
    # Save final model and history
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, val_result,
        os.path.join(output_dir, "final_model.pt")
    )
    
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Best Val ICBHI Score: {best_val_icbhi*100:.2f}%")
    print(f"  Outputs: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()