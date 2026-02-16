"""
Generate figures for SoftFiLM:
    1. Mask disentanglement analysis (6-panel) — from trained SoftFiLM model
    2. t-SNE feature space comparison: Baseline vs SoftFiLM (colored by class, device, site)
    3. FiLM parameter distributions (γ, β) by device and site

Usage:
    python visualize.py \
        --config configs/best_params_config.yaml \
        --baseline-ckpt checkpoints/ast_baseline_best.pt \
        --softfilm-ckpt checkpoints/ast_filmpp_soft_best.pt \
        --output-dir figures
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns
import torch
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from tqdm import tqdm
from collections import defaultdict

from ls.config.loader import load_config
from ls.data.dataloaders import build_dataloaders
from ls.engine.utils import get_device, set_seed
from ls.models.builder import build_model

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# Color palette — consistent across all figures
C_DEV  = '#e74c3c'
C_SITE = '#2980b9'
C_REST = '#27ae60'

C_NORMAL  = '#3498db'
C_CRACKLE = '#e74c3c'
C_WHEEZE  = '#f39c12'
C_BOTH    = '#8e44ad'

CLASS_COLORS = {'Normal': C_NORMAL, 'Crackle': C_CRACKLE, 'Wheeze': C_WHEEZE,
                'Both': C_BOTH, 'Abnormal': C_CRACKLE}

DEVICE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
DEVICE_NAMES  = ['Littmann 3200', 'Littmann 4000', 'Meditron', 'AKGC417L']

SITE_COLORS = ['#1abc9c', '#e67e22', '#9b59b6', '#e74c3c', '#3498db', '#2c3e50', '#f1c40f']
SITE_NAMES  = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']


# =============================================================================
# DATA EXTRACTION FROM TRAINED MODELS
# =============================================================================

def extract_softfilm_masks(model):
    """Extract soft masks from trained SoftFiLM model."""
    masks = {}
    with torch.no_grad():
        if model.per_layer_masks:
            for layer_idx in model.conditioned_layers:
                w_dev = torch.sigmoid(model.mask_dev[str(layer_idx)]).cpu().numpy()
                w_site = torch.sigmoid(model.mask_site[str(layer_idx)]).cpu().numpy()
                w_rest = torch.sigmoid(model.mask_rest[str(layer_idx)]).cpu().numpy()
                masks[f'layer_{layer_idx}'] = {'device': w_dev, 'site': w_site, 'rest': w_rest}
        else:
            w_dev = torch.sigmoid(model.mask_dev).cpu().numpy()
            w_site = torch.sigmoid(model.mask_site).cpu().numpy()
            w_rest = torch.sigmoid(model.mask_rest).cpu().numpy()
            masks['shared'] = {'device': w_dev, 'site': w_site, 'rest': w_rest}
    return masks


def labels_to_classes(labels):
    """Convert label arrays to class name strings."""
    classes = []
    for label in labels:
        if label.ndim == 0 or len(label) == 1:
            classes.append('Abnormal' if label else 'Normal')
        else:
            c, w = label
            if c == 0 and w == 0: classes.append('Normal')
            elif c == 1 and w == 0: classes.append('Crackle')
            elif c == 0 and w == 1: classes.append('Wheeze')
            else: classes.append('Both')
    return classes


def extract_features(model, dataloader, device, num_batches=None):
    """Extract CLS token features from model for all samples."""
    model.eval()
    all_features, all_labels, all_devices, all_sites = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if num_batches and batch_idx >= num_batches:
                break

            inputs = batch["input_values"].to(device)
            labels = batch["label"].numpy()
            devices_batch = batch["device"]
            sites_batch = batch["site"]

            if hasattr(model, 'forward_features'):
                if hasattr(model, 'condition_on_device') or hasattr(model, 'ast'):
                    device_ids = batch["device_id"].to(device)
                    site_ids = batch["site_id"].to(device)
                    m_rest = batch["m_rest"].to(device)
                    try:
                        features = model.forward_features(inputs, device_ids, site_ids, m_rest)
                    except TypeError:
                        features = model.forward_features(inputs)
                else:
                    features = model.forward_features(inputs)
            else:
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1).transpose(2, 3)
                features = model.ast.forward_features(inputs) if hasattr(model, 'ast') else model.forward_features(inputs)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            all_devices.extend(devices_batch)
            all_sites.extend(sites_batch)

    return {
        'features': np.vstack(all_features),
        'labels': np.array(all_labels),
        'devices': all_devices,
        'sites': all_sites,
    }


def collect_film_parameters(model, dataloader, device, num_batches=None):
    """Collect FiLM γ/β parameters grouped by device, site, class."""
    model.eval()
    params_by_device = defaultdict(lambda: {'gamma': [], 'beta': []})
    params_by_site = defaultdict(lambda: {'gamma': [], 'beta': []})
    params_by_class = defaultdict(lambda: {'gamma': [], 'beta': []})

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting FiLM params")):
            if num_batches and batch_idx >= num_batches:
                break

            inputs = batch["input_values"].to(device)
            device_ids = batch["device_id"].to(device)
            site_ids = batch["site_id"].to(device)
            m_rest = batch["m_rest"].to(device)
            labels = batch["label"].numpy()
            devices_batch = batch["device"]
            sites_batch = batch["site"]

            if not hasattr(model, 'forward_with_film_info'):
                continue
            _, film_info = model.forward_with_film_info(
                inputs, device_id=device_ids, site_id=site_ids, m_rest=m_rest
            )

            if 'gamma' in film_info:
                layer_idx = list(film_info['gamma'].keys())[0]
                gamma = film_info['gamma'][layer_idx].cpu().numpy()
                beta = film_info['beta'][layer_idx].cpu().numpy()
            elif 'gamma_cls' in film_info:
                layer_idx = list(film_info['gamma_cls'].keys())[0]
                gamma = film_info['gamma_cls'][layer_idx].cpu().numpy()
                beta = film_info['beta_cls'][layer_idx].cpu().numpy()
            elif 'params' in film_info:
                layer_idx = list(film_info['params'].keys())[0]
                gamma = film_info['params'][layer_idx]['gamma'].cpu().numpy()
                beta = film_info['params'][layer_idx]['beta'].cpu().numpy()
            else:
                continue

            for i in range(len(devices_batch)):
                dev = devices_batch[i]
                site = sites_batch[i]
                if labels.ndim == 1:
                    cls = 'Abnormal' if labels[i] == 1 else 'Normal'
                else:
                    c, w = labels[i]
                    if c == 0 and w == 0: cls = 'Normal'
                    elif c == 1 and w == 0: cls = 'Crackle'
                    elif c == 0 and w == 1: cls = 'Wheeze'
                    else: cls = 'Both'

                params_by_device[dev]['gamma'].append(gamma[i])
                params_by_device[dev]['beta'].append(beta[i])
                params_by_site[site]['gamma'].append(gamma[i])
                params_by_site[site]['beta'].append(beta[i])
                params_by_class[cls]['gamma'].append(gamma[i])
                params_by_class[cls]['beta'].append(beta[i])

    return {'by_device': dict(params_by_device), 'by_site': dict(params_by_site), 'by_class': dict(params_by_class)}


# FIGURE 1: Mask Disentanglement Analysis
def plot_figure1_masks(model, save_path='figures/fig_mask_disentanglement.pdf'):

    masks = extract_softfilm_masks(model)
    key = list(masks.keys())[0]
    w_dev = masks[key]['device']
    w_site = masks[key]['site']
    w_rest = masks[key]['rest']
    D = len(w_dev)

    fig = plt.figure(figsize=(15, 10), facecolor='white')
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.4, 1, 1],
        hspace=0.42, wspace=0.30, left=0.08, right=0.92, top=0.93, bottom=0.07
    )

    # (a) Mask activation heatmap
    ax_heat = fig.add_subplot(gs[0, :])
    mask_matrix = np.stack([w_dev, w_site, w_rest], axis=0)
    im = ax_heat.imshow(mask_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_yticklabels(['Device', 'Site', 'Continuous'], fontsize=12, fontweight='bold')
    n_ticks = 8
    tick_pos = np.linspace(0, D-1, n_ticks, dtype=int)
    ax_heat.set_xticks(tick_pos); ax_heat.set_xticklabels(tick_pos, fontsize=9)
    ax_heat.set_xlabel('Feature Dimension', fontsize=11)
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.015, aspect=15)
    cbar.set_label('Mask Weight', fontsize=10); cbar.ax.tick_params(labelsize=9)

    # Region annotations — find actual dominant regions
    dev_active = np.where(w_dev > 0.5)[0]
    site_active = np.where(w_site > 0.5)[0]
    rest_active = np.where(w_rest > 0.5)[0]

    for active_dims, color, label in [
        (dev_active, C_DEV, 'Device\nregion'),
        (site_active, C_SITE, 'Site\nregion'),
        (rest_active, C_REST, 'Continuous\nregion'),
    ]:
        if len(active_dims) > 0:
            start, end = active_dims[0], active_dims[-1]
            mid = (start + end) / 2
            ax_heat.annotate('', xy=(start, -0.6), xytext=(end, -0.6),
                             arrowprops=dict(arrowstyle='<->', color=color, lw=2),
                             annotation_clip=False)
            ax_heat.text(mid, -0.9, label, ha='center', va='top', fontsize=8,
                         color=color, fontweight='bold', clip_on=False)

    # (b) Sorted mask profiles
    ax_sorted = fig.add_subplot(gs[1, 0])
    for arr, name, color in [(w_dev, 'Device', C_DEV), (w_site, 'Site', C_SITE), (w_rest, 'Continuous', C_REST)]:
        ax_sorted.plot(np.sort(arr)[::-1], color=color, lw=2, label=name)
    ax_sorted.axhline(0.5, color='#999', ls=':', alpha=0.6, lw=1)
    ax_sorted.fill_between(range(D), 0.5, 1.0, alpha=0.03, color='green')
    ax_sorted.fill_between(range(D), 0.0, 0.5, alpha=0.03, color='red')
    ax_sorted.text(D*0.75, 0.85, 'Active', fontsize=8, color='#666', style='italic')
    ax_sorted.text(D*0.75, 0.15, 'Inactive', fontsize=8, color='#666', style='italic')
    ax_sorted.set_xlabel('Dimension (sorted)', fontsize=10)
    ax_sorted.set_ylabel('Mask Weight', fontsize=10)
    ax_sorted.legend(fontsize=9, loc='center right')
    ax_sorted.set_ylim(-0.05, 1.05); ax_sorted.set_xlim(0, D)

    # (c) Weight distributions
    gs_hist = gs[1, 1].subgridspec(1, 3, wspace=0.25)
    bins = np.linspace(0, 1, 45)
    for idx, (arr, name, color) in enumerate([(w_dev, 'Device', C_DEV), (w_site, 'Site', C_SITE), (w_rest, 'Continuous', C_REST)]):
        ax = fig.add_subplot(gs_hist[0, idx])
        ax.hist(arr, bins=bins, color=color, edgecolor='white', alpha=0.85, lw=0.5)
        ax.axvline(0.5, color='#999', ls=':', alpha=0.6, lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(name, fontsize=10, fontweight='bold', color=color)
        if idx == 0: ax.set_ylabel('Count', fontsize=9)
        else: ax.set_yticklabels([])
        if idx == 1: ax.set_xlabel('Mask Weight', fontsize=9)
        n_active = (arr > 0.5).sum()
        ax.text(0.95, 0.92, f'Active:\n{n_active}/{D}', transform=ax.transAxes, fontsize=8,
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#ddd'))

    # Hide container axis
    ax_hist_container = fig.add_subplot(gs[1, 1]); ax_hist_container.axis('off')

    # Compute stats
    dominant = np.argmax(np.stack([w_dev, w_site, w_rest]), axis=0)
    active = [(w_dev > 0.5).sum(), (w_site > 0.5).sum(), (w_rest > 0.5).sum()]
    dom_counts = [(dominant == i).sum() for i in range(3)]

    # (d) Active dimensions bar chart
    ax_bar = fig.add_subplot(gs[2, 0])
    names = ['Device', 'Site', 'Continuous']; fc = [C_DEV, C_SITE, C_REST]
    x_pos = np.arange(3); width = 0.32
    bars1 = ax_bar.bar(x_pos - width/2, active, width, color=fc, edgecolor='white', lw=1.5, label='Active ($w > 0.5$)')
    bars2 = ax_bar.bar(x_pos + width/2, dom_counts, width, color=fc, edgecolor='white', lw=1.5, alpha=0.45, hatch='///', label='Dominant (argmax)')
    ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(names, fontsize=10)
    ax_bar.set_ylabel('# Dimensions', fontsize=10)
    ax_bar.axhline(D / 3, color='#999', ls=':', alpha=0.5, lw=1)
    ax_bar.text(2.55, D / 3 + 8, f'$D/3={D//3}$', fontsize=8, color='#888')
    ax_bar.legend(fontsize=8, loc='upper left', framealpha=0.9)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2, h + 5, f'{int(h)}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_bar.set_ylim(0, D * 0.6)

    # (e) Overlap matrix
    ax_overlap = fig.add_subplot(gs[2, 1])
    overlap_ds = (w_dev * w_site).mean()
    overlap_dr = (w_dev * w_rest).mean()
    overlap_sr = (w_site * w_rest).mean()
    overlap_matrix = np.array([
        [w_dev.mean(), overlap_ds, overlap_dr],
        [overlap_ds, w_site.mean(), overlap_sr],
        [overlap_dr, overlap_sr, w_rest.mean()]])
    labels_short = ['Dev', 'Site', 'Cont']
    mask_diag = np.eye(3, dtype=bool)
    display = np.ma.masked_where(mask_diag, overlap_matrix)
    im2 = ax_overlap.imshow(display, cmap='YlOrRd', vmin=0, vmax=0.3, aspect='equal')
    ax_overlap.set_xticks([0, 1, 2]); ax_overlap.set_xticklabels(labels_short, fontsize=10)
    ax_overlap.set_yticks([0, 1, 2]); ax_overlap.set_yticklabels(labels_short, fontsize=10)
    for i in range(3):
        for j in range(3):
            if i == j:
                ax_overlap.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='#f0f0f0', zorder=2))
                ax_overlap.text(j, i, f'$\\mu$={overlap_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9, color='#888')
            else:
                ax_overlap.text(j, i, f'{overlap_matrix[i,j]:.3f}', ha='center', va='center', fontsize=10,
                                fontweight='bold', color='white' if overlap_matrix[i,j] > 0.15 else 'black')
    cbar2 = plt.colorbar(im2, ax=ax_overlap, shrink=0.7, pad=0.05)
    cbar2.set_label('Overlap', fontsize=9); cbar2.ax.tick_params(labelsize=8)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    # Print stats
    coverage = ((w_dev > 0.5) | (w_site > 0.5) | (w_rest > 0.5)).sum() / D
    dead_dims = ((w_dev < 0.5) & (w_site < 0.5) & (w_rest < 0.5)).sum()
    shared_dims = ((w_dev > 0.5).astype(int) + (w_site > 0.5).astype(int) + (w_rest > 0.5).astype(int) > 1).sum()
    binariness = (2 * np.abs(np.concatenate([w_dev, w_site, w_rest]) - 0.5)).mean()

    print(f"\n{'='*50}\nMask Statistics (D={D})\n{'='*50}")
    print(f"Active: Device={active[0]} ({100*active[0]/D:.1f}%), Site={active[1]} ({100*active[1]/D:.1f}%), Cont={active[2]} ({100*active[2]/D:.1f}%)")
    print(f"Coverage: {coverage*100:.1f}% | Dead: {dead_dims} | Shared: {shared_dims} | Binariness: {binariness:.3f}")
    print(f"Overlap: Dev-Site={overlap_ds:.4f}, Dev-Cont={overlap_dr:.4f}, Site-Cont={overlap_sr:.4f}")


# FIGURE 2: t-SNE Comparison — Baseline vs SoftFiLM
def plot_figure2_tsne(baseline_data, softfilm_data, save_path='figures/fig_tsne_comparison.pdf',
                      perplexity=30, n_iter=1000):
    """
    2×3 grid: Row 1 = Baseline, Row 2 = SoftFiLM
    Columns: colored by Class, Device, Site
    """
    print("Computing t-SNE for baseline...")
    emb_base = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=42, init='pca', learning_rate='auto').fit_transform(baseline_data['features'])
    print("Computing t-SNE for SoftFiLM...")
    emb_film = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=42, init='pca', learning_rate='auto').fit_transform(softfilm_data['features'])

    base_cls = labels_to_classes(baseline_data['labels'])
    film_cls = labels_to_classes(softfilm_data['labels'])

    unique_devices = sorted(set(baseline_data['devices']))
    unique_sites = sorted(set(baseline_data['sites']))

    # Build color maps from actual data
    device_cmap = {dev: DEVICE_COLORS[i % len(DEVICE_COLORS)] for i, dev in enumerate(unique_devices)}
    site_cmap = {site: SITE_COLORS[i % len(SITE_COLORS)] for i, site in enumerate(unique_sites)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), facecolor='white')

    col_configs = [
        {'name': 'Class', 'base_vals': base_cls, 'film_vals': film_cls,
         'color_map': CLASS_COLORS, 'base_meta': base_cls, 'film_meta': film_cls},
        {'name': 'Device', 'base_vals': baseline_data['devices'], 'film_vals': softfilm_data['devices'],
         'color_map': device_cmap},
        {'name': 'Recording Site', 'base_vals': baseline_data['sites'], 'film_vals': softfilm_data['sites'],
         'color_map': site_cmap},
    ]

    row_data = [
        {'emb': emb_base, 'label': 'AST Baseline'},
        {'emb': emb_film, 'label': 'AST + SoftFiLM'},
    ]

    for row_idx, row in enumerate(row_data):
        for col_idx, col in enumerate(col_configs):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#fafbfc')

            vals = col['base_vals'] if row_idx == 0 else col['film_vals']
            color_map = col['color_map']
            unique_vals = sorted(set(vals))

            for val in unique_vals:
                mask = np.array(vals) == val
                c = color_map.get(val, '#888888')
                ax.scatter(row['emb'][mask, 0], row['emb'][mask, 1],
                           c=c, s=14, alpha=0.55, edgecolors='white',
                           linewidths=0.2, label=val, rasterized=True, zorder=2)

            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.08, lw=0.5)
            for spine in ax.spines.values():
                spine.set_color('#ccc'); spine.set_linewidth(0.8)

            if row_idx == 0:
                ax.set_title(f'Colored by {col["name"]}', fontsize=12, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(row['label'], fontsize=12, fontweight='bold', labelpad=10)

            ncol = 2 if len(unique_vals) > 5 else 1
            fs = 7 if len(unique_vals) > 5 else 8
            leg = ax.legend(fontsize=fs, loc='upper right', framealpha=0.85,
                            edgecolor='#ddd', markerscale=1.2, ncol=ncol)
            leg.get_frame().set_linewidth(0.5)

    fig.suptitle('t-SNE Visualization of Learned Feature Representations',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.955,
             'SoftFiLM learns device- and site-invariant features while preserving class discriminability',
             ha='center', fontsize=11, style='italic', color='#555')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(hspace=0.28, wspace=0.08)

    annotations = [
        ('↓  Class clusters become separable  ↓', '#27ae60'),
        ('↓  Device bias removed  ↓', '#e74c3c'),
        ('↓  Site variation absorbed  ↓', '#2980b9'),
    ]
    for col_idx, (text, color) in enumerate(annotations):
        fig.text(0.19 + col_idx * 0.295, 0.49, text,
                 ha='center', va='center', fontsize=9.5, fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           edgecolor=color, alpha=0.95, lw=1.5))

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# FIGURE 3: FiLM Parameter Distributions
def plot_figure3_film_params(film_params, save_path='figures/fig_film_params.pdf'):
    """FiLM γ/β violin plots by device and site."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    # γ by device
    ax = axes[0, 0]
    device_data, device_labels = [], []
    for dev, params in film_params['by_device'].items():
        device_data.append(np.array(params['gamma']).mean(axis=1))
        device_labels.append(dev)
    colors = plt.cm.Set2(np.linspace(0, 1, len(device_labels)))
    parts = ax.violinplot(device_data, positions=range(len(device_labels)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']): pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(device_labels))); ax.set_xticklabels(device_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean γ (scale)'); ax.set_title('γ Distribution by Device', fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Identity (γ=1)'); ax.legend(fontsize=9)

    # β by device
    ax = axes[0, 1]
    device_data = [np.array(p['beta']).mean(axis=1) for p in film_params['by_device'].values()]
    parts = ax.violinplot(device_data, positions=range(len(device_labels)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']): pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(device_labels))); ax.set_xticklabels(device_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean β (shift)'); ax.set_title('β Distribution by Device', fontweight='bold')
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Identity (β=0)'); ax.legend(fontsize=9)

    # γ by site
    ax = axes[1, 0]
    site_data, site_labels = [], []
    for site, params in film_params['by_site'].items():
        site_data.append(np.array(params['gamma']).mean(axis=1))
        site_labels.append(site)
    colors_s = plt.cm.Set3(np.linspace(0, 1, len(site_labels)))
    parts = ax.violinplot(site_data, positions=range(len(site_labels)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']): pc.set_facecolor(colors_s[i]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(site_labels))); ax.set_xticklabels(site_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean γ (scale)'); ax.set_title('γ Distribution by Site', fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # β by site
    ax = axes[1, 1]
    site_data = [np.array(p['beta']).mean(axis=1) for p in film_params['by_site'].values()]
    parts = ax.violinplot(site_data, positions=range(len(site_labels)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']): pc.set_facecolor(colors_s[i]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(site_labels))); ax.set_xticklabels(site_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean β (shift)'); ax.set_title('β Distribution by Site', fontweight='bold')
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def load_model_from_checkpoint(checkpoint_path, model_class, cfg, device):
    """Load model from checkpoint with key remapping."""
    model = build_model(cfg.models, model_class)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    fixed = {}
    for k, v in state_dict.items():
        fixed[k.replace("ast.mlp_head.", "classifier.")] = v

    model.load_state_dict(fixed)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_class} from {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}, ICBHI: {ckpt.get('icbhi_score', 'N/A')}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate figures from trained models')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--baseline-ckpt', type=str, required=True, help='Path to baseline AST checkpoint')
    parser.add_argument('--softfilm-ckpt', type=str, required=True, help='Path to SoftFiLM checkpoint')
    parser.add_argument('--output-dir', type=str, default='figures', help='Output directory')
    parser.add_argument('--num-batches', type=int, default=None, help='Limit batches (for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(device_id=0, verbose=True)

    print("\nLoading data...")
    train_loader, test_loader = build_dataloaders(cfg.dataset, cfg.audio)

    print("\nLoading models...")
    baseline_model = load_model_from_checkpoint(args.baseline_ckpt, 'ast', cfg, device)
    softfilm_model = load_model_from_checkpoint(args.softfilm_ckpt, 'ast_filmpp_soft', cfg, device)

    # Figure 1: Mask Analysis
    print(f"\n{'='*50}\nFigure 1: Mask Disentanglement\n{'='*50}")
    plot_figure1_masks(softfilm_model, save_path=os.path.join(args.output_dir, 'fig_mask_disentanglement.pdf'))

    # Figure 2: t-SNE
    print(f"\n{'='*50}\nFigure 2: t-SNE Comparison\n{'='*50}")
    baseline_data = extract_features(baseline_model, test_loader, device, num_batches=args.num_batches)
    softfilm_data = extract_features(softfilm_model, test_loader, device, num_batches=args.num_batches)
    plot_figure2_tsne(baseline_data, softfilm_data, save_path=os.path.join(args.output_dir, 'fig_tsne_comparison.pdf'))

    # Figure 3: FiLM Parameters
    print(f"\n{'='*50}\nFigure 3: FiLM Parameters\n{'='*50}")
    film_params = collect_film_parameters(softfilm_model, test_loader, device, num_batches=args.num_batches)
    plot_figure3_film_params(film_params, save_path=os.path.join(args.output_dir, 'fig_film_params.pdf'))

    print(f"\n{'='*50}\nAll figures saved to {args.output_dir}/\n{'='*50}")


if __name__ == '__main__':
    main()