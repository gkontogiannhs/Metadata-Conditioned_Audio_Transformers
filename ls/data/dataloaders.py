import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Tuple, Optional, List

from ls.data.icbhi import ICBHIDataset
from ls.data.transforms import build_transforms
from ls.config.dataclasses import DatasetConfig, AudioConfig


# ============================================================
# IMPROVED WEIGHTED SAMPLING
# ============================================================

def get_sample_weights(samples, multi_label=True, normal_weight=0.3):
    """
    Improved multi-label weighted sampling.
    
    Uses hybrid approach:
    - Balances 4 composite classes
    - Balances individual labels (crackle/wheeze)
    - Doesn't overweight "Both" samples
    
    Args:
        samples: List of dataset samples
        multi_label: Whether using multi-label (True) or 4-class (False)
        normal_weight: Weight for normal samples (default: 0.3)
    
    Returns:
        np.ndarray: Sample weights normalized to sum to 1
    """
    
    if not multi_label:
        # Original 4-class weighting
        labels = [s["label"] for s in samples]
        class_counts = np.bincount(labels, minlength=4)
        inv_freq = 1.0 / (class_counts + 1e-6)
        weights = [inv_freq[label] for label in labels]
        return np.array(weights) / np.sum(weights)
    
    else:
        # Improved multi-label weighting
        labels_ml = np.array([s["multi_label"] for s in samples])
        
        # 1. Class-based weights (4 composite classes)
        class_labels = labels_ml[:, 0] * 2 + labels_ml[:, 1]
        class_counts = np.bincount(class_labels, minlength=4)
        class_inv_freq = 1.0 / (class_counts + 1e-6)
        class_weights = class_inv_freq[class_labels]
        
        # 2. Label-based weights (per label)
        label_freq = labels_ml.sum(axis=0).astype(float)
        label_inv_freq = 1.0 / (label_freq + 1e-6)
        
        label_weights = []
        for vec in labels_ml:
            if vec.sum() == 0:  # Normal
                w = normal_weight
            else:
                # AVERAGE not SUM (prevents overweighting "Both")
                w = np.mean(label_inv_freq[vec == 1])
            label_weights.append(w)
        label_weights = np.array(label_weights)
        
        # 3. Blend both approaches (50/50)
        sample_weights = (class_weights + label_weights) / 2.0
        
        return sample_weights / sample_weights.sum()


# ============================================================
# MAIN DATALOADER BUILDER
# ============================================================

def build_dataloaders(
    dataset_cfg: DatasetConfig,
    audio_cfg: AudioConfig,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for ICBHI with proper statistics sharing.
    
    Args:
        dataset_cfg: Dataset configuration
        audio_cfg: Audio configuration
        verbose: Print detailed info
    
    Returns:
        train_loader, test_loader
    """
    
    if verbose:
        print("\n" + "="*80)
        print("BUILDING DATALOADERS")
        print("="*80)
    
    # Transforms
    train_transform = build_transforms(dataset_cfg, audio_cfg, train=True)
    test_transform = build_transforms(dataset_cfg, audio_cfg, train=False)
    
    # Build training dataset (computes statistics)
    train_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg,
        train=True, 
        transform=train_transform, 
        print_info=verbose,
        continuous_stats=None  # Will compute from training data
    )
    
    # Get training statistics
    train_continuous_stats = train_ds.continuous_stats
    
    if verbose:
        print("\n[DataLoader] Training statistics:")
        for feat, (mean, std) in train_continuous_stats.items():
            print(f"  {feat}: mean={mean:.2f}, std={std:.2f}")
    
    # Build test dataset (uses training statistics)
    test_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg,
        train=False, 
        transform=test_transform, 
        print_info=verbose,
        continuous_stats=train_continuous_stats  # Use training stats
    )
    
    # Weighted sampling setup
    if dataset_cfg.get("weighted_sampler", False):
        if verbose:
            print("\n[DataLoader] Using weighted sampling")
        
        # Use improved weighting
        sample_weights = get_sample_weights(
            train_ds.samples,
            multi_label=dataset_cfg.multi_label,
            normal_weight=getattr(dataset_cfg, 'normal_weight', 0.3)
        )
        
        if verbose:
            # Show sampling distribution
            print("\n[DataLoader] Expected sampling distribution:")
            labels_ml = np.array([s["multi_label"] for s in train_ds.samples])
            class_labels = labels_ml[:, 0] * 2 + labels_ml[:, 1]
            class_names = ["Normal (00)", "Wheeze (01)", "Crackle (10)", "Both (11)"]
            
            for i, name in enumerate(class_names):
                mask = (class_labels == i)
                original_pct = mask.sum() / len(class_labels) * 100
                expected_pct = sample_weights[mask].sum() * 100
                print(f"  {name}: {original_pct:5.1f}% → {expected_pct:5.1f}%")
        
        sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        shuffle = False
    else:
        if verbose:
            print("\n[DataLoader] Using uniform sampling (no weighting)")
        sampler = None
        shuffle = True
    
    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=dataset_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
        drop_last=False,  # Don't drop last batch in test
    )
    
    if verbose:
        print(f"\n[DataLoader] Created:")
        print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"  Test:  {len(test_ds)} samples, {len(test_loader)} batches")
        print("="*80 + "\n")
    
    return train_loader, test_loader


# ============================================================
# K-FOLD WITH STATISTICS SHARING (IMPROVED)
# ============================================================

def build_train_val_kfold(
    dataset_cfg: DatasetConfig,
    audio_cfg: AudioConfig,
    n_splits: int = 5,
    max_retries: int = 50,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[Tuple[DataLoader, DataLoader]], DataLoader]:
    """    
    Returns:
        folds: list of (train_loader, val_loader) for each fold
        test_loader: fixed test loader
    """
    
    if verbose:
        print("\n" + "="*80)
        print(f"BUILDING {n_splits}-FOLD CROSS-VALIDATION")
        print("="*80)
    
    # Build datasets
    train_transform = build_transforms(dataset_cfg, audio_cfg, train=True)
    test_transform = build_transforms(dataset_cfg, audio_cfg, train=False)
    
    # Full training dataset (computes statistics)
    full_train_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg, 
        train=True,
        transform=train_transform,
        print_info=verbose,
        continuous_stats=None
    )
    
    # Get training statistics for test set
    train_stats = full_train_ds.continuous_stats
    
    # Test dataset (uses training statistics)
    test_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg, 
        train=False,
        transform=test_transform,
        print_info=verbose,
        continuous_stats=train_stats
    )
    
    # Extract patient IDs and labels
    patient_ids = [int(s["filename"].split("_")[0]) for s in full_train_ds.samples]
    
    if dataset_cfg.multi_label:
        # For multi-label, use composite class for stratification
        labels_ml = np.array([s["multi_label"] for s in full_train_ds.samples])
        labels = (labels_ml[:, 0] * 2 + labels_ml[:, 1]).tolist()
    else:
        labels = [s["label"] for s in full_train_ds.samples]
    
    indices = list(range(len(full_train_ds)))
    n_cls = 4 if dataset_cfg.multi_label else dataset_cfg.n_cls
    
    # Try to create valid folds
    for attempt in range(max_retries):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=seed + attempt
        )
        folds = []
        valid = True
        
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(indices, labels, groups=patient_ids)):
            # Create subsets
            train_subset = Subset(full_train_ds, train_idx)
            val_subset = Subset(full_train_ds, val_idx)
            
            # Validate class coverage in validation set
            val_labels = [labels[i] for i in val_idx]
            val_counts = np.bincount(val_labels, minlength=n_cls)
            
            if np.any(val_counts == 0):
                if verbose:
                    print(f"[KFold] Attempt {attempt+1}: fold {fold_idx+1} missing classes, retrying...")
                valid = False
                break
            
            # Optional weighted sampling for training fold
            if dataset_cfg.get("weighted_sampler", False):
                train_samples = [full_train_ds.samples[i] for i in train_idx]
                sample_weights = get_sample_weights(
                    train_samples,
                    multi_label=dataset_cfg.multi_label,
                    normal_weight=getattr(dataset_cfg, 'normal_weight', 0.3)
                )
                sampler = WeightedRandomSampler(
                    sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False
            else:
                sampler = None
                shuffle = True
            
            # Build loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=dataset_cfg.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=dataset_cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=dataset_cfg.batch_size,
                shuffle=False,
                num_workers=dataset_cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            
            folds.append((train_loader, val_loader))
            
            # Debug info
            if verbose:
                train_labels = [labels[i] for i in train_idx]
                train_counts = np.bincount(train_labels, minlength=n_cls)
                
                print(f"\n[KFold] Fold {fold_idx+1}/{n_splits} (attempt {attempt+1})")
                print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
                
                class_names = ["Normal (00)", "Wheeze (01)", "Crackle (10)", "Both (11)"] if dataset_cfg.multi_label else [f"Class {i}" for i in range(n_cls)]
                for c in range(n_cls):
                    print(f"  {class_names[c]}: train={train_counts[c]:4d}, val={val_counts[c]:4d}")
        
        if valid:
            if verbose:
                print(f"\n✓ K-fold creation successful after {attempt+1} attempt(s)")
            break
    else:
        raise RuntimeError(f"Failed to create valid folds after {max_retries} attempts")
    
    # Fixed test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    if verbose:
        print("="*80 + "\n")
    
    return folds, test_loader


# ============================================================
# STATISTICS COMPUTATION (UNCHANGED)
# ============================================================

def compute_and_cache_stats(
    dataset: DataLoader,
    cache_file: str,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[float, float]:
    """
    Compute dataset mean/std for fbank features and cache them to disk.
    If cache exists, just load it.
    """
    if os.path.exists(cache_file):
        print(f"[Stats] Loading cached mean/std from {cache_file}")
        with open(cache_file, "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    
    print("[Stats] Computing dataset-wide mean/std...")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    total_sum, total_sq, total_count = 0.0, 0.0, 0
    for batch in tqdm(loader):
        fbanks = batch["input_values"]  # (B, 1, F, T)
        fbanks = fbanks.view(fbanks.size(0), -1)
        
        total_sum += fbanks.sum().item()
        total_sq += (fbanks ** 2).sum().item()
        total_count += fbanks.numel()
    
    mean = total_sum / total_count
    var = (total_sq / total_count) - (mean ** 2)
    std = np.sqrt(var)
    
    stats = {"mean": mean, "std": std}
    with open(cache_file, "w") as f:
        json.dump(stats, f)
    
    print(f"[Stats] Cached mean={mean:.4f}, std={std:.4f} to {cache_file}")
    return mean, std

# import torch
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import StratifiedGroupKFold
# import numpy as np
# import os
# import json
# from tqdm import tqdm
# from typing import Tuple

# from ls.data.icbhi import ICBHIDataset
# from ls.data.transforms import build_transforms
# from ls.config.dataclasses import DatasetConfig, AudioConfig


# def build_dataloaders(
#     dataset_cfg: DatasetConfig, 
#     audio_cfg: AudioConfig
# ) -> Tuple[DataLoader, DataLoader]: # TODO: Make scalable for more datasets
#     """
#     Build PyTorch DataLoaders for ICBHI. This way does not account for a validation set.
#     This is not the recommended way if you want to do proper model selection. This is just
#     for comparison with past work.

#     Args:
#         dataset_cfg: dataset configuration (with dataset/audio/loader fields).
#         audio_cfg: audio configuration (with audio-specific fields).

#     Returns:
#         train_loader, test_loader, (optional) test_loader
#     """
#     # Transforms (feature-domain)
#     train_transform = build_transforms(dataset_cfg, audio_cfg, train=True)
#     test_transform = build_transforms(dataset_cfg, audio_cfg, train=False)

#     # Datasets
#     train_ds = ICBHIDataset(dataset_cfg, audio_cfg,
#                             train=True, transform=train_transform, print_info=True)
#     test_ds = ICBHIDataset(dataset_cfg, audio_cfg,
#                            train=False, transform=test_transform, print_info=True)
    
#     if dataset_cfg.get("weighted_sampler", False):
#         sample_weights = train_ds.get_sample_weights()
#         sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
#         shuffle = False
#     else:
#         sampler = None
#         shuffle = True
    
#     # DataLoaders
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=dataset_cfg.batch_size,
#         shuffle=shuffle if sampler is None else False,
#         sampler=sampler,
#         num_workers=dataset_cfg.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=dataset_cfg.batch_size,
#         shuffle=False,
#         num_workers=dataset_cfg.num_workers,
#         pin_memory=True,
#     )

#     return train_loader, test_loader


# def build_train_val_kfold(
#     dataset_cfg: DatasetConfig, audio_cfg: AudioConfig, 
#     n_splits: int = 5, max_retries: int = 50, seed: int = 42
#     ) -> Tuple[list[Tuple[DataLoader, DataLoader]], DataLoader]:
#     """
#     Patient-level stratified K-fold with safety checks:
#     - Prevents patient leakage
#     - Ensures each validation fold has all classes represented
#     - Retries until valid folds are found (or max_retries exceeded)

#     Returns:
#         folds: list of (train_loader, val_loader)
#         test_loader: fixed test loader
#     """
#     # Step 1: outer split (train vs test)
#     train_ds = ICBHIDataset(
#         dataset_cfg, audio_cfg, train=True, 
#         transform=build_transforms(dataset_cfg, audio_cfg, train=True), 
#         print_info=True
#     )
#     test_ds = ICBHIDataset(
#         dataset_cfg, audio_cfg, train=False, 
#         transform=build_transforms(dataset_cfg, audio_cfg, train=False), 
#         print_info=True
#     )

#     patient_ids = [int(s["filename"].split("_")[0]) for s in train_ds.samples]
#     labels = [s["label"] for s in train_ds.samples]
#     indices = list(range(len(train_ds)))
#     n_cls = dataset_cfg.n_cls

#     for attempt in range(max_retries):
#         sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
#         folds = []
#         valid = True

#         for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(indices, labels, groups=patient_ids)):
#             train_subset = Subset(train_ds, train_idx)
#             val_subset = Subset(train_ds, val_idx)

#             # Validate class coverage in val set
#             val_labels = [labels[i] for i in val_idx]
#             val_counts = np.bincount(val_labels, minlength=n_cls)

#             if np.any(val_counts == 0):
#                 print(f"[KFold] Attempt {attempt+1}: fold {fold_idx+1} missing classes, retrying...")
#                 valid = False
#                 break

#             # Build loaders
#             train_loader = DataLoader(
#                 train_subset,
#                 batch_size=dataset_cfg.batch_size,
#                 shuffle=True,
#                 num_workers=dataset_cfg.num_workers,
#                 pin_memory=True,
#                 drop_last=True,
#             )
#             val_loader = DataLoader(
#                 val_subset,
#                 batch_size=dataset_cfg.batch_size,
#                 shuffle=False,
#                 num_workers=dataset_cfg.num_workers,
#                 pin_memory=True,
#             )
#             folds.append((train_loader, val_loader))

#             # Debug info
#             train_labels = [labels[i] for i in train_idx]
#             train_counts = np.bincount(train_labels, minlength=n_cls)
#             print(f"\n[KFold] Fold {fold_idx+1}/{n_splits} (attempt {attempt+1})")
#             for c in range(n_cls):
#                 print(f"  Class {c}: train={train_counts[c]}, val={val_counts[c]}")

#         if valid:
#             print(f"[KFold] Success after {attempt+1} attempt(s)")
#             break
#     else:
#         raise RuntimeError("Failed to create valid folds after max_retries attempts")

#     # Fixed test loader
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=dataset_cfg.batch_size,
#         shuffle=False,
#         num_workers=dataset_cfg.num_workers,
#         pin_memory=True,
#     )

#     return folds, test_loader


# def compute_and_cache_stats(dataset: DataLoader, cache_file: str, batch_size: int = 32, num_workers: int = 0) -> Tuple[float, float]:
#     """
#     Compute dataset mean/std for fbank features and cache them to disk.
#     If cache exists, just load it.
#     """
#     if os.path.exists(cache_file):
#         print(f"[Stats] Loading cached mean/std from {cache_file}")
#         with open(cache_file, "r") as f:
#             stats = json.load(f)
#         return stats["mean"], stats["std"]

#     print("[Stats] Computing dataset-wide mean/std...")
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     total_sum, total_sq, total_count = 0.0, 0.0, 0
#     for batch in tqdm(loader):
#         fbanks = batch["input_values"]  # (B, 1, F, T)
#         fbanks = fbanks.view(fbanks.size(0), -1)

#         total_sum += fbanks.sum().item()
#         total_sq += (fbanks ** 2).sum().item()
#         total_count += fbanks.numel()

#     mean = total_sum / total_count
#     var = (total_sq / total_count) - (mean ** 2)
#     std = np.sqrt(var)

#     stats = {"mean": mean, "std": std}
#     with open(cache_file, "w") as f:
#         json.dump(stats, f)

#     print(f"[Stats] Cached mean={mean:.4f}, std={std:.4f} to {cache_file}")
#     return mean, std

