import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Tuple, List, Optional

from ls.data.icbhi import ICBHIDataset
from ls.data.transforms import build_transforms
from ls.config.dataclasses import DatasetConfig, AudioConfig


# ============================================================
# MAIN DATALOADER BUILDER (with statistics sharing)
# ============================================================

def build_dataloaders(
    dataset_cfg: DatasetConfig, 
    audio_cfg: AudioConfig,
    use_weighted_sampler: Optional[bool] = None,
    normal_weight: float = 0.3,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for ICBHI with proper statistics sharing.
    
    Args:
        dataset_cfg: Dataset configuration
        audio_cfg: Audio configuration
        use_weighted_sampler: Override config setting (default: use config value)
    
    Returns:
        train_loader, test_loader
    """
    
    # Transforms (feature-domain)
    train_transform = build_transforms(dataset_cfg, audio_cfg, train=True)
    test_transform = build_transforms(dataset_cfg, audio_cfg, train=False)

    print("\n" + "="*80)
    print("BUILDING DATALOADERS")
    print("="*80)

    # Step 1: Build training dataset (computes statistics)
    print("\n[1/3] Building training dataset...")
    train_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg,
        train=True, 
        transform=train_transform, 
        print_info=True,
        continuous_stats=None  # Will compute from training data
    )
    
    # Step 2: Get statistics from training set
    train_continuous_stats = train_ds.continuous_stats
    
    print("\n[2/3] Training statistics computed:")
    for feat, (mean, std) in train_continuous_stats.items():
        print(f"  {feat}: mean={mean:.4f}, std={std:.4f}")
    
    # Step 3: Build test dataset (uses training statistics)
    print("\n[3/3] Building test dataset with training statistics...")
    test_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg,
        train=False, 
        transform=test_transform, 
        print_info=True,
        continuous_stats=train_continuous_stats  # Use training stats
    )
    
    # Weighted sampling configuration
    if use_weighted_sampler is None:
        use_weighted_sampler = dataset_cfg.get("weighted_sampler", False)
    
    if use_weighted_sampler:
        print("\n[Sampling] Using weighted sampling...")
        sample_weights = train_ds.get_sample_weights()
        
        # Validate weights
        if sample_weights is None:
            print("Warning: get_sample_weights() returned None, using uniform sampling")
            sampler = None
            shuffle = True
        elif np.isnan(sample_weights).any() or np.isinf(sample_weights).any():
            print("Warning: Sample weights contain NaN/Inf, using uniform sampling")
            sampler = None
            shuffle = True
        else:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
            
            # Show sampling effect
            print(f"  Sample weights: min={sample_weights.min():.6f}, max={sample_weights.max():.6f}")
            print(f"  Total samples: {len(sample_weights)}")
    else:
        print("\n[Sampling] Using uniform sampling (shuffle=True)")
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
        drop_last=False,
    )
    
    print(f"\n[Loaders] Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print("="*80 + "\n")

    return train_loader, test_loader


# ============================================================
# K-FOLD CROSS-VALIDATION (with statistics sharing)
# ============================================================

def build_train_val_kfold(
    dataset_cfg: DatasetConfig, 
    audio_cfg: AudioConfig, 
    n_splits: int = 5, 
    max_retries: int = 50, 
    seed: int = 42,
    use_weighted_sampler: bool = False,
) -> Tuple[List[Tuple[DataLoader, DataLoader]], DataLoader]:
    """
    Patient-level stratified K-fold with proper statistics handling.
    
    Args:
        dataset_cfg: Dataset configuration
        audio_cfg: Audio configuration
        n_splits: Number of folds
        max_retries: Max attempts to find valid folds
        seed: Random seed
        use_weighted_sampler: Use weighted sampling in training
    
    Returns:
        folds: List of (train_loader, val_loader) for each fold
        test_loader: Fixed test set loader
    """
    
    print("\n" + "="*80)
    print(f"BUILDING {n_splits}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Step 1: Build full training dataset (to get samples)
    print("\n[1/2] Loading full training set...")
    full_train_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg, 
        train=True, 
        transform=build_transforms(dataset_cfg, audio_cfg, train=True),
        print_info=True,
        continuous_stats=None  # Will compute per-fold
    )
    
    # Step 2: Build test dataset (uses full training statistics)
    print("\n[2/2] Loading test set...")
    test_ds = ICBHIDataset(
        dataset_cfg, 
        audio_cfg, 
        train=False, 
        transform=build_transforms(dataset_cfg, audio_cfg, train=False),
        print_info=True,
        continuous_stats=full_train_ds.continuous_stats
    )
    
    # Extract metadata for splitting
    patient_ids = [int(s["filename"].split("_")[0]) for s in full_train_ds.samples]
    
    if dataset_cfg.multi_label:
        # For multi-label, use composite class for stratification
        labels_ml = np.array([s["multi_label"] for s in full_train_ds.samples])
        labels = labels_ml[:, 0] * 2 + labels_ml[:, 1]  # Convert to 0-3
    else:
        labels = [s["label"] for s in full_train_ds.samples]
    
    indices = list(range(len(full_train_ds)))
    n_cls = 4 if dataset_cfg.multi_label else dataset_cfg.n_cls
    
    print(f"\n[K-Fold] Creating {n_splits} folds with patient-level splitting...")
    print(f"  Total samples: {len(full_train_ds)}")
    print(f"  Unique patients: {len(set(patient_ids))}")
    print(f"  Classes: {n_cls}")
    
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
            # Create fold-specific dataset with recomputed statistics
            # This ensures validation uses training statistics from THIS fold only
            
            # Get training samples for this fold
            train_samples = [full_train_ds.samples[i] for i in train_idx]
            
            # Compute statistics from training fold
            fold_train_ages = [s["age"] for s in train_samples]
            fold_train_bmis = [s["bmi"] for s in train_samples]
            fold_train_durs = [s["duration"] for s in train_samples]
            
            fold_stats = {
                "age": (np.mean(fold_train_ages), np.std(fold_train_ages) + 1e-6),
                "bmi": (np.mean(fold_train_bmis), np.std(fold_train_bmis) + 1e-6),
                "duration": (np.mean(fold_train_durs), np.std(fold_train_durs) + 1e-6),
            }
            
            # Create subsets
            train_subset = Subset(full_train_ds, train_idx)
            val_subset = Subset(full_train_ds, val_idx)
            
            # Update continuous_stats for validation subset
            # We need to ensure val subset uses fold training statistics
            # This is done by monkey-patching the dataset's statistics
            original_stats = full_train_ds.continuous_stats
            full_train_ds.continuous_stats = fold_stats

            # Validate class coverage in val set
            val_labels = [labels[i] for i in val_idx]
            val_counts = np.bincount(val_labels, minlength=n_cls)

            if np.any(val_counts == 0):
                print(f"  [Attempt {attempt+1}] Fold {fold_idx+1} missing classes, retrying...")
                valid = False
                full_train_ds.continuous_stats = original_stats
                break

            # Weighted sampling for training
            if use_weighted_sampler:
                train_samples_for_weights = [full_train_ds.samples[i] for i in train_idx]
                sample_weights = compute_fold_sample_weights(
                    train_samples_for_weights, 
                    dataset_cfg.multi_label
                )
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                train_shuffle = False
            else:
                train_sampler = None
                train_shuffle = True

            # Build loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=dataset_cfg.batch_size,
                shuffle=train_shuffle,
                sampler=train_sampler,
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
            
            # Restore original stats
            full_train_ds.continuous_stats = original_stats

            # Debug info
            train_labels = [labels[i] for i in train_idx]
            train_counts = np.bincount(train_labels, minlength=n_cls)
            
            print(f"\n  Fold {fold_idx+1}/{n_splits} (attempt {attempt+1}):")
            print(f"    Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
            print(f"    Train patients: {len(set([patient_ids[i] for i in train_idx]))}")
            print(f"    Val patients: {len(set([patient_ids[i] for i in val_idx]))}")
            print(f"    Class distribution:")
            for c in range(n_cls):
                print(f"      Class {c}: train={train_counts[c]}, val={val_counts[c]}")

        if valid:
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
    
    print(f"\n[Test] Test loader: {len(test_loader)} batches")
    print("="*80 + "\n")

    return folds, test_loader


# ============================================================
# HELPER: COMPUTE SAMPLE WEIGHTS FOR A FOLD
# ============================================================

def compute_sample_weights(samples, multi_label=True, normal_weight=0.3):
    """
    Helper function to compute sample weights.
    Can be called from dataset or directly from dataloader.
    """
    if not multi_label:
        labels = np.array([s["label"] for s in samples], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=4)
        inv_freq = 1.0 / (class_counts.astype(np.float64) + 1e-6)
        weights = inv_freq[labels]
        return weights / weights.sum()
    else:
        labels_ml = np.array([s["multi_label"] for s in samples], dtype=np.int64)
        
        # Class weights
        class_labels = labels_ml[:, 0] * 2 + labels_ml[:, 1]
        class_counts = np.bincount(class_labels, minlength=4)
        class_inv_freq = 1.0 / (class_counts.astype(np.float64) + 1e-6)
        class_weights = np.array([class_inv_freq[cls] for cls in class_labels], dtype=np.float64)
        
        # Label weights
        label_freq = labels_ml.sum(axis=0).astype(np.float64)
        label_inv_freq = 1.0 / (label_freq + 1e-6)
        
        label_weights = []
        for vec in labels_ml:
            if vec.sum() == 0:
                w = float(normal_weight)
            else:
                w = float(np.mean(label_inv_freq[vec == 1]))
            label_weights.append(w)
        label_weights = np.array(label_weights, dtype=np.float64)
        
        # Blend
        sample_weights = (class_weights + label_weights) / 2.0
        return sample_weights / sample_weights.sum()

    
def compute_fold_sample_weights(samples: List[dict], multi_label: bool, normal_weight: float = 0.3) -> np.ndarray:
    """
    Compute sample weights for a specific fold using the improved hybrid method.
    
    Args:
        samples: List of sample dictionaries
        multi_label: Whether using multi-label classification
        normal_weight: Weight for normal samples (default: 0.3)
    
    Returns:
        Sample weights (normalized)
    """
    if not multi_label:
        # 4-class weighting
        labels = [s["label"] for s in samples]
        class_counts = np.bincount(labels, minlength=4)
        inv_freq = 1.0 / (class_counts + 1e-6)
        weights = np.array([inv_freq[lbl] for lbl in labels])
        return weights / weights.sum()
    else:
        # MULTI-LABEL WEIGHTING (Hybrid method)
        labels_ml = np.array([s["multi_label"] for s in samples])
        
        # 1. Class-based weights (4 composite classes)
        class_labels = labels_ml[:, 0] * 2 + labels_ml[:, 1]
        class_counts = np.bincount(class_labels, minlength=4)
        class_inv_freq = 1.0 / (class_counts + 1e-6)
        class_weights = np.array(class_inv_freq[class_labels])  # Ensure numpy array
        
        # 2. Label-based weights (per label)
        label_freq = labels_ml.sum(axis=0).astype(float)
        label_inv_freq = 1.0 / (label_freq + 1e-6)
        
        label_weights = []
        for vec in labels_ml:
            if vec.sum() == 0:  # Normal
                w = float(normal_weight)  # Ensure float
            else:
                w = float(np.mean(label_inv_freq[vec == 1]))  # Ensure float
            label_weights.append(w)
        label_weights = np.array(label_weights, dtype=np.float64)  # Ensure numpy array
        
        # 3. Blend both approaches
        sample_weights = (class_weights + label_weights) / 2.0
        
        return sample_weights / sample_weights.sum()


# ============================================================
# STATISTICS COMPUTATION (unchanged, for reference)
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
    
    Note: This computes statistics for FBANK features (spectrograms),
    which is different from continuous metadata statistics.
    """
    if os.path.exists(cache_file):
        print(f"[Stats] Loading cached mean/std from {cache_file}")
        with open(cache_file, "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    print("[Stats] Computing dataset-wide mean/std for fbank features...")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    total_sum, total_sq, total_count = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="Computing stats"):
        fbanks = batch["input_values"]  # (B, 1, F, T)
        fbanks = fbanks.view(fbanks.size(0), -1)

        total_sum += fbanks.sum().item()
        total_sq += (fbanks ** 2).sum().item()
        total_count += fbanks.numel()

    mean = total_sum / total_count
    var = (total_sq / total_count) - (mean ** 2)
    std = np.sqrt(var)

    stats = {"mean": mean, "std": std}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(stats, f)

    print(f"[Stats] Cached mean={mean:.4f}, std={std:.4f} to {cache_file}")
    return mean, std