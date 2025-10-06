import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Tuple

from ls.data.icbhi import ICBHIDataset
from ls.data.transforms import build_transforms
from ls.config.dataclasses import DatasetConfig, AudioConfig


def build_dataloaders(
    dataset_cfg: DatasetConfig, 
    audio_cfg: AudioConfig
) -> Tuple[DataLoader, DataLoader]: # TODO: Make scalable for more datasets
    """
    Build PyTorch DataLoaders for ICBHI. This way does not account for a validation set.
    This is not the recommended way if you want to do proper model selection. This is just
    for comparison with past work.

    Args:
        dataset_cfg: dataset configuration (with dataset/audio/loader fields).
        audio_cfg: audio configuration (with audio-specific fields).

    Returns:
        train_loader, test_loader, (optional) test_loader
    """
    # Transforms (feature-domain)
    train_transform = build_transforms(dataset_cfg, audio_cfg, train=True)
    test_transform = build_transforms(dataset_cfg, audio_cfg, train=False)

    # Datasets
    train_ds = ICBHIDataset(dataset_cfg, audio_cfg,
                            train=True, transform=train_transform, print_info=True)
    test_ds = ICBHIDataset(dataset_cfg, audio_cfg,
                           train=False, transform=test_transform, print_info=True)

    # Sampler (optional weighted)
    if dataset_cfg.get("weighted_sampler", False):
        class_weights = 1.0 / (train_ds.class_counts + 1e-6) # avoid div by zero
        sample_weights = [class_weights[s["label"]] for s in train_ds.samples] # assign weight to each sample for proper use of WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=dataset_cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
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
    )

    return train_loader, test_loader


def build_train_val_kfold(
    dataset_cfg: DatasetConfig, audio_cfg: AudioConfig, 
    n_splits: int = 5, max_retries: int = 50, seed: int = 42
    ) -> Tuple[list[Tuple[DataLoader, DataLoader]], DataLoader]:
    """
    Patient-level stratified K-fold with safety checks:
    - Prevents patient leakage
    - Ensures each validation fold has all classes represented
    - Retries until valid folds are found (or max_retries exceeded)

    Returns:
        folds: list of (train_loader, val_loader)
        test_loader: fixed test loader
    """
    # Step 1: outer split (train vs test)
    train_ds = ICBHIDataset(
        dataset_cfg, audio_cfg, train=True, 
        transform=build_transforms(dataset_cfg, audio_cfg, train=True), 
        print_info=True
    )
    test_ds = ICBHIDataset(
        dataset_cfg, audio_cfg, train=False, 
        transform=build_transforms(dataset_cfg, audio_cfg, train=False), 
        print_info=True
    )

    patient_ids = [int(s["filename"].split("_")[0]) for s in train_ds.samples]
    labels = [s["label"] for s in train_ds.samples]
    indices = list(range(len(train_ds)))
    n_cls = dataset_cfg.n_cls

    for attempt in range(max_retries):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        folds = []
        valid = True

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(indices, labels, groups=patient_ids)):
            train_subset = Subset(train_ds, train_idx)
            val_subset = Subset(train_ds, val_idx)

            # Validate class coverage in val set
            val_labels = [labels[i] for i in val_idx]
            val_counts = np.bincount(val_labels, minlength=n_cls)

            if np.any(val_counts == 0):
                print(f"[KFold] Attempt {attempt+1}: fold {fold_idx+1} missing classes, retrying...")
                valid = False
                break

            # Build loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=dataset_cfg.batch_size,
                shuffle=True,
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
            )
            folds.append((train_loader, val_loader))

            # Debug info
            train_labels = [labels[i] for i in train_idx]
            train_counts = np.bincount(train_labels, minlength=n_cls)
            print(f"\n[KFold] Fold {fold_idx+1}/{n_splits} (attempt {attempt+1})")
            for c in range(n_cls):
                print(f"  Class {c}: train={train_counts[c]}, val={val_counts[c]}")

        if valid:
            print(f"[KFold] Success after {attempt+1} attempt(s)")
            break
    else:
        raise RuntimeError("Failed to create valid folds after max_retries attempts")

    # Fixed test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
    )

    return folds, test_loader


def compute_and_cache_stats(dataset: DataLoader, cache_file: str, batch_size: int = 32, num_workers: int = 0) -> Tuple[float, float]:
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

