import torch
import random
import numpy as np
import torch.nn as nn
from ls.config.dataclasses import TrainingConfig


def set_seed(seed: int = 42, deterministic: bool = True, verbose: bool = True):
    """
    Fix random seeds for reproducibility across NumPy, Python, and PyTorch.
    Works safely with CUDA, MPS, or CPU.

    Args:
        seed (int): Random seed.
        deterministic (bool): Whether to enforce deterministic behavior.
        verbose (bool): Print confirmation if True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MPS note (Apple Silicon)
    elif torch.backends.mps.is_available():
        # MPS currently doesn't support fully deterministic ops, but seed it anyway
        torch.mps.manual_seed(seed)

    if verbose:
        device_type = (
            "CUDA" if torch.cuda.is_available()
            else "MPS" if torch.backends.mps.is_available()
            else "CPU"
        )
        print(f"[Seed] Fixed all random seeds to {seed} ({device_type})")


def get_device(device_id: int = 0, verbose: bool = True) -> torch.device:
    """
    Returns the best available device among CUDA, MPS, and CPU.
    Automatically detects hardware availability.

    Args:
        verbose (bool): If True, prints the chosen device.

    Returns:
        torch.device: torch.device("cuda"|"mps"|"cpu")
    """
    # device_id = int(device_id)
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of available CUDA devices: {num_devices}")
        if device_id >= num_devices:
            raise ValueError(f"Requested CUDA device {device_id}, but only {num_devices} available.")
        device = torch.device(f"cuda:{device_id}")
        if verbose:
            print(f"[Device] Using CUDA:{device_id} -> {torch.cuda.get_device_name(device_id)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        if verbose:
            print("[Device] Using Apple Metal (MPS) acceleration")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] Using CPU (no GPU backend found)")

    return device

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
            # weights = class_weights.to(device)
            # weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"[INFO] Using dynamically computed class weights: {class_weights}")
        elif getattr(cfg, "class_weights", []) == list and len(cfg.class_weights) > 0:
            class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32).to(device)
            print(f"[INFO] Using static class weights from config: {class_weights}")
        else:
            raise ValueError(
                "Weighted CE loss selected but no class weights provided. "
                "Either compute them dynamically or specify cfg.loss.class_weights."
            )
        return nn.CrossEntropyLoss(weight=class_weights).to(device)

    # -------------------------------
    # 3) Focal Loss
    # -------------------------------
    elif loss_type == "focal":

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, weight=None):
                super().__init__()
                self.gamma = gamma
                if weight is not None:
                    print(f"[INFO] Using dynamically computed class weights: {class_weights.cpu().numpy().round(3).tolist()}")
                self.ce = nn.CrossEntropyLoss(weight=weight)

            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs.float(), targets)
                pt = torch.exp(-ce_loss)
                return ((1 - pt) ** self.gamma * ce_loss).mean()

        gamma = float(getattr(cfg, "gamma", 2.0))
        # weights = class_weights.to(device) if class_weights is not None else None
        return FocalLoss(gamma=gamma, weight=class_weights).to(device)

    # -------------------------------
    # 4) Unknown loss
    # -------------------------------
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")