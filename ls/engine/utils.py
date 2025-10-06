import torch
import random
import numpy as np


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


def get_device(verbose: bool = True) -> torch.device:
    """
    Returns the best available device among CUDA, MPS, and CPU.
    Automatically detects hardware availability.

    Args:
        verbose (bool): If True, prints the chosen device.

    Returns:
        torch.device: torch.device("cuda"|"mps"|"cpu")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        if verbose:
            print("[Device] Using Apple Metal (MPS) acceleration")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] Using CPU (no GPU backend found)")

    return device