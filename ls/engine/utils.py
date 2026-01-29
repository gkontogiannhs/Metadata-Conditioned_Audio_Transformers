import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Focuses learning on hard examples by down-weighting easy ones.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Balancing factor for positive/negative examples
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=1, clip=0.01, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping: SUBTRACT margin from negative probabilities
        # This down-weights easy negatives (where prob is already low)
        probs_neg = (probs - self.clip).clamp(min=0)
        
        # Positive part: standard focal-style loss
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * ((1 - probs) ** self.gamma_pos)
        
        # Negative part: use clipped probabilities
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps)) * (probs_neg ** self.gamma_neg)
        
        loss = -loss_pos - loss_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLossCE(nn.Module):
    """
    Focal Loss for multi-class (single-label) classification.
    
    Args:
        gamma: Focusing parameter
        weight: Optional class weights tensor
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def build_criterion(cfg, device=None):
    """
    Build loss function based on configuration.
    
    Supports:
        Multi-label mode:
            - 'bce': Standard BCEWithLogitsLoss
            - 'weighted_bce': BCEWithLogitsLoss with pos_weight
            - 'focal': Focal Loss
            - 'asymmetric': Asymmetric Loss
        
        Multi-class mode:
            - 'ce': Standard CrossEntropyLoss
            - 'weighted_ce': CrossEntropyLoss with class weights
            - 'focal_ce': Focal Loss for multi-class
    
    Config example:
        loss:
            type: asymmetric
            gamma_neg: 4
            gamma_pos: 1
            clip: 0.05
    
    Args:
        cfg: Configuration object with cfg.dataset.multi_label and cfg.loss
        device: Device to move weight tensors to
    
    Returns:
        nn.Module: Loss function
    """
    multi_label = True # cfg.dataset.multi_label
    # Get loss config (with defaults)
    loss_cfg = getattr(cfg, 'loss', None)
    print(f"User wants to use {loss_cfg} loss function.")
    # if loss_cfg is None:
    #     loss_type = 'default'
    # else:
    #     loss_type = getattr(loss_cfg, 'type', 'default')
    
    print(f"[Loss] Building criterion: type={loss_cfg}, multi_label={multi_label}")
    
    # ============================================================
    # MULTI-LABEL LOSSES
    # ============================================================
    if multi_label:
        if loss_cfg in ['default', 'bce']:
            print(f"[Loss] Using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss()
        
        elif loss_cfg == 'weighted_bce':
            # pos_weight: weight for positive class (>1 increases recall)
            # e.g., [2.0, 3.0] means crackle pos weighted 2x, wheeze 3x
            pos_weight = getattr(loss_cfg, 'pos_weight', [1.0, 1.0])
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            if device:
                pos_weight = pos_weight.to(device)
            print(f"[Loss] Using Weighted BCEWithLogitsLoss with pos_weight={pos_weight.tolist()}")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        elif loss_cfg == 'focal':
            alpha = getattr(loss_cfg, 'alpha', 0.25)
            gamma = getattr(loss_cfg, 'gamma', 2.0)
            print(f"[Loss] Using FocalLoss with alpha={alpha}, gamma={gamma}")
            return FocalLoss(alpha=alpha, gamma=gamma)
        
        elif loss_cfg == 'asymmetric':
            gamma_neg = getattr(loss_cfg, 'gamma_neg', 4)
            gamma_pos = getattr(loss_cfg, 'gamma_pos', 1)
            clip = getattr(loss_cfg, 'clip', 0.05)
            print(f"[Loss] Using AsymmetricLoss with gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}")
            return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        
        else:
            raise ValueError(f"Unknown multi-label loss type: {loss_cfg}")
    
    # ============================================================
    # MULTI-CLASS LOSSES
    # ============================================================
    else:
        if loss_cfg in ['default', 'ce']:
            print(f"[Loss] Using CrossEntropyLoss")
            return nn.CrossEntropyLoss()
        
        elif loss_cfg == 'weighted_ce':
            # class_weights: inverse frequency weights
            # e.g., [0.5, 1.5, 2.0, 2.5] for [normal, crackle, wheeze, both]
            class_weights = getattr(loss_cfg, 'class_weights', None)
            if class_weights is None:
                raise ValueError("weighted_ce requires loss.class_weights in config")
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            if device:
                class_weights = class_weights.to(device)
            print(f"[Loss] Using Weighted CrossEntropyLoss with weights={class_weights.tolist()}")
            return nn.CrossEntropyLoss(weight=class_weights)
        
        elif loss_cfg in ['focal', 'focal_ce']:
            gamma = getattr(loss_cfg, 'gamma', 2.0)
            class_weights = getattr(loss_cfg, 'class_weights', None)
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
                if device:
                    class_weights = class_weights.to(device)
            print(f"[Loss] Using FocalLossCE with gamma={gamma}, weights={class_weights}")
            return FocalLossCE(gamma=gamma, weight=class_weights)
        
        else:
            raise ValueError(f"Unknown multi-class loss type: {loss_cfg}")
        

def set_visible_gpus(gpus: str, verbose: bool = True):
    """Restrict which GPUs PyTorch can see."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if verbose:
        print(f"[CUDA] Visible devices set to: {gpus}")
    torch.cuda.device_count()


def set_seed(seed: int = 42, deterministic: bool = True, verbose: bool = True):
    """
    Fix random seeds for reproducibility across NumPy, Python, and PyTorch.
    Works safely with CUDA, MPS, or CPU.

    Args:
        seed (int): Random seed.
        deterministic (bool): Whether to enforce deterministic behavior.
        verbose (bool): Print confirmation if True.
    """
    import random
    import numpy as np
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