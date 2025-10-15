import torch
import random
import numpy as np
import torch.nn as nn
from ls.config.dataclasses import TrainingConfig


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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
            # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"[INFO] Using dynamically computed class weights: {class_weights.cpu().numpy().round(3).tolist()}")
        elif hasattr(cfg, "class_weights") and len(cfg.class_weights) > 0:
            class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32).to(device)
            print(f"[INFO] Using static class weights from config: {class_weights.cpu().numpy().round(3).tolist()}")
        else:
            raise ValueError(
                "Weighted CE loss selected but no class weights provided. "
                "Either compute them dynamically or specify cfg.class_weights."
            )
        return nn.CrossEntropyLoss(weight=class_weights).to(device)

    # -------------------------------
    # 3) Focal Loss
    # -------------------------------
    elif loss_type == "focal":
        gamma = float(getattr(cfg, "gamma", 2.0))
        # alpha = float(getattr(cfg, "alpha", None))
        if class_weights is not None:
            print(f"[INFO] Using dynamically computed class weights for Focal Loss: {class_weights.cpu().numpy().round(3).tolist()}")
        else:
            class_weights = torch.tensor([0.25] * 4, dtype=torch.float32).to(device)
        num_classes = getattr(cfg, "n_cls", None)

        return FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            num_classes=num_classes,
            task_type="multi-class",
        ).to(device)

    # -------------------------------
    # 4) Unknown loss
    # -------------------------------
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")