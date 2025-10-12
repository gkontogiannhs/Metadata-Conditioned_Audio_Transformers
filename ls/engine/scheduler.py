import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

from ls.config.dataclasses.schedulers import SchedulerConfig

def build_scheduler(cfg: SchedulerConfig, epochs: int, optimizer: optim.Optimizer):
    """Return learning rate scheduler according to config."""
    # sch_cfg = cfg.training.scheduler
    # epochs = cfg.training.epochs

    if cfg.type == "none":
        return None

    # --- Cosine decay only ---
    if cfg.type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(epochs),
            eta_min=float(cfg.min_lr),
        )

    elif cfg.type == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=int(cfg.T_0), 
            T_mult=1, 
            eta_min=float(cfg.min_lr)
        )

    # --- Warmup + Cosine decay ---
    elif cfg.type == "cosine_warmup":
        warmup_epochs = int(cfg.warmup_epochs)
        cosine_epochs = epochs - warmup_epochs

        warmup = LinearLR(
            optimizer,
            start_factor=float(cfg.start_linear_warmup),
            end_factor=float(cfg.end_linear_warmup),
            total_iters=warmup_epochs,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=float(cfg.min_lr),
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    # --- Reduce on Plateau ---
    elif cfg.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.reduce_mode,
            factor=cfg.reduce_factor,
            patience=cfg.reduce_patience,
            min_lr=float(cfg.reduce_min_lr),
        )

    else:
        raise ValueError(f"Unknown scheduler type: {cfg.use}")
