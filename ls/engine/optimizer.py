import torch
import torch.optim as optim

from ls.config.dataclasses.optimizer import OptimizerConfig

def build_optimizer(model: torch.nn.Module, cfg: OptimizerConfig):
    """Instantiate optimizer based on OptimizerConfig."""
    params = model.parameters()

    if cfg.type == "adam":
        return optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas) if cfg.betas else (0.9, 0.999)
        )

    elif cfg.type == "adamw":
        return optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas) if cfg.betas else (0.9, 0.999)
        )

    elif cfg.type == "sgd":
        return optim.SGD(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum or 0.9,
            nesterov=True
        )

    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.type}")