from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class OptimizerConfig:
    type: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 3e-5
    weight_decay: float = 0.05
    momentum: Optional[float] = None
    betas: Optional[list] = None