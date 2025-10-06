from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

from ls.config.dataclasses.schedulers import SchedulerConfig


@dataclass
class TrainingConfig:
    loss: str = "weighted_ce"
    use_class_weights: bool = True
    epochs: int = 100
    lr: float = 3e-5
    weight_decay: float = 0.05
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)