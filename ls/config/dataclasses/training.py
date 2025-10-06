from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

from ls.config.dataclasses.schedulers import SchedulerConfig
from ls.config.dataclasses.optimizer import OptimizerConfig

@dataclass
class TrainingConfig:
    loss: str = "weighted_ce"
    use_class_weights: bool = True
    epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)