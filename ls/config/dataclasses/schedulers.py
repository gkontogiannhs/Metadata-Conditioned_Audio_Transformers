from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class SchedulerConfig:
    use: Literal["none", "cosine", "cosine_warmup", "reduce_on_plateau"] = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-8
    cosine_weight_decay: bool = True
    final_weight_decay: float = 0.0
    reduce_factor: Optional[float] = 0.5
    reduce_patience: Optional[int] = 5
    reduce_min_lr: Optional[float] = 1e-8
    reduce_metric: Optional[str] = "icbhi_score"
    reduce_mode: Optional[str] = "max"