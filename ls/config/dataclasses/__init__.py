from dataclasses import dataclass, field

from ls.config.dataclasses.dataset import DatasetConfig
from ls.config.dataclasses.audio import AudioConfig
from ls.config.dataclasses.models import ModelsConfig
from ls.config.dataclasses.training import TrainingConfig
from ls.config.dataclasses.mlflow import MLflowConfig

__all__ = [
    "Config",
    "DatasetConfig",
    "AudioConfig",
    "ModelsConfig",
    "TrainingConfig",
    "MLflowConfig",
]

@dataclass
class Config:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)