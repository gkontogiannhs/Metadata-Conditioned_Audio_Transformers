import yaml
from box import Box
import os

from ls.config.dataclasses import (
    Config, DatasetConfig, AudioConfig,
    ModelsConfig, TrainingConfig, MLflowConfig
)

from ls.config.dataclasses.audio import AugmentationConfig
from ls.config.dataclasses.schedulers import SchedulerConfig
from ls.config.dataclasses.models import CNN6Config, ASTConfig
from ls.config.dataclasses.optimizer import OptimizerConfig


def load_config(path: str = "configs/config.yaml") -> Box:
    """
    Load a YAML config into a Box object for dot access.
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return Box(cfg_dict, default_box=True, box_dots=True)