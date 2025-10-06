import yaml
from box import Box

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


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: str = "configs/base.yaml") -> Config:
    raw = load_yaml(path)
    print(raw)
    # Dataset
    dataset_cfg = DatasetConfig(**raw["dataset"])

    # Audio
    audio_raw = raw["audio"]
    wave_aug = [
        AugmentationConfig(
            type=a["type"], p=a["p"],
            params={k: v for k, v in a.items() if k not in ["type", "p"]}
        ) for a in audio_raw.get("wave_aug", [])
    ]
    spec_aug = [
        AugmentationConfig(
            type=a["type"], p=a["p"],
            params={k: v for k, v in a.items() if k not in ["type", "p"]}
        ) for a in audio_raw.get("spec_aug", [])
    ]
    audio_cfg = AudioConfig(**{k: v for k, v in audio_raw.items() if k not in ["wave_aug", "spec_aug"]},
                            wave_aug=wave_aug, spec_aug=spec_aug)

    # Models
    cnn6_cfg = CNN6Config(**raw["cnn6"])
    ast_cfg = ASTConfig(**raw["ast"])
    models_cfg = ModelsConfig(cnn6=cnn6_cfg, ast=ast_cfg)

    # ---- Training ----
    training_raw = load_yaml(raw["training"])
    optimizer_cfg = OptimizerConfig(**load_yaml(training_raw["optimizer"]))
    scheduler_cfg = SchedulerConfig(**load_yaml(training_raw["scheduler"]))
    training_cfg = TrainingConfig(
        loss=training_raw["loss"],
        use_class_weights=training_raw["use_class_weights"],
        epochs=training_raw["epochs"],
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
    )

    # ---- MLflow ----
    mlflow_cfg = MLflowConfig(**load_yaml(raw["mlflow"]))

    return Config(
        seed=raw.get("seed", 42),
        dataset=dataset_cfg,
        audio=audio_cfg,
        models=models_cfg,
        training=training_cfg,
        mlflow=mlflow_cfg,
    )