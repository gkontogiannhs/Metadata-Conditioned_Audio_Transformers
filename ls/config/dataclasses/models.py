from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class CNN6Config:
    name: str = "cnn6"
    do_dropout: bool = True
    label_dim: int = 4
    backbone_only: bool = False
    cpt_path: Optional[str] = None


@dataclass
class ASTConfig:
    name: str = "ast"
    audioset_pretrained: bool = True
    input_fdim: int = 128
    input_tdim: int = 1024
    model_size: str = "base384"
    fstride: int = 10
    tstride: int = 10
    imagenet_pretrained: bool = False
    label_dim: int = 4
    backbone_only: bool = False
    cpt_path: Optional[str] = None


@dataclass
class ModelsConfig:
    """Container for multiple model configs."""
    model1: CNN6Config = field(default_factory=CNN6Config)
    model2: ASTConfig = field(default_factory=ASTConfig)