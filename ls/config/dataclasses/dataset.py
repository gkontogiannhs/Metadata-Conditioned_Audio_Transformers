from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class DatasetConfig:
    name: str = "icbhi"
    data_folder: str = ""
    cycle_metadata_path: str = ""
    class_split: Literal["lungsound", "diagnosis"] = "lungsound"
    split_strategy: Literal["official", "foldwise", "random"] = "official"
    test_fold: int = 0
    n_cls: int = 4
    weighted_sampler: bool = True
    batch_size: int = 32
    num_workers: int = 0
    h: int = 128
    w: int = 1024
    test_fold: int = 0
    multi_label: bool = True
    n_cls: int = 4
    weighted_sampler: bool = True