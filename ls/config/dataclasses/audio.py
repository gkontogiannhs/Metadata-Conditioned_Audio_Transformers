from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class AugmentationConfig:
    type: str
    p: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioConfig:
    # core
    sample_rate: int = 16000
    desired_length: float = 10.0
    pad_type: Literal["zero", "repeat", "aug"] = "repeat"
    use_fade: bool = True
    fade_samples_ratio: int = 64

    # features
    n_mels: int = 128
    frame_length: int = 40
    frame_shift: int = 10
    low_freq: int = 100
    high_freq: int = 8000
    window_type: str = "hanning"
    use_energy: bool = False
    dither: float = 0.0
    mel_norm: Optional[str] = "mit"
    resz: float = 1.0

    # augmentations
    raw_augment: int = 0
    wave_aug: List[AugmentationConfig] = field(default_factory=list)
    spec_aug: List[AugmentationConfig] = field(default_factory=list)