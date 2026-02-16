from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class AugmentationConfig:
    type: str
    p: float = 0.0
    sampling_rate: Optional[int] = None
    zone: Optional[List[float]] = None
    coverage: Optional[float] = None
    color: Optional[str] = None
    factor: Optional[Any] = None
    fhi: Optional[int] = None
    policy: Optional[str] = None
    mask: Optional[str] = None


@dataclass
class AudioConfig:
    # Core
    sample_rate: int = 16000
    desired_length: float = 10.0
    remove_dc: bool = True
    normalize: bool = False
    pad_type: Literal["zero", "repeat", "aug"] = "repeat"
    use_fade: bool = True
    fade_samples_ratio: int = 64

    # Features
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

    # Augmentations (use List[Dict] for compatibility with existing builder)
    raw_augment: int = 1
    wave_aug: List[Dict[str, Any]] = field(default_factory=list)
    spec_aug: List[Dict[str, Any]] = field(default_factory=list)