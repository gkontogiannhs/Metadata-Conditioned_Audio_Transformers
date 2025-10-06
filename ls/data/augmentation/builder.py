from .wave import Crop, Noise, Speed, Loudness, VTLP, Pitch
from .spec import SpecAugment
from ls.config.dataclasses import AudioConfig


AUGMENT_REGISTRY = {
    "Crop": Crop,
    "Noise": Noise,
    "Speed": Speed,
    "Loudness": Loudness,
    "VTLP": VTLP,
    "Pitch": Pitch,
    "SpecAugment": SpecAugment,
}


def build_augmentations(audio_cfg: AudioConfig):
    """
    Build augmentation pipeline from config list.
    
    Args:
        audio_cfg: Audio configuration containing augmentation settings
        
    Returns:
        tuple: (waveform_augmentations, spectrogram_augmentations)
    """
    waveform_augs = getattr(audio_cfg, "wave_aug", [])
    spec_augs = getattr(audio_cfg, "spec_aug", [])
    # print(f"[Augmentations] Waveform augs: {waveform_augs}, Spec augs: {spec_augs}")

    if not waveform_augs and not spec_augs:
        return [], []

    def _build_aug_list(aug_configs):
        """Helper to build augmentation list from config."""
        augs = []
        for aug_cfg in aug_configs:
            aug_cfg = dict(aug_cfg)
            aug_type = aug_cfg.pop("type")
            
            if aug_type not in AUGMENT_REGISTRY:
                raise ValueError(f"Unknown augmentation type: {aug_type}. "
                               f"Available: {list(AUGMENT_REGISTRY.keys())}")
            
            prob = aug_cfg.get("p", 0.0)
            if 0.0 < prob < 1.0:
                aug = AUGMENT_REGISTRY[aug_type](**aug_cfg)
                augs.append(aug)
        return augs

    aug_1d = _build_aug_list(waveform_augs)
    aug_2d = _build_aug_list(spec_augs)
    
    return aug_1d, aug_2d

