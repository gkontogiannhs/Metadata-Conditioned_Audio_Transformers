import torch
import torchaudio
from torchvision import transforms as T

from ls.config.dataclasses import DatasetConfig, AudioConfig


def build_transforms(dataset_cfg: DatasetConfig, audio_cfg: AudioConfig, train: bool = True):
    """
    Build transform pipeline for log-mel features.
    Args:
        cfg: config object with dataset/audio settings
        train: whether for training (can include more transforms)
    """
    h, w = cfg.dataset.h, cfg.dataset.w  # store in cfg after dataset init
    resz = getattr(cfg.audio, "resz", 1.0)
    print(f"[Transforms] Input spectrogram resize factor: {resz}, target size: ({int(h*resz)}, {int(w*resz)})")
    tfms = []

    # Convert to tensor if not already
    # tfms.append(T.ToTensor())

    # Resize to match model input (scaled by resz if needed)
    tfms.append(T.Resize((int(h * resz), int(w * resz))))

    # Normalize spectrogram here if ast mean and std are custom computed
    # custom_mean = getattr(cfg.audio, "custom_mean", None)
    # custom_std = getattr(cfg.audio, "custom_std", None)
    # if custom_mean is not None and custom_std is not None:
    #     print(f"[Transforms] Using custom mean/std for spectrogram normalization.")
    #     tfms.append(T.Normalize(mean=[custom_mean], std=[custom_std]))

    return T.Compose(tfms)


def generate_fbank(audio: torch.Tensor, audio_cfg: AudioConfig) -> torch.Tensor:
    """
    Generate mel filterbank features for AST using torchaudio.kaldi fbank.

    Args:
        audio (torch.Tensor): Input waveform (T,) or (1, T), 16kHz mono.
        audio_cfg: Audio configuration object with parameters.

    Returns:
        torch.Tensor: Log-mel filterbank tensor of shape (time, freq, 1).
    """
    assert audio_cfg.sample_rate == 16000, "AST requires 16kHz input."
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2 and audio.shape[0] > 1:
        raise ValueError("Audio must be mono (T,) or (1, T).")

    # Kaldi fbank extraction
    fbank = torchaudio.compliance.kaldi.fbank(
        audio,
        htk_compat=True,
        sample_frequency=audio_cfg.sample_rate,
        use_energy=audio_cfg.use_energy,
        window_type=audio_cfg.window_type,
        num_mel_bins=audio_cfg.n_mels,
        dither=audio_cfg.dither,
        frame_shift=audio_cfg.frame_shift,
        frame_length=audio_cfg.frame_length,
        low_freq=audio_cfg.low_freq,
        high_freq=audio_cfg.high_freq,
    )
    
    # Normalization
    mean, std = -4.2677393, 4.5689974
    mel_norm = getattr(audio_cfg, "mel_norm", "hf").lower()

    if mel_norm == "hf":
        fbank = (fbank - mean) / std
    elif mel_norm == "mit":
        fbank = (fbank - mean) / (std * 2)
    elif mel_norm == "custom":
        custom_mean = getattr(audio_cfg, "custom_mean", None)
        custom_std = getattr(audio_cfg, "custom_std", None)
        if custom_mean is None or custom_std is None:
            raise ValueError("Custom normalization selected but no mean/std provided in audio_cfg.")
        fbank = (fbank - custom_mean) / custom_std
    elif mel_norm in ("none", "off"):
        pass  # leave raw
    else:
        raise ValueError(f"Unknown mel_norm: {mel_norm}. Use 'hf', 'mit', 'custom' or 'none'.")

    return fbank.unsqueeze(-1)  # (time, freq, 1)
