import torch
import torch.nn.functional as F
from torchvision import transforms as T
from ls.config.dataclasses import AudioConfig


def slice_data(start: float, end: float, data: torch.Tensor, sample_rate: int) -> torch.Tensor:
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return data[:, start_ind:end_ind]

def cut_pad_waveform(
    waveform: torch.Tensor,
    audio_cfg: AudioConfig,
) -> torch.Tensor:
    """
    Pad or truncate waveform to target length using audio_cfg.
    """
    target_len = int(audio_cfg.desired_length * audio_cfg.sample_rate)
    cur_len = waveform.shape[-1]

    if cur_len == 0:
        return torch.zeros((waveform.shape[0], target_len), device=waveform.device)
    if cur_len >= target_len:
        return waveform[..., :target_len]

    if audio_cfg.pad_type == "zero":
        pad_left = (target_len - cur_len) // 2
        pad_right = target_len - cur_len - pad_left
        return F.pad(waveform, (pad_left, pad_right))

    if audio_cfg.pad_type == "repeat":
        if audio_cfg.use_fade:
            fade_samples = max(1, int(audio_cfg.sample_rate / audio_cfg.fade_samples_ratio))
            fade_in_t = T.Fade(fade_in_len=fade_samples, fade_out_len=0, fade_shape="linear")
            fade_out_t = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape="linear")
            total, chunks = 0, []
            while total < target_len:
                chunk = waveform.clone()
                if total > 0 and fade_samples < cur_len:
                    chunk = fade_in_t(chunk)
                if total + cur_len < target_len and fade_samples < cur_len:
                    chunk = fade_out_t(chunk)
                chunks.append(chunk)
                total += cur_len
            return torch.cat(chunks, dim=-1)[..., :target_len]
        else:
            repeats = (target_len + cur_len - 1) // cur_len
            return waveform.repeat(1, repeats)[..., :target_len]

    raise ValueError(f"Unknown pad_type: {audio_cfg.pad_type}")