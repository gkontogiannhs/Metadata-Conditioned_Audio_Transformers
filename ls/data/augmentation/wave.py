import random
import numpy as np
import nlpaug.augmenter.audio as naa
import torch


class Crop:
    def __init__(self, sampling_rate=16000, zone=(0.0, 1.0), coverage=1.0, p=1.0):
        self.p = p
        self.aug = naa.CropAug(sampling_rate=sampling_rate, zone=zone, coverage=coverage)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        return torch.tensor(self.aug.augment(waveform.squeeze().cpu().numpy())[0]).unsqueeze(0)


class Noise:
    def __init__(self, color="white", p=1.0):
        self.p = p
        self.aug = naa.NoiseAug(color=color)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        return torch.tensor(self.aug.augment(waveform.squeeze().cpu().numpy())[0]).unsqueeze(0)


class Speed:
    def __init__(self, factor=(0.9, 1.1), p=1.0):
        self.p = p
        self.aug = naa.SpeedAug(factor=factor)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        return torch.tensor(self.aug.augment(waveform.squeeze().cpu().numpy())[0]).unsqueeze(0)


class Loudness:
    def __init__(self, factor=(0.5, 2.0), p=1.0):
        self.p = p
        self.aug = naa.LoudnessAug(factor=factor)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        return torch.tensor(self.aug.augment(waveform.squeeze().cpu().numpy())[0]).unsqueeze(0)


class VTLP:
    def __init__(self, sampling_rate=16000, zone=(0.0, 1.0), fhi=4800, factor=(0.9, 1.1), p=1.0):
        self.p = p
        self.aug = naa.VtlpAug(sampling_rate=sampling_rate, zone=zone, fhi=fhi, factor=factor)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        augmented = self.aug.augment(waveform.squeeze().cpu().numpy())[0]
        return torch.tensor(augmented).unsqueeze(0)


class Pitch:
    def __init__(self, sampling_rate=16000, factor=(-1, 3), p=1.0):
        self.p = p
        self.aug = naa.PitchAug(sampling_rate=sampling_rate, factor=factor)

    def __call__(self, waveform: torch.Tensor):
        if random.random() >= self.p:
            return waveform
        return torch.tensor(self.aug.augment(waveform.squeeze().cpu().numpy())[0]).unsqueeze(0)
