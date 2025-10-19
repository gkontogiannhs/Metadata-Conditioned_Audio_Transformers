"""
ICBHI utilities for processing annotations, slicing recordings,
generating features, and evaluating predictions.
"""

import os
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms as T
from typing import Dict, List, Tuple

from ls.config.dataclasses import AudioConfig, DatasetConfig
from ls.data.preprocessing import slice_data


def get_annotations(data_folder: str, class_split: str) -> Dict[str, pd.DataFrame]:
    """
    Extract annotation data from ICBHI files.
    """
    if class_split in ["lungsound", "lungsound_meta", "meta"]:
        exclude_list = [
            "patient_diagnosis.txt",
            "metadata.txt",
            "official_split.txt",
            "README.txt",
            "patient_list_foldwise.txt",
            "ICBHI_challenge_train_test.txt",
            "filename_format.txt",
            "filename_differences.txt",
        ]
        filenames = [
            f.split(".")[0]
            for f in os.listdir(data_folder)
            if f.endswith(".txt") and f not in exclude_list
        ]
        return {f: _extract_lungsound_annotation(f, data_folder)[1] for f in filenames}

    if class_split == "diagnosis":
        filenames = [f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".txt")]
        diagnosis = pd.read_csv(
            os.path.join(data_folder, "patient_diagnosis.txt"),
            names=["Disease"],
            delimiter="\t",
        )
        ann_dict = {}
        for f in filenames:
            _, ann = _extract_lungsound_annotation(f, data_folder)
            ann = ann.drop(["Crackles", "Wheezes"], axis=1)
            disease = diagnosis.loc[int(f.split("_")[0]), "Disease"]
            ann["Disease"] = disease
            ann_dict[f] = ann
        return ann_dict

    return {}


def get_individual_cycles(
    recording_annotations: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    audio_cfg: AudioConfig,
    filename: str,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Split a recording into individual respiratory cycles with labels.
    """
    fpath = os.path.join(dataset_cfg.data_folder, filename + ".wav")
    data, sr = torchaudio.load(fpath)
    
    if sr != audio_cfg.sample_rate:
        data = T.Resample(sr, audio_cfg.sample_rate)(data)

    if audio_cfg.remove_dc:
        data -= data.mean()

    if audio_cfg.normalize:
        data /= data.abs().max()

    # if audio_cfg.use_fade:
    #     fade_len = int(audio_cfg.sample_rate / audio_cfg.fade_samples_ratio)
    #     fade = T.Fade(fade_in_len=fade_len, fade_out_len=fade_len, fade_shape="linear")
    #     data = fade(data)
    sample_data = []
    for _, row in recording_annotations.iterrows():
        chunk = slice_data(row["Start"], row["End"], data, audio_cfg.sample_rate)

        if dataset_cfg.class_split == "lungsound":
            label = _get_lungsound_label(row["Crackles"], row["Wheezes"], dataset_cfg.n_cls)
        else:  # diagnosis
            label = _get_diagnosis_label(row["Disease"], dataset_cfg.n_cls)
        # padded = cut_pad_waveform(chunk, audio_cfg)

        sample_data.append((chunk, label))
    return sample_data


def _extract_lungsound_annotation(file_name: str, data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tokens = file_name.strip().split("_")
    recording_info = pd.DataFrame(
        data=[tokens],
        columns=[
            "Patient Number",
            "Recording index",
            "Chest location",
            "Acquisition mode",
            "Recording equipment",
        ],
    )
    recording_annotations = pd.read_csv(
        os.path.join(data_folder, file_name + ".txt"),
        names=["Start", "End", "Crackles", "Wheezes"],
        delimiter="\t",
    )
    return recording_info, recording_annotations


def _get_lungsound_label(crackle: int, wheeze: int, n_cls: int) -> int:
    if n_cls == 4:
        if crackle == 0 and wheeze == 0:
            return 0  # normal
        elif crackle == 1 and wheeze == 0:
            return 1  # crackle
        elif crackle == 0 and wheeze == 1:
            return 2  # wheeze
        elif crackle == 1 and wheeze == 1:
            return 3  # both
    elif n_cls == 2:
        return 0 if (crackle == 0 and wheeze == 0) else 1
    raise ValueError(f"Unsupported n_cls: {n_cls}")


def _get_diagnosis_label(disease: str, n_cls: int) -> int:
    if n_cls == 3:
        if disease in ["COPD", "Bronchiectasis", "Asthma"]:
            return 1
        elif disease in ["URTI", "LRTI", "Pneumonia", "Bronchiolitis"]:
            return 2
        else:
            return 0
    elif n_cls == 2:
        return 0 if disease == "Healthy" else 1
    raise ValueError(f"Unsupported n_cls: {n_cls}")


def _convert_4class_to_multilabel(label: int) -> List[int]:
    """Map 4-class integer (0–3) to 2D binary multi-label [crackle, wheeze]."""
    mapping = {
        0: [0, 0],  # Normal
        1: [1, 0],  # Crackle
        2: [0, 1],  # Wheeze
        3: [1, 1],  # Both
    }
    return mapping.get(int(label), [0, 0])