import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List

from ls.data.icbhi_utils import get_annotations, get_individual_cycles
from ls.data.preprocessing import cut_pad_waveform
from ls.data.augmentation import build_augmentations
from ls.data.transforms import generate_fbank
from ls.config.dataclasses import DatasetConfig, AudioConfig


class ICBHIDataset(Dataset):
    def __init__(
        self, 
        dataset_cfg: DatasetConfig,
        audio_cfg: AudioConfig,
        train: bool, 
        transform=None, 
        mean_std: bool = False, 
        print_info: bool = True
    ):
        self.dataset_cfg = dataset_cfg
        self.audio_cfg = audio_cfg
        self.train = train
        self.augment = self.audio_cfg.raw_augment
        self.transform = transform
        self.mean_std = mean_std
        self.print_info = print_info
        
        # build augmentation pipelines from config
        self.waveform_augs, self.spec_augs = build_augmentations(audio_cfg) if self.train and self.augment else ([], [])

        # 1. Collect files & split
        # TODO: Can be optimized to avoid listing files twice
        all_filenames = self._list_base_filenames()
        self.split_map = self._make_split(all_filenames)
        self.filenames = [f for f in all_filenames if self.split_map.get(f) == ("train" if train else "test")]

        # 2. Load annotations & cycles
        annotation_dict = get_annotations(self.dataset_cfg.data_folder, self.dataset_cfg.class_split)
        self.cycle_list = self._build_cycles(annotation_dict)

        # 3. Build log-mel spectrograms (original + augments if any)
        self.samples = self._build_logmels()

        # Input shape
        self.h, self.w, self.c = self.samples[0]["fbank"].shape
        print(f"[ICBHI] Input spectrogram shape: ({self.h}, {self.w}, {self.c})")

        if print_info:
            print(f"[ICBHI] {len(self.samples)} cycles (base cycles only, aug handled dynamically)")

            self.class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
            total = self.class_counts.sum()
            for i, c in enumerate(self.class_counts):
                ratio = 100 * c / total
                print(f"  Class {i}: {c} ({ratio:.1f}%)")
            if self.waveform_augs and self.augment:
                wf_list = [f"{aug.__class__.__name__}" for aug in self.waveform_augs]
                print(f"[ICBHI] Active waveform augmentations: {', '.join(wf_list)}")
            else:
                print("[ICBHI] No waveform augmentations")

            if self.spec_augs and self.augment:
                spec_list = [f"{aug.__class__.__name__}" for aug in self.spec_augs]
                print(f"[ICBHI] Active spectrogram augmentations: {', '.join(spec_list)}")
            else:
                print("[ICBHI] No spectrogram augmentations")
            if self.train and self.augment:
                self.expected_p_aug = self._expected_aug_prob()
                n = len(self.cycle_list)
                print(f"[ICBHI] Expected P(augmented) ≈ {self.expected_p_aug:.2f} "
                    f"(~{int(round(self.expected_p_aug*n))}/{n} per epoch)")

    # ---------------------------
    # Split helpers
    # ---------------------------
    def _list_base_filenames(self) -> List[str]:
        items = os.listdir(self.dataset_cfg.data_folder)
        bases_wav = {f.split(".")[0] for f in items if f.endswith(".wav")}
        bases_txt = {f.split(".")[0] for f in items if f.endswith(".txt")}
        return sorted(bases_wav & bases_txt)

    def _make_split(self, filenames: List[str]) -> Dict[str, str]:
        strategy = str(self.dataset_cfg.split_strategy).lower()
        if strategy == "official":
            return self._split_official(split_file=os.path.join(
                self.dataset_cfg.data_folder, "official_split.txt"))
        elif strategy == "foldwise":
            fold_id = int(self.dataset_cfg.test_fold)
            return self._split_foldwise(
                filenames, fold_id,
                fold_file=os.path.join(self.dataset_cfg.data_folder, "patient_list_foldwise.txt"))
        elif strategy == "random":
            return self._split_random(filenames, ratio=0.6)
        else:
            raise ValueError(f"Unknown split_strategy: {self.dataset_cfg.split_strategy}")

    def _split_official(self, split_file: str) -> Dict[str, str]:
        split_map = {}
        with open(split_file, "r") as f:
            for line in f:
                fname, split = line.strip().split()
                split_map[fname] = split
        return split_map

    def _split_foldwise(self, filenames: List[str], fold_id: int, fold_file: str) -> Dict[str, str]:
        patient_to_fold = {}
        with open(fold_file, "r") as f:
            for line in f:
                pid, fold = line.strip().split()
                patient_to_fold[int(pid)] = int(fold)

        split_map = {}
        for fn in filenames:
            pid = int(fn.split("_")[0])
            pfold = patient_to_fold.get(pid, None)
            if pfold is None:
                split_map[fn] = "train"
            else:
                split_map[fn] = "test" if pfold == fold_id else "train"
        return split_map

    def _split_random(self, filenames: List[str], ratio: float = 0.6, seed: int = 1) -> Dict[str, str]:
        rng = random.Random(seed)
        shuffled = filenames.copy()
        rng.shuffle(shuffled)
        train_size = int(len(shuffled) * ratio)
        return {fn: ("train" if i < train_size else "test")for i, fn in enumerate(shuffled)}

    # ---------------------------
    # Cycle & variants
    # ---------------------------
    def _build_cycles(self, annotation_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        cycles = []
        for filename in self.filenames:
            if filename not in annotation_dict:
                continue
            recording_annotations = annotation_dict[filename]
            sample_data = get_individual_cycles(
                recording_annotations, self.dataset_cfg, self.audio_cfg, filename
            )
            for cycle_index, (audio_tensor, label) in enumerate(sample_data):
                row = recording_annotations.iloc[cycle_index]
                start, end = row["Start"], row["End"]
                crackles, wheezes = row.get("Crackles", None), row.get("Wheezes", None)
                cycles.append({
                    "audio": audio_tensor,
                    "label": label,
                    "filename": filename,
                    "cycle_index": cycle_index,
                    "duration": float(np.round(end - start, 3)),
                    "start_time": float(start),
                    "end_time": float(end),
                    "crackle": int(crackles) if crackles is not None else None,
                    "wheeze": int(wheezes) if wheezes is not None else None,
                })
        if self.print_info:
            print(f"[ICBHI] Extracted {len(cycles)} respiratory cycles "
                  f"from {len(self.filenames)} recordings")
        return cycles

    def _build_logmels(self):
        samples = []
        for cycle in self.cycle_list:
            # cycle["audio"] contains the original (non-preprocessed) waveform
            waveform = cut_pad_waveform(cycle["audio"], self.audio_cfg)
            cycle["audio"] = waveform

            # Only the base feature (no augmentations here)
            fbank = generate_fbank(waveform, self.audio_cfg)
            cycle["fbank"] = fbank

            samples.append(cycle)
        return samples
    
    def _expected_aug_prob(self):
        """Expected P(sample augmented) assuming independent aug decisions."""
        def safe_p(a): 
            return float(getattr(a, "p", 0.0))
        no_wf = 1.0
        for a in self.waveform_augs:
            no_wf *= (1.0 - safe_p(a))
        no_spec = 1.0
        for a in self.spec_augs:
            no_spec *= (1.0 - safe_p(a))
        return 1.0 - (no_wf * no_spec)

    def __getitem__(self, idx):
        """
        Fetch a single sample.
        Steps:
        1. Get stored waveform + fbank (precomputed base).
        2. If training & augmenting: apply waveform augmentations → recompute fbank.
        3. Permute to (1, F, T) for AST/torchvision.
        4. If training & augmenting: apply spectrogram augmentations.
        5. Apply any final transform (e.g. normalization).
        """
        base_sample = self.samples[idx]
        # Copy to avoid in-place mutation (important for DataLoader workers)
        sample = dict(base_sample)

        # ========== Waveform processing ==========
        if self.train and self.augment and not self.mean_std:
            waveform = sample["audio"].clone()
            for aug in self.waveform_augs:
                waveform = aug(waveform)

            # Pad/cut again in case augmentations changed length
            waveform = cut_pad_waveform(waveform, self.audio_cfg)
            sample["aug_audio"] = waveform  # keep for debugging

            # Recompute fbank after waveform aug
            fbank = generate_fbank(waveform, self.audio_cfg)  # (T, F, 1)
        else:
            fbank = sample["fbank"]

        # Reorder dimensions
        # (T, F, 1) → (1, F, T)
        fbank = fbank.permute(2, 1, 0)

        # Spectrogram augmentations
        if self.train and self.augment and not self.mean_std:
            for aug in self.spec_augs:  # SpecAug expects (1, F, T)
                fbank = aug(fbank)
            sample["aug_fbank"] = fbank  # keep for debugging

        # Final transform
        if self.transform:
            fbank = self.transform(fbank)

        # Normal training retur
        return {
            "input_values": fbank, # .permute(0, 2, 1).squeeze(0),  # (1, F, T) -> (1, T, F) for AST
            "labels": sample["label"],
            **sample  # keep metadata (filename, crackle, etc.)
        }

    def __len__(self):
        return len(self.samples)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(train={self.train}, "
                f"n_samples={len(self)}, "
                f"input_shape=({self.h}, {self.w}), "
                f"n_wave_augs={len(self.waveform_augs)}, "
                f"n_spec_augs={len(self.spec_augs)})")

