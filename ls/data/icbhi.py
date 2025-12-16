import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List
import torch

from ls.data.icbhi_utils import get_annotations, get_individual_cycles
from ls.data.preprocessing import cut_pad_waveform
from ls.data.augmentation import build_augmentations
from ls.data.transforms import generate_fbank
from ls.config.dataclasses import DatasetConfig, AudioConfig
from ls.data.icbhi_utils import _convert_4class_to_multilabel

import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import torch
import torchaudio
from torchaudio import transforms as T

from ls.config.dataclasses import DatasetConfig, AudioConfig
from ls.data.preprocessing import slice_data, cut_pad_waveform
from ls.data.augmentation import build_augmentations
from ls.data.transforms import generate_fbank

# ---------------------------
# Dataset with strict cycle metadata TSV
# ---------------------------
class ICBHIDataset(Dataset):
    """
    Uses a strict, user-defined cycle metadata file with columns:

      PID, Filename, CycleIndex, CycleStart, CycleEnd,
      Crackles, Wheezes, Split, Device, Fold,
      Age, Sex, BMI, CW, CH, Disease, AuscLoc

    Join key: (Filename, CycleIndex)
    - Site comes from AuscLoc
    - Device comes from Device
    - If BMI missing, compute BMI = CW / (CH/100)^2 (requires CW, CH, and Age present),
      otherwise raise.
    """

    REQUIRED_META_COLS = [
        "PID", "Filename", "CycleIndex", "CycleStart", "CycleEnd",
        "Crackles", "Wheezes", "Split", "Device", "Fold",
        "Age", "Sex", "BMI", "CW", "CH", "Disease", "AuscLoc"
    ]

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        audio_cfg: AudioConfig,
        train: bool,
        transform=None,
        mean_std: bool = False,
        print_info: bool = True,
    ):
        self.dataset_cfg = dataset_cfg
        self.audio_cfg = audio_cfg
        self.train = train
        self.augment = self.audio_cfg.raw_augment
        self.transform = transform
        self.mean_std = mean_std
        self.print_info = print_info
        self.multi_label = dataset_cfg.multi_label

        # augmentations
        self.waveform_augs, self.spec_augs = build_augmentations(audio_cfg) if self.train and self.augment else ([], [])

        # load your cycle-level metadata TSV (must be provided)
        self.cycle_metadata_path = getattr(self.dataset_cfg, "cycle_metadata_path", None)
        if not self.cycle_metadata_path:
            raise ValueError("dataset_cfg.cycle_metadata_path must point to your cycle metadata TSV.")
        self.cycle_meta, self.site2id, self.device2id = self._load_cycle_metadata_tsv(self.cycle_metadata_path)

        # 1) file list + split
        all_filenames = self._list_base_filenames()
        self.split_map = self._make_split(all_filenames)
        self.filenames = [f for f in all_filenames if self.split_map.get(f) == ("train" if train else "test")]

        # 2) annotations + cycles
        annotation_dict = get_annotations(self.dataset_cfg.data_folder, self.dataset_cfg.class_split)
        self.cycle_list = self._build_cycles(annotation_dict)

        # 3) features
        self.samples = self._build_logmels()

        # info
        self.h, self.w, self.c = self.samples[0]["fbank"].shape
        if self.print_info:
            print(f"[ICBHI] Input spectrogram shape: ({self.h}, {self.w}, {self.c})")
            print(f"[ICBHI] {len(self.samples)} cycles")
            class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
            total = class_counts.sum()
            for i, c in enumerate(class_counts):
                print(f"  Class {i}: {c} ({(100*c/total if total else 0):.1f}%)")

    # ---------------------------
    # Strict TSV loader
    # ---------------------------
    def _load_cycle_metadata_tsv(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cycle metadata TSV not found: {path}")

        df = pd.read_csv(path)

        missing = [c for c in self.REQUIRED_META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata TSV missing columns: {missing}. Found: {list(df.columns)}")

        # normalize types
        df["Filename"] = df["Filename"].astype(str)
        df["CycleIndex"] = df["CycleIndex"].astype(int)
        df["Device"] = df["Device"].astype(str)
        df["AuscLoc"] = df["AuscLoc"].astype(str)

        def _sex_to_float(v):
            s = str(v).strip().lower()
            if s in ("m", "male", "1"):
                return 1.0
            if s in ("f", "female", "0"):
                return 0.0
            raise ValueError(f"Unexpected Sex value: {v} (expected M/F or 0/1)")

        def _compute_bmi_or_impute(bmi, cw, ch, default_bmi):
            # if bmi present, use it
            if bmi is not None and not pd.isna(bmi):
                return float(bmi), 0.0  # (value, missing_flag)

            # try compute from cw/ch if both present
            if cw is not None and ch is not None and (not pd.isna(cw)) and (not pd.isna(ch)):
                cw = float(cw)     # kg
                ch = float(ch)     # cm
                h_m = ch / 100.0
                if h_m > 0:
                    return float(cw / (h_m * h_m)), 0.0

            # otherwise impute
            return float(default_bmi), 1.0

        # Build dict keyed by (Filename, CycleIndex)
        meta = {}
        bmi_series = df["BMI"].dropna().astype(float)
        default_bmi = float(bmi_series.median()) if len(bmi_series) else 0.0

        for _, r in df.iterrows():
            bmi_val, bmi_missing = _compute_bmi_or_impute(
                r["BMI"], r["CW"], r["CH"], default_bmi
            )
            key = (r["Filename"], int(r["CycleIndex"]))
            meta[key] = {
                "pid": str(r["PID"]),
                "split": str(r["Split"]),
                "fold": int(r["Fold"]),
                "device": str(r["Device"]),
                "site": str(r["AuscLoc"]),
                "age": float(r["Age"]),
                # "sex": _sex_to_float(r["Sex"]),
                "bmi": bmi_val,
                "bmi_missing": float(bmi_missing),
                "cw": float(r["CW"]) if not pd.isna(r["CW"]) else None,
                "ch": float(r["CH"]) if not pd.isna(r["CH"]) else None,
                "disease": str(r["Disease"]),
            }

        # vocab from TSV (stable & no guessing)
        site2id = {"<UNK>": 0}
        for s in sorted(set(df["AuscLoc"].astype(str).tolist())):
            site2id[s] = len(site2id)

        device2id = {"<UNK>": 0}
        for d in sorted(set(df["Device"].astype(str).tolist())):
            device2id[d] = len(device2id)

        if self.print_info:
            print(f"[ICBHI] Loaded cycle metadata TSV: {len(meta)} rows")
            print(f"[ICBHI] #Sites={len(site2id)-1}, #Devices={len(device2id)-1}")

        return meta, site2id, device2id

    # ---------------------------
    # Split helpers (unchanged)
    # ---------------------------
    def _list_base_filenames(self) -> List[str]:
        items = os.listdir(self.dataset_cfg.data_folder)
        bases_wav = {f.split(".")[0] for f in items if f.endswith(".wav")}
        bases_txt = {f.split(".")[0] for f in items if f.endswith(".txt")}
        return sorted(bases_wav & bases_txt)

    def _make_split(self, filenames: List[str]) -> Dict[str, str]:
        strategy = str(self.dataset_cfg.split_strategy).lower()
        if strategy == "official":
            return self._split_official(os.path.join(self.dataset_cfg.data_folder, "official_split.txt"))
        elif strategy == "foldwise":
            fold_id = int(self.dataset_cfg.test_fold)
            return self._split_foldwise(filenames, fold_id, os.path.join(self.dataset_cfg.data_folder, "patient_list_foldwise.txt"))
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
            split_map[fn] = "test" if (pfold == fold_id) else "train"
        return split_map

    def _split_random(self, filenames: List[str], ratio: float = 0.6, seed: int = 1) -> Dict[str, str]:
        rng = random.Random(seed)
        shuffled = filenames.copy()
        rng.shuffle(shuffled)
        train_size = int(len(shuffled) * ratio)
        return {fn: ("train" if i < train_size else "test") for i, fn in enumerate(shuffled)}

    # ---------------------------
    # Build cycles with TSV join
    # ---------------------------
    def _build_cycles(self, annotation_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        cycles = []
        missing_meta = 0

        for filename in self.filenames:
            if filename not in annotation_dict:
                continue

            recording_annotations = annotation_dict[filename]
            sample_data = get_individual_cycles(recording_annotations, self.dataset_cfg, self.audio_cfg, filename)

            for cycle_index, (audio_tensor, label) in enumerate(sample_data):
                row = recording_annotations.iloc[cycle_index]
                start, end = float(row["Start"]), float(row["End"])

                meta_key = (filename, cycle_index)
                mrow = self.cycle_meta.get(meta_key, None)
                if mrow is None:
                    missing_meta += 1
                    # you asked strictness mostly for BMI; join missing is likely a bug
                    raise KeyError(f"Missing metadata for key (Filename={filename}, CycleIndex={cycle_index})")

                # duration from annotation (seconds)
                duration = float(np.round(end - start, 3))

                cycles.append({
                    "audio": audio_tensor,
                    "label": label,
                    "filename": filename,
                    "cycle_index": cycle_index,

                    "pid": mrow["pid"],
                    "duration": duration,
                    "start_time": start,
                    "end_time": end,

                    "crackle": int(row.get("Crackles")),
                    "wheeze": int(row.get("Wheezes")),

                    # from TSV
                    "site": mrow["site"],       # AuscLoc
                    "device": mrow["device"],
                    "age": mrow["age"],
                    # "sex": mrow["sex"],         # 0/1 float
                    "bmi": mrow["bmi"],
                    "cw": mrow.get("cw", None),
                    "ch": mrow.get("ch", None),
                })

        if self.print_info:
            print(f"[ICBHI] Extracted {len(cycles)} cycles from {len(self.filenames)} recordings")
            print(f"[ICBHI] Metadata join missing: {missing_meta} (strict join; should be 0)")
        return cycles

    def _build_logmels(self):
        samples = []
        for cycle in self.cycle_list:
            waveform = cut_pad_waveform(cycle["audio"], self.audio_cfg)
            cycle["audio"] = waveform

            fbank = generate_fbank(waveform, self.audio_cfg)  # (T, F, 1)
            cycle["fbank"] = fbank

            if self.multi_label:
                cycle["multi_label"] = _convert_4class_to_multilabel(cycle["label"])

            samples.append(cycle)
        return samples
    
    def get_sample_weights(self):
        """
        Compute per-sample weights dynamically based on current mode.
        - In multi-class mode → inverse class frequency
        - In multi-label mode → inverse label frequency
        """
        if not self.train:
            return None

        if not self.multi_label:
            # Original 4-class weighting
            class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
            inv_freq = 1.0 / (class_counts + 1e-6)
            weights = [inv_freq[s["label"]] for s in self.samples]
            return np.array(weights) / np.sum(weights)
        else:
            # Multi-label weighting
            labels_ml = np.array([s["multi_label"] for s in self.samples])
            label_freq = labels_ml.sum(axis=0).astype(float)
            inv_freq = 1.0 / (label_freq + 1e-6)
            inv_freq = inv_freq / inv_freq.sum()

            sample_weights = []
            for vec in labels_ml:
                if vec.sum() == 0:  # Normal
                    w = min(inv_freq) * 0.5
                else:
                    w = np.sum(inv_freq[vec == 1])
                sample_weights.append(w)

            return np.array(sample_weights) / np.sum(sample_weights)
        
    # ---------------------------
    # Dataset protocol
    # ---------------------------
    def __getitem__(self, idx):
        base_sample = self.samples[idx]
        sample = dict(base_sample)

        # waveform aug
        if self.train and self.augment and not self.mean_std:
            waveform = sample["audio"].clone()
            for aug in self.waveform_augs:
                waveform = aug(waveform)
            waveform = cut_pad_waveform(waveform, self.audio_cfg)
            fbank = generate_fbank(waveform, self.audio_cfg)
        else:
            fbank = sample["fbank"]

        # (T, F, 1) → (1, F, T)
        fbank = fbank.permute(2, 1, 0)

        # spec aug
        if self.train and self.augment and not self.mean_std:
            for aug in self.spec_augs:
                fbank = aug(fbank)

        if self.transform:
            fbank = self.transform(fbank)

        # categorical ids
        site_id = self.site2id.get(sample["site"], 0)
        device_id = self.device2id.get(sample["device"], 0)

        # continuous (NO fitting; raw values)
        age = float(sample["age"])
        bmi = float(sample["bmi"])
        # dur = float(sample["duration"])
        # sex = float(sample["sex"])  # already 0/1

        # cont_feats = torch.tensor([age, bmi, dur], dtype=torch.float32)
        # m_rest = torch.tensor([sex, age, bmi, dur], dtype=torch.float32)  # FiLM++ rest vector

        out = {
            "input_values": fbank,  # (1, F, T)
            "audio": sample["audio"],  # (1, N)
            "filename": sample["filename"],
            "cycle_index": sample["cycle_index"],
            "pid": sample["pid"],

            "duration": sample["duration"],
            "start_time": sample["start_time"],
            "end_time": sample["end_time"],

            "site": sample["site"],
            "device": sample["device"],
            "site_id": torch.tensor(site_id, dtype=torch.long),
            "device_id": torch.tensor(device_id, dtype=torch.long),

            "age": torch.tensor(age, dtype=torch.float32),
            "bmi": torch.tensor(bmi, dtype=torch.float32),

            # "sex": torch.tensor(sex, dtype=torch.float32),
            # "cont_feats": cont_feats,  # (3,)
            # "m_rest": m_rest,          # (4,)
        }

        if self.multi_label:
            out["label"] = torch.tensor(sample["multi_label"], dtype=torch.float32)
        else:
            out["label"] = sample["label"]

        return out

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (f"{self.__class__.__name__}(train={self.train}, "
                f"n_samples={len(self)}, "
                f"input_shape=({self.h}, {self.w}), "
                f"n_wave_augs={len(self.waveform_augs)}, "
                f"n_spec_augs={len(self.spec_augs)})")


# class ICBHIDataset(Dataset):
#     def __init__(
#         self, 
#         dataset_cfg: DatasetConfig,
#         audio_cfg: AudioConfig,
#         train: bool, 
#         transform=None, 
#         mean_std: bool = False, 
#         print_info: bool = True
#     ):
#         self.dataset_cfg = dataset_cfg
#         self.audio_cfg = audio_cfg
#         self.train = train
#         self.augment = self.audio_cfg.raw_augment
#         self.transform = transform
#         self.mean_std = mean_std
#         self.print_info = print_info
#         self.multi_label = dataset_cfg.multi_label
        
#         # build augmentation pipelines from config
#         self.waveform_augs, self.spec_augs = build_augmentations(audio_cfg) if self.train and self.augment else ([], [])

#         # 1. Collect files & split
#         # TODO: Can be optimized to avoid listing files twice
#         all_filenames = self._list_base_filenames()
#         self.split_map = self._make_split(all_filenames)
#         self.filenames = [f for f in all_filenames if self.split_map.get(f) == ("train" if train else "test")]

#         # 2. Load annotations & cycles
#         annotation_dict = get_annotations(self.dataset_cfg.data_folder, self.dataset_cfg.class_split)
#         self.cycle_list = self._build_cycles(annotation_dict)

#         # 3. Build log-mel spectrograms (original + augments if any)
#         self.samples = self._build_logmels()

#         # Input shape
#         self.h, self.w, self.c = self.samples[0]["fbank"].shape
#         print(f"[ICBHI] Input spectrogram shape: ({self.h}, {self.w}, {self.c})")

#         if print_info:
#             print(f"[ICBHI] {len(self.samples)} cycles (base cycles only, aug handled dynamically)")

#             self.class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
#             total = self.class_counts.sum()
#             for i, c in enumerate(self.class_counts):
#                 ratio = 100 * c / total
#                 print(f"  Class {i}: {c} ({ratio:.1f}%)")
#             if self.waveform_augs and self.augment:
#                 wf_list = [f"{aug.__class__.__name__}" for aug in self.waveform_augs]
#                 print(f"[ICBHI] Active waveform augmentations: {', '.join(wf_list)}")
#             else:
#                 print("[ICBHI] No waveform augmentations")

#             if self.spec_augs and self.augment:
#                 spec_list = [f"{aug.__class__.__name__}" for aug in self.spec_augs]
#                 print(f"[ICBHI] Active spectrogram augmentations: {', '.join(spec_list)}")
#             else:
#                 print("[ICBHI] No spectrogram augmentations")
#             if self.train and self.augment:
#                 self.expected_p_aug = self._expected_aug_prob()
#                 n = len(self.cycle_list)
#                 print(f"[ICBHI] Expected P(augmented) ≈ {self.expected_p_aug:.2f} "
#                     f"(~{int(round(self.expected_p_aug*n))}/{n} per epoch)")

#     # ---------------------------
#     # Split helpers
#     # ---------------------------
#     def _list_base_filenames(self) -> List[str]:
#         items = os.listdir(self.dataset_cfg.data_folder)
#         bases_wav = {f.split(".")[0] for f in items if f.endswith(".wav")}
#         bases_txt = {f.split(".")[0] for f in items if f.endswith(".txt")}
#         return sorted(bases_wav & bases_txt)

#     def _make_split(self, filenames: List[str]) -> Dict[str, str]:
#         strategy = str(self.dataset_cfg.split_strategy).lower()
#         if strategy == "official":
#             return self._split_official(split_file=os.path.join(
#                 self.dataset_cfg.data_folder, "official_split.txt"))
#         elif strategy == "foldwise":
#             fold_id = int(self.dataset_cfg.test_fold)
#             return self._split_foldwise(
#                 filenames, fold_id,
#                 fold_file=os.path.join(self.dataset_cfg.data_folder, "patient_list_foldwise.txt"))
#         elif strategy == "random":
#             return self._split_random(filenames, ratio=0.6)
#         else:
#             raise ValueError(f"Unknown split_strategy: {self.dataset_cfg.split_strategy}")

#     def _split_official(self, split_file: str) -> Dict[str, str]:
#         split_map = {}
#         with open(split_file, "r") as f:
#             for line in f:
#                 fname, split = line.strip().split()
#                 split_map[fname] = split
#         return split_map

#     def _split_foldwise(self, filenames: List[str], fold_id: int, fold_file: str) -> Dict[str, str]:
#         patient_to_fold = {}
#         with open(fold_file, "r") as f:
#             for line in f:
#                 pid, fold = line.strip().split()
#                 patient_to_fold[int(pid)] = int(fold)

#         split_map = {}
#         for fn in filenames:
#             pid = int(fn.split("_")[0])
#             pfold = patient_to_fold.get(pid, None)
#             if pfold is None:
#                 split_map[fn] = "train"
#             else:
#                 split_map[fn] = "test" if pfold == fold_id else "train"
#         return split_map

#     def _split_random(self, filenames: List[str], ratio: float = 0.6, seed: int = 1) -> Dict[str, str]:
#         rng = random.Random(seed)
#         shuffled = filenames.copy()
#         rng.shuffle(shuffled)
#         train_size = int(len(shuffled) * ratio)
#         return {fn: ("train" if i < train_size else "test")for i, fn in enumerate(shuffled)}

#     # ---------------------------
#     # Cycle & variants
#     # ---------------------------
#     def _build_cycles(self, annotation_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
#         cycles = []
#         for filename in self.filenames:
#             if filename not in annotation_dict:
#                 continue
#             recording_annotations = annotation_dict[filename]
#             sample_data = get_individual_cycles(
#                 recording_annotations, self.dataset_cfg, self.audio_cfg, filename
#             )
#             for cycle_index, (audio_tensor, label) in enumerate(sample_data):
#                 row = recording_annotations.iloc[cycle_index]
#                 start, end = row["Start"], row["End"]
#                 crackles, wheezes = row.get("Crackles"), row.get("Wheezes")
#                 site, device = row["Site"], row["Device"]
#                 cycles.append({
#                     "audio": audio_tensor,
#                     "label": label,
#                     "filename": filename,
#                     "cycle_index": cycle_index,
#                     "duration": float(np.round(end - start, 3)),
#                     "start_time": float(start),
#                     "end_time": float(end),
#                     "crackle": int(crackles) if crackles is not None else None,
#                     "wheeze": int(wheezes) if wheezes is not None else None,
#                     "site": site,
#                     "device": device,
#                 })
#         if self.print_info:
#             print(f"[ICBHI] Extracted {len(cycles)} respiratory cycles "
#                   f"from {len(self.filenames)} recordings")
#         return cycles

#     def _build_logmels(self):
#         samples = []
#         for cycle in self.cycle_list:
#             # cycle["audio"] contains the original (non-preprocessed) waveform
#             waveform = cut_pad_waveform(cycle["audio"], self.audio_cfg)
#             cycle["audio"] = waveform

#             # Only the base feature (no augmentations here)
#             fbank = generate_fbank(waveform, self.audio_cfg)
#             cycle["fbank"] = fbank

#             if self.multi_label:
#                 cycle["multi_label"] = _convert_4class_to_multilabel(cycle["label"])

#             samples.append(cycle)
#         return samples
    
#     def _expected_aug_prob(self):
#         """Expected P(sample augmented) assuming independent aug decisions."""
#         def safe_p(a): 
#             return float(getattr(a, "p", 0.0))
#         no_wf = 1.0
#         for a in self.waveform_augs:
#             no_wf *= (1.0 - safe_p(a))
#         no_spec = 1.0
#         for a in self.spec_augs:
#             no_spec *= (1.0 - safe_p(a))
#         return 1.0 - (no_wf * no_spec)
    
#     def get_sample_weights(self):
#         """
#         Compute per-sample weights dynamically based on current mode.
#         - In multi-class mode → inverse class frequency
#         - In multi-label mode → inverse label frequency
#         """
#         if not self.train:
#             return None

#         if not self.multi_label:
#             # Original 4-class weighting
#             class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
#             inv_freq = 1.0 / (class_counts + 1e-6)
#             weights = [inv_freq[s["label"]] for s in self.samples]
#             return np.array(weights) / np.sum(weights)
#         else:
#             # Multi-label weighting
#             labels_ml = np.array([s["multi_label"] for s in self.samples])
#             label_freq = labels_ml.sum(axis=0).astype(float)
#             inv_freq = 1.0 / (label_freq + 1e-6)
#             inv_freq = inv_freq / inv_freq.sum()

#             sample_weights = []
#             for vec in labels_ml:
#                 if vec.sum() == 0:  # Normal
#                     w = min(inv_freq) * 0.5
#                 else:
#                     w = np.sum(inv_freq[vec == 1])
#                 sample_weights.append(w)

#             return np.array(sample_weights) / np.sum(sample_weights)

#     def __getitem__(self, idx):
#         """
#         Fetch a single sample.
#         Steps:
#         1. Get stored waveform + fbank (precomputed base).
#         2. If training & augmenting: apply waveform augmentations → recompute fbank.
#         3. Permute to (1, F, T) for AST/torchvision.
#         4. If training & augmenting: apply spectrogram augmentations.
#         5. Apply any final transform (e.g. normalization).
#         """
#         base_sample = self.samples[idx]
#         # Copy to avoid in-place mutation (important for DataLoader workers)
#         sample = dict(base_sample)

#         # ========== Waveform processing ==========
#         if self.train and self.augment and not self.mean_std:
#             waveform = sample["audio"].clone()
#             for aug in self.waveform_augs:
#                 waveform = aug(waveform)

#             # Pad/cut again in case augmentations changed length
#             waveform = cut_pad_waveform(waveform, self.audio_cfg)
#             sample["aug_audio"] = waveform  # keep for debugging

#             # Recompute fbank after waveform aug
#             fbank = generate_fbank(waveform, self.audio_cfg)  # (T, F, 1)
#         else:
#             fbank = sample["fbank"]

#         # Reorder dimensions
#         # (T, F, 1) → (1, F, T)
#         fbank = fbank.permute(2, 1, 0)

#         # Spectrogram augmentations
#         if self.train and self.augment and not self.mean_std:
#             for aug in self.spec_augs:  # SpecAug expects (1, F, T)
#                 fbank = aug(fbank)
#             sample["aug_fbank"] = fbank  # keep for debugging

#         # Final transform
#         if self.transform:
#             fbank = self.transform(fbank)

#         out = {
#             "input_values": fbank,   # (1, F, T) -> (1, T, F) for AST
#             "audio": sample["audio"],
#             "filename": sample["filename"],
#             "cycle_index": sample["cycle_index"],
#             "duration": sample["duration"],
#             "start_time": sample["start_time"],
#             "end_time": sample["end_time"],
#             "crackle": sample["crackle"],
#             "wheeze": sample["wheeze"],
#             "site": sample["site"],
#             "device": sample["device"],
#         }

#         if self.multi_label:
#             out["label"] = torch.tensor(sample["multi_label"], dtype=torch.float32)
#         else:
#             out["label"] = sample["label"]

#         return out

#     def __len__(self):
#         return len(self.samples)
    
#     def __repr__(self):
#         return (f"{self.__class__.__name__}(train={self.train}, "
#                 f"n_samples={len(self)}, "
#                 f"input_shape=({self.h}, {self.w}), "
#                 f"n_wave_augs={len(self.waveform_augs)}, "
#                 f"n_spec_augs={len(self.spec_augs)})")

