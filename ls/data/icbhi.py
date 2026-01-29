import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import torch

from ls.config.dataclasses import DatasetConfig, AudioConfig
from ls.data.preprocessing import cut_pad_waveform
from ls.data.augmentation import build_augmentations
from ls.data.transforms import generate_fbank
from ls.data.icbhi_utils import get_annotations, get_individual_cycles, _convert_4class_to_multilabel


class ICBHIDataset(Dataset):
    """
    ICBHI Dataset with missing value handling.
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
        continuous_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.dataset_cfg = dataset_cfg
        self.audio_cfg = audio_cfg
        self.train = train
        self.augment = self.audio_cfg.raw_augment
        self.transform = transform
        self.mean_std = mean_std
        self.print_info = print_info
        self.multi_label = dataset_cfg.multi_label

        # Augmentations
        self.waveform_augs, self.spec_augs = build_augmentations(audio_cfg) if self.train and self.augment else ([], [])

        # Load cycle metadata
        self.cycle_metadata_path = getattr(self.dataset_cfg, "cycle_metadata_path", None)
        if not self.cycle_metadata_path:
            raise ValueError("dataset_cfg.cycle_metadata_path must point to your cycle metadata TSV.")
        
        self.cycle_meta, self.site2id, self.device2id = self._load_cycle_metadata_tsv(
            self.cycle_metadata_path
        )

        # Split
        all_filenames = self._list_base_filenames()
        self.split_map = self._make_split(all_filenames)
        self.filenames = [f for f in all_filenames if self.split_map.get(f) == ("train" if train else "test")]

        # Build cycles
        annotation_dict = get_annotations(self.dataset_cfg.data_folder, self.dataset_cfg.class_split)
        self.cycle_list = self._build_cycles(annotation_dict)

        # Build features
        self.samples = self._build_logmels()

        # Compute statistics for normalization
        if continuous_stats is None and train:
            # Training set: compute statistics
            self.continuous_stats = self._compute_continuous_statistics()
        elif continuous_stats is not None:
            # Test set: use provided statistics from training
            self.continuous_stats = continuous_stats
        else:
            # Fallback (shouldn't happen with proper usage)
            self.continuous_stats = self._compute_continuous_statistics()
        
        # Validate dataset
        self._validate_dataset()

        # Info
        self.h, self.w, self.c = self.samples[0]["fbank"].shape
        if self.print_info:
            print(f"\n{'='*70}")
            print(f"[ICBHI] Dataset: {'TRAIN' if train else 'TEST'}")
            print(f"{'='*70}")
            print(f"Input spectrogram shape: ({self.h}, {self.w}, {self.c})")
            print(f"Total cycles: {len(self.samples)}")
            
            # Class distribution
            if self.multi_label:
                labels_ml = np.array([s["multi_label"] for s in self.samples])
                print(f"\nMulti-label distribution:")
                print(f"  Normal (00): {((labels_ml.sum(axis=1) == 0).sum())} ({((labels_ml.sum(axis=1) == 0).sum()/len(labels_ml)*100):.1f}%)")
                print(f"  Crackle (10): {((labels_ml[:, 0] == 1) & (labels_ml[:, 1] == 0)).sum()} ({((labels_ml[:, 0] == 1) & (labels_ml[:, 1] == 0)).sum()/len(labels_ml)*100:.1f}%)")
                print(f"  Wheeze (01): {((labels_ml[:, 0] == 0) & (labels_ml[:, 1] == 1)).sum()} ({((labels_ml[:, 0] == 0) & (labels_ml[:, 1] == 1)).sum()/len(labels_ml)*100:.1f}%)")
                print(f"  Both (11): {((labels_ml[:, 0] == 1) & (labels_ml[:, 1] == 1)).sum()} ({((labels_ml[:, 0] == 1) & (labels_ml[:, 1] == 1)).sum()/len(labels_ml)*100:.1f}%)")
            else:
                class_counts = np.bincount([s["label"] for s in self.samples], minlength=self.dataset_cfg.n_cls)
                total = class_counts.sum()
                for i, c in enumerate(class_counts):
                    print(f"  Class {i}: {c} ({(100*c/total if total else 0):.1f}%)")
            
            # Continuous statistics
            print(f"\nContinuous feature statistics:")
            for feat, (mean, std) in self.continuous_stats.items():
                print(f"  {feat}: mean={mean:.2f}, std={std:.2f}")
            
            print(f"{'='*70}\n")

    # ============================================================
    # METADATA LOADING
    # ============================================================

    def _load_cycle_metadata_tsv(self, path: str):
        """Load metadata TSV with missing value handling."""
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cycle metadata TSV not found: {path}")

        df = pd.read_csv(path)

        # Validate required columns
        missing = [c for c in self.REQUIRED_META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata TSV missing columns: {missing}. Found: {list(df.columns)}")

        # Normalize types
        df["Filename"] = df["Filename"].astype(str)
        df["CycleIndex"] = df["CycleIndex"].astype(int)
        df["Device"] = df["Device"].astype(str)
        df["AuscLoc"] = df["AuscLoc"].astype(str)

        # Compute imputation values BEFORE processing rows
        age_median = df["Age"].median()
        bmi_median = df["BMI"].median()
        
        # If BMI column is mostly NaN, compute from CW/CH where possible
        if df["BMI"].isna().sum() > len(df) * 0.5:  # More than 50% missing
            print("Mostly missing BMI values - computing from CW/CH where possible...")
            computed_bmis = []
            for _, row in df.iterrows():
                if pd.notna(row["CW"]) and pd.notna(row["CH"]) and row["CH"] > 0:
                    h_m = float(row["CH"]) / 100.0
                    computed_bmi = float(row["CW"]) / (h_m * h_m)
                    computed_bmis.append(computed_bmi)
            
            if computed_bmis:
                bmi_median = np.median(computed_bmis)
        
        if pd.isna(age_median):
            age_median = 50.0  # Fallback
        if pd.isna(bmi_median):
            bmi_median = 25.0  # Fallback

        if self.print_info:
            print(f"\n[ICBHI] Imputation values:")
            print(f"  Age median: {age_median:.1f}")
            print(f"  BMI median: {bmi_median:.1f}")

        # Build metadata dictionary
        meta = {}
        
        for _, r in df.iterrows():
            # Handle Age with imputation
            if pd.isna(r["Age"]):
                age = float(age_median)
                age_missing = 1.0
            else:
                age = float(r["Age"])
                age_missing = 0.0
            
            # Handle BMI with computation or imputation
            bmi, bmi_missing = self._compute_or_impute_bmi(
                r["BMI"], r["CW"], r["CH"], bmi_median
            )
            
            key = (r["Filename"], int(r["CycleIndex"]))
            meta[key] = {
                "pid": str(r["PID"]),
                "split": str(r["Split"]),
                "fold": int(r["Fold"]),
                "device": str(r["Device"]),
                "site": str(r["AuscLoc"]),
                
                # Continuous features with missing flags
                "age": age,
                "age_missing": age_missing,
                "bmi": bmi,
                "bmi_missing": bmi_missing,
                
                "cw": float(r["CW"]) if pd.notna(r["CW"]) else None,
                "ch": float(r["CH"]) if pd.notna(r["CH"]) else None,
                "disease": str(r["Disease"]),
            }

        # Build vocabularies/vectors for categorical features (site, device)
        site2id = {s: i for i, s in enumerate(sorted(set(df["AuscLoc"].astype(str))))}
        device2id = {d: i for i, d in enumerate(sorted(set(df["Device"].astype(str))))}

        if self.print_info:
            print(f"[ICBHI] Loaded cycle metadata: {len(meta)} rows")
            print(f"[ICBHI] Sites: {len(site2id)} - {site2id}")
            print(f"[ICBHI] Devices: {len(device2id)} - {device2id}")

        return meta, site2id, device2id

    def _compute_or_impute_bmi(
        self, 
        bmi: Optional[float], 
        cw: Optional[float], 
        ch: Optional[float], 
        default_bmi: float
    ) -> Tuple[float, float]:
        """
        Compute BMI from weight/height or impute.
        
        Returns:
            (bmi_value, missing_flag)
            - bmi_value: computed or imputed BMI
            - missing_flag: 0.0 if real/computed, 1.0 if imputed
        """
        # If BMI is present, use it
        if pd.notna(bmi):
            return float(bmi), 0.0
        
        # Try to compute from CW and CH
        if pd.notna(cw) and pd.notna(ch) and ch > 0:
            cw_kg = float(cw)
            ch_cm = float(ch)
            h_m = ch_cm / 100.0
            computed_bmi = cw_kg / (h_m * h_m)
            
            # Sanity check (BMI should be reasonable)
            if 10.0 <= computed_bmi <= 60.0:
                return computed_bmi, 0.0
        
        # Otherwise, impute with median
        return float(default_bmi), 1.0

    # ============================================================
    # SPLIT METHODS
    # ============================================================

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

    # ============================================================
    # BUILD CYCLES
    # ============================================================

    def _build_cycles(self, annotation_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Build cycle list with metadata join."""
        
        cycles = []
        missing_meta = 0

        for filename in self.filenames:
            if filename not in annotation_dict:
                continue

            recording_annotations = annotation_dict[filename]
            sample_data = get_individual_cycles(
                recording_annotations, self.dataset_cfg, self.audio_cfg, filename
            )

            for cycle_index, (audio_tensor, label) in enumerate(sample_data):
                row = recording_annotations.iloc[cycle_index]
                start, end = float(row["Start"]), float(row["End"])

                # Join with metadata
                meta_key = (filename, cycle_index)
                mrow = self.cycle_meta.get(meta_key, None)
                
                if mrow is None:
                    missing_meta += 1
                    if self.print_info and missing_meta <= 5:
                        print(f"Missing metadata for {filename}, cycle {cycle_index}")
                    continue  # Skip this cycle

                # Compute duration
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

                    "crackle": int(row.get("Crackles", 0)),
                    "wheeze": int(row.get("Wheezes", 0)),

                    # From metadata TSV
                    "site": mrow["site"],
                    "device": mrow["device"],
                    "age": mrow["age"],
                    "age_missing": mrow["age_missing"],
                    "bmi": mrow["bmi"],
                    "bmi_missing": mrow["bmi_missing"],
                    "cw": mrow.get("cw", None),
                    "ch": mrow.get("ch", None),
                })

        if self.print_info:
            print(f"[ICBHI] Extracted {len(cycles)} cycles from {len(self.filenames)} recordings")
            if missing_meta > 0:
                print(f"Skipped {missing_meta} cycles due to missing metadata")

        return cycles

    def _build_logmels(self):
        """Build log-mel spectrograms for all cycles."""
        
        samples = []
        for cycle in self.cycle_list:
            waveform = cut_pad_waveform(cycle["audio"], self.audio_cfg)
            cycle["audio"] = waveform

            fbank = generate_fbank(waveform, self.audio_cfg)
            cycle["fbank"] = fbank

            if self.multi_label:
                cycle["multi_label"] = _convert_4class_to_multilabel(cycle["label"])

            samples.append(cycle)
        
        return samples

    # ============================================================
    # STATISTICS COMPUTATION
    # ============================================================

    def _compute_continuous_statistics(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean and std for continuous features from current samples.
        
        Returns:
            Dict with keys: 'age', 'bmi', 'duration'
            Each value is (mean, std)
        """
        ages = [s["age"] for s in self.samples]
        bmis = [s["bmi"] for s in self.samples]
        durations = [s["duration"] for s in self.samples]

        stats = {
            "age": (np.mean(ages), np.std(ages) + 1e-6),  # +eps to prevent division by zero
            "bmi": (np.mean(bmis), np.std(bmis) + 1e-6),
            "duration": (np.mean(durations), np.std(durations) + 1e-6),
        }

        return stats
    
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

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_dataset(self):
        """Validate that dataset has no NaN values."""
        
        print(f"\n[ICBHI] Validating dataset...")
        
        issues = []
        
        for i, sample in enumerate(self.samples):
            # Check age
            if np.isnan(sample["age"]) or np.isinf(sample["age"]):
                issues.append(f"Sample {i}: Invalid age={sample['age']}")
            
            # Check BMI
            if np.isnan(sample["bmi"]) or np.isinf(sample["bmi"]):
                issues.append(f"Sample {i}: Invalid bmi={sample['bmi']}")
            
            # Check duration
            if np.isnan(sample["duration"]) or np.isinf(sample["duration"]):
                issues.append(f"Sample {i}: Invalid duration={sample['duration']}")
        
        if issues:
            print(f"Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Print first 10
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
            raise ValueError("Dataset validation failed! Fix missing values in metadata.")
        else:
            print(f"Dataset validation passed - no NaN/Inf in continuous features")

    # ============================================================
    # DATASET PROTOCOL
    # ============================================================

    def __getitem__(self, idx):
        """Get a single sample with proper normalization and validation."""
        
        base_sample = self.samples[idx]
        sample = dict(base_sample)

        # Waveform augmentation
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

        # Spectrogram augmentation
        if self.train and self.augment and not self.mean_std:
            for aug in self.spec_augs:
                fbank = aug(fbank)

        if self.transform:
            fbank = self.transform(fbank)

        # Categorical IDs
        site_id = self.site2id[sample["site"]]
        device_id = self.device2id[sample["device"]]

        # Normalize continuous features
        age_mean, age_std = self.continuous_stats["age"]
        bmi_mean, bmi_std = self.continuous_stats["bmi"]
        dur_mean, dur_std = self.continuous_stats["duration"]

        age_norm = (float(sample["age"]) - age_mean) / age_std
        bmi_norm = (float(sample["bmi"]) - bmi_mean) / bmi_std
        dur_norm = (float(sample["duration"]) - dur_mean) / dur_std

        # Create continuous metadata vector (normalized)
        m_rest = torch.tensor([age_norm, bmi_norm, dur_norm], dtype=torch.float32)

        # Validation: Check for NaN/Inf
        if torch.isnan(m_rest).any() or torch.isinf(m_rest).any():
            print(f"NaN/Inf detected in m_rest for sample {idx}:")
            print(f"   age: {sample['age']} → {age_norm}")
            print(f"   bmi: {sample['bmi']} → {bmi_norm}")
            print(f"   duration: {sample['duration']} → {dur_norm}")
            print(f"   Stats: age=({age_mean:.2f}, {age_std:.2f}), bmi=({bmi_mean:.2f}, {bmi_std:.2f}), dur=({dur_mean:.2f}, {dur_std:.2f})")
            # Replace with zeros to prevent crash (better than NaN)
            m_rest = torch.zeros(3, dtype=torch.float32)

        out = {
            "input_values": fbank,
            "audio": sample["audio"],
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

            "age": torch.tensor(sample["age"], dtype=torch.float32),
            "bmi": torch.tensor(sample["bmi"], dtype=torch.float32),
            
            "m_rest": m_rest,  # Normalized continuous features
        }

        if self.multi_label:
            out["label"] = torch.tensor(sample["multi_label"], dtype=torch.float32)
        else:
            out["label"] = torch.tensor(sample["label"], dtype=torch.long)

        return out

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (f"{self.__class__.__name__}(train={self.train}, "
                f"n_samples={len(self)}, "
                f"input_shape=({self.h}, {self.w}), "
                f"n_wave_augs={len(self.waveform_augs)}, "
                f"n_spec_augs={len(self.spec_augs)})")