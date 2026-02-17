# Metadata-Conditioned Audio Transformers for Adaptive Respiratory Sound Classification

<!-- <p align="center">
  <a href="#"><img src="https://img.shields.io/badge/EUSIPCO-2026-blue.svg" alt="EUSIPCO 2026"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-green.svg" alt="BibTeX"></a>
  <a href="https://github.com/gkontogiannhs/Metadata-Conditioned_Audio_Transformers"><img src="https://img.shields.io/badge/GitHub-Repository-black.svg" alt="GitHub"></a>
</p> -->

<p align="center">
  <b>Official Implementation</b> of <i>"Metadata-Conditioned Audio Transformers for Adaptive Respiratory Sound Classification"</i>
  <br>
  Submitted to <b>EUSIPCO 2026</b>
</p>

<p align="center">
  <a href="mailto:g.kontogiannis@ac.upatras.gr"><b>George Kontogiannis</b></a><sup>†</sup>, 
  <a href="mailto:tzamalis@ceid.upatras.gr"><b>Pantelis Tzamalis</b></a>, 
  <a href="mailto:nikole@ceid.upatras.gr"><b>Sotiris Nikoletseas</b></a>
  <br>
  <i>Computer Engineering and Informatics Department, University of Patras, Greece</i>
  <br>
  <sup>†</sup> Corresponding author
</p>

---

## 🎯 Highlights

- **Family of metadata-conditioning mechanisms** for the Audio Spectrogram Transformer (AST), progressing from gated residual fusion to Feature-wise Linear Modulation (FiLM) adapted to the Transformer architecture.

- **Two novel conditioning variants**:
  - **Token-Aware FiLM (TAFiLM)**: Applies distinct modulation to classification and patch tokens
  - **Soft-Factorized FiLM (SoftFiLM)**: Learns soft masks to align metadata factors (device, site, demographics) with disentangled feature subspaces

- **State-of-the-art on 2-class ICBHI** (72.40%) and **competitive performance on 4-class** (64.11%, +3.94% over baseline)

- **Interpretable disentanglement**: Learned soft masks partition the feature space with <3.2% pairwise overlap (vs. 33% expected by chance)

---

## 📊 Results

Comparison with state-of-the-art methods on the ICBHI lung sound classification task using the official 60-40% train-test split.

<p align="center">
  <img src="figs/icbhi_sota_comparison.png" alt="SOTA Comparison" width="100%">
</p>

### 4-Class Evaluation

| Method | Backbone | Meta | $S_p$ (%) | $S_e$ (%) | Score (%) |
|--------|----------|:----:|-----------|-----------|-----------|
| Dong *et al.* (multi-view) | AST+Time-Domain | ✗ | **85.99** | 49.11 | 67.55* |
| MVST | AST | ✗ | 81.99 | 51.10 | 66.55 |
| Dong *et al.* (AFF+DDL) | AST | ✗ | 85.13 | 45.94 | 65.53 |
| **GRF-AST [ours]** | AST | ✓ | 74.47±1.10 | 47.91±0.85 | 61.19±0.90 |
| **FiLM-AST [ours]** | AST | ✓ | 77.26±1.25 | 47.49±0.95 | 62.37±1.05 |
| **TAFiLM-AST [ours]** | AST | ✓ | 78.84±1.30 | 48.20±1.10 | 63.52±1.20 |
| **SoftFiLM-AST [ours]** | AST | ✓ | 78.92±1.40 | **49.30±1.25** | **64.11±1.15** |

### 2-Class Evaluation (Normal vs. Abnormal)

| Method | Backbone | Meta | $S_p$ (%) | $S_e$ (%) | Score (%) |
|--------|----------|:----:|-----------|-----------|-----------|
| Fraihi *et al.* (Patch-Mix CL + FBS) | AST | ✓ | 75.17 | 65.0 | 70.08* |
| SG-SCL | AST | ✓ | 79.87 | 57.97 | 68.93 |
| Bae *et al.* (Patch-Mix CL) | AST | ✗ | 81.66 | 55.77 | 68.71 |
| **GRF-AST [ours]** | AST | ✓ | 79.10±1.20 | 61.25±0.90 | 70.17±0.74 |
| **FiLM-AST [ours]** | AST | ✓ | 75.49±1.10 | **67.45±1.30** | 71.47±1.00 |
| **TAFiLM-AST [ours]** | AST | ✓ | 77.80±1.35 | 66.40±1.25 | 71.10±0.95 |
| **SoftFiLM-AST [ours]** | AST | ✓ | 77.20±1.40 | 67.60±1.15 | **72.40±0.80** 🏆 |

> *\* denotes previous state-of-the-art*  
> Score = ($S_p$ + $S_e$) / 2

---

## 🏗️ Architecture

<p align="center">
  <img src="figures/architecture.png" alt="Architecture" width="80%">
</p>

Our proposed methods condition the Audio Spectrogram Transformer on recording metadata (device, auscultation site, patient demographics) through:

1. **Gated Residual Fusion (GRF)**: Projects metadata embeddings and adds to audio features with learnable gate
2. **FiLM**: Generates layer-wise scale (γ) and shift (β) parameters from metadata
3. **TAFiLM**: Separate FiLM parameters for CLS token vs. patch tokens
4. **SoftFiLM**: Learns soft masks to partition features into factor-aligned subspaces

---

## 🛠️ Requirements

```bash
# Create conda environment
conda create -n icbhi-ast python=3.10
conda activate icbhi-ast

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- torchaudio >= 2.0
- transformers
- timm
- librosa
- scikit-learn
- mlflow
- omegaconf
- pandas
- numpy

---

## 📁 Data Preparation

### 1. Download ICBHI Dataset

Download the ICBHI 2017 Respiratory Sound Database from the [official page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge):

```bash
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
unzip ICBHI_final_database.zip -d /path/to/icbhi_dataset
```

The dataset consists of 6,898 respiratory cycles from 126 subjects:
- 1,864 contain crackles
- 886 contain wheezes  
- 506 contain both
- 3,642 are normal

### 2. Generate Metadata File

Run the metadata creation notebook to generate the required CSV file:

```bash
jupyter notebook NBs/00-ICBHI_metadata_creation.ipynb
```

This creates `icbhi_metadata.csv` containing device, site, age, BMI, and duration for each cycle.

### 3. Update Configuration

Edit the config file to point to your data location:

```yaml
# configs/ast_config.yaml
dataset:
  name: icbhi
  data_folder: /path/to/icbhi_dataset
  cycle_metadata_path: /path/to/icbhi_dataset/icbhi_metadata.csv
  class_split: lungsound
  split_strategy: official  # official 60/40 split
  n_cls: 4                  # 4-class or 2-class
  batch_size: 16
  h: 128
  w: 1024
```

---

## 🚀 Training

### Baseline AST (no metadata)

```bash
python ast_train.py \
    --config configs/ast_config.yaml \
    --mlflow-config configs/mlflow.yaml
```

### Gated Residual Fusion (GRF-AST)

```bash
python ast_meta_fus_train.py \
    --config configs/ast_fus_config.yaml \
    --mlflow-config configs/mlflow.yaml
```

### FiLM-AST

```bash
python ast_film_train.py \
    --config configs/ast_film_config.yaml \
    --mlflow-config configs/mlflow.yaml
```

### TAFiLM-AST

```bash
python ast_tafilm_train.py \
    --config configs/ast_tafilm_config.yaml \
    --mlflow-config configs/mlflow.yaml
```

### SoftFiLM-AST

```bash
python ast_film_soft.py \
    --config configs/ast_film_soft_config.yaml \
    --mlflow-config configs/mlflow.yaml
```

---

## ⚙️ Configuration

All hyperparameters are defined in YAML config files. Key settings:

```yaml
# Model configuration
models:
  ast_softfilm:
    label_dim: 2
    input_fdim: 128
    input_tdim: 1024
    dev_embed_dim: 8        # Device embedding dimension
    site_embed_dim: 8       # Site embedding dimension
    rest_dim: 3             # Continuous features (age, BMI, duration)
    film_hidden_dim: 64     # FiLM encoder hidden dimension
    conditioned_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # All layers
    dropout: 0.4
    mask_init_scale: 2.0    # Soft mask initialization scale
    mask_overlap_lambda: 0.01  # Overlap regularization weight

# Training configuration
training:
  epochs: 100
  lr: 3e-5
  warmup_epochs: 10
  weight_decay: 0.01
  sensitivity_bias: 1.5
```

---

## 📊 Visualization

Generate analysis figures:

```bash
# Mask analysis for SoftFiLM
python visualize.py --mask-analysis --checkpoint checkpoints/softfilm_best.pt

# t-SNE comparison
python visualize.py --tsne --baseline-ckpt checkpoints/ast_best.pt --softfilm-ckpt checkpoints/softfilm_best.pt

# FiLM parameter distribution
python visualize.py --film-params --checkpoint checkpoints/softfilm_best.pt
```

---

## 🔬 Ablation Studies

### Backbone Freezing

| Trainable Blocks | Baseline | GRF | FiLM | TAFiLM | SoftFiLM |
|------------------|----------|-----|------|--------|----------|
| n=0 (head only)  | 50.04 | 46.67 | 53.09 | 54.21 | 55.03 |
| n=4              | 58.00 | 59.34 | 58.73 | 59.48 | 60.05 |
| n=8              | 59.00 | 60.58 | 60.85 | 61.73 | 62.34 |
| n=12 (all)       | **60.17** | **61.19** | **62.37** | **63.52** | **64.11** |

### Conditioned Layers (SoftFiLM, n=12)

| Conditioned Layers | Score (%) |
|--------------------|-----------|
| Last 2 {10, 11}    | 62.45 |
| Last 6 {6-11}      | 63.31 |
| All 12 {0-11}      | **64.11** |

---