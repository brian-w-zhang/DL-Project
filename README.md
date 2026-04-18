# Activation-Based Jailbreak Anomaly Detection

**50.039 Deep Learning, Y2026 — Sabrina Wang, Brian Zhang, Albert Nguyen-Tran**

We investigate whether frozen LLM internal activations can detect harmful and jailbreak prompts as anomalies, without fine-tuning. Using **Qwen2.5-1.5B** as a feature extractor, we train lightweight linear probes on hidden-state vectors across all 28 transformer layers on an imbalanced dataset (90:10 benign:harmful). Our best model achieves **ROC-AUC 0.971** on the internal test set and **ROC-AUC 0.889** on the held-out WildGuardTest benchmark.

**Report:** `report/report.pdf`  
**GitHub:** https://github.com/brian-w-zhang/DL-Project  
**Pre-built artifacts (features, checkpoints, data):** https://drive.google.com/drive/folders/14Yj8SMPvBkft1TrUVPoHSoe5n05mNBwx?usp=sharing

---

## Quickest path to reproducing results

The feature extraction step requires a GPU and takes ~60 minutes per pooling strategy. **To skip it**, download the pre-built artifacts from Google Drive (link above) and place them in the repo root:

```
features/          ← pre-extracted activation tensors (all 29 layers, 2 pooling strategies)
data/              ← processed train/val/test CSVs + raw HuggingFace snapshots
outputs/           ← saved checkpoints (.pt), metrics (.json), and figures (.png)
```

With those in place, jump directly to [Step 3: Train probes](#step-3-train-probes-and-baselines) and [Step 4: Run notebooks](#step-4-run-notebooks).

---

## Environment

Python 3.10+ required. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch transformers datasets pandas scikit-learn matplotlib tqdm pyarrow
```

All scripts must be run from the **repo root** (not from `src/`), as imports resolve relative to it.

---

## Dataset

**WildGuardMix** ([`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix)) — a large-scale prompt safety dataset from AI2. Access may require accepting terms on Hugging Face.

The processed splits are available pre-built on Google Drive. To rebuild from scratch:

```bash
# 1. Download the raw HuggingFace datasets to disk
python download_wildguardmix.py

# 2. Build imbalanced train/val/test splits from WildGuardTrain (90:10, 70/10/20)
python make_imbalance_split.py

# 3. Build the external test set from WildGuardTest
python process_wildguardtest.py
```

Outputs land in `data/processed/wildguardmix_imbalance/` (`train.csv`, `val.csv`, `test.csv`, `test_external.csv`).

---

## Step 1: Extract activation features  *(skip if using Drive artifacts)*

Runs Qwen2.5-1.5B (downloaded automatically from Hugging Face on first run, ~3 GB) and saves hidden-state vectors per layer. Use a GPU — extraction takes ~60 min per pooling strategy on a Colab T4, or ~40 min on Apple MPS.

```bash
# Last-token pooling, all 29 layers (used for main results)
python src/extract_features.py \
    --layers $(python -c "print(' '.join(map(str, range(29))))") \
    --pool last --batch_size 32 --max_length 256

# Mean pooling, all 29 layers (pooling ablation)
python src/extract_features.py \
    --layers $(python -c "print(' '.join(map(str, range(29))))") \
    --pool mean --batch_size 32 --max_length 256 \
    --out_slug qwen2.5-1.5b-mean
```

Outputs: `features/<slug>/layer_<n>/{X,y}_{train,val,test,test_external}.pt`

For Colab extraction, see `notebooks/colab_extract.ipynb`.

---

## Step 2: Train probes and baselines

```bash
# Text baseline (no LLM)
python src/baselines.py

# Full layer sweep — trains a linear probe on every extracted layer
python src/sweep_all_layers.py --probe_type linear

# Named probes for the main results table
python src/train_probe.py --layer 19 --probe_type linear
python src/train_probe.py --layer 14 --probe_type linear
python src/train_probe.py --layer 14 --probe_type mlp
python src/train_probe.py --layer 4  --probe_type linear
python src/train_probe.py --layer 26 --probe_type linear

# Mean-pool probes (pooling ablation)
python src/train_probe.py --layer 19 --probe_type linear --model_slug qwen2.5-1.5b-mean
python src/train_probe.py --layer 15 --probe_type linear --model_slug qwen2.5-1.5b-mean

# One-class Mahalanobis baseline
python src/one_class.py --layer 14

# Hyperparameter sweep (pos_weight × lr × weight_decay grid)
python src/hparam_sweep.py --layer 19 --probe_type linear
```

Checkpoints (`*.pt`) and metrics (`*.json`) are saved to `outputs/checkpoints/` and `outputs/metrics/`.

---

## Step 3: Run notebooks

After training is complete, run notebooks top-to-bottom to reproduce all figures and tables:

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda.ipynb` | Exploratory data analysis (class distribution, prompt length, adversarial breakdown) |
| `notebooks/02_main_experiment.ipynb` | Main results table, ROC/PR curves, confusion matrices, layer sweep, t-SNE, per-category recall, adversarial subgroup analysis, hyperparameter sweep, extended experiment (v2) |
| `notebooks/colab_extract.ipynb` | GPU feature extraction on Google Colab (reference only) |

The main experiment notebook **only reads** from `outputs/` — it does not retrain any model.

---

## Reproducing a saved model without retraining

Each checkpoint saved by `train_probe.py` includes the probe weights, normalisation statistics, and decision threshold. To reload and evaluate:

```python
import torch
from src.train_probe import load_and_evaluate   # or use eval_probe.py
```

Or via CLI:

```bash
python src/eval_probe.py --layer 19 --probe_type linear
```

The checkpoint at `outputs/checkpoints/qwen2.5-1.5b_layer19_linear.pt` is the v1 best model; v2 checkpoints follow the same format under the same directory. All checkpoints are available on Google Drive.

---

## Repository structure

```
.
├── src/
│   ├── extract_features.py   # LLM forward pass → activation tensors
│   ├── train_probe.py        # Train linear/MLP probe on saved features
│   ├── sweep_all_layers.py   # Probe training across all layers
│   ├── hparam_sweep.py       # Hyperparameter grid search
│   ├── baselines.py          # TF-IDF + Logistic Regression text baseline
│   ├── one_class.py          # Mahalanobis one-class anomaly detector
│   ├── eval_probe.py         # Reload checkpoint and re-evaluate
│   └── utils.py              # Shared paths, seeds, data loaders
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_main_experiment.ipynb
│   └── colab_extract.ipynb
├── make_imbalance_split.py   # Build train/val/test splits
├── process_wildguardtest.py  # Build external test set from WildGuardTest
├── download_wildguardmix.py  # Download raw HuggingFace dataset
├── report/report.pdf         # Final report
└── requirements.txt
```

`data/`, `features/`, and `outputs/` are gitignored. Download from Google Drive link above.
