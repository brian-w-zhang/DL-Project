# Deep Learning Project

Code and materials for an activation-probe study on harmful vs benign prompts (frozen **Qwen2.5-1.5B**). The write-up lives in `report/report.tex`; raw numbers and figures are under `outputs/`.

---

## For teammates (catch up quickly)

**If someone shares pre-built artifacts on Drive** (recommended so you skip slow steps):

1. Clone this repo and create the venv (see [Environment](#environment)).
2. Download and merge into the project root so paths match:
   - **`features/`** — pre-extracted activation tensors (`qwen2.5-1.5b/layer_*`, etc.). With this in place you **do not** need to run `extract_features.py` or download the base LM for extraction.
   - **`data/processed/wildguardmix_imbalance/`** — optional but useful for EDA / baselines / reproducing CSV-based steps (`baselines.py`, `01_eda.ipynb`).
   - **`outputs/`** — optional; if included, notebooks can show numbers immediately. If missing, generate metrics by running the scripts below.

**What to run when `features/` for all layers is already there:**

```bash
python src/baselines.py
python src/sweep_all_layers.py --probe_type linear
python src/train_probe.py --layer 19 --probe_type linear
python src/train_probe.py --layer 14 --probe_type mlp
python src/one_class.py --layer 14
```

- **`sweep_all_layers.py`** trains a probe on **every** layer folder under `features/qwen2.5-1.5b/` (the full depth curve). Use this instead of calling `train_probe.py` once per layer by hand.
- **`train_probe.py`** is still needed for **standard result filenames** the notebook expects for the main table (e.g. `qwen2.5-1.5b_layer19_linear.json`), and for one-off layers / MLP. Sweep saves files named `*_layer{N}_linear_sweep.json`; the summary table loads names **without** `_sweep`.
- You **do not** need only layer 4 — that was a minimal demo. With full features, the sweep **is** the “train on all layers” step.

First-time Hugging Face download is only required if you run **`extract_features.py`** yourself.

---

## What the main scripts do

| Script | Role |
|--------|------|
| **`baselines.py`** | **Text-only baseline:** TF‑IDF features + sklearn logistic regression on raw prompts. No LLM. Writes metrics to `outputs/metrics/baseline_tfidf.json`. |
| **`train_probe.py`** | **Activation probe:** trains a tiny **linear** or **MLP** classifier on vectors loaded from `features/<slug>/layer_<L>/`. Uses frozen LM activations already saved as `.pt` files — it does **not** run the LM. Saves a **checkpoint** under `outputs/checkpoints/` (e.g. `qwen2.5-1.5b_layer19_linear.pt`) with probe weights, train-set mean/std for normalization, and threshold, plus matching **metrics** JSON in `outputs/metrics/`. |
| **`sweep_all_layers.py`** | Same probe training as `train_probe`, but loops **all** extracted layers and writes sweep plot + `full_sweep_linear.json`. Checkpoints/metrics use a `_sweep` suffix so they do not overwrite single-layer runs. |
| **`extract_features.py`** | Runs **Qwen2.5-1.5B** once per batch, pools hidden states (last/mean token), saves `X_*.pt` per layer. Heavy; skip if you use shared `features/`. |
| **`one_class.py`** | Scores test prompts with **Mahalanobis distance** to normal-only training activations (no harmful labels in training). |
| **`make_imbalance_split.py`** | Builds stratified CSV splits from the raw HF dataset. Skip if you use shared `data/processed/`. |

---

## Dataset

Download assets from Google Drive:

- [DL Project Dataset](https://drive.google.com/drive/folders/14Yj8SMPvBkft1TrUVPoHSoe5n05mNBwx?usp=sharing)

Place files under `data/` as needed. `data/` is gitignored.

**WildGuardMix-style setup:** scripts expect an on-disk Hugging Face dataset for building splits (default path: `data/raw/wildguardmix/wildguardtrain_hf`). See `make_imbalance_split.py` and `WILDGUARDMIX_DATASET_REFERENCE.md` if you obtain the data elsewhere.

## Environment

Use Python 3.10+ and a virtual environment from the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch transformers datasets pandas scikit-learn matplotlib tqdm
```

Adjust the `pip install` line if your platform needs a specific PyTorch wheel (CUDA, CPU-only, or MPS on Apple Silicon).

First run of feature extraction downloads **Qwen/Qwen2.5-1.5B** from Hugging Face (~few GB cache under `~/.cache/huggingface`).

## Pipeline overview

| Step | What it does | Main output |
|------|----------------|-------------|
| 1. Splits | Imbalanced train/val/test CSVs | `data/processed/wildguardmix_imbalance/` |
| 2. Features | LM forward pass, pooled hidden states | `features/<model_slug>/layer_*/` |
| 3. Probes & baselines | Train/eval lightweight classifiers | `outputs/checkpoints/`, `outputs/metrics/` |
| 4. Layer sweep (optional) | Train a probe per layer | `outputs/figures/full_layer_sweep_*.png`, metrics |
| 5. Notebook | Tables + plots from saved files | Reads `outputs/`, writes figures there |
| 6. Report | PDF from LaTeX | `report/report.pdf` |

---

## 1. Build processed splits

From the repo root:

```bash
python make_imbalance_split.py
```

Defaults write to `data/processed/wildguardmix_imbalance/` (`train.csv`, `val.csv`, `test.csv`). Override paths and class balance with `--input_hf_path`, `--out_dir`, `--ratio_normal_to_abnormal`, etc. (`python make_imbalance_split.py --help`).

---

## 2. Extract activation features

Scripts import `utils` from `src/`; run them with the project root as the working directory:

```bash
cd "/path/to/DL Project"

# Default: last-token pooling, layers 4, 14, 26 (quick / paper table subset)
python src/extract_features.py --batch_size 16

# Full 28-layer sweep (Qwen2.5-1.5B uses hidden_states indices 0..27 for transformer layers)
python src/extract_features.py --pool last --batch_size 16 --layers $(seq 0 27)

# Mean pooling only (e.g. ablation) — use a distinct slug so you do not overwrite defaults
python src/extract_features.py --pool mean --layers 19 --batch_size 16 --out_slug qwen2.5-1.5b-meanpool
```

Outputs: `features/<slug>/layer_<n>/X_{train,val,test}.pt` plus `extraction_config.json`.

- Default feature directory slug is `qwen2.5-1.5b` (see `src/utils.py`).
- **`--pool`:** `last` (default, matches the report) or `mean`.

Inference is heavy; use GPU/MPS when available. Extraction time scales with dataset size and is **not** multiplied by the number of layers in a meaningful way (one forward pass per batch).

---

## 3. Train probes and baselines

**If you already extracted features for layers `0..27`:** start with the sweep — it replaces running `train_probe.py` separately for every layer.

```bash
python src/sweep_all_layers.py --probe_type linear
```

Auto-detects `features/qwen2.5-1.5b/layer_*`. Writes `outputs/figures/full_layer_sweep_linear.png`, per-layer checkpoints/metrics with a `_sweep` suffix, and `outputs/metrics/full_sweep_linear.json`.

Then train **named** probes for the main table / notebook rows (these file names match `02_main_experiment.ipynb`):

```bash
python src/baselines.py
python src/train_probe.py --layer 19 --probe_type linear
python src/train_probe.py --layer 14 --probe_type mlp
python src/train_probe.py --layer 4  --probe_type linear   # optional table row
python src/train_probe.py --layer 14 --probe_type linear
python src/train_probe.py --layer 26 --probe_type linear
python src/one_class.py --layer 14
```

**If you only extracted a few layers** (e.g. default `4 14 26`), use `train_probe.py` per layer instead of the full sweep.

```bash
# Custom feature directory (e.g. mean pooling):
python src/train_probe.py --layer 19 --probe_type linear --model_slug qwen2.5-1.5b-meanpool
```

Checkpoints and JSON metrics land in `outputs/checkpoints/` and `outputs/metrics/`. Common flags: `--pos_weight`, `--epochs`, `--patience` (`python src/train_probe.py --help`).

Optional re-evaluation / figure export for a single probe:

```bash
python src/eval_probe.py --layer 19 --probe_type linear
```

---

## 4. Notebooks vs command line

### `notebooks/02_main_experiment.ipynb`

- **Does:** loads `outputs/metrics/*.json`, builds the results table, and regenerates plots (ROC/PR, confusion, layer sweep chart, t-SNE, etc.) into `outputs/figures/`.
- **Does not:** run Hugging Face extraction or probe training. Those must be done via the scripts above first; otherwise cells will show `n/a` or fail when files are missing.

The first markdown cell lists an **outdated minimal** CLI (only a few layers). With **all** layers on disk, use **`sweep_all_layers.py`** plus the `train_probe` / `baselines` / `one_class` commands in [§3](#3-train-probes-and-baselines).

### `notebooks/01_eda.ipynb`

Exploratory analysis on the processed CSVs; run after splits exist.

---

## 5. Compile the report

From `report/`:

```bash
cd report
latexmk -pdf report.tex
# or: pdflatex report.tex (run twice if references don’t resolve)
```

---

## Quick reproduction checklists

**Teammates with Drive `features/` (+ optional `data/processed/`):**

1. Unpack artifacts into repo root; activate venv.
2. `python src/baselines.py`
3. `python src/sweep_all_layers.py`
4. `python src/train_probe.py --layer 19 --probe_type linear` (and other rows from §3 as needed)
5. `python src/one_class.py --layer 14`
6. Run `notebooks/02_main_experiment.ipynb`

**From scratch (no shared tensors):**

1. `python make_imbalance_split.py`
2. `python src/extract_features.py --layers $(seq 0 27)` (or a subset)
3. Same training steps as above.

The **layer sweep figure** requires all layers extracted **or** restored from Drive under `features/qwen2.5-1.5b/layer_*`.
