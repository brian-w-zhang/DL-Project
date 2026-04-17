# Activation-Based Jailbreak Anomaly Detection Plan

## Goal

Build a complete, reproducible deep learning project that detects harmful/jailbreak prompts using frozen LLM activations, with clear comparisons and report-ready outputs.

## Dataset and Splits

- Dataset: WildGuardMix
- Splits (already prepared):  
  - Train: 2,700 benign + 300 harmful (90:10)  
  - Val: 540 benign + 60 harmful  
  - Test: 540 benign + 60 harmful
- Files: `data/processed/wildguardmix_imbalance/{train,val,test}.csv`

## Model Choice

- Primary model: `Qwen/Qwen2.5-1.5B`
- Fallback if resource-limited: `Qwen/Qwen2.5-0.5B`
- In all extraction runs, model is frozen (no weight updates).

## Scope Priority

1. **MVP (must finish)**
   - One-layer activation probe (linear)
   - TF-IDF + logistic regression baseline
   - Core metrics + plots
2. **Layer sweep (high priority)**
   - Compare multiple layers (early/mid/late)
   - Plot layer index vs AUC
3. **One-class anomaly variant (optional but recommended)**
   - Train on benign only
   - Score outliers at test time
4. **Paraphrase robustness (time permitting)**
   - Evaluate on paraphrased subset

## Project Structure

```text
src/
  utils.py
  extract_features.py
  train_probe.py
  baselines.py
  eval_probe.py
  one_class.py              # optional if time is tight
notebooks/
  02_main_experiment.ipynb  # primary report/demo notebook
outputs/
  checkpoints/
  metrics/
  figures/
features/
  qwen2.5-1.5b/
    layer_4/
    layer_14/
    layer_26/
```

## Script Responsibilities

- `src/utils.py`
  - Seed setup
  - Shared paths
  - CSV loading utilities

- `src/extract_features.py`
  - Load frozen model with `output_hidden_states=True`
  - Extract hidden states for selected layers (e.g. 4, 14, 26)
  - Pool token representations to one vector per prompt
  - Save `X_{split}.pt`, `y_{split}.pt`

- `src/train_probe.py`
  - Train linear probe (and optional MLP probe)
  - Handle class imbalance with weighted loss
  - Save best checkpoint + validation/test metrics JSON

- `src/baselines.py`
  - TF-IDF + logistic regression baseline on raw text
  - Save metrics JSON for direct comparison

- `src/eval_probe.py`
  - Generate final metrics and figures
  - ROC, PR, confusion matrix, layer-sweep chart

- `src/one_class.py` (optional)
  - Benign-only anomaly detector on activation vectors
  - Compare to binary probe metrics

## Core Experiments

- **E1:** Linear probe at one layer (default: layer 14)
- **E2:** Layer sweep (layers 4, 14, 26)
- **E3:** MLP vs linear ablation (same layer)
- **Baseline:** TF-IDF + logistic regression
- **E4 (optional):** One-class anomaly detection

## Metrics to Report

- ROC-AUC (primary)
- PR-AUC (important for class imbalance)
- F1 (especially harmful class)
- Precision / Recall
- Confusion matrix

## Figures to Produce

- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `layer_sweep_auc.png`

## Reproducibility Commands

```bash
source .venv/bin/activate
python src/extract_features.py --layers 4 14 26
python src/train_probe.py --layer 14 --probe_type linear
python src/train_probe.py --layer 14 --probe_type mlp
python src/train_probe.py --layer 4 --probe_type linear
python src/train_probe.py --layer 26 --probe_type linear
python src/baselines.py
python src/eval_probe.py --layer 14 --probe_type linear
# Optional:
python src/one_class.py --layer 14
```

## Time-Boxed Execution (Today)

- Phase 1 (must ship): extraction + one linear probe + baseline + final evaluation
- Phase 2 (strong extension): layer sweep + layer-vs-AUC figure
- Phase 3 (optional): one-class anomaly, then paraphrase robustness

## Report Story (Suggested)

1. Frozen LLM activations contain safety-relevant signal.  
2. Signal is stronger in mid/late layers than early layers.  
3. Activation probing outperforms a text-only baseline on imbalanced detection.
