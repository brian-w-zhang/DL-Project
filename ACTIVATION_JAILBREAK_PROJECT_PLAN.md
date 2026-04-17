# Activation-Based Jailbreak Anomaly Detection (1-Day Plan)

## 0) Quick Summary

Project idea: detect jailbreak/harmful prompts as anomalies using **internal activations** from a frozen LLM, rather than relying only on prompt text or generated output text.

Core claim to test:
- A lightweight probe trained on hidden states can separate benign vs jailbreak/harmful prompts with useful accuracy.

Constraints:
- Must be anomaly detection (class imbalance) -> satisfied.
- Must use PyTorch -> satisfied.
- 1-day execution -> keep scope narrow and reproducible.

---

## 1) Problem Statement (Draft)

Given an input prompt, predict whether it is:
- `0 = benign/normal` (majority class)
- `1 = harmful/jailbreak` (minority class, anomaly)

using:
- Frozen LLM internal activation features extracted from selected transformer layers.

This is framed as anomaly detection under class imbalance.

---

## 2) Why This Is Interesting

- Moves beyond keyword/prompt pattern matching.
- Uses model-internal representations ("what the model internally encodes").
- Connected to current AI safety/interpretability work (activation probing, representation engineering, latent-space safety signals).
- Feasible with small model and linear/tiny probe in 1 day.

---

## 3) Scope Lock (Important for 1 Day)

### Minimum Viable Scope (must finish)

1. Build imbalanced dataset:
   - benign prompts (majority)
   - harmful/jailbreak prompts (minority)
2. Extract activations from 1-3 layers of one small open LLM.
3. Train simple PyTorch classifier (logistic head or 2-layer MLP).
4. Evaluate with AUC/F1/PR/recall and confusion matrix.
5. Save:
   - extracted feature tensors
   - trained probe weights
   - reproducible scripts + report figures.

### Nice-to-have (only if time)

- Compare multiple layers (early/mid/late) and show which performs best.
- Add text-only baseline.
- Add cross-dataset test split or transfer test.
- Add attention-entropy feature ablation (optional).

---

## 4) Terminology (for report)

- **Prompt-level detector**: classifier over raw prompt text.
- **Output-level detector**: classifier over generated response text.
- **Activation-level detector** (this project): classifier over hidden vectors from inside model.
- **Internal activations**: hidden state vectors produced at each transformer layer for each token.
- **Probe**: lightweight classifier trained on frozen model representations.

Note: avoid saying "weight-space" unless editing model parameters; this project is mostly activation/representation-space probing.

---

## 5) Dataset Options (Practical)

## Harmful/Jailbreak class (minority)

Candidate sources:
- AdvBench harmful instruction prompts
- HarmBench prompts (if easy to access)
- JailbreakBench prompts (if easy to access)

Pick one primary source quickly (AdvBench easiest).

## Benign class (majority)

Candidate sources:
- Alpaca instruction prompts
- Dolly prompts
- Safe/normal instruction datasets

## Class imbalance target

Use explicit imbalance for anomaly framing:
- Example: 90:10 or 85:15 benign:harmful

Suggested sizes for speed:
- Train: 1800 benign + 200 harmful
- Val: 360 benign + 40 harmful
- Test: 360 benign + 40 harmful

If compute/time tight, halve all numbers.

---

## 6) Model Choice

Use one small open model that fits local resources:
- Qwen2.5-1.5B (or 0.5B if needed)
- Llama-3.2-1B (if available)
- TinyLlama as fallback

Requirements:
- Fast forward pass
- Can return hidden states
- No fine-tuning needed (frozen backbone)

---

## 7) Feature Extraction Design

For each prompt:
1. Tokenize prompt.
2. Forward pass with `output_hidden_states=True` and no grad.
3. Select hidden states from target layer(s).
4. Pool token representations into one fixed-size vector.

Pooling choices:
- last-token hidden state (simple and common)
- mean over prompt tokens (robust)

Start with:
- last-token at one mid-late layer (e.g., 16 or equivalent).

Then optionally compare:
- early vs middle vs late layers.

Output:
- `X.npy` or `.pt` feature tensor (N x D)
- `y.npy` label vector (N)

---

## 8) Probe Architecture (PyTorch)

Baseline probe:
- Logistic regression equivalent in PyTorch: `Linear(D,1)` + BCEWithLogitsLoss.

Slightly stronger:
- Tiny MLP: `Linear(D,256)->ReLU->Dropout->Linear(256,1)`.

Training details:
- weighted BCE or focal loss (because imbalance)
- AdamW
- early stopping on validation AUC/F1

Recommended metrics:
- ROC-AUC
- PR-AUC (important for anomaly imbalance)
- F1 (macro and positive-class)
- precision, recall
- confusion matrix

---

## 9) Baselines (for comparison)

Minimum baseline:
- Prompt-text baseline using TF-IDF + Logistic Regression (sklearn) OR simple embedding+linear classifier.

Optional baseline:
- Output-text safety heuristic (if generation time allows).

Main narrative:
- Activation probe vs prompt baseline under same split.
- Not claiming SOTA; show promising and interpretable direction.

---

## 10) Experiments to Run

## E1: Main experiment
- One layer, one pooling, one probe.
- Report final test metrics.

## E2: Layer sweep (optional but valuable)
- Compare 3 layers: early/mid/late.
- Plot metric vs layer index.

## E3: Ablation (optional)
- last-token vs mean-pooling.
- linear probe vs tiny MLP.

## E4: Generalization sanity check (optional)
- Train on one jailbreak set style, test on another style.

---

## 11) Reproducibility Checklist

- Fixed random seeds.
- Save train/val/test splits.
- Save extracted features and labels.
- Save best probe checkpoint.
- Save config JSON (model name, layers, pooling, LR, batch size, seed).
- One script/notebook to regenerate all report figures.

---

## 12) Project Folder Draft

```text
project/
  README.md
  requirements.txt
  data/
    raw/              # downloaded prompt datasets (or links only)
    processed/
      train.csv
      val.csv
      test.csv
  features/
    model_<name>/
      layer_<id>_pool_<type>/
        X_train.pt
        y_train.pt
        X_val.pt
        y_val.pt
        X_test.pt
        y_test.pt
  src/
    prepare_data.py
    extract_features.py
    train_probe.py
    eval_probe.py
    baselines.py
    utils.py
  notebooks/
    01_quick_eda.ipynb
    02_results_plots.ipynb
  outputs/
    checkpoints/
    figures/
    metrics/
  report/
    report.pdf
```

---

## 13) One-Day Timeline (Aggressive)

### Hour 0-1: Setup
- environment + dependencies
- choose model
- collect prompt datasets quickly

### Hour 1-3: Data + feature extraction
- create imbalanced split
- run feature extraction for chosen layer
- save tensors

### Hour 3-5: Probe training
- train linear and tiny MLP
- tune only 1-2 key hyperparams

### Hour 5-6: Baseline
- run text baseline quickly

### Hour 6-7: Evaluation/plots
- confusion matrix, ROC, PR curves
- layer comparison if possible

### Hour 7-9: Report writing
- method, data, metrics, results, limitations, reproducibility steps

### Hour 9-10: Cleanup
- final run from clean state
- verify loading checkpoints + reproducing key plot

---

## 14) Risks + Mitigations

Risk: model too slow/OOM
- Mitigation: smaller model, batch size 1-4, shorter prompts, fewer samples.

Risk: poor class separation
- Mitigation: try different layer, mean pooling, weighted loss, normalize features.

Risk: dataset noise / mislabeled harmful prompts
- Mitigation: quick manual sanity check of 50 samples per class.

Risk: no time for advanced analysis
- Mitigation: lock MVP and report optionals as future work.

---

## 15) Report Structure (Draft)

1. Introduction + motivation
2. Related work
3. Dataset construction and imbalance framing
4. Method
   - frozen LLM feature extraction
   - probe architecture
   - baselines
5. Experimental setup
   - splits, metrics, hardware, hyperparams
6. Results
   - main table + key plots
   - ablations/layer analysis
7. Failure cases + limitations
8. Reproducibility instructions
9. Conclusion + future work

---

## 16) Related Work Pointers (Start Here)

- Representation Engineering / activation steering intros
- ALERT: Zero-shot LLM Jailbreak Detection via Internal Discrepancy Amplification (arXiv:2601.03600)
- Jailbreaking Leaves a Trace: internal representation based detection (arXiv:2602.11495)
- Goal/framing disentanglement work for concealed jailbreak detection (FrameShield/ReDAct line)

Note for writeup:
- Compare directionally to published ideas and results.
- Avoid claiming direct apples-to-apples SOTA unless exact same data/protocol.

---

## 17) "Interesting + Original" Angles You Can Pick

Pick one extra angle (small but distinctive):
- Layer-wise "where does safety signal emerge?" analysis.
- Time-to-detect: can detection happen pre-generation?
- Transfer: train on one jailbreak style, test on another.
- Robustness: does simple paraphrasing fool activation probe?

Even one of these makes the project feel more original.

---

## 18) Minimal Commands Sketch (to refine)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm

python src/prepare_data.py --harmful_source advbench --benign_source alpaca --imbalance 0.1
python src/extract_features.py --model qwen2.5-1.5b --layer 16 --pool last --split train,val,test
python src/train_probe.py --model_type linear --class_weight balanced
python src/eval_probe.py --ckpt outputs/checkpoints/best.pt
python src/baselines.py --method tfidf_logreg
```

---

## 19) What To Say If Results Are Mixed

If performance is not amazing, still frame as valid:
- Activation features show detectable signal above baseline/random.
- Mid/late layers often outperform early layers (if observed).
- Data quality and attack diversity likely limit robustness.
- Strong case for larger-scale follow-up.

This can still score well if method, documentation, and analysis are clean.

---

## 20) Final Deliverables Checklist

- [ ] PDF report
- [ ] PyTorch training code for probe
- [ ] Code to load saved model + reproduce metrics
- [ ] Saved checkpoint(s)
- [ ] Script/notebook to regenerate figures
- [ ] README with setup + run steps
- [ ] Dataset links (not full raw dataset upload if large)
- [ ] Short demo for presentation

---

## 21) Optional Extensions (Only if Extra Time)

- Multi-layer concatenated features
- Per-head attention entropy features
- One-class anomaly detector variant (train mostly on benign)
- Calibration + threshold tuning for high recall safety setting
- Basic interpretability on probe weights

---

## 22) Fast Decision Defaults (So You Don't Stall)

If uncertain, use:
- Model: Qwen2.5-1.5B (or smallest available)
- Harmful set: AdvBench
- Benign set: Alpaca
- Imbalance: 90:10
- Feature: last-token, middle-late layer
- Probe: linear first, MLP second
- Metrics: ROC-AUC, PR-AUC, F1, recall

Ship this first, improve second.

