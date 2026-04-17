# Results So Far

## What we did

1. Downloaded WildGuardMix (AllenAI) — real user prompts labeled harmful/unharmful
2. Created imbalanced splits: 90% normal, 10% harmful (train: 2700/300, val/test: 540/60)
3. Loaded frozen Qwen2.5-1.5B (no fine-tuning — weights never updated)
4. Extracted hidden states from layers 4, 14, 26 for every prompt
5. Trained lightweight probes on top of those hidden vectors
6. Ran TF-IDF text baseline and one-class anomaly variant

---

## Full Results Table

| Method | ROC-AUC | PR-AUC | F1 (harmful) |
|--------|---------|--------|--------------|
| TF-IDF + LogReg (text baseline) | 0.877 | 0.652 | **0.589** |
| Activation probe, Layer 4 (early) | 0.847 | 0.667 | — |
| Activation probe, Layer 14 (mid) — Linear | **0.917** | **0.767** | 0.412* |
| Activation probe, Layer 14 — MLP | 0.909 | 0.721 | — |
| Activation probe, Layer 26 (late) | 0.899 | 0.747 | — |
| One-class Mahalanobis, Layer 14 | 0.705 | 0.180 | 0.263 |

*F1 at default threshold 0.5 — sub-optimal due to class imbalance; ROC/PR-AUC are the reliable metrics here.

---

## Key Findings

### 1. Activation probe beats text baseline on ROC-AUC and PR-AUC
- Probe ROC-AUC 0.917 vs TF-IDF 0.877 (+0.04)
- Probe PR-AUC 0.767 vs TF-IDF 0.652 (+0.115)
- This confirms that internal model representations contain safety signal beyond what surface text patterns capture.

### 2. Mid-layer (14) is the sweet spot
- Layer 4 → 14 → 26 ROC-AUC: 0.847 → **0.917** → 0.899
- Safety signal peaks in the middle of the network, then slightly degrades in later layers.
- Interpretation: mid layers encode rich semantic content; late layers shift toward next-token prediction rather than meaning representation.

### 3. Linear probe beats MLP
- Linear: ROC-AUC 0.917, PR-AUC 0.767
- MLP: ROC-AUC 0.909, PR-AUC 0.721
- The harmful/benign classes are roughly linearly separable in activation space. MLP likely overfits slightly given only 300 harmful training examples.

### 4. One-class variant is weaker but still above chance
- ROC-AUC 0.705 — meaningful signal but harder task (no harmful examples during training).
- This is expected and honest: pure anomaly detection without any harmful labels is a harder problem.
- Good to include in report as the "strictest" anomaly framing.

### 5. The F1 gap is a threshold issue, not a model failure
- The probe's F1 (0.412) looks worse than TF-IDF (0.589) at threshold 0.5.
- This is because `pos_weight=9` during training shifts the probability calibration.
- At a lower threshold (e.g. 0.3), probe F1 would be higher. ROC-AUC and PR-AUC are threshold-independent and are the correct metrics for imbalanced anomaly detection.

---

## Figures generated

- `outputs/figures/roc_pr_layer14_linear.png` — ROC + PR curves, probe vs baseline
- `outputs/figures/confusion_layer14_linear.png` — confusion matrix
- `outputs/figures/layer_sweep_linear.png` — bar chart: layer vs AUC

---

## Raw metrics files

All numbers saved in `outputs/metrics/`:
- `qwen2.5-1.5b_layer4_linear.json`
- `qwen2.5-1.5b_layer14_linear.json`
- `qwen2.5-1.5b_layer26_linear.json`
- `qwen2.5-1.5b_layer14_mlp.json`
- `baseline_tfidf.json`
- `one_class_layer14.json`

---

## Resampling experiments (layer 14 linear, pos_weight=3)

| Training balance | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **90:10 (original)** | **0.9224** | **0.7698** | 0.660 |
| 75:25 | 0.9173 | 0.7412 | 0.685 |
| 67:33 | 0.9064 | 0.7399 | 0.672 |

**Conclusion:** Original 90:10 wins. Resampling didn't help — bottleneck is 300 harmful examples total, not the ratio.

## Remaining work

- [ ] `notebooks/01_eda.ipynb` — data exploration, class distribution, prompt length plots
- [ ] `notebooks/02_main_experiment.ipynb` — all results, figures, live demo widget
- [ ] PDF report
