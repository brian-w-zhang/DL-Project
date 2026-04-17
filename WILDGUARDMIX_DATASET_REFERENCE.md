# WildGuardMix Dataset Reference

## Source

- Dataset: `allenai/wildguardmix`
- Hugging Face: [https://huggingface.co/datasets/allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
- Paper: **WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**  
  [https://arxiv.org/abs/2406.18495](https://arxiv.org/abs/2406.18495)

---

## Dataset Card Notes (copied/condensed for quick reference)

### Disclaimer

The data includes examples that might be disturbing, harmful or upsetting. It includes a range of harmful topics such as discriminatory language and discussions about abuse, violence, self-harm, sexual content, misinformation among other high-risk categories. The main goal of this data is for advancing research in building safe LLMs. It is recommended not to train a LLM exclusively on the harmful examples.

### Dataset Summary

WildGuardMix consists of two splits, WildGuardTrain and WildGuardTest.

#### WildGuardTrain (used to train WildGuard)

- Data size: **86,759** examples
  - **48,783** prompt-only
  - **37,976** prompt+response
- Data types:
  - synthetic data (**87%**)
  - in-the-wild user-LLM interactions (**11%**)
  - existing annotator-written data (**2%**)
- Prompt types:
  - vanilla and adversarial
  - covers both harmful and benign scenarios
- Response generation:
  - for synthetic adversarial/vanilla prompts, matched refusal/compliance responses generated via multiple LLMs
- Labels:
  - prompt harmfulness
  - response harmfulness
  - response refusal
  - (obtained via GPT-4 in train data pipeline)
- Data audit:
  - filtering by assigning labels for target tasks and recategorizing mismatched items
  - human audit on sampled items: agreement reported as
    - **92%** (prompt harm)
    - **82%** (response harm)
    - **95%** (refusal)

#### WildGuardTest (used to evaluate safety classifiers)

- Data size: **1,725** items for prompt harm / response harm / response refusal tasks
- Composition: around **55% vanilla** and **45% adversarial**
- Similar source style to train: synthetic + in-the-wild interactions, with both benign and harmful scenarios
- Labels:
  - collected from three independent annotators
- Label quality:
  - Fleiss Kappa:
    - **0.55** (prompt harm)
    - **0.72** (response refusal)
    - **0.50** (response harm)
  - additional GPT-4 auditing on mismatches

### Usage Snippet

```python
from datasets import load_dataset

# Load the wildguardtrain dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtrain")

# Load the wildguardtest dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
```

### Dataset Fields

- `prompt`: `str` - user request
- `adversarial`: `bool` - whether prompt is adversarial/jailbreak-style
- `response`: `str` or `None` - absent for prompt-only items in `wildguardtrain`
- `prompt_harm_label`: `str` in {`harmful`, `unharmful`} or `None` if no agreement
- `response_harm_label`: `str` in {`harmful`, `unharmful`} or `None`
- `response_refusal_label`: `str` in {`refusal`, `compliance`} or `None`
- `subcategory`: `str` - fine-grained risk category
- (test extras) agreement columns for prompt harm / response harm / refusal

### Citation

```bibtex
@misc{wildguard2024,
      title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs}, 
      author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18495}, 
}
```

---

## How This Relates to This Project

Project theme: activation-based anomaly detection for harmful/jailbreak prompts.

### Core label definition for project

- **Normal (majority):** `prompt_harm_label == "unharmful"`
- **Abnormal / anomaly (minority):** `prompt_harm_label == "harmful"`

This is a direct anomaly-detection formulation aligned with project requirements.

### Why WildGuardMix is a strong choice

1. Single-source distribution for both benign and harmful prompts reduces cross-dataset style artifacts.
2. Includes adversarial and non-adversarial prompts, enabling harder/realistic evaluation.
3. Provides high-quality safety labels and explicit metadata useful for analysis.
4. Large enough to downsample into controlled imbalance ratios.

### Important distinction used in analysis

- `prompt_harm_label` answers **what is being asked** (harmful intent).
- `adversarial` answers **how it is being asked** (jailbreak-style framing).

So for this project:
- primary anomaly label = `prompt_harm_label`
- optional slice analysis = compare adversarial vs non-adversarial behavior

### Imbalance setup for anomaly detection

Because WildGuardMix is not guaranteed to be naturally heavily imbalanced for the chosen label in every split, we will enforce imbalance during sampling:

- Suggested ratio: **90:10** (normal:abnormal), optionally **95:5**
- Example split:
  - train: 1,800 normal + 200 abnormal
  - val: 360 normal + 40 abnormal
  - test: 360 normal + 40 abnormal

### Minimal filtering rules (recommended)

1. Drop rows where `prompt` is missing/empty.
2. Keep rows where `prompt_harm_label` is not `None`.
3. Use only prompt text for input feature extraction.
4. Stratify by label when creating train/val/test.
5. Fix random seed for reproducibility.

### Model-facing framing

Input to frozen LLM:
- prompt text only

Feature:
- hidden activation vector(s) from selected layer(s), e.g., last-token representation

Classifier (probe):
- binary classifier for anomaly prediction on top of frozen activations.

---

## Practical Notes

- Access is gated on Hugging Face and requires accepting AI2 Responsible Use terms.
- Local login required (`huggingface-cli login`) before loading with `datasets`.
- For quick reproducibility, save filtered/sampled splits to local parquet/csv once prepared.

