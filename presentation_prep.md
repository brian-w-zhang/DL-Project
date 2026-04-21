# Project Presentation

## Part 1: Bottom-Up Project Explanation

To truly grasp what your team built, we must look at it layer by layer (no pun intended):

### 1. The Real-World Problem
Large Language Models (LLMs) have a vulnerability: **jailbreaks**. Users can craft clever prompts that bypass safety guardrails, tricking the model into providing harmful instructions (e.g., "Write a poem about hacking a bank"). Most current defenses try to solve this by analyzing the **prompt's text** (which hackers just obfuscate) or the **generated response's text** (which is computationally wasteful since the model already did the work).

### 2. The Core Hypothesis: "Look Inside the Brain"
Instead of looking at surface-level words, what if we look at what the model is *thinking*? As a prompt travels through the layers of a frozen LLM (like `Qwen2.5-1.5B`), it is converted into high-dimensional numerical vectors called **hidden states** or **activations**. The hypothesis is that even if a prompt is disguised, the model natively understands it is malicious—and that realization is embedded in its internal activations long before it actually speaks.

### 3. The Framing: Anomaly Detection
In the wild, genuine jailbreaks are rare; mostly, people ask benign questions. To mirror this reality, we set this up as an **anomaly detection problem**. We used the `WildGuardMix` dataset but intentionally enforced a severe **90:10 imbalance** (2700 benign vs 300 harmful). 

### 4. The Engineering (Feature Extraction)
The execution is elegant:
1. We pass prompts into a **frozen** `Qwen2.5-1.5B` (its behavior is not altered).
2. We halt it and extract the 1536-dimensional hidden state vectors from different depths.
3. We evaluated two extraction methodologies: **last-token pooling**, which takes the final activation of the prompt, and a heavily superior **mean pooling** strategy, which mathematically averages every word's activation in the sequence.

### 5. Probes & Baselines: The Tripwires
With these extracted "brain waves," we train incredibly lightweight, traditional classifiers—called **probes**—on top of these vectors.
- We tested a **Linear Probe** and a **Multi-Layer Perceptron (MLP)**.
- We tested an entirely unsupervised **Mahalanobis One-Class Detector** (trained without seeing any harmful labels!).
- We explicitly compared everything against a standard **TF-IDF + Logistic Regression text baseline**.

### 6. The Verdict & The "Mean Pooling" Discovery
The experiment was a massive success:
- **Baseline Dominance:** Linear probing of internal representations heavily outperforms standard text methods (0.960 vs 0.877 ROC-AUC).
- **The "Mean Pooling" Phenomenon:** When using single last-tokens, the safety signal peaked at **Layer 19**. But when we averaged the entire prompt context (Mean Pooling), we discovered the harm signal natively crystallizes even earlier at **Layer 15**! Upper-middle layers logically solidify semantic meaning before final layers drift into grammar prediction.
- **Unsupervised Anomaly Baseline:** Even without giving the detector ANY harmful labels during training, the pure Mahalanobis distance calculation flagged jailbreaks at a 0.705 ROC-AUC rate, proving harmful prompts actively deviate from the "normal" LLM brain-state.
- **Generalization:** Scaling training up to 7,000 examples hit a huge 0.889 ROC-AUC on a completely independent benchmark (`WildGuardTest`).

### 7. Real-World Deployment Reality
While Layer 19 is vastly superior in accuracy, we investigated the real-world operational costs to frame this as an engineering solution:
- **Safety Guarantees (FPR at 95% Recall):** If a company wants to aggressively catch exactly 95% of jailbreaks, deploying the TF-IDF model would incorrectly flag ~61% of all normal user queries as harmful (rendering it useless). The Layer 19 internal probe massively halves this error rate down to ~27%.
- **Robustness to Obfuscation (Prompt Length):** Injecting long role-play scripts mathematically breaks term-frequency models because the payload gets diluted. Our Layer 19 probe is completely immune to this, maintaining a perfectly flat recall across heavily obfuscated prompts spanning 300+ characters.
- **The Efficiency Trade-off:** The TF-IDF model evaluates in an instantaneous ~0.22ms on CPU. Deploying the Qwen2.5-1.5B (up to Layer 19) takes ~463ms and ~2.8 GB of VRAM. It acts as a premium, compute-heavy security filter.

---

## Part 2: Terms Cheatsheet

| Concept | What it means in this project |
| :--- | :--- |
| **Qwen2.5-1.5B** | The small open-source LLM we probed. It was kept completely frozen (no fine-tuning). |
| **WildGuardMix** | The dataset by AllenAI containing a mix of benign and harmful user-LLM interactions. |
| **Hidden States/Activations** | The numerical vectors passed between layers in the LLM. It's the model's internal representation of the prompt. |
| **Probe** | A lightweight classifier (Linear or MLP) trained purely on the extracted hidden states to act as a binary tripwire (Benign = 0, Harmful = 1). |
| **Layer 19** | The "Sweet Spot". The layer where the safety signal is mathematically clearest before degrading into grammatical predictions. |
| **ROC-AUC (0.934)** | The probability that the probe ranks a random harmful prompt higher than a random benign one. Excellent score. |
| **PR-AUC (0.781)** | The metric that actually cares about the severe imbalance. 0.781 means precision and recall hold up extremely well, crushing the TF-IDF baseline (0.652). |
| **Thresholding (F1 Score)** | Because of class imbalance, standard 0.5 thresholding looks bad. Adjusting the threshold to 0.84 yielded top F1 results without breaking the raw AUC scores. |
| **FPR @ 95% Recall (TPR)** | A strict business metric. To catch 95% of bad guys, how many innocent people do we accidentally block? TF-IDF blocks ~61%, Layer 19 blocks ~27%. |
| **Obfuscation (Jailbreak Wrappers)** | Attackers burying harmful requests inside long, benign scenarios (e.g. "Write a poem about..."). Our tests proved this mathematically breaks TF-IDF (payload dilution) but fails against the Layer 19 self-attention mechanisms. |
| **Inference Latency Trade-off** | The massive $2000\times$ speed difference between running a simple text equation (0.22ms) vs a 1.5 Billion parameter AI brain structure (463ms) up to Layer 19. |

---

## Part 3: Presenter Talking Points

### 1. Introduction & The Problem
* **The Hook:** Currently, detecting jailbreaks is a cat-and-mouse game. Hackers write clever text to bypass text-filters. So we asked: what if we stop looking at the text, and start looking at the model's internal brain?
* **The Problem:** Explain that text-based filters (like TF-IDF models) fail because they only see surface words. They miss the hidden intent. Furthermore, waiting for the LLM to generate a toxic output means the damage is already processing.
* **Our Approach:** We opted for an **Activation-Based** detection method. We look at the hidden states of a frozen model.
* **Defining the Environment:** We framed this realistically as an **anomaly detection problem**. Real-world harmful prompts are rare, so we forced a 90:10 imbalance using the WildGuardMix dataset (2700 benign to 300 harmful prompts).

### 2. Methodology & Implementation
* **The Model Setup:** We used Qwen2.5-1.5B as our base model. Notably, we kept the weights frozen—we didn't fine-tune it. We just used it as a feature extractor.
* **Extracting the State:** When a prompt goes into Qwen, it travels through 28 layers. We extracted the internal hidden states of these layers using 'last-token pooling', which acts as a dense numerical summary of the whole prompt.
* **Building the Tripwire (Probes):** With these extracted features, we trained lightweight probes—essentially, a simple Linear classifier and an MLP—to classify if those internals looked benign (0) or harmful (1).
* **The Baseline:** To prove our point, we also ran a standard text-based TF-IDF Logistic Regression pipeline on the actual text to compare surface-level vs internal representation.

### 3. Results, The 'Layer 15' Insight, & Obfuscation Resilience
* **The Headline Result:** The internal activations completely outperformed the text baselines. Our absolute best probe (Mean Pooling) hit an ROC-AUC of **0.960** and a PR-AUC of **0.842**, actively crushing the TF-IDF baseline (0.877 ROC-AUC).
* **Beating Obfuscation:** Hackers hide payloads in long role-play scripts. We empirically proved that as prompt length increases past 100 characters, TF-IDF fails catastrophically because the harmful words get mathematically diluted by benign padding. The deeper L15/19 representations maintained a structurally perfect, flat detection rate regardless of length.
* **The 'Sweet Spot' Discovery (Layer 15 vs 19):** What was most fascinating was *where* we found the signal. The safety signal doesn't go up linearly. For single tokens, it peaks at **Layer 19**. But when we averaged the entire prompt context (Mean Pooling), we discovered the harm signal natively crystallizes even earlier at **Layer 15**! This perfectly maps to the deep learning theory that upper-mid layers solidify semantic meaning and harm, while final layers just worry about next-token generation.
* **Unsupervised Anomaly Baseline (Mahalanobis):** We even proved that without giving the detector ANY harmful labels during training, a pure distance-calculation (Mahalanobis) still hits an ROC-AUC of 0.705. The jailbreaks practically flag themselves.

### 4. Constraints, Deployment Reality, & Conclusion
* **Operational False Positives:** ROC-AUC is a theoretical metric. In a real company, if you demand a 95% safety standard (catching 95% of bad queries), the TF-IDF baseline will incorrectly block >60% of your real users. The Activation Probe halves that failure rate to ~27%, representing a massive operational improvement despite the imbalanced data bottleneck.
* **Sub-category Limitations & Blatant Harm:** We must acknowledge error cases. We hit perfect 100% recall on 7 subcategories (cyberattack, defamation, fraud, etc.), but struggle (44% recall) on subtle 'social stereotypes'. Furthermore, because our training data is full of sneaky jailbreak wrappers, the model occasionally misses *blatant, direct* requests (like "how to build a bomb") while successfully catching wrapped ones.
* **The Efficiency Cost:** We also must be honest about deployment costs as engineers. The superior accuracy from the LLM backbone comes at a steep price: evaluating it takes ~463ms (a $>2000\times$ latency increase) and requires dedicated GPU VRAM. 
* **Conclusion & Future Directions:** Ultimately, we validated the extraction architecture and proved its scale hitting 0.889 ROC-AUC on a 7k-sample benchmark. Moving forward, we recommend a cascading architecture: use lightweight text filters for the obvious 80% of traffic, and strictly route the highly suspicious, nuanced 20% to the potent Layer 19/15 activation probe.
