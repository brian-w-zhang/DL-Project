#!/usr/bin/env python3
"""
Evaluate a saved probe checkpoint and generate all report figures.

Usage:
  python src/eval_probe.py --layer 14 --probe_type linear
  python src/eval_probe.py --layer 14 --probe_type linear --layer_sweep
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from utils import MODEL_SLUG, PATHS, load_features, load_metrics, set_seed
from train_probe import LinearProbe, MLPProbe

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def load_probe(layer: int, probe_type: str) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    ckpt_name = f"{MODEL_SLUG}_layer{layer}_{probe_type}"
    ckpt_path = PATHS["checkpoints"] / f"{ckpt_name}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if probe_type == "linear":
        probe = LinearProbe(ckpt["in_dim"])
    else:
        probe = MLPProbe(ckpt["in_dim"])
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe.eval()
    return probe, ckpt["norm_mean"], ckpt["norm_std"]


def get_scores(probe: nn.Module, X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    X = X.float()
    X_norm = (X - mean) / std.clamp(min=1e-8)
    with torch.no_grad():
        logits = probe(X_norm)
    return torch.sigmoid(logits).numpy()


def plot_roc_pr(layer: int, probe_type: str) -> None:
    probe, mean, std = load_probe(layer, probe_type)
    X_test, y_test = load_features(MODEL_SLUG, layer, "test")
    scores = get_scores(probe, X_test, mean, std)
    y_np = y_test.numpy()

    # Load baseline metrics for comparison
    baseline = load_metrics("baseline_tfidf")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_np, scores)
    auc = roc_auc_score(y_np, scores)
    axes[0].plot(fpr, tpr, lw=2, label=f"Activation probe L{layer} (AUC={auc:.3f})")
    axes[0].axhline(baseline["test_roc_auc"], color="gray", linestyle="--",
                    label=f"TF-IDF baseline (AUC={baseline['test_roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR
    prec, rec, _ = precision_recall_curve(y_np, scores)
    pr_auc = average_precision_score(y_np, scores)
    axes[1].plot(rec, prec, lw=2, label=f"Activation probe L{layer} (AP={pr_auc:.3f})")
    axes[1].axhline(baseline["test_pr_auc"], color="gray", linestyle="--",
                    label=f"TF-IDF baseline (AP={baseline['test_pr_auc']:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    out = PATHS["figures"] / f"roc_pr_layer{layer}_{probe_type}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_confusion(layer: int, probe_type: str, threshold: float = 0.5) -> None:
    probe, mean, std = load_probe(layer, probe_type)
    X_test, y_test = load_features(MODEL_SLUG, layer, "test")
    scores = get_scores(probe, X_test, mean, std)
    preds = (scores >= threshold).astype(int)
    y_np = y_test.numpy().astype(int)

    cm = confusion_matrix(y_np, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Harmful"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — Layer {layer} {probe_type} probe")
    plt.tight_layout()
    out = PATHS["figures"] / f"confusion_layer{layer}_{probe_type}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")

    f1 = f1_score(y_np, preds, average="binary")
    print(f"  F1 (harmful): {f1:.4f} at threshold {threshold}")


def plot_layer_sweep(probe_type: str = "linear", layers: list[int] = None) -> None:
    if layers is None:
        layers = [4, 14, 26]

    roc_aucs, pr_aucs = [], []
    for layer in layers:
        try:
            m = load_metrics(f"{MODEL_SLUG}_layer{layer}_{probe_type}")
            roc_aucs.append(m["test_roc_auc"])
            pr_aucs.append(m["test_pr_auc"])
        except FileNotFoundError:
            print(f"  Warning: no metrics for layer {layer}, skipping.")
            roc_aucs.append(0)
            pr_aucs.append(0)

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, roc_aucs, width, label="ROC-AUC", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, pr_aucs, width, label="PR-AUC", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Layer Sweep — {probe_type.upper()} Probe")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2)
    ax.bar_label(bars2, fmt="%.3f", padding=2)

    plt.tight_layout()
    out = PATHS["figures"] / f"layer_sweep_{probe_type}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def full_report(layer: int, probe_type: str) -> None:
    probe, mean, std = load_probe(layer, probe_type)
    X_test, y_test = load_features(MODEL_SLUG, layer, "test")
    scores = get_scores(probe, X_test, mean, std)
    y_np = y_test.numpy()
    preds = (scores >= 0.5).astype(int)

    roc = roc_auc_score(y_np, scores)
    pr = average_precision_score(y_np, scores)
    f1 = f1_score(y_np, preds, average="binary")

    print(f"\n=== Test Results: {probe_type} probe, layer {layer} ===")
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(f"  F1 (harmful): {f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--layer_sweep", action="store_true", help="Also generate layer sweep chart")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 14, 26])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    PATHS["figures"].mkdir(parents=True, exist_ok=True)

    full_report(args.layer, args.probe_type)
    plot_roc_pr(args.layer, args.probe_type)
    plot_confusion(args.layer, args.probe_type)

    if args.layer_sweep:
        plot_layer_sweep(probe_type=args.probe_type, layers=args.layers)


if __name__ == "__main__":
    main()
