#!/usr/bin/env python3
"""
One-class anomaly detection on activation features.
Trains only on benign (normal) samples; scores test prompts by distance from the benign cluster.

Usage:
  python src/one_class.py --layer 14
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from utils import MODEL_SLUG, load_features, save_metrics


def mahalanobis_scores(X_train_benign: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Higher score = more anomalous."""
    cov = EmpiricalCovariance().fit(X_train_benign)
    scores = cov.mahalanobis(X_test)
    return scores


def run_one_class(layer: int, seed: int) -> None:
    np.random.seed(seed)

    X_train, y_train = load_features(MODEL_SLUG, layer, "train")
    X_test, y_test = load_features(MODEL_SLUG, layer, "test")

    X_train_np = X_train.float().numpy()
    X_test_np = X_test.float().numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    # Train only on benign
    benign_mask = y_train_np == 0
    X_benign = X_train_np[benign_mask]
    print(f"One-class: training on {X_benign.shape[0]} benign samples only (layer {layer})")

    # Mahalanobis distance from benign cluster
    scores = mahalanobis_scores(X_benign, X_test_np)

    roc = roc_auc_score(y_test_np, scores)
    pr = average_precision_score(y_test_np, scores)

    # Threshold at 95th percentile of benign train scores
    benign_scores = mahalanobis_scores(X_benign, X_benign)
    threshold = np.percentile(benign_scores, 95)
    preds = (scores > threshold).astype(int)
    f1 = f1_score(y_test_np, preds, average="binary")

    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(f"  F1 (harmful, thresh=p95): {f1:.4f}")

    metrics = {
        "method": "one_class_mahalanobis",
        "layer": layer,
        "model_slug": MODEL_SLUG,
        "n_benign_train": int(X_benign.shape[0]),
        "threshold_percentile": 95,
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "test_f1_harmful": float(f1),
    }
    save_metrics(metrics, f"one_class_layer{layer}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_one_class(args.layer, args.seed)


if __name__ == "__main__":
    main()
