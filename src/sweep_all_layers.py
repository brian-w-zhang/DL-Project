#!/usr/bin/env python3
"""
Train linear probes for every extracted layer and plot the full layer sweep.

Usage:
  python src/sweep_all_layers.py
  python src/sweep_all_layers.py --layers 0 4 8 12 16 20 24 27  # subset
  python src/sweep_all_layers.py --probe_type mlp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils import MODEL_SLUG, PATHS, load_features, set_seed

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ---------------------------------------------------------------------------
# Inline probe (avoids circular import issues)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Train one layer
# ---------------------------------------------------------------------------

def train_one_layer(
    layer: int,
    probe_type: str,
    pos_weight: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    model_slug: str | None = None,
) -> dict:
    set_seed(seed)
    device = torch.device("cpu")
    slug = model_slug or MODEL_SLUG

    # Check features exist
    feat_dir = PATHS["features"] / slug / f"layer_{layer}"
    if not (feat_dir / "X_train.pt").exists():
        return None

    X_train, y_train = load_features(slug, layer, "train")
    X_val,   y_val   = load_features(slug, layer, "val")
    X_test,  y_test  = load_features(slug, layer, "test")
    X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()

    mean = X_train.mean(dim=0)
    std  = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    in_dim = X_train.shape[1]
    probe  = LinearProbe(in_dim) if probe_type == "linear" else MLPProbe(in_dim)

    pw        = torch.tensor([pos_weight], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    loader    = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_prauc, patience_counter, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss(pos_weight=pw)(probe(xb), yb).backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(probe(X_val)).numpy()
        prauc = average_precision_score(y_val.numpy(), val_scores)

        if prauc > best_prauc:
            best_prauc   = prauc
            best_state   = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(probe(X_test)).numpy()
        val_scores_final = torch.sigmoid(probe(X_val)).numpy()

    y_test_np = y_test.numpy()
    y_val_np  = y_val.numpy()

    # Optimal threshold on val
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val_np, (val_scores_final >= t).astype(int),
                      average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    test_preds = (test_scores >= best_t).astype(int)

    result = {
        "layer":        layer,
        "probe_type":   probe_type,
        "test_roc_auc": float(roc_auc_score(y_test_np, test_scores)),
        "test_pr_auc":  float(average_precision_score(y_test_np, test_scores)),
        "test_f1":      float(f1_score(y_test_np, test_preds, average="binary", zero_division=0)),
        "best_threshold": float(best_t),
    }

    # Save checkpoint
    ckpt_name = f"{slug}_layer{layer}_{probe_type}_sweep"
    ckpt_path = PATHS["checkpoints"] / f"{ckpt_name}.pt"
    torch.save({
        "probe_state_dict": best_state,
        "probe_type":  probe_type,
        "in_dim":      in_dim,
        "layer":       layer,
        "model_slug":  slug,
        "norm_mean":   mean,
        "norm_std":    std,
        "best_threshold": best_t,
    }, ckpt_path)

    # Save metrics
    metrics_path = PATHS["metrics"] / f"{ckpt_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(results: list[dict], probe_type: str) -> None:
    results = sorted(results, key=lambda r: r["layer"])
    layers   = [r["layer"]        for r in results]
    roc_aucs = [r["test_roc_auc"] for r in results]
    pr_aucs  = [r["test_pr_auc"]  for r in results]
    f1s      = [r["test_f1"]      for r in results]

    best_layer = layers[int(np.argmax(roc_aucs))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC-AUC + PR-AUC line plot
    axes[0].plot(layers, roc_aucs, "o-", color="#4C72B0", lw=2, label="ROC-AUC")
    axes[0].plot(layers, pr_aucs,  "s-", color="#DD8452", lw=2, label="PR-AUC")
    axes[0].axvline(best_layer, color="gray", linestyle="--", lw=1,
                    label=f"Best layer ({best_layer})")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Full Layer Sweep ({probe_type.upper()} probe) — ROC-AUC & PR-AUC")
    axes[0].legend()
    axes[0].set_xticks(layers[::2])
    axes[0].grid(alpha=0.3)

    # F1 bar chart
    colors = ["#C44E52" if l == best_layer else "#8C8C8C" for l in layers]
    axes[1].bar(layers, f1s, color=colors, width=0.7)
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("F1 (harmful class)")
    axes[1].set_title(f"Full Layer Sweep ({probe_type.upper()} probe) — F1 at Optimal Threshold")
    axes[1].set_xticks(layers[::2])
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Where does safety signal emerge in {MODEL_SLUG}?\n"
        f"Best ROC-AUC at layer {best_layer}: {max(roc_aucs):.4f}",
        fontsize=12
    )
    plt.tight_layout()

    out = PATHS["figures"] / f"full_layer_sweep_{probe_type}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to sweep. Default: all extracted layers 0-27.")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_slug", type=str, default=None,
                        help="Feature slug to load from (default: MODEL_SLUG). "
                             "Use e.g. qwen2.5-1.5b-meanpool for mean-pool ablation.")
    args = parser.parse_args()

    slug = args.model_slug or MODEL_SLUG

    # Auto-detect available layers
    if args.layers is None:
        base = PATHS["features"] / slug
        args.layers = sorted([
            int(p.name.split("_")[1])
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith("layer_")
        ])
    print(f"Slug: {slug}  |  Sweeping {len(args.layers)} layers: {args.layers}")

    results = []
    for i, layer in enumerate(args.layers):
        print(f"\n[{i+1}/{len(args.layers)}] Layer {layer} ...", flush=True)
        r = train_one_layer(
            layer=layer,
            probe_type=args.probe_type,
            pos_weight=args.pos_weight,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            model_slug=slug,
        )
        if r is None:
            print(f"  Skipped (features not found)")
            continue
        results.append(r)
        print(f"  ROC-AUC={r['test_roc_auc']:.4f}  PR-AUC={r['test_pr_auc']:.4f}  F1={r['test_f1']:.4f}")

    print(f"\nDone. Best layer by ROC-AUC: {max(results, key=lambda r: r['test_roc_auc'])['layer']}")

    # Save full sweep summary — separate file per slug so results don't overwrite
    slug_suffix = "" if slug == MODEL_SLUG else f"_{slug.replace('/', '-')}"
    summary_path = PATHS["metrics"] / f"full_sweep_{args.probe_type}{slug_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary -> {summary_path}")

    plot_sweep(results, args.probe_type)


if __name__ == "__main__":
    main()
