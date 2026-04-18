#!/usr/bin/env python3
"""
Train a linear or MLP probe on frozen LLM activation features.

Usage:
  python src/train_probe.py --layer 14 --probe_type linear
  python src/train_probe.py --layer 14 --probe_type mlp
  python src/train_probe.py --layer 4 --probe_type linear
  python src/train_probe.py --layer 26 --probe_type linear
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    MODEL_SLUG,
    PATHS,
    load_features,
    save_metrics,
    set_seed,
)


# ---------------------------------------------------------------------------
# Probe architectures
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def resample_balanced(X: torch.Tensor, y: torch.Tensor, harmful_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample training set so harmful class = harmful_frac of total."""
    rng = np.random.default_rng(seed)
    idx_normal = np.where(y.numpy() == 0)[0]
    idx_harmful = np.where(y.numpy() == 1)[0]
    n_harmful = len(idx_harmful)
    n_normal = int(n_harmful * (1 - harmful_frac) / harmful_frac)
    n_normal = min(n_normal, len(idx_normal))
    chosen_normal = rng.choice(idx_normal, size=n_normal, replace=False)
    idx = np.concatenate([chosen_normal, idx_harmful])
    rng.shuffle(idx)
    print(f"  Resampled train: {n_normal} normal + {n_harmful} harmful "
          f"({harmful_frac*100:.0f}% harmful)")
    return X[idx], y[idx]


def train(
    layer: int,
    probe_type: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    pos_weight: float,
    harmful_frac: float | None,
    run_name: str | None,
    model_slug: str | None = None,
) -> None:
    set_seed(seed)
    device = torch.device("cpu")  # probes are tiny, CPU is fine

    slug = model_slug or MODEL_SLUG
    X_train, y_train = load_features(slug, layer, "train")
    X_val, y_val = load_features(slug, layer, "val")
    X_test, y_test = load_features(slug, layer, "test")

    X_train = X_train.float()
    X_val = X_val.float()
    X_test = X_test.float()

    if harmful_frac is not None:
        X_train, y_train = resample_balanced(X_train, y_train, harmful_frac, seed)

    # Normalize features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    in_dim = X_train.shape[1]
    probe = LinearProbe(in_dim) if probe_type == "linear" else MLPProbe(in_dim)
    probe.to(device)

    pw = torch.tensor([pos_weight], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_prauc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val.to(device)).cpu()
            val_scores = torch.sigmoid(val_logits).numpy()
        val_prauc = average_precision_score(y_val.numpy(), val_scores)

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | val PR-AUC: {val_prauc:.4f} (best: {best_val_prauc:.4f})")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    # Load best weights and evaluate on test
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test.to(device)).cpu()
        test_scores = torch.sigmoid(test_logits).numpy()
        val_logits_final = probe(X_val.to(device)).cpu()
        val_scores_final = torch.sigmoid(val_logits_final).numpy()

    y_test_np = y_test.numpy()
    y_val_np = y_val.numpy()

    test_roc = roc_auc_score(y_test_np, test_scores)
    test_pr = average_precision_score(y_test_np, test_scores)
    val_roc = roc_auc_score(y_val_np, val_scores_final)

    # Find optimal threshold on val set (maximises F1 for harmful class)
    best_thresh, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (val_scores_final >= t).astype(int)
        f1 = f1_score(y_val_np, preds, average="binary", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1, best_thresh = f1, float(t)

    test_preds = (test_scores >= best_thresh).astype(int)
    test_f1 = f1_score(y_test_np, test_preds, average="binary", zero_division=0)

    print(f"\nTest ROC-AUC: {test_roc:.4f} | Test PR-AUC: {test_pr:.4f}")
    print(f"Optimal threshold (val F1): {best_thresh:.2f} -> Test F1(harmful): {test_f1:.4f}")

    # Save checkpoint
    ckpt_name = run_name if run_name else f"{MODEL_SLUG}_layer{layer}_{probe_type}"
    ckpt_path = PATHS["checkpoints"] / f"{ckpt_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "probe_state_dict": best_state,
        "probe_type": probe_type,
        "in_dim": in_dim,
        "layer": layer,
        "model_slug": MODEL_SLUG,
        "norm_mean": mean,
        "norm_std": std,
        "best_epoch": best_epoch,
        "best_threshold": best_thresh,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    metrics = {
        "model_slug": slug,
        "layer": layer,
        "probe_type": probe_type,
        "best_epoch": best_epoch,
        "val_roc_auc": float(val_roc),
        "val_pr_auc": float(best_val_prauc),
        "test_roc_auc": float(test_roc),
        "test_pr_auc": float(test_pr),
        "best_threshold": float(best_thresh),
        "test_f1_harmful_at_best_thresh": float(test_f1),
        "hyperparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "pos_weight": pos_weight,
            "batch_size": batch_size,
            "seed": seed,
            "harmful_frac": harmful_frac,
        },
    }
    save_metrics(metrics, ckpt_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmful_frac", type=float, default=None,
                        help="Resample train so harmful=this fraction e.g. 0.25 for 75:25")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom name for checkpoint/metrics files (default: auto)")
    parser.add_argument("--model_slug", type=str, default=None,
                        help="Override feature slug (default: MODEL_SLUG from utils). "
                             "Use to load from e.g. qwen2.5-1.5b-meanpool.")
    args = parser.parse_args()

    print(f"Training {args.probe_type} probe on layer {args.layer} features...")
    train(
        layer=args.layer,
        probe_type=args.probe_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        pos_weight=args.pos_weight,
        harmful_frac=args.harmful_frac,
        run_name=args.run_name,
        model_slug=args.model_slug,
    )


if __name__ == "__main__":
    main()
