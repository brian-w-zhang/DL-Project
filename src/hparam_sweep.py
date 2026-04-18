#!/usr/bin/env python3
"""
Automated hyperparameter sweep for the linear probe.

Tries combinations of pos_weight, lr, and weight_decay on the val set,
then prints and saves a comparison table. Best config is highlighted.

Individual per-run metric JSONs are written to a temporary subfolder and
deleted after the sweep - only the consolidated summary JSON and CSV are kept.

Usage:
  python src/hparam_sweep.py --layer 19
  python src/hparam_sweep.py --layer 19 --probe_type mlp
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import PATHS, MODEL_SLUG
from train_probe import train

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def run_sweep(layer: int, probe_type: str, seed: int) -> None:

    # --- Search grid ---
    # pos_weight: how much extra penalty for missing a harmful sample.
    #   1.0 = treat both classes equally (ignores imbalance)
    #   3.0 = missing a harmful prompt costs 3x more than a false alarm
    #   9.0 = very aggressive, matches the 9:1 class ratio exactly
    pos_weights = [1.0, 3.0, 9.0]

    # lr: how fast the probe learns. Too high = unstable, too low = slow
    lrs = [1e-3, 3e-4]

    # weight_decay: regularisation strength, prevents overfitting
    weight_decays = [1e-4, 1e-3]

    configs = list(product(pos_weights, lrs, weight_decays))
    print(f"Running {len(configs)} hyperparameter combinations on layer {layer}...\n")

    # Use a temporary directory for per-run metric files so they don't
    # clutter outputs/metrics/. Only the consolidated summary is kept.
    tmp_metrics_dir = Path(tempfile.mkdtemp(prefix="hparam_sweep_tmp_"))
    original_metrics_path = PATHS["metrics"]

    results = []

    try:
        for i, (pw, lr, wd) in enumerate(configs):
            run_name = f"hparam_sweep_layer{layer}_{probe_type}_pw{pw}_lr{lr}_wd{wd}"
            print(f"[{i+1}/{len(configs)}] pos_weight={pw}, lr={lr}, weight_decay={wd}")

            # Temporarily redirect metrics writes to the temp folder so
            # save_metrics() inside train() doesn't pollute outputs/metrics/
            PATHS["metrics"] = tmp_metrics_dir

            metrics = train(
                layer=layer,
                probe_type=probe_type,
                lr=lr,
                weight_decay=wd,
                epochs=50,
                batch_size=64,
                patience=7,
                seed=seed,
                pos_weight=pw,
                harmful_frac=None,
                run_name=run_name,
                save_curves=False,   # don't save a curve png for every single run
            )

            # Restore real metrics path immediately after each run
            PATHS["metrics"] = original_metrics_path

            results.append({
                "pos_weight": pw,
                "lr": lr,
                "weight_decay": wd,
                "val_pr_auc": round(metrics["val_pr_auc"], 4),
                "val_roc_auc": round(metrics["val_roc_auc"], 4),
                "test_roc_auc": round(metrics["test_roc_auc"], 4),
                "test_pr_auc": round(metrics["test_pr_auc"], 4),
                "test_f1": round(metrics["test_f1_harmful_at_best_thresh"], 4),
                "best_epoch": metrics["best_epoch"],
            })
            print()

    finally:
        # Always restore the real path and delete temp folder, even if a run crashes
        PATHS["metrics"] = original_metrics_path
        shutil.rmtree(tmp_metrics_dir, ignore_errors=True)

    # Sort by val PR-AUC (what we optimise during training)
    results.sort(key=lambda r: r["val_pr_auc"], reverse=True)
    best = results[0]

    # Print table
    df = pd.DataFrame(results)
    print("\n========== HYPERPARAMETER SWEEP RESULTS ==========")
    print(df.to_string(index=False))
    print(f"\nBest config (by val PR-AUC):")
    print(f"  pos_weight={best['pos_weight']}, lr={best['lr']}, "
          f"weight_decay={best['weight_decay']}")
    print(f"  val PR-AUC={best['val_pr_auc']}, test ROC-AUC={best['test_roc_auc']}, "
          f"test F1={best['test_f1']}")

    # Save consolidated results JSON 
    out_json = PATHS["metrics"] / f"hparam_sweep_layer{layer}_{probe_type}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved -> {out_json}")

    # Save results CSV 
    out_csv = PATHS["metrics"] / f"hparam_sweep_layer{layer}_{probe_type}.csv"
    df.to_csv(out_csv, index=False)
    print(f"CSV saved -> {out_csv}")

    # --- Plot: pos_weight effect on val PR-AUC (main knob for imbalance) ---
    # Average over lr/wd to isolate pos_weight effect
    pw_results: dict[float, list[float]] = {}
    for r in results:
        pw = r["pos_weight"]
        if pw not in pw_results:
            pw_results[pw] = []
        pw_results[pw].append(r["val_pr_auc"])

    pw_vals = sorted(pw_results.keys())
    pw_means = [sum(pw_results[pw]) / len(pw_results[pw]) for pw in pw_vals]
    pw_bests = [max(pw_results[pw]) for pw in pw_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: effect of pos_weight
    axes[0].plot(pw_vals, pw_means, "o-", color="#4C72B0", lw=2, label="Mean val PR-AUC")
    axes[0].plot(pw_vals, pw_bests, "s--", color="#DD8452", lw=2, label="Best val PR-AUC")
    axes[0].set_xlabel("pos_weight")
    axes[0].set_ylabel("Val PR-AUC")
    axes[0].set_title("Effect of pos_weight on Val PR-AUC\n(averaged over lr & weight_decay)")
    axes[0].set_xticks(pw_vals)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: full result scatter (test ROC-AUC vs val PR-AUC, coloured by pos_weight)
    colors = {1.0: "#4C72B0", 3.0: "#DD8452", 9.0: "#55A868"}
    seen_pw: set[float] = set()
    for r in results:
        pw = r["pos_weight"]
        axes[1].scatter(
            r["val_pr_auc"], r["test_roc_auc"],
            color=colors[pw],
            s=80, zorder=3,
            label=f"pw={pw}" if pw not in seen_pw else "",
        )
        seen_pw.add(pw)

    # Mark best
    axes[1].scatter(best["val_pr_auc"], best["test_roc_auc"],
                    color="red", s=200, marker="*", zorder=5, label="Best")
    axes[1].set_xlabel("Val PR-AUC")
    axes[1].set_ylabel("Test ROC-AUC")
    axes[1].set_title("All Configs: Val PR-AUC vs Test ROC-AUC")
    axes[1].grid(alpha=0.3)

    # Deduplicate legend
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys())

    plt.suptitle(
        f"Hyperparameter Sweep — {probe_type.upper()} probe, Layer {layer}\n"
        f"Best: pos_weight={best['pos_weight']}, lr={best['lr']}, wd={best['weight_decay']}",
        fontsize=11
    )
    plt.tight_layout()

    out_fig = PATHS["figures"] / f"hparam_sweep_layer{layer}_{probe_type}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Sweep plot saved -> {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=19,
                        help="Which layer's features to sweep over.")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PATHS["figures"].mkdir(parents=True, exist_ok=True)
    PATHS["metrics"].mkdir(parents=True, exist_ok=True)
    PATHS["checkpoints"].mkdir(parents=True, exist_ok=True)

    run_sweep(layer=args.layer, probe_type=args.probe_type, seed=args.seed)


if __name__ == "__main__":
    main()