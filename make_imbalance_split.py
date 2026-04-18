#!/usr/bin/env python3
"""
Create imbalanced train/val/test splits from WildGuardMix (WildGuardTrain only).

Default label mapping:
- normal (0): prompt_harm_label == "unharmful"
- abnormal (1): prompt_harm_label == "harmful"

Default split (90:10 imbalance, 70/10/20 train/val/test of 10k total):
  train: 7000 (700 harmful, 6300 benign)
  val:   1000 (100 harmful, 900 benign)
  test:  2000 (200 harmful, 1800 benign)  <- internal test, same distribution

For the external benchmark test use process_wildguardtest.py to generate test_external.csv.

Usage:
  python3 make_imbalance_split.py
  python3 make_imbalance_split.py --ratio_normal_to_abnormal 9 --train_abnormal 700 --val_abnormal 100 --test_abnormal 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def _sample_block(df: pd.DataFrame, n_normal: int, n_abnormal: int, seed: int) -> pd.DataFrame:
    normal_df = df[df["label"] == 0]
    abnormal_df = df[df["label"] == 1]
    sampled_normal = normal_df.sample(n=n_normal, random_state=seed, replace=False)
    sampled_abnormal = abnormal_df.sample(n=n_abnormal, random_state=seed, replace=False)
    out = pd.concat([sampled_normal, sampled_abnormal], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hf_path", type=str, default="data/raw/wildguardmix/wildguardtrain_hf")
    parser.add_argument("--out_dir", type=str, default="data/processed/wildguardmix_imbalance")
    parser.add_argument("--ratio_normal_to_abnormal", type=int, default=9, help="9 means 90:10")
    parser.add_argument("--train_abnormal", type=int, default=700)
    parser.add_argument("--val_abnormal", type=int, default=100)
    parser.add_argument("--test_abnormal", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input_hf_path)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset from {in_path} ...")
    ds = load_from_disk(str(in_path))
    df = ds.to_pandas()

    # Keep rows usable for prompt-level harm classification.
    df = df[df["prompt"].notna()].copy()
    df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.strip() != ""]
    df = df[df["prompt_harm_label"].isin(["harmful", "unharmful"])].copy()

    # Map labels: normal=0, abnormal=1
    df["label"] = (df["prompt_harm_label"] == "harmful").astype(int)

    # Keep only columns needed downstream; preserve metadata for analysis.
    keep_cols = ["prompt", "label", "prompt_harm_label", "adversarial", "subcategory"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    total_normal = int((df["label"] == 0).sum())
    total_abnormal = int((df["label"] == 1).sum())
    print(f"Available rows after filtering -> normal: {total_normal}, abnormal: {total_abnormal}")

    ratio = args.ratio_normal_to_abnormal
    n_ab_train = args.train_abnormal
    n_ab_val = args.val_abnormal
    n_ab_test = args.test_abnormal

    n_no_train = n_ab_train * ratio
    n_no_val = n_ab_val * ratio
    n_no_test = n_ab_test * ratio

    need_abnormal = n_ab_train + n_ab_val + n_ab_test
    need_normal = n_no_train + n_no_val + n_no_test

    if total_abnormal < need_abnormal:
        raise ValueError(f"Not enough abnormal rows. Need {need_abnormal}, found {total_abnormal}.")
    if total_normal < need_normal:
        raise ValueError(f"Not enough normal rows. Need {need_normal}, found {total_normal}.")

    # Global shuffle once, then take disjoint chunks.
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    abnormal_pool = df[df["label"] == 1].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    normal_pool = df[df["label"] == 0].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    ab_train = abnormal_pool.iloc[:n_ab_train]
    ab_val = abnormal_pool.iloc[n_ab_train:n_ab_train + n_ab_val]
    ab_test = abnormal_pool.iloc[n_ab_train + n_ab_val:n_ab_train + n_ab_val + n_ab_test]

    no_train = normal_pool.iloc[:n_no_train]
    no_val = normal_pool.iloc[n_no_train:n_no_train + n_no_val]
    no_test = normal_pool.iloc[n_no_train + n_no_val:n_no_train + n_no_val + n_no_test]

    train_df = pd.concat([ab_train, no_train], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = pd.concat([ab_val, no_val], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat([ab_test, no_test], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save CSV + parquet
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    meta = {
        "input_hf_path": str(in_path),
        "label_definition": {"normal_0": "prompt_harm_label == unharmful", "abnormal_1": "prompt_harm_label == harmful"},
        "ratio_normal_to_abnormal": ratio,
        "seed": args.seed,
        "train": {"rows": len(train_df), "normal": int((train_df["label"] == 0).sum()), "abnormal": int((train_df["label"] == 1).sum())},
        "val": {"rows": len(val_df), "normal": int((val_df["label"] == 0).sum()), "abnormal": int((val_df["label"] == 1).sum())},
        "test": {"rows": len(test_df), "normal": int((test_df["label"] == 0).sum()), "abnormal": int((test_df["label"] == 1).sum())},
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved split files:")
    print(f"- {out_dir / 'train.csv'}")
    print(f"- {out_dir / 'val.csv'}")
    print(f"- {out_dir / 'test.csv'}")
    print(f"- {out_dir / 'metadata.json'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

