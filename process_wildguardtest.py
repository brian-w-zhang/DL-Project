#!/usr/bin/env python3
"""
Convert the WildGuardTest HuggingFace dataset into test_external.csv.

This is the official held-out benchmark with human multi-annotator labels —
kept completely separate from train/val/test splits derived from WildGuardTrain.

Usage:
  python3 process_wildguardtest.py
  python3 process_wildguardtest.py --input_hf_path data/raw/wildguardmix/wildguardtest_hf --out_dir data/processed/wildguardmix_imbalance
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hf_path", type=str, default="data/raw/wildguardmix/wildguardtest_hf")
    parser.add_argument("--out_dir", type=str, default="data/processed/wildguardmix_imbalance")
    args = parser.parse_args()

    in_path = Path(args.input_hf_path)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading WildGuardTest from {in_path} ...")
    ds = load_from_disk(str(in_path))
    df = ds.to_pandas()

    df = df[df["prompt"].notna()].copy()
    df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.strip() != ""]
    df = df[df["prompt_harm_label"].isin(["harmful", "unharmful"])].copy()

    df["label"] = (df["prompt_harm_label"] == "harmful").astype(int)

    keep_cols = ["prompt", "label", "prompt_harm_label", "adversarial", "subcategory"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    out_csv = out_dir / "test_external.csv"
    out_parquet = out_dir / "test_external.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)

    n_harmful = int((df["label"] == 1).sum())
    n_benign = int((df["label"] == 0).sum())

    meta = {
        "source": "allenai/wildguardmix wildguardtest",
        "input_hf_path": str(in_path),
        "label_definition": {"normal_0": "prompt_harm_label == unharmful", "abnormal_1": "prompt_harm_label == harmful"},
        "test_external": {"rows": len(df), "normal": n_benign, "abnormal": n_harmful},
    }
    with open(out_dir / "metadata_test_external.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {out_csv}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
