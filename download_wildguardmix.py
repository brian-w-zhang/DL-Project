#!/usr/bin/env python3
"""
Download WildGuardMix and save project-local copies.

Usage:
  python3 download_wildguardmix.py
  python3 download_wildguardmix.py --out_dir data/raw/wildguardmix
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/raw/wildguardmix",
        help="Output directory for saved datasets.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WildGuardMix train split...")
    train = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")

    print("Loading WildGuardMix test split...")
    test = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    # Save HF-native format for fast reload.
    train_hf_path = out_dir / "wildguardtrain_hf"
    test_hf_path = out_dir / "wildguardtest_hf"
    print(f"Saving HF format to: {train_hf_path} and {test_hf_path}")
    train.save_to_disk(str(train_hf_path))
    test.save_to_disk(str(test_hf_path))

    # Save parquet for easier ad-hoc inspection.
    train_parquet = out_dir / "wildguardtrain.parquet"
    test_parquet = out_dir / "wildguardtest.parquet"
    print(f"Saving parquet to: {train_parquet} and {test_parquet}")
    train.to_parquet(str(train_parquet))
    test.to_parquet(str(test_parquet))

    print("Done.")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Files saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

