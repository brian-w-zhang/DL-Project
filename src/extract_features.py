#!/usr/bin/env python3
"""
Extract hidden-state features from a frozen LLM for each prompt split.

Usage:
  python src/extract_features.py
  python src/extract_features.py --layers 4 14 26 --pool last --batch_size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import (
    MODEL_NAME,
    MODEL_SLUG,
    PATHS,
    get_device,
    load_split,
    save_features,
    set_seed,
)

DEFAULT_SPLITS = ["train", "val", "test", "test_external"]


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    """
    hidden: (batch, seq_len, hidden_dim)
    Returns: (batch, hidden_dim)
    """
    if method == "last":
        # Index of last non-padding token for each sequence
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        idx = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)
    elif method == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def extract_all(
    layers: list[int],
    pool: str,
    batch_size: int,
    max_length: int,
    seed: int,
    out_slug: str | None = None,
    splits: list[str] | None = None,
) -> None:
    active_splits = splits if splits is not None else DEFAULT_SPLITS
    # Skip splits whose CSV doesn't exist yet (e.g. test_external before wildguardtest processing)
    active_splits = [s for s in active_splits if (PATHS["data_processed"] / f"{s}.csv").exists()]

    set_seed(seed)
    device = get_device()
    slug = out_slug or MODEL_SLUG
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME}  |  saving as slug: {slug}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    config_out = {
        "model": MODEL_NAME,
        "model_slug": slug,
        "layers": layers,
        "pool": pool,
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": seed,
    }
    config_path = PATHS["features"] / slug / "extraction_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)

    for split in active_splits:
        print(f"\n--- Split: {split} ---")
        prompts, labels = load_split(split)
        n = len(prompts)

        # Pre-allocate storage per layer
        # We'll collect into lists then stack
        layer_vecs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        for start in tqdm(range(0, n, batch_size), desc=split):
            batch_prompts = prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            # out.hidden_states is a tuple of (num_layers+1) tensors, each (batch, seq, dim)
            for layer_idx in layers:
                h = out.hidden_states[layer_idx]  # (batch, seq, dim)
                vec = pool_hidden(h, attention_mask, pool)  # (batch, dim)
                layer_vecs[layer_idx].append(vec.cpu())

        y_tensor = torch.tensor(labels, dtype=torch.float32)

        for layer_idx in layers:
            X = torch.cat(layer_vecs[layer_idx], dim=0)  # (N, dim)
            save_features(X, y_tensor, slug, layer_idx, split)
            print(f"  Layer {layer_idx}: saved X={tuple(X.shape)}, y={tuple(y_tensor.shape)}")

    print("\nExtraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 14, 26])
    parser.add_argument("--pool", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_slug", type=str, default=None,
                        help="Override save slug (default: model slug). Use to avoid overwriting existing features.")
    parser.add_argument("--splits", nargs="+", type=str, default=None,
                        help=f"Splits to extract (default: {DEFAULT_SPLITS}). Missing CSVs are skipped.")
    args = parser.parse_args()

    extract_all(
        layers=args.layers,
        pool=args.pool,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        out_slug=args.out_slug,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
