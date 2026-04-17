import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "data_processed": ROOT / "data" / "processed" / "wildguardmix_imbalance",
    "features": ROOT / "features",
    "checkpoints": ROOT / "outputs" / "checkpoints",
    "figures": ROOT / "outputs" / "figures",
    "metrics": ROOT / "outputs" / "metrics",
}

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_SLUG = "qwen2.5-1.5b"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str) -> tuple[list[str], list[int]]:
    """Load a CSV split and return (prompts, labels)."""
    path = PATHS["data_processed"] / f"{split}.csv"
    df = pd.read_csv(path)
    prompts = df["prompt"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return prompts, labels


# ---------------------------------------------------------------------------
# Feature paths
# ---------------------------------------------------------------------------

def feature_dir(model_slug: str, layer: int) -> Path:
    d = PATHS["features"] / model_slug / f"layer_{layer}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_features(X: torch.Tensor, y: torch.Tensor, model_slug: str, layer: int, split: str) -> None:
    d = feature_dir(model_slug, layer)
    torch.save(X, d / f"X_{split}.pt")
    torch.save(y, d / f"y_{split}.pt")


def load_features(model_slug: str, layer: int, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = feature_dir(model_slug, layer)
    X = torch.load(d / f"X_{split}.pt", weights_only=True)
    y = torch.load(d / f"y_{split}.pt", weights_only=True)
    return X, y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, name: str) -> None:
    path = PATHS["metrics"] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {path}")


def load_metrics(name: str) -> dict:
    path = PATHS["metrics"] / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
