#!/usr/bin/env bash
# Extract pooled hidden states for every Qwen2.5-1.5B layer index returned by
# Hugging Face (embedding + 28 blocks => indices 0..28).
#
# Usage (from anywhere):
#   bash scripts/extract_all_layers.sh
#
# Or from project root:
#   bash scripts/extract_all_layers.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python src/extract_features.py --layers $(seq 0 28) "$@"
