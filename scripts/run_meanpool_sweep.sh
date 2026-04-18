#!/usr/bin/env bash
# Run the full mean-pooling ablation sweep:
#   1. Extract all 28 layers with mean pooling → features/qwen2.5-1.5b-meanpool/
#   2. Train a linear probe for every layer and produce the sweep plot.
#
# Existing last-token features and checkpoints are NOT touched.
# Results save to:
#   outputs/metrics/full_sweep_linear_qwen2.5-1.5b-meanpool.json
#   outputs/figures/full_layer_sweep_linear_meanpool.png  (after notebook cell)
#
# Usage (from project root, with .venv activated):
#   bash scripts/run_meanpool_sweep.sh

set -e
cd "$(dirname "$0")/.."

SLUG="qwen2.5-1.5b-meanpool"

echo "============================================"
echo "Step 1/2  Extracting all 28 layers (mean pool)"
echo "============================================"
python src/extract_features.py \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
    --pool mean \
    --batch_size 16 \
    --out_slug "$SLUG"

echo ""
echo "============================================"
echo "Step 2/2  Sweeping all layers (linear probe)"
echo "============================================"
python src/sweep_all_layers.py \
    --probe_type linear \
    --model_slug "$SLUG"

echo ""
echo "Done. Summary: outputs/metrics/full_sweep_linear_${SLUG}.json"
