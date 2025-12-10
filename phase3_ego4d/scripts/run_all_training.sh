#!/bin/bash
# Run EgoHands training, then unified training sequentially

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=========================================="
echo "Step 1: Training on EgoHands alone"
echo "=========================================="
python scripts/train_with_visualization.py

echo ""
echo "=========================================="
echo "Step 2: Training on Unified Dataset"
echo "=========================================="
python scripts/train_unified.py

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="

