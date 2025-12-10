#!/bin/bash
# Monitor Ego4D training progress

cd "$(dirname "$0")/.."

echo "Checking training status..."
echo "================================"

# Check if process is running
if ps aux | grep -v grep | grep -q "train_ego4d.py"; then
    echo "✅ Training is RUNNING"
    ps aux | grep -v grep | grep "train_ego4d.py" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Runtime:", $10}'
else
    echo "❌ Training process NOT FOUND"
fi

echo ""
echo "Checkpoints:"
ls -lht results/ego4d_distill/*.pt 2>/dev/null | head -5 | awk '{print "  " $9, "(" $5, $6, $7, $8 ")"}'

echo ""
echo "Training history:"
if [ -f results/ego4d_distill/training_history.json ]; then
    echo "✅ Found training_history.json"
    # Count epochs
    epochs=$(python3 -c "import json; d=json.load(open('results/ego4d_distill/training_history.json')); print(f\"  Epochs completed: {len(d.get('train_loss', []))}\")" 2>/dev/null)
    echo "$epochs"
else
    echo "  ⏳ Not yet created (training in progress)"
fi

echo ""
echo "================================"

