#!/bin/bash
# Quick status check script

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Unified Training Status Check"
echo "=========================================="

# Check if process is running
if ps aux | grep -v grep | grep -q "train_unified.py"; then
    echo "✓ Training process is running"
else
    echo "⚠ Training process not found"
fi

# Check current epoch
if [ -f "results/unified_egocentric/training_history.json" ]; then
    python3 -c "
import json
from pathlib import Path
path = Path('results/unified_egocentric/training_history.json')
data = json.load(open(path))
if isinstance(data, dict) and 'train_acc' in data:
    epoch = len(data['train_acc'])
    train_acc = data['train_acc'][-1] * 100
    val_acc = data['val_acc'][-1] * 100
    print(f'Current Epoch: {epoch}/30')
    print(f'Latest Train Acc: {train_acc:.2f}%')
    print(f'Latest Val Acc: {val_acc:.2f}%')
    
    if epoch >= 30:
        print('')
        print('✅ TRAINING COMPLETE!')
    else:
        remaining = 30 - epoch
        print(f'Remaining: ~{remaining * 2} minutes (estimated)')
"
else
    echo "Training history not available yet"
fi

# Check log for completion
if [ -f "results/unified_egocentric/training.log" ]; then
    if grep -q "Training complete!" "results/unified_egocentric/training.log"; then
        echo ""
        echo "✅ Training marked as complete in log!"
    fi
fi

echo "=========================================="


