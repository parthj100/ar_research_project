#!/bin/bash
# Automated monitoring that checks every minute and notifies when complete

cd "$(dirname "$0")/.."

echo "🔍 Starting automated training monitor..."
echo "   Checking every 60 seconds"
echo "   Will notify when training completes"
echo ""

CHECK_INTERVAL=60
LAST_EPOCH=0

while true; do
    # Check if training is still running
    if ! ps aux | grep -v grep | grep -q "train_unified.py"; then
        # Process stopped, check if it completed successfully
        if [ -f "results/unified_egocentric/training_history.json" ]; then
            EPOCH=$(python3 -c "
import json
from pathlib import Path
try:
    data = json.load(open('results/unified_egocentric/training_history.json'))
    if isinstance(data, dict) and 'train_acc' in data:
        print(len(data['train_acc']))
    else:
        print(0)
except:
    print(0)
")
            
            if [ "$EPOCH" -ge 30 ]; then
                echo ""
                echo "═══════════════════════════════════════════════════════════════"
                echo "✅ TRAINING COMPLETE!"
                echo "═══════════════════════════════════════════════════════════════"
                echo ""
                
                # Get final metrics
                python3 -c "
import json
from pathlib import Path
data = json.load(open('results/unified_egocentric/training_history.json'))
if isinstance(data, dict) and 'train_acc' in data:
    train_acc = data['train_acc'][-1] * 100
    val_acc = data['val_acc'][-1] * 100
    train_loss = data['train_loss'][-1]
    val_loss = data['val_loss'][-1]
    print(f'Final Epoch: {len(data[\"train_acc\"])}/30')
    print(f'Final Train Accuracy: {train_acc:.2f}%')
    print(f'Final Val Accuracy: {val_acc:.2f}%')
    print(f'Final Train Loss: {train_loss:.4f}')
    print(f'Final Val Loss: {val_loss:.4f}')
"
                
                echo ""
                echo "═══════════════════════════════════════════════════════════════"
                echo "Next Steps:"
                echo "  1. Measure deployment metrics (latency, bandwidth)"
                echo "  2. Compare EgoHands vs Unified results"
                echo "  3. Generate comprehensive summary"
                echo "═══════════════════════════════════════════════════════════════"
                
                # Make a sound notification (if on macOS)
                if command -v say &> /dev/null; then
                    say "Training complete!"
                fi
                
                break
            else
                echo "⚠ Training process stopped but not at 30 epochs (at $EPOCH)"
                echo "   Check logs: results/unified_egocentric/training.log"
                break
            fi
        else
            echo "⚠ Training process stopped but no history file found"
            echo "   Check logs: results/unified_egocentric/training.log"
            break
        fi
    fi
    
    # Check current progress
    if [ -f "results/unified_egocentric/training_history.json" ]; then
        CURRENT_EPOCH=$(python3 -c "
import json
from pathlib import Path
try:
    data = json.load(open('results/unified_egocentric/training_history.json'))
    if isinstance(data, dict) and 'train_acc' in data:
        epoch = len(data['train_acc'])
        train_acc = data['train_acc'][-1] * 100
        val_acc = data['val_acc'][-1] * 100
        print(f'{epoch}|{train_acc:.1f}|{val_acc:.1f}')
    else:
        print('0|0|0')
except:
    print('0|0|0')
")
        
        EPOCH=$(echo $CURRENT_EPOCH | cut -d'|' -f1)
        TRAIN_ACC=$(echo $CURRENT_EPOCH | cut -d'|' -f2)
        VAL_ACC=$(echo $CURRENT_EPOCH | cut -d'|' -f3)
        
        if [ "$EPOCH" -gt "$LAST_EPOCH" ]; then
            echo "[$(date +%H:%M:%S)] Epoch $EPOCH/30 - Train: ${TRAIN_ACC}% | Val: ${VAL_ACC}%"
            LAST_EPOCH=$EPOCH
            
            if [ "$EPOCH" -ge 30 ]; then
                echo ""
                echo "✅ Training reached 30 epochs! Waiting for final checkpoint..."
                sleep 10
            fi
        else
            echo -n "."
        fi
    else
        echo -n "."
    fi
    
    sleep $CHECK_INTERVAL
done


