"""
Monitor training progress and notify when complete
"""

import time
import json
import subprocess
from pathlib import Path
import sys


def check_training_status():
    """Check if unified training is still running."""
    # Check for running process
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )
    
    is_running = 'train_unified.py' in result.stdout
    
    # Check training log for completion
    log_path = Path('results/unified_egocentric/training.log')
    is_complete = False
    last_epoch = None
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            content = f.read()
            if 'Training complete!' in content or 'Epoch 30/30' in content:
                is_complete = True
                # Extract last epoch info
                lines = content.split('\n')
                for line in reversed(lines):
                    if 'Epoch 30/30' in line:
                        last_epoch = 30
                        break
    
    # Check training history
    history_path = Path('results/unified_egocentric/training_history.json')
    current_epoch = 0
    if history_path.exists():
        try:
            with open(history_path) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'train_acc' in data:
                    current_epoch = len(data['train_acc'])
                elif isinstance(data, list):
                    current_epoch = len(data)
        except:
            pass
    
    return {
        'is_running': is_running,
        'is_complete': is_complete,
        'current_epoch': current_epoch,
        'last_epoch': last_epoch,
    }


def get_training_metrics():
    """Get current training metrics if available."""
    history_path = Path('results/unified_egocentric/training_history.json')
    
    if not history_path.exists():
        return None
    
    try:
        with open(history_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'train_acc' in data:
            train_acc = data['train_acc']
            val_acc = data['val_acc']
            train_loss = data['train_loss']
            val_loss = data['val_loss']
            
            if len(train_acc) > 0:
                return {
                    'epoch': len(train_acc),
                    'train_acc': train_acc[-1] * 100,
                    'val_acc': val_acc[-1] * 100,
                    'train_loss': train_loss[-1],
                    'val_loss': val_loss[-1],
                }
    except Exception as e:
        print(f"Error reading metrics: {e}")
    
    return None


def monitor_training(check_interval=60):
    """Monitor training with periodic checks."""
    print("="*70)
    print("Training Monitor - Unified Dataset (EgoHands + Ego4D)")
    print("="*70)
    print(f"Checking every {check_interval} seconds...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_epoch = 0
    
    try:
        while True:
            status = check_training_status()
            metrics = get_training_metrics()
            
            if status['is_complete']:
                print("\n" + "="*70)
                print("✅ TRAINING COMPLETE!")
                print("="*70)
                
                if metrics:
                    print(f"\nFinal Results:")
                    print(f"  Total Epochs: {metrics['epoch']}")
                    print(f"  Final Train Accuracy: {metrics['train_acc']:.2f}%")
                    print(f"  Final Val Accuracy: {metrics['val_acc']:.2f}%")
                    print(f"  Final Train Loss: {metrics['train_loss']:.4f}")
                    print(f"  Final Val Loss: {metrics['val_loss']:.4f}")
                
                print("\n" + "="*70)
                print("Next Steps:")
                print("  1. Measure deployment metrics (latency, bandwidth)")
                print("  2. Compare EgoHands vs Unified results")
                print("  3. Generate comprehensive summary")
                print("="*70)
                break
            
            elif not status['is_running'] and status['current_epoch'] == 0:
                print("⚠ Training process not found. It may have completed or crashed.")
                print("   Check logs: results/unified_egocentric/training.log")
                break
            
            else:
                # Show progress
                current = status['current_epoch'] or 0
                if current > last_epoch:
                    print(f"\n📊 Progress Update:")
                    print(f"  Current Epoch: {current}/30")
                    if metrics:
                        print(f"  Train Acc: {metrics['train_acc']:.2f}%")
                        print(f"  Val Acc: {metrics['val_acc']:.2f}%")
                    last_epoch = current
                else:
                    print(".", end="", flush=True)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Training may still be running in the background.")


if __name__ == '__main__':
    check_interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    monitor_training(check_interval)


