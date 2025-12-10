"""
Compare EgoHands vs Unified training results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_history(path):
    """Load training history from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'train_acc' in data:
        return {
            'train_acc': [x * 100 for x in data['train_acc']],
            'val_acc': [x * 100 for x in data['val_acc']],
            'train_loss': data['train_loss'],
            'val_loss': data['val_loss'],
        }
    return None


def compare_experiments():
    """Compare EgoHands vs Unified training."""
    
    # Load histories
    egohands_path = Path('results/egohands_visualized/training_history.json')
    unified_path = Path('results/unified_egocentric/training_history.json')
    
    egohands_data = load_training_history(egohands_path) if egohands_path.exists() else None
    unified_data = load_training_history(unified_path) if unified_path.exists() else None
    
    if not egohands_data or not unified_data:
        print("⚠ Missing training history files")
        return
    
    print("="*70)
    print("EgoHands vs Unified Training Comparison")
    print("="*70)
    
    # Final metrics
    print("\n📊 Final Training Metrics:")
    print(f"{'Metric':<25} {'EgoHands':<20} {'Unified':<20} {'Difference':<15}")
    print("-" * 70)
    
    egohands_final_train_acc = egohands_data['train_acc'][-1]
    egohands_final_val_acc = egohands_data['val_acc'][-1]
    egohands_final_train_loss = egohands_data['train_loss'][-1]
    egohands_final_val_loss = egohands_data['val_loss'][-1]
    
    unified_final_train_acc = unified_data['train_acc'][-1]
    unified_final_val_acc = unified_data['val_acc'][-1]
    unified_final_train_loss = unified_data['train_loss'][-1]
    unified_final_val_loss = unified_data['val_loss'][-1]
    
    print(f"{'Final Train Accuracy':<25} {egohands_final_train_acc:<20.2f} {unified_final_train_acc:<20.2f} {unified_final_train_acc - egohands_final_train_acc:<15.2f}")
    print(f"{'Final Val Accuracy':<25} {egohands_final_val_acc:<20.2f} {unified_final_val_acc:<20.2f} {unified_final_val_acc - egohands_final_val_acc:<15.2f}")
    print(f"{'Final Train Loss':<25} {egohands_final_train_loss:<20.4f} {unified_final_train_loss:<20.4f} {unified_final_train_loss - egohands_final_train_loss:<15.4f}")
    print(f"{'Final Val Loss':<25} {egohands_final_val_loss:<20.4f} {unified_final_val_loss:<20.4f} {unified_final_val_loss - egohands_final_val_loss:<15.4f}")
    
    # Overfitting analysis
    egohands_gap = egohands_final_train_acc - egohands_final_val_acc
    unified_gap = unified_final_train_acc - unified_final_val_acc
    
    print(f"\n📈 Overfitting Analysis:")
    print(f"{'Metric':<25} {'EgoHands':<20} {'Unified':<20}")
    print("-" * 65)
    print(f"{'Train-Val Gap':<25} {egohands_gap:<20.2f} {unified_gap:<20.2f}")
    print(f"{'Gap Reduction':<25} {'-':<20} {egohands_gap - unified_gap:<20.2f}")
    
    # Best validation accuracy
    egohands_best_val_acc = max(egohands_data['val_acc'])
    unified_best_val_acc = max(unified_data['val_acc'])
    
    print(f"\n🏆 Best Validation Accuracy:")
    print(f"  EgoHands: {egohands_best_val_acc:.2f}%")
    print(f"  Unified: {unified_best_val_acc:.2f}%")
    print(f"  Improvement: {unified_best_val_acc - egohands_best_val_acc:.2f}%")
    
    # Training stability
    egohands_val_std = np.std(egohands_data['val_acc'][-10:])  # Last 10 epochs
    unified_val_std = np.std(unified_data['val_acc'][-10:])
    
    print(f"\n📉 Training Stability (std of last 10 epochs):")
    print(f"  EgoHands Val Acc Std: {egohands_val_std:.2f}%")
    print(f"  Unified Val Acc Std: {unified_val_std:.2f}%")
    print(f"  {'Unified is more stable' if unified_val_std < egohands_val_std else 'EgoHands is more stable'}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_egohands = range(1, len(egohands_data['train_acc']) + 1)
    epochs_unified = range(1, len(unified_data['train_acc']) + 1)
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs_egohands, egohands_data['train_acc'], 'b-', label='EgoHands Train', linewidth=2)
    ax1.plot(epochs_egohands, egohands_data['val_acc'], 'b--', label='EgoHands Val', linewidth=2)
    ax1.plot(epochs_unified, unified_data['train_acc'], 'r-', label='Unified Train', linewidth=2)
    ax1.plot(epochs_unified, unified_data['val_acc'], 'r--', label='Unified Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss comparison
    ax2 = axes[0, 1]
    ax2.plot(epochs_egohands, egohands_data['train_loss'], 'b-', label='EgoHands Train', linewidth=2)
    ax2.plot(epochs_egohands, egohands_data['val_loss'], 'b--', label='EgoHands Val', linewidth=2)
    ax2.plot(epochs_unified, unified_data['train_loss'], 'r-', label='Unified Train', linewidth=2)
    ax2.plot(epochs_unified, unified_data['val_loss'], 'r--', label='Unified Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Validation accuracy only
    ax3 = axes[1, 0]
    ax3.plot(epochs_egohands, egohands_data['val_acc'], 'b-', label='EgoHands', linewidth=2, marker='o', markersize=4)
    ax3.plot(epochs_unified, unified_data['val_acc'], 'r-', label='Unified', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Validation Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax4 = axes[1, 1]
    egohands_gaps = [t - v for t, v in zip(egohands_data['train_acc'], egohands_data['val_acc'])]
    unified_gaps = [t - v for t, v in zip(unified_data['train_acc'], unified_data['val_acc'])]
    ax4.plot(epochs_egohands, egohands_gaps, 'b-', label='EgoHands', linewidth=2)
    ax4.plot(epochs_unified, unified_gaps, 'r-', label='Unified', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train-Val Gap (%)')
    ax4.set_title('Overfitting Gap Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path('results/egohands_vs_unified_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    
    # Save comparison data
    comparison_data = {
        'egohands': {
            'final_train_acc': egohands_final_train_acc,
            'final_val_acc': egohands_final_val_acc,
            'best_val_acc': egohands_best_val_acc,
            'final_train_loss': egohands_final_train_loss,
            'final_val_loss': egohands_final_val_loss,
            'train_val_gap': egohands_gap,
            'val_std_last_10': float(egohands_val_std),
        },
        'unified': {
            'final_train_acc': unified_final_train_acc,
            'final_val_acc': unified_final_val_acc,
            'best_val_acc': unified_best_val_acc,
            'final_train_loss': unified_final_train_loss,
            'final_val_loss': unified_final_val_loss,
            'train_val_gap': unified_gap,
            'val_std_last_10': float(unified_val_std),
        },
        'improvements': {
            'val_acc_improvement': unified_final_val_acc - egohands_final_val_acc,
            'gap_reduction': egohands_gap - unified_gap,
            'stability_improvement': egohands_val_std - unified_val_std,
        }
    }
    
    comparison_path = Path('results/egohands_vs_unified_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"✓ Comparison data saved to: {comparison_path}")
    print("="*70)


if __name__ == '__main__':
    compare_experiments()

