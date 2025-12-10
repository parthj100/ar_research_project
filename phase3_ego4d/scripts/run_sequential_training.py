"""
Run EgoHands training, then automatically start unified training
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("="*70)
    print("Sequential Training: EgoHands → Unified")
    print("="*70)
    
    # Step 1: Train on EgoHands
    print("\n" + "="*70)
    print("Step 1: Training on EgoHands alone (with visualization)")
    print("="*70)
    
    result1 = subprocess.run([
        sys.executable, 
        str(Path(__file__).parent / 'train_with_visualization.py')
    ])
    
    if result1.returncode != 0:
        print("\n⚠ EgoHands training had errors, but continuing to unified training...")
    
    # Step 2: Train on unified dataset
    print("\n" + "="*70)
    print("Step 2: Training on Unified Dataset (EgoHands + Ego4D)")
    print("="*70)
    
    result2 = subprocess.run([
        sys.executable,
        str(Path(__file__).parent / 'train_unified.py')
    ])
    
    if result2.returncode != 0:
        print("\n⚠ Unified training had errors.")
        return 1
    
    print("\n" + "="*70)
    print("All Training Completed Successfully!")
    print("="*70)
    return 0

if __name__ == '__main__':
    sys.exit(main())

