"""
Download egocentric datasets from HuggingFace (easier than manual download)

This script helps download datasets that are available on HuggingFace.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
from pathlib import Path


def download_epic_kitchens_100():
    """Download EPIC-KITCHENS-100 from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ HuggingFace datasets not installed.")
        print("  Install with: pip install datasets")
        return False
    
    print("\n" + "="*70)
    print("Downloading EPIC-KITCHENS-100 from HuggingFace")
    print("="*70)
    
    output_dir = Path('data/epic_kitchens')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\nLoading dataset (this may take a while)...")
        dataset = load_dataset("epic-kitchens-100", split="train")
        
        print(f"✓ Loaded {len(dataset)} samples")
        print("\nConverting to our format...")
        
        # Create annotations
        annotations = []
        for i, sample in enumerate(dataset):
            annotations.append({
                'video_id': sample.get('video_id', f'video_{i}'),
                'start_frame': sample.get('start_frame', 0),
                'stop_frame': sample.get('stop_frame', 30),
                'verb': sample.get('verb', 'other'),
                'noun': sample.get('noun', 'object'),
            })
        
        # Save annotations
        ann_file = output_dir / 'annotations' / 'EPIC_train_action_labels.json'
        ann_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ann_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Saved annotations to {ann_file}")
        print(f"  Total clips: {len(annotations)}")
        
        # Note about videos
        print("\n⚠ Note: Videos need to be downloaded separately from EPIC-KITCHENS website")
        print("  Or you can extract frames from the HuggingFace dataset")
        print("  The annotations are ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAlternative: Download manually from https://epic-kitchens.github.io/")
        return False


def main():
    """Main download function."""
    print("="*70)
    print("HuggingFace Dataset Downloader")
    print("="*70)
    
    print("\nAvailable datasets:")
    print("  1. EPIC-KITCHENS-100 (smaller subset, easier to download)")
    print("  2. Full EPIC-KITCHENS (requires manual download)")
    print("  3. EgoHands (requires manual download)")
    
    choice = input("\nWhich dataset to download? (1 for EPIC-KITCHENS-100, or 'q' to quit): ").strip()
    
    if choice == '1':
        download_epic_kitchens_100()
    elif choice.lower() == 'q':
        print("Exiting...")
    else:
        print("Invalid choice. Use '1' for EPIC-KITCHENS-100")


if __name__ == '__main__':
    main()

