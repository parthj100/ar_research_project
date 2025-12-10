"""
Helper script to download and prepare egocentric datasets

Provides instructions and helper functions for downloading:
- EPIC-KITCHENS
- EgoHands
- Additional Ego4D data
"""

import os
import subprocess
from pathlib import Path


def print_epic_kitchens_instructions():
    """Print instructions for downloading EPIC-KITCHENS."""
    print("\n" + "="*70)
    print("EPIC-KITCHENS Dataset Download Instructions")
    print("="*70)
    print("""
1. Visit: https://epic-kitchens.github.io/
2. Sign the data agreement (free for research)
3. Download annotations:
   - Go to "Downloads" section
   - Download "EPIC_train_action_labels.pkl" or JSON format
   - Place in: data/epic_kitchens/annotations/

4. Download videos (optional, can use pre-extracted frames):
   - Download videos from the official repository
   - Place in: data/epic_kitchens/videos/

5. Or extract frames from videos:
   - Use provided scripts or extract manually
   - Place frames in: data/epic_kitchens/frames/{video_id}/

Alternative: Use EPIC-KITCHENS-100 (smaller subset)
  - 100 videos subset for faster testing
  - Available on HuggingFace: datasets/epic-kitchens-100
    """)
    print("="*70)


def print_egohands_instructions():
    """Print instructions for downloading EgoHands."""
    print("\n" + "="*70)
    print("EgoHands Dataset Download Instructions")
    print("="*70)
    print("""
1. Visit: https://vision.soic.indiana.edu/projects/egohands/
2. Download the dataset (free, no signup required)
3. Extract to: data/egohands/
4. Expected structure:
   data/egohands/
   └── _LABELLED_SAMPLES/
       ├── CARDS_COURTYARD_B_T/
       │   ├── frame_0000.jpg
       │   └── ...
       ├── JENGA_B_S/
       └── ...

The dataset includes:
- 48 videos
- 4,800 frames
- Hand segmentation masks (optional for our use case)
    """)
    print("="*70)


def check_dataset_availability():
    """Check which datasets are available."""
    print("\n" + "="*70)
    print("Dataset Availability Check")
    print("="*70)
    
    datasets_status = {}
    
    # Check Ego4D
    ego4d_path = Path('data/ego4d/annotations.json')
    if ego4d_path.exists():
        with open(ego4d_path) as f:
            import json
            data = json.load(f)
            count = len(data) if isinstance(data, list) else 0
        datasets_status['Ego4D'] = {
            'available': True,
            'samples': count,
            'path': 'data/ego4d/'
        }
    else:
        datasets_status['Ego4D'] = {
            'available': False,
            'samples': 0,
            'path': 'data/ego4d/'
        }
    
    # Check EPIC-KITCHENS
    epic_path = Path('data/epic_kitchens')
    if epic_path.exists():
        # Try to find annotation files
        ann_files = list(epic_path.rglob('*.json')) + list(epic_path.rglob('*.pkl'))
        if ann_files:
            datasets_status['EPIC-KITCHENS'] = {
                'available': True,
                'samples': 'unknown',
                'path': 'data/epic_kitchens/'
            }
        else:
            datasets_status['EPIC-KITCHENS'] = {
                'available': False,
                'samples': 0,
                'path': 'data/epic_kitchens/'
            }
    else:
        datasets_status['EPIC-KITCHENS'] = {
            'available': False,
            'samples': 0,
            'path': 'data/epic_kitchens/'
        }
    
    # Check EgoHands
    hands_path = Path('data/egohands/_LABELLED_SAMPLES')
    if hands_path.exists():
        video_dirs = [d for d in hands_path.iterdir() if d.is_dir()]
        total_frames = sum(len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png'))) for d in video_dirs)
        datasets_status['EgoHands'] = {
            'available': True,
            'samples': f"{len(video_dirs)} videos, ~{total_frames} frames",
            'path': 'data/egohands/'
        }
    else:
        datasets_status['EgoHands'] = {
            'available': False,
            'samples': 0,
            'path': 'data/egohands/'
        }
    
    # Print status
    for name, status in datasets_status.items():
        if status['available']:
            print(f"✅ {name}: Available")
            print(f"   Samples: {status['samples']}")
            print(f"   Path: {status['path']}")
        else:
            print(f"❌ {name}: Not found")
            print(f"   Path: {status['path']}")
    
    print("\n" + "="*70)
    
    return datasets_status


def try_download_epic_kitchens_hf():
    """Try to download EPIC-KITCHENS from HuggingFace (if available)."""
    try:
        from datasets import load_dataset
        print("\nAttempting to download EPIC-KITCHENS from HuggingFace...")
        
        # Try EPIC-KITCHENS-100 (smaller subset)
        dataset = load_dataset("epic-kitchens-100", split="train")
        print(f"✓ Loaded {len(dataset)} samples from HuggingFace")
        print("  Note: You'll need to process this into our format")
        return True
    except ImportError:
        print("⚠ HuggingFace datasets not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"⚠ Could not load from HuggingFace: {e}")
        return False


def main():
    """Main function to check and provide download instructions."""
    print("="*70)
    print("Egocentric Dataset Download Helper")
    print("="*70)
    
    # Check availability
    status = check_dataset_availability()
    
    # Count available datasets
    available = sum(1 for s in status.values() if s['available'])
    
    if available == 0:
        print("\n⚠ No datasets found!")
        print("\nYou need to download at least one dataset:")
        print_epic_kitchens_instructions()
        print_egohands_instructions()
    elif available < 3:
        print(f"\n✓ Found {available}/3 datasets")
        print("\nTo add more datasets:")
        if not status['EPIC-KITCHENS']['available']:
            print_epic_kitchens_instructions()
        if not status['EgoHands']['available']:
            print_egohands_instructions()
    else:
        print("\n✅ All datasets available!")
        print("You can now train on the combined dataset.")
    
    # Try HuggingFace as alternative
    print("\n" + "="*70)
    print("Alternative: HuggingFace Datasets")
    print("="*70)
    print("Some datasets may be available on HuggingFace:")
    print("  - EPIC-KITCHENS-100: datasets/epic-kitchens-100")
    print("  - Install: pip install datasets")
    print("  - Then use: from datasets import load_dataset")
    print("="*70)


if __name__ == '__main__':
    main()

