"""
Download EgoHands Dataset

EgoHands is freely available and doesn't require signup.
This script helps download and set it up.
"""

import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def download_egohands():
    """Download EgoHands dataset."""
    print("="*70)
    print("EgoHands Dataset Download")
    print("="*70)
    
    data_dir = Path('data/egohands')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEgoHands Download Instructions:")
    print("="*70)
    print("""
1. Visit: https://vision.soic.indiana.edu/projects/egohands/
2. Click on "Download Dataset" (direct link usually available)
3. Download the zip file
4. Extract to: data/egohands/

OR use direct download if URL is available:
    """)
    
    # Try to find if there's a direct download URL
    print("\nChecking for direct download...")
    
    # Common EgoHands download URLs (may change)
    possible_urls = [
        "https://vision.soic.indiana.edu/egohands_files/egohands_data.zip",
        "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip",
    ]
    
    download_url = None
    for url in possible_urls:
        try:
            # Check if URL exists
            response = urllib.request.urlopen(url, timeout=5)
            if response.status == 200:
                download_url = url
                break
        except:
            continue
    
    if download_url:
        print(f"✓ Found download URL: {download_url}")
        print("\nDownloading automatically...")
        
        zip_path = data_dir / 'egohands_data.zip'
        print(f"\nDownloading to {zip_path}...")
        print("This may take a while (dataset is ~500MB)...")
        
        try:
            urllib.request.urlretrieve(download_url, zip_path)
            print("✓ Download complete!")
            
            print("\nExtracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print("✓ Extraction complete!")
            print(f"\nDataset should be at: {data_dir}/_LABELLED_SAMPLES/")
            
            # Clean up zip
            zip_path.unlink()
            print("✓ Cleaned up zip file")
            
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nPlease download manually from the website.")
            return False
    else:
        print("⚠ Direct download not available.")
        print("\nPlease download manually:")
        print("  1. Go to: https://vision.soic.indiana.edu/projects/egohands/")
        print("  2. Download the dataset zip file")
        print("  3. Extract to: data/egohands/")
        print("  4. Expected structure: data/egohands/_LABELLED_SAMPLES/...")
    
    return False


def verify_egohands():
    """Verify EgoHands dataset is properly set up."""
    data_dir = Path('data/egohands/_LABELLED_SAMPLES')
    
    if not data_dir.exists():
        print("❌ EgoHands dataset not found!")
        print(f"   Expected: {data_dir}")
        return False
    
    # Count videos and frames
    video_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    total_frames = 0
    
    for video_dir in video_dirs:
        frames = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
        total_frames += len(frames)
    
    print("="*70)
    print("EgoHands Dataset Verification")
    print("="*70)
    print(f"✓ Dataset found at: {data_dir}")
    print(f"  Videos: {len(video_dirs)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Estimated clips (8 frames each): ~{total_frames // 8}")
    
    if len(video_dirs) > 0:
        print(f"\nSample videos:")
        for video_dir in sorted(video_dirs)[:5]:
            frames = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
            print(f"  - {video_dir.name}: {len(frames)} frames")
    
    return True


if __name__ == '__main__':
    print("="*70)
    print("EgoHands Dataset Setup")
    print("="*70)
    
    # Check if already downloaded
    if verify_egohands():
        print("\n✅ EgoHands dataset is already set up!")
    else:
        print("\nDownloading EgoHands...")
        if download_egohands():
            verify_egohands()

