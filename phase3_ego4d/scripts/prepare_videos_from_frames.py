"""
Helper script to create test videos from extracted frames

This allows testing the video evaluation pipeline even when original
video files aren't available. It reconstructs videos from frame sequences.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
import json


def frames_to_video(
    frame_paths: List[Path],
    output_path: Path,
    fps: int = 30,
    image_size: tuple = (224, 224),
):
    """Convert a sequence of frames to a video file."""
    if len(frame_paths) == 0:
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        return False
    
    h, w = first_frame.shape[:2]
    
    # Resize if needed
    if image_size:
        h, w = image_size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Resize if needed
        if image_size:
            frame = cv2.resize(frame, image_size)
        
        out.write(frame)
    
    out.release()
    return True


def create_videos_from_ego4d_frames(
    frames_dir: Path = Path('data/ego4d/frames'),
    videos_dir: Path = Path('data/ego4d/videos'),
    annotations_path: Path = Path('data/ego4d/annotations.json'),
    max_videos: int = 10,
):
    """Create videos from Ego4D frame sequences."""
    
    if not frames_dir.exists():
        print(f"⚠ Frames directory not found: {frames_dir}")
        return
    
    if not annotations_path.exists():
        print(f"⚠ Annotations not found: {annotations_path}")
        return
    
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    print(f"Creating videos from {len(annotations)} clips...")
    print(f"  (Limited to {max_videos} for testing)")
    
    created = 0
    for i, ann in enumerate(annotations[:max_videos]):
        video_id = ann.get('video_id', f'video_{i:03d}')
        action_label = ann.get('action_label', 'unknown')
        
        # Find frames for this video - try multiple patterns
        frames = []
        patterns = [
            f"{video_id}_frame_*.jpg",
            f"{video_id}_frame_*.png",
            f"*{video_id}*frame*.jpg",
            f"*{video_id}*frame*.png",
        ]
        
        for pattern in patterns:
            frames = sorted(list(frames_dir.glob(pattern)))
            if len(frames) >= 8:
                break
        
        # If still no frames, use sampled_frames from annotation
        if len(frames) < 8 and 'sampled_frames' in ann:
            frame_paths = [Path(f) for f in ann['sampled_frames']]
            frames = [f for f in frame_paths if f.exists()]
        
        # If still no frames, try to find any frames with video_id in name
        if len(frames) < 8:
            all_frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
            video_id_str = f"video_{video_id:03d}" if isinstance(video_id, int) else str(video_id)
            frames = [f for f in all_frames if video_id_str in f.name]
            frames = sorted(frames)[:32]  # Limit to reasonable number
        
        if len(frames) < 8:
            print(f"  ⚠ Skipping {video_id}: only {len(frames)} frames found")
            continue
        
        # Create video
        output_path = videos_dir / f"{video_id}_{action_label}.mp4"
        
        if frames_to_video(frames, output_path, fps=30, image_size=(224, 224)):
            created += 1
            print(f"  ✓ Created: {output_path.name} ({len(frames)} frames)")
    
    print(f"\n✓ Created {created} videos in {videos_dir}")


def create_videos_from_egohands_frames(
    frames_dir: Path = Path('data/egohands/_LABELLED_SAMPLES'),
    videos_dir: Path = Path('data/egohands/videos'),
    max_videos: int = 10,
):
    """Create videos from EgoHands frame sequences."""
    
    if not frames_dir.exists():
        print(f"⚠ Frames directory not found: {frames_dir}")
        return
    
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video directories
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    
    print(f"Creating videos from {len(video_dirs)} video directories...")
    print(f"  (Limited to {max_videos} for testing)")
    
    created = 0
    for video_dir in video_dirs[:max_videos]:
        # Find frames
        frames = sorted(list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png')))
        
        if len(frames) < 8:
            continue
        
        # Create video
        video_name = video_dir.name
        output_path = videos_dir / f"{video_name}.mp4"
        
        if frames_to_video(frames, output_path, fps=30, image_size=(224, 224)):
            created += 1
            print(f"  ✓ Created: {output_path.name} ({len(frames)} frames)")
    
    print(f"\n✓ Created {created} videos in {videos_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test videos from frames')
    parser.add_argument('--dataset', type=str, choices=['ego4d', 'egohands', 'both'],
                       default='both', help='Which dataset to process')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='Maximum number of videos to create')
    
    args = parser.parse_args()
    
    if args.dataset in ['ego4d', 'both']:
        print("="*70)
        print("Creating Ego4D Videos")
        print("="*70)
        create_videos_from_ego4d_frames(max_videos=args.max_videos)
    
    if args.dataset in ['egohands', 'both']:
        print("\n" + "="*70)
        print("Creating EgoHands Videos")
        print("="*70)
        create_videos_from_egohands_frames(max_videos=args.max_videos)
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Run: python scripts/eval_on_videos.py")
    print("  2. Or specify video directory:")
    print("     python scripts/eval_on_videos.py --video-dir data/ego4d/videos")
    print("="*70)

