"""
Evaluate models on actual video files from datasets

This script processes actual video files (not pre-extracted frames) to:
1. Load videos directly
2. Extract frames on-the-fly
3. Run inference on video clips
4. Compare performance with frame-based approach
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ opencv-python not available. Install with: pip install opencv-python")

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    # Decord may not be available on all platforms (e.g., macOS ARM)


class VideoDataset:
    """Dataset for loading and processing video files directly."""
    
    def __init__(
        self,
        video_paths: List[Path],
        action_labels: List[str],
        frames_per_clip: int = 8,
        image_size: int = 224,
        use_decord: bool = True,
    ):
        self.video_paths = video_paths
        self.action_labels = action_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(action_labels)}
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.use_decord = use_decord and DECORD_AVAILABLE
        
        # Build video clips
        self.clips = []
        for video_path in video_paths:
            # Extract action label from video path or filename
            label = self._extract_label(video_path)
            if label in self.label_to_idx:
                self.clips.append({
                    'video_path': video_path,
                    'label': label,
                    'label_idx': self.label_to_idx[label],
                })
    
    def _extract_label(self, video_path: Path) -> str:
        """Extract action label from video path."""
        # Try to extract from path structure
        parts = video_path.parts
        for part in parts:
            # Common action names
            actions = ['walking', 'turning', 'looking_around', 'manipulating',
                      'picking_up', 'putting_down', 'reaching', 'standing',
                      'cards', 'jenga', 'puzzle', 'chess']
            for action in actions:
                if action.lower() in part.lower():
                    return action
        return 'other'
    
    def load_video_frames(self, video_path: Path, num_frames: int = 8) -> torch.Tensor:
        """Load frames from video file."""
        if self.use_decord:
            return self._load_with_decord(video_path, num_frames)
        elif CV2_AVAILABLE:
            return self._load_with_opencv(video_path, num_frames)
        else:
            raise RuntimeError("No video loading library available")
    
    def _load_with_decord(self, video_path: Path, num_frames: int) -> torch.Tensor:
        """Load frames using decord (faster)."""
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        
        # Convert to (T, C, H, W) and normalize
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Resize frames
        import torchvision.transforms as T
        resize = T.Resize((self.image_size, self.image_size))
        frames = torch.stack([resize(frame) for frame in frames])
        
        # Normalize
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        frames = torch.stack([normalize(frame) for frame in frames])
        
        return frames  # (T, C, H, W)
    
    def _load_with_opencv(self, video_path: Path, num_frames: int) -> torch.Tensor:
        """Load frames using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                # Convert to tensor and normalize
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            # Pad with last frame
            while len(frames) < num_frames:
                frames.append(frames[-1])
        
        frames = torch.stack(frames)  # (T, C, H, W)
        
        # Normalize
        import torchvision.transforms as T
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        frames = torch.stack([normalize(frame) for frame in frames])
        
        return frames


def find_video_files(data_dir: Path, extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> List[Path]:
    """Find all video files in a directory."""
    video_files = []
    for ext in extensions:
        video_files.extend(list(data_dir.rglob(f'*{ext}')))
    return sorted(video_files)


def evaluate_on_videos(
    checkpoint_path: str,
    video_dir: Path,
    num_actions: int,
    frames_per_clip: int = 8,
    device: torch.device = None,
    max_videos: int = 10,
):
    """Evaluate model on actual video files."""
    
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("="*70)
    print("Video-based Evaluation")
    print("="*70)
    print(f"Video directory: {video_dir}")
    print(f"Device: {device}")
    
    # Find video files
    video_files = find_video_files(video_dir)
    
    if len(video_files) == 0:
        print(f"\n⚠ No video files found in {video_dir}")
        print("   Looking for: .mp4, .avi, .mov, .mkv")
        print("\n   To test with videos:")
        print("   1. Download video files to the dataset directory")
        print("   2. Place them in: data/ego4d/videos/ or data/egohands/videos/")
        print("   3. Run this script again")
        return None
    
    print(f"\n✓ Found {len(video_files)} video files")
    video_files = video_files[:max_videos]  # Limit for testing
    print(f"  Using first {len(video_files)} videos for evaluation")
    
    # Create models
    print("\n📦 Loading models...")
    teacher = create_clip_teacher(
        model_size='base',
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device).eval()
    
    student = create_mobilevit_student(
        model_size='xxs',
        num_actions=num_actions,
        teacher_embed_dim=512,
    ).to(device).eval()
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Create video dataset
    action_labels = ['walking', 'turning', 'looking_around', 'manipulating',
                     'picking_up', 'putting_down', 'reaching', 'standing']
    dataset = VideoDataset(
        video_paths=video_files,
        action_labels=action_labels,
        frames_per_clip=frames_per_clip,
    )
    
    print(f"\n🎬 Processing {len(dataset.clips)} video clips...")
    
    # Process videos
    results = {
        'teacher_predictions': [],
        'student_predictions': [],
        'labels': [],
        'teacher_latencies': [],
        'student_latencies': [],
        'video_paths': [],
    }
    
    with torch.no_grad():
        for i, clip in enumerate(dataset.clips):
            video_path = clip['video_path']
            label_idx = clip['label_idx']
            
            print(f"\n  [{i+1}/{len(dataset.clips)}] {video_path.name}")
            
            try:
                # Load frames from video
                frames = dataset.load_video_frames(video_path, frames_per_clip)
                frames = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)
                
                # Teacher inference
                t_start = time.perf_counter()
                teacher_output = teacher(frames, return_features=False)
                if device.type == 'mps':
                    torch.mps.synchronize()
                teacher_latency = (time.perf_counter() - t_start) * 1000
                teacher_pred = teacher_output['logits'].argmax(dim=1).item()
                
                # Student inference
                s_start = time.perf_counter()
                student_output = student(frames, return_features=False)
                if device.type == 'mps':
                    torch.mps.synchronize()
                student_latency = (time.perf_counter() - s_start) * 1000
                student_pred = student_output['logits'].argmax(dim=1).item()
                
                results['teacher_predictions'].append(teacher_pred)
                results['student_predictions'].append(student_pred)
                results['labels'].append(label_idx)
                results['teacher_latencies'].append(teacher_latency)
                results['student_latencies'].append(student_latency)
                results['video_paths'].append(str(video_path))
                
                print(f"    Label: {action_labels[label_idx]}")
                print(f"    Teacher: {action_labels[teacher_pred]} ({teacher_latency:.2f}ms)")
                print(f"    Student: {action_labels[student_pred]} ({student_latency:.2f}ms)")
                
            except Exception as e:
                print(f"    ⚠ Error processing video: {e}")
                continue
    
    # Calculate metrics
    if len(results['labels']) > 0:
        teacher_correct = sum(1 for p, l in zip(results['teacher_predictions'], results['labels']) if p == l)
        student_correct = sum(1 for p, l in zip(results['student_predictions'], results['labels']) if p == l)
        
        teacher_acc = 100 * teacher_correct / len(results['labels'])
        student_acc = 100 * student_correct / len(results['labels'])
        
        teacher_lat_mean = np.mean(results['teacher_latencies'])
        student_lat_mean = np.mean(results['student_latencies'])
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Videos processed: {len(results['labels'])}")
        print(f"\nAccuracy:")
        print(f"  Teacher: {teacher_acc:.2f}% ({teacher_correct}/{len(results['labels'])})")
        print(f"  Student: {student_acc:.2f}% ({student_correct}/{len(results['labels'])})")
        print(f"\nLatency (mean):")
        print(f"  Teacher: {teacher_lat_mean:.2f}ms")
        print(f"  Student: {student_lat_mean:.2f}ms")
        print(f"  Speedup: {teacher_lat_mean / student_lat_mean:.2f}x")
        print("="*70)
        
        # Save results
        output_path = Path('results/video_evaluation.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'num_videos': len(results['labels']),
                'teacher_accuracy': teacher_acc,
                'student_accuracy': student_acc,
                'teacher_latency_ms': teacher_lat_mean,
                'student_latency_ms': student_lat_mean,
                'speedup': teacher_lat_mean / student_lat_mean,
                'results': results,
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        return results
    
    return None


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models on video files')
    parser.add_argument('--checkpoint', type=str, default='results/unified_egocentric/best_student.pt',
                       help='Path to student checkpoint')
    parser.add_argument('--video-dir', type=str, default='data/ego4d/videos',
                       help='Directory containing video files')
    parser.add_argument('--num-actions', type=int, default=12,
                       help='Number of action classes')
    parser.add_argument('--frames-per-clip', type=int, default=8,
                       help='Number of frames per clip')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    evaluate_on_videos(
        checkpoint_path=args.checkpoint,
        video_dir=Path(args.video_dir),
        num_actions=args.num_actions,
        frames_per_clip=args.frames_per_clip,
        device=device,
        max_videos=args.max_videos,
    )


if __name__ == '__main__':
    main()

