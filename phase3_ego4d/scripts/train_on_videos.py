"""
Train on actual video files (not pre-extracted frames)

This script trains the teacher-student distillation model using
actual video files, which is more realistic for AR applications.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.video_loader import create_video_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train on video files')
    parser.add_argument('--video-dir', type=str, default='data/ego4d/videos',
                       help='Directory containing video files')
    parser.add_argument('--annotations', type=str, default='data/ego4d/annotations.json',
                       help='Path to annotations JSON (optional)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (smaller for videos)')
    parser.add_argument('--frames-per-clip', type=int, default=8,
                       help='Number of frames per clip')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--output-dir', type=str, default='results/video_training',
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, mps, cuda)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("Video-based Training")
    print("="*70)
    print(f"Video directory: {args.video_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frames per clip: {args.frames_per_clip}")
    print("="*70)
    
    # Create dataloaders
    print("\n📦 Loading video datasets...")
    try:
        train_loader, val_loader, action_labels = create_video_dataloaders(
            video_dir=Path(args.video_dir),
            annotations_path=Path(args.annotations) if Path(args.annotations).exists() else None,
            batch_size=args.batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with video loading
            image_size=args.image_size,
            frames_per_clip=args.frames_per_clip,
            train_split=0.8,
            augment_train=True,
        )
    except Exception as e:
        print(f"❌ Error loading videos: {e}")
        print("\nMake sure:")
        print("  1. Video files exist in the specified directory")
        print("  2. Videos are in supported formats (.mp4, .avi, .mov, .mkv)")
        print("  3. opencv-python is installed: pip install opencv-python")
        return
    
    num_actions = len(action_labels)
    print(f"\n✓ Loaded {num_actions} action classes:")
    for i, label in enumerate(action_labels):
        print(f"  {i}: {label}")
    
    # Create models
    print("\n🏗️ Creating models...")
    teacher = create_clip_teacher(
        model_size='base',
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device)
    
    student = create_mobilevit_student(
        model_size='xxs',
        num_actions=num_actions,
        teacher_embed_dim=512,
    ).to(device)
    
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"  Student: {sum(p.numel() for p in student.parameters()):,} params")
    
    # Create trainer
    print("\n🚀 Starting training...")
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        lr=5e-5,
        alpha=0.5,  # Feature loss weight
        beta=1.0,   # Task loss weight
        gamma=0.1,  # KL divergence weight
        temperature=3.0,
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Best model: {args.output_dir}/best_student.pt")
    print("="*70)


if __name__ == '__main__':
    main()

