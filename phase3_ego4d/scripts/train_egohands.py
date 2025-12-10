"""
Train CLIP → MobileViT distillation on EgoHands dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.egohands_loader import create_egohands_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*60)
    print("EgoHands Egocentric Hand-Object Interaction Distillation")
    print("="*60)
    
    # Load EgoHands data
    print("\nLoading EgoHands dataset...")
    train_loader, val_loader, action_labels = create_egohands_dataloaders(
        data_dir='data/egohands',
        batch_size=16,
        num_workers=0,
        image_size=224,
        frames_per_clip=8,
        train_split=0.8,
    )
    
    num_actions = len(action_labels)
    print(f"\n✓ Loaded {num_actions} action classes:")
    for i, label in enumerate(action_labels):
        print(f"  {i}: {label}")
    
    # Create models
    print("\nCreating models...")
    teacher = create_clip_teacher(
        model_size='base',
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    )
    
    student = create_mobilevit_student(
        model_size='xxs',
        num_actions=num_actions,
        teacher_embed_dim=512,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-5,
        alpha=0.5,
        beta=1.0,
        gamma=1.0,
        temperature=3.0,
        output_dir='results/egohands',
    )
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print("="*60)
    
    # Train
    trainer.train(num_epochs=30)

