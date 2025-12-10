"""
Train on Combined Egocentric Datasets (Ego4D + EPIC-KITCHENS + EgoHands)

This script trains on multiple egocentric datasets combined for better generalization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.combined_egocentric_loader import create_combined_egocentric_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*70)
    print("Combined Egocentric Video Distillation")
    print("Ego4D + EPIC-KITCHENS + EgoHands")
    print("="*70)
    
    # Determine which datasets are available
    available_datasets = []
    if Path('data/ego4d/annotations.json').exists():
        available_datasets.append('ego4d')
    if Path('data/epic_kitchens').exists():
        available_datasets.append('epic_kitchens')
    if Path('data/egohands/_LABELLED_SAMPLES').exists():
        available_datasets.append('egohands')
    
    if not available_datasets:
        print("\n⚠ No datasets found!")
        print("\nTo download datasets:")
        print("  1. Ego4D: Already have some data in data/ego4d/")
        print("  2. EPIC-KITCHENS: https://epic-kitchens.github.io/")
        print("  3. EgoHands: https://vision.soic.indiana.edu/projects/egohands/")
        print("\nUsing Ego4D only for now...")
        available_datasets = ['ego4d']
    
    print(f"\nUsing datasets: {available_datasets}")
    
    # Load combined data
    print("\nLoading combined egocentric datasets...")
    train_loader, val_loader, action_labels = create_combined_egocentric_dataloaders(
        datasets_to_use=available_datasets,
        batch_size=16,
        num_workers=0,
        image_size=224,
        frames_per_clip=8,
        train_split=0.8,
        max_samples_per_dataset=None,  # Use all available data
    )
    
    num_actions = len(action_labels)
    print(f"\n✓ Total action classes: {num_actions}")
    
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
    
    # Create trainer with optimized hyperparameters
    print("\nInitializing trainer...")
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-5,  # Optimized LR
        alpha=0.5,  # Feature distillation weight
        beta=1.0,   # Response distillation weight
        gamma=1.0,  # Task loss weight
        temperature=3.0,
        output_dir='results/combined_egocentric',
    )
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print(f"Datasets: {', '.join(available_datasets)}")
    print("="*70)
    
    # Train
    trainer.train(num_epochs=30)

