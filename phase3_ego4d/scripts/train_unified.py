"""
Unified Training: EgoHands + Ego4D

Train on combined egocentric datasets for better generalization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.egohands_loader import create_egohands_dataloaders as create_egohands
from data.ego4d_loader import create_ego4d_dataloaders as create_ego4d
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer
from torch.utils.data import ConcatDataset, DataLoader


def create_unified_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    frames_per_clip: int = 8,
):
    """Create unified dataloaders combining EgoHands and Ego4D."""
    datasets_available = []
    all_action_labels = []
    
    # Try EgoHands
    try:
        egohands_train, egohands_val, egohands_labels = create_egohands(
            data_dir='data/egohands',
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            frames_per_clip=frames_per_clip,
        )
        datasets_available.append(('egohands', egohands_train.dataset, egohands_val.dataset, egohands_labels))
        all_action_labels.extend(egohands_labels)
        print(f"✓ EgoHands: {len(egohands_train.dataset)} train, {len(egohands_val.dataset)} val")
    except FileNotFoundError:
        print("⚠ EgoHands not found, skipping...")
    
    # Try Ego4D
    try:
        ego4d_train, ego4d_val, ego4d_labels = create_ego4d(
            data_dir='data/ego4d',
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            frames_per_clip=frames_per_clip,
        )
        datasets_available.append(('ego4d', ego4d_train.dataset, ego4d_val.dataset, ego4d_labels))
        all_action_labels.extend(ego4d_labels)
        print(f"✓ Ego4D: {len(ego4d_train.dataset)} train, {len(ego4d_val.dataset)} val")
    except FileNotFoundError:
        print("⚠ Ego4D not found, skipping...")
    
    if not datasets_available:
        raise ValueError("No datasets available! Please download at least one dataset.")
    
    # Merge action labels (remove duplicates, keep order)
    unique_labels = []
    seen = set()
    for label in all_action_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    # Create unified datasets
    train_datasets = [ds[1] for ds in datasets_available]
    val_datasets = [ds[2] for ds in datasets_available]
    
    unified_train = ConcatDataset(train_datasets)
    unified_val = ConcatDataset(val_datasets)
    
    # Create unified dataloaders
    def collate_fn(batch):
        """Collate function for unified dataset."""
        frames = torch.stack([item['frames'] for item in batch])  # (B, T, C, H, W)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return {
            'frames': frames,
            'label': labels,
        }
    
    train_loader = DataLoader(
        unified_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        unified_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Store action labels
    train_loader.dataset.action_labels = unique_labels
    val_loader.dataset.action_labels = unique_labels
    
    print(f"\n✓ Unified dataset:")
    print(f"  Train samples: {len(unified_train)}")
    print(f"  Val samples: {len(unified_val)}")
    print(f"  Total action classes: {len(unique_labels)}")
    print(f"  Classes: {unique_labels}")
    
    return train_loader, val_loader, unique_labels


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*70)
    print("Unified Egocentric Training: EgoHands + Ego4D")
    print("="*70)
    
    # Load unified data
    print("\nLoading unified egocentric datasets...")
    train_loader, val_loader, action_labels = create_unified_dataloaders(
        batch_size=16,
        num_workers=0,
        image_size=224,
        frames_per_clip=8,
    )
    
    num_actions = len(action_labels)
    
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
        output_dir='results/unified_egocentric',
    )
    
    print("\n" + "="*70)
    print("Starting Unified Training")
    print("="*70)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print("="*70)
    
    # Train
    trainer.train(num_epochs=30)

