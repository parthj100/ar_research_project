"""
Standalone distillation training on GTEA.
Teacher: CLIP ViT-B/32 (frozen)
Student: MobileViT-XXS (default)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.gtea_loader import create_gtea_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n" + "="*70)
    print("GTEA Distillation Training")
    print("="*70)

    # Load data
    print("\nLoading GTEA dataset...")
    train_loader, val_loader, action_labels = create_gtea_dataloaders(
        data_dir="data/gtea",
        batch_size=16,
        num_workers=0,
        image_size=224,
        frames_per_clip=8,
        train_split=0.8,
    )
    num_actions = len(action_labels)
    print(f"✓ Loaded {num_actions} action classes")

    # Models
    teacher = create_clip_teacher(
        model_size="base",
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device)

    student = create_mobilevit_student(
        model_size="xxs",
        num_actions=num_actions,
        teacher_embed_dim=512,
    ).to(device)

    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Trainer
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
        output_dir="results/gtea_distill",
    )

    print("\n" + "="*70)
    print("Starting GTEA Training")
    print("="*70)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print("="*70)

    trainer.train(num_epochs=30)


if __name__ == "__main__":
    main()

