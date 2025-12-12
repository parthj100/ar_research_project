"""
Standalone distillation experiment on EGTEA Gaze+ (cropped clips).

Default: verb classification (19 classes) for a fast, meaningful result.
You can switch to action classification (106 classes) via --label-type action.

Teacher: CLIP (ViT-B/32 by default)
Student: MobileNetV3 or EfficientNet-B0 via DistillableStudent (timm backbone)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from data.egtea_gaze_loader import create_egtea_gaze_dataloaders
from models.clip_teacher import create_clip_teacher
from scripts.train_distill import DistillationTrainer
from scripts.train_distill_custom import DistillableStudent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--label-type", type=str, default="verb", choices=["verb", "action"])
    parser.add_argument("--student", type=str, default="mobilenetv3",
                        choices=["mobilenetv3", "mobilenetv3_large", "efficientnet_b0", "efficientnet_b1", "mobilevit_xxs"])
    parser.add_argument("--teacher", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    egtea_root = "/Users/nypv/Desktop/ar-teacher-student-research/EGTEA Gaze+"

    print("\n" + "=" * 70)
    print("EGTEA Gaze+ Distillation Training")
    print("=" * 70)
    print(f"Split: {args.split_id} | Label: {args.label_type} | Student: {args.student} | Teacher: {args.teacher}")

    train_loader, val_loader, label_names = create_egtea_gaze_dataloaders(
        egtea_root=egtea_root,
        split_id=args.split_id,
        label_type=args.label_type,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=224,
        frames_per_clip=args.frames_per_clip,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
    )
    num_classes = len(label_names)
    print(f"✓ Classes: {num_classes}")

    teacher = create_clip_teacher(
        model_size=args.teacher,
        num_actions=num_classes,
        for_distillation=True,
        freeze_clip=True,
        use_temporal=True,
    ).to(device)

    student = DistillableStudent(
        model_name=args.student,
        num_actions=num_classes,
        teacher_embed_dim=512,
    ).to(device)

    out_dir = Path("results/egtea_gaze") / f"split{args.split_id}_{args.label_type}_{args.student}_teacher-{args.teacher}"
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
        output_dir=str(out_dir),
    )

    trainer.train(num_epochs=args.epochs)
    print(f"\n✓ Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()


