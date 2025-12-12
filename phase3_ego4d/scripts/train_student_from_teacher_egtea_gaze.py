"""
Stage 2 (from-scratch workflow):
Distill a student using a *trained* EGTEA teacher checkpoint.

Teacher: CLIPTeacherForDistillation loaded from results/egtea_teacher/.../best_teacher.pt
Student: MobileNetV3-Large / EfficientNet-B1 / etc.
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
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="Path to best_teacher.pt")
    parser.add_argument("--split-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--label-type", type=str, default="verb", choices=["verb", "action"])
    parser.add_argument("--teacher", type=str, default="large", choices=["base", "large"])
    parser.add_argument("--student", type=str, default="mobilenetv3_large",
                        choices=["mobilenetv3", "mobilenetv3_large", "efficientnet_b0", "efficientnet_b1", "mobilevit_xxs"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    egtea_root = "/Users/nypv/Desktop/ar-teacher-student-research/EGTEA Gaze+"
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

    # Build teacher for distillation and load trained weights
    teacher = create_clip_teacher(
        model_size=args.teacher,
        num_actions=num_classes,
        for_distillation=True,
        freeze_clip=True,  # teacher is fixed during distillation
        use_temporal=True,
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    teacher.load_state_dict(state, strict=False)
    teacher.eval()

    teacher_embed_dim = getattr(teacher, "embed_dim", 512)
    print(f"Teacher embed dim: {teacher_embed_dim}")

    student = DistillableStudent(
        model_name=args.student,
        num_actions=num_classes,
        teacher_embed_dim=teacher_embed_dim,
    ).to(device)

    out_dir = Path("results/egtea_from_scratch") / f"split{args.split_id}_{args.label_type}_student-{args.student}_teacher-{args.teacher}"
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
    print(f"\n✓ Distillation done. Results: {out_dir}")


if __name__ == "__main__":
    main()


