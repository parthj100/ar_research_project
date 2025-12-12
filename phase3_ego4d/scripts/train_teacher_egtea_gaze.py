"""
Train (fine-tune) a stronger teacher model on EGTEA Gaze+ first, then we can distill it.

Teacher: CLIPTeacher (ViT-B/32 or ViT-L/14) + action_head.

This is supervised training (cross-entropy) on EGTEA verb/action labels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.egtea_gaze_loader import create_egtea_gaze_dataloaders
from models.clip_teacher import create_clip_teacher


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        out = model(frames, return_features=False)
        logits = out["logits"]
        loss = ce(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--label-type", type=str, default="verb", choices=["verb", "action"])
    parser.add_argument("--teacher", type=str, default="large", choices=["base", "large"])
    parser.add_argument("--freeze-clip", action="store_true", help="Freeze CLIP backbone (train only heads).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    # Build teacher (for_distillation=False is fine; we just need logits)
    teacher = create_clip_teacher(
        model_size=args.teacher,
        num_actions=num_classes,
        for_distillation=False,
        freeze_clip=args.freeze_clip,
        use_temporal=True,
    ).to(device)

    # Optimizer: only trainable params
    params = [p for p in teacher.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    out_dir = Path("results/egtea_teacher") / f"split{args.split_id}_{args.label_type}_teacher-{args.teacher}_freezeclip-{int(args.freeze_clip)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc = 0.0
    best_path = out_dir / "best_teacher.pt"

    print("\n" + "=" * 70)
    print("Teacher Training (EGTEA Gaze+)")
    print("=" * 70)
    print(f"Classes: {num_classes} | Split: {args.split_id} | Label: {args.label_type}")
    print(f"Teacher: CLIP-{args.teacher} | Freeze CLIP: {args.freeze_clip}")
    print(f"Train clips: {len(train_loader.dataset)} | Test clips: {len(val_loader.dataset)}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        teacher.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch}")
        for batch in pbar:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            out = teacher(frames, return_features=False)
            logits = out["logits"]
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/max(total,1):.1f}%")

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate(teacher, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)
        print(
            f"\nTeacher Epoch {epoch}/{args.epochs}\n"
            f"  Train Loss: {train_loss:.4f}  Acc: {100*train_acc:.2f}%\n"
            f"  Val   Loss: {val_metrics['loss']:.4f}  Acc: {100*val_metrics['acc']:.2f}%\n"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": teacher.state_dict(),
                    "config": vars(args),
                    "num_classes": num_classes,
                    "label_names": label_names,
                },
                best_path,
            )
            print(f"  ✓ New best teacher: {100*best_acc:.2f}% saved to {best_path}")

        with open(out_dir / "teacher_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Teacher training complete. Best val acc: {100*best_acc:.2f}%")
    print(f"Saved: {best_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


