"""
Distillation training with alternative student backbones:
- mobilevit_xxs (default, reference)
- mobilenetv3_small_100
- efficientnet_b0

Outputs saved under: results/distilled_custom/<model_name>/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import timm

from scripts.train_unified import create_unified_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer


class DistillableStudent(nn.Module):
    """
    Student wrapper that supports different backbones and outputs
    in the format expected by DistillationTrainer:
      {'logits': ..., 'pooled_features': ..., 'features': ...}
    """

    def __init__(self, model_name: str, num_actions: int, teacher_embed_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.teacher_embed_dim = teacher_embed_dim

        if model_name == "mobilevit_xxs":
            self.is_mobilevit = True
            self.backbone = create_mobilevit_student(
                model_size="xxs", num_actions=num_actions, teacher_embed_dim=teacher_embed_dim
            )
        else:
            self.is_mobilevit = False
            if model_name == "mobilenetv3":
                backbone_name = "mobilenetv3_small_100"
            elif model_name == "mobilenetv3_large":
                backbone_name = "mobilenetv3_large_100"
            elif model_name == "efficientnet_b0":
                backbone_name = "efficientnet_b0"
            elif model_name == "efficientnet_b1":
                backbone_name = "efficientnet_b1"
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0, global_pool=""
            )
            self.proj = nn.Linear(self._infer_feat_dim(), teacher_embed_dim)
            self.classifier = nn.Linear(teacher_embed_dim, num_actions)

    def _infer_feat_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self._extract_feats(dummy)
            return feats.shape[-1]

    def _extract_feats(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])  # global avg pool
        return feats

    def forward(self, frames: torch.Tensor, return_features: bool = False):
        # frames: (B, T, C, H, W)
        if self.is_mobilevit:
            out = self.backbone(frames, return_features=True)
            return out

        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feats = self._extract_feats(x)              # (B*T, D)
        feats = feats.view(B, T, -1).mean(dim=1)    # temporal mean -> (B, D)
        pooled = nn.functional.normalize(feats, dim=-1)
        proj = self.proj(pooled) if feats.shape[-1] != self.teacher_embed_dim else pooled
        logits = self.classifier(proj)
        return {
            "logits": logits,
            "pooled_features": proj,
            "features": proj if return_features else None,
        }


def run_distillation(model_name: str, epochs: int = 10, batch_size: int = 16):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, action_labels = create_unified_dataloaders(
        batch_size=batch_size, num_workers=0, image_size=224, frames_per_clip=8
    )
    num_actions = len(action_labels)

    # Models
    teacher = create_clip_teacher(
        model_size="base",
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device)

    student = DistillableStudent(
        model_name=model_name, num_actions=num_actions, teacher_embed_dim=512
    ).to(device)

    # Trainer
    output_dir = Path("results/distilled_custom") / model_name
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
        output_dir=str(output_dir),
    )

    print("\n" + "="*70)
    print(f"Starting Distillation ({model_name})")
    print("="*70)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print("="*70)

    trainer.train(num_epochs=epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mobilenetv3",
                        choices=["mobilevit_xxs", "mobilenetv3", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_distillation(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)

