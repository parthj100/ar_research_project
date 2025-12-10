"""
Baseline (no-distillation) training on the unified egocentric dataset.

Supports multiple lightweight student backbones:
- mobilevit_xxs (default)
- mobilenetv3_small_100
- efficientnet_b0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

import timm

from scripts.train_unified import create_unified_dataloaders
from models.mobilevit_student import create_mobilevit_student


class BaselineStudent(nn.Module):
    """
    Wrapper to support multiple backbones and temporal pooling.
    Expects input frames shaped (B, T, C, H, W).
    """

    def __init__(self, model_name: str, num_actions: int):
        super().__init__()
        self.model_name = model_name

        if model_name == "mobilevit_xxs":
            # Reuse existing student implementation (with temporal mean)
            self.backbone = create_mobilevit_student(
                model_size="xxs", num_actions=num_actions, teacher_embed_dim=512
            )
            self.temporal_head = None  # handled inside backbone
            self.is_mobilevit = True
        elif model_name == "mobilenetv3":
            self.is_mobilevit = False
            self.backbone = timm.create_model(
                "mobilenetv3_small_100", pretrained=True, num_classes=0, global_pool=""
            )
            feat_dim = self._infer_feat_dim()
            self.classifier = nn.Linear(feat_dim, num_actions)
        elif model_name == "efficientnet_b0":
            self.is_mobilevit = False
            self.backbone = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=0, global_pool=""
            )
            feat_dim = self._infer_feat_dim()
            self.classifier = nn.Linear(feat_dim, num_actions)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def _infer_feat_dim(self):
        """Run a dummy forward to infer feature dimension after pooling."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self._extract_feats(dummy)
            return feats.shape[-1]

    def _extract_feats(self, x: torch.Tensor):
        """Extract spatial features then pool to (B, D)."""
        feats = self.backbone.forward_features(x)
        # forward_features may return tensor or list
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        # Global average pool
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        return feats

    def forward(self, frames: torch.Tensor):
        # frames: (B, T, C, H, W)
        if self.is_mobilevit:
            # MobileViT student already supports temporal mean pooling internally
            out = self.backbone(frames, return_features=False)
            return out["logits"]

        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feats = self._extract_feats(x)  # (B*T, D)
        feats = feats.view(B, T, -1).mean(dim=1)  # temporal mean
        logits = self.classifier(feats)
        return logits


def train_baseline(
    model_name: str = "mobilevit_xxs",
    batch_size: int = 16,
    num_epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    output_dir: str = "results/baseline_unified",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load unified data (Ego4D + EgoHands; EPIC if available)
    train_loader, val_loader, action_labels = create_unified_dataloaders(
        batch_size=batch_size, num_workers=0, image_size=224, frames_per_clip=8
    )
    num_actions = len(action_labels)

    # Create model
    model = BaselineStudent(model_name=model_name, num_actions=num_actions).to(device)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        train_loss = total_loss / total
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)
                logits = model(frames)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc / 100)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc / 100)

        scheduler.step()

        print(
            f"\nEpoch {epoch}/{num_epochs}\n"
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%\n"
            f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%\n"
        )

    # Save history
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), out_dir / "final_student.pt")
    print(f"\n✓ Saved results to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mobilevit_xxs",
                        choices=["mobilevit_xxs", "mobilenetv3", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_baseline(
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir="results/baseline_unified",
    )

