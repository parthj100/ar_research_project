"""
Inference Demo Script

Load best model checkpoint and run inference on sample frames.
Displays predictions with confidence scores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

# ============================================================
# CONFIG
# ============================================================

# Best model checkpoint
CHECKPOINT_PATH = "results/distilled_custom/mobilenetv3/best_student.pt"
NUM_CLASSES = 12
FRAMES_PER_CLIP = 8
IMAGE_SIZE = 224

# Action labels for Unified dataset
ACTION_LABELS = [
    'CARDS', 'CHESS', 'JENGA', 'PUZZLE',  # EgoHands
    'looking_around', 'manipulating', 'picking_up', 'putting_down',  # Ego4D
    'reaching', 'standing', 'turning', 'walking'
]

# Output directory
OUT_DIR = Path("results/presentation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL DEFINITION (must match training)
# ============================================================

class DistilledStudent(nn.Module):
    """MobileNetV3-based student for action recognition."""
    
    def __init__(self, num_classes: int = 12, frames_per_clip: int = 8):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        
        # Backbone
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            feat = self.features(dummy)
            feat = self.avgpool(feat)
            feat_dim = feat.view(1, -1).shape[1]
        
        # Temporal pooling (match checkpoint: 512 hidden)
        self.temporal_pool = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video frames
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, C, H, W)
        feats = self.features(x)
        feats = self.avgpool(feats).view(B * T, -1)
        
        # Temporal aggregation (mean pooling)
        feats = feats.view(B, T, -1).mean(dim=1)  # (B, feat_dim)
        
        # Classification
        feats = self.temporal_pool(feats)
        logits = self.classifier(feats)
        
        return logits


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    model = DistilledStudent(num_classes=NUM_CLASSES, frames_per_clip=FRAMES_PER_CLIP)
    
    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found, using random weights for demo")
    
    model.to(device)
    model.eval()
    return model


def get_transform():
    """Get inference transform."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_sample_images():
    """Find sample images from existing datasets."""
    sample_paths = []
    
    # Try EgoHands (per-video folders)
    egohands_dir = Path("data/egohands/frames")
    if egohands_dir.exists():
        for video_dir in list(egohands_dir.iterdir())[:2]:
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("*.jpg"))[:FRAMES_PER_CLIP]
                if len(frames) >= FRAMES_PER_CLIP:
                    sample_paths.append((video_dir.name, frames))
    
    # Try Ego4D (flat folder with video_XXX_frame_YY.jpg naming)
    ego4d_dir = Path("data/ego4d/frames")
    if ego4d_dir.exists():
        all_frames = sorted(ego4d_dir.glob("*.jpg"))
        if all_frames:
            # Group by video ID
            from collections import defaultdict
            video_frames = defaultdict(list)
            for f in all_frames:
                # Parse video_XXX_frame_YY.jpg
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    video_id = f"video_{parts[1]}"
                    video_frames[video_id].append(f)
            
            # Take first 2 videos with enough frames
            for video_id, frames in list(video_frames.items())[:4]:
                frames = sorted(frames)[:FRAMES_PER_CLIP]
                if len(frames) >= 4:  # Need at least some frames
                    # Pad if needed
                    while len(frames) < FRAMES_PER_CLIP:
                        frames.append(frames[-1])
                    sample_paths.append((video_id, frames[:FRAMES_PER_CLIP]))
                if len(sample_paths) >= 2:
                    break
    
    # Try Human Action
    human_action_dir = Path("data/human_action")
    if human_action_dir.exists():
        for class_dir in list(human_action_dir.iterdir())[:2]:
            if class_dir.is_dir():
                frames = sorted(class_dir.glob("*.jpg"))[:FRAMES_PER_CLIP]
                if len(frames) >= 1:
                    # Duplicate single frame for video input
                    frames = frames * FRAMES_PER_CLIP
                    frames = frames[:FRAMES_PER_CLIP]
                    sample_paths.append((class_dir.name, frames))
    
    return sample_paths


def run_inference(model, frames: list, transform, device) -> tuple:
    """
    Run inference on a list of frame paths.
    
    Returns:
        predicted_label, confidence, all_probs
    """
    # Load and transform frames
    imgs = []
    for frame_path in frames[:FRAMES_PER_CLIP]:
        img = Image.open(frame_path).convert('RGB')
        img = transform(img)
        imgs.append(img)
    
    # Pad if needed
    while len(imgs) < FRAMES_PER_CLIP:
        imgs.append(imgs[-1])
    
    # Stack: (T, C, H, W) -> (1, T, C, H, W)
    x = torch.stack(imgs).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_idx = probs.argmax()
    pred_label = ACTION_LABELS[pred_idx] if pred_idx < len(ACTION_LABELS) else f"class_{pred_idx}"
    confidence = probs[pred_idx]
    
    return pred_label, confidence, probs


def visualize_prediction(frames: list, pred_label: str, confidence: float, 
                         probs: np.ndarray, sample_name: str, save_path: Path):
    """Create visualization of prediction."""
    fig = plt.figure(figsize=(14, 6))
    
    # Show sample frames (4 of them)
    n_show = min(4, len(frames))
    for i in range(n_show):
        ax = fig.add_subplot(2, 4, i + 1)
        img = Image.open(frames[i * (len(frames) // n_show)]).convert('RGB')
        ax.imshow(img)
        ax.set_title(f"Frame {i+1}", fontsize=10)
        ax.axis('off')
    
    # Show prediction bar chart
    ax_bar = fig.add_subplot(1, 2, 2)
    
    # Top 5 predictions
    top_k = 5
    top_indices = probs.argsort()[-top_k:][::-1]
    top_labels = [ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"class_{i}" for i in top_indices]
    top_probs = probs[top_indices]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(top_k)]
    bars = ax_bar.barh(range(top_k), top_probs * 100, color=colors, edgecolor='black')
    ax_bar.set_yticks(range(top_k))
    ax_bar.set_yticklabels(top_labels, fontsize=11)
    ax_bar.set_xlabel("Confidence (%)", fontsize=12)
    ax_bar.set_title(f"Prediction: {pred_label} ({confidence*100:.1f}%)", fontsize=14, fontweight='bold')
    ax_bar.set_xlim(0, 100)
    ax_bar.invert_yaxis()
    
    # Add value labels
    for bar, prob in zip(bars, top_probs):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%', va='center', fontsize=10)
    
    plt.suptitle(f"Sample: {sample_name}", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Inference Demo")
    print("=" * 60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(CHECKPOINT_PATH, device)
    transform = get_transform()
    
    # Find sample images
    print("\nFinding sample images...")
    samples = find_sample_images()
    
    if not samples:
        print("⚠ No sample images found. Creating synthetic demo...")
        # Create a synthetic demo with random predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Synthetic predictions
        labels = ACTION_LABELS[:6]
        probs = np.random.dirichlet(np.ones(6)) * 100
        probs = np.sort(probs)[::-1]
        
        colors = ['#2ecc71'] + ['#3498db'] * 5
        bars = ax.barh(range(6), probs, color=colors, edgecolor='black')
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Confidence (%)", fontsize=12)
        ax.set_title(f"Demo: Model Prediction Format\n(Synthetic data - no sample images found)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / "inference_demo_synthetic.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {OUT_DIR / 'inference_demo_synthetic.png'}")
        return
    
    # Run inference on each sample
    print(f"\nRunning inference on {len(samples)} samples...")
    
    results = []
    for i, (sample_name, frames) in enumerate(samples):
        print(f"\n  Sample {i+1}: {sample_name}")
        
        pred_label, confidence, probs = run_inference(model, frames, transform, device)
        print(f"    Prediction: {pred_label} ({confidence*100:.1f}%)")
        
        # Visualize
        save_path = OUT_DIR / f"inference_demo_{i+1}.png"
        visualize_prediction(frames, pred_label, confidence, probs, sample_name, save_path)
        
        results.append({
            "sample": sample_name,
            "prediction": pred_label,
            "confidence": float(confidence),
            "top_probs": {ACTION_LABELS[j]: float(probs[j]) for j in probs.argsort()[-5:][::-1]}
        })
    
    # Save results
    with open(OUT_DIR / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {OUT_DIR / 'inference_results.json'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

