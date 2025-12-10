"""
Visualize model predictions on Human Action Recognition dataset
Shows sample images, predictions, confusion matrix, and per-class accuracy
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.human_action import create_human_action_dataloaders, ACTION_LABELS
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student


def load_models(device, checkpoint_path=None):
    """Load teacher and student models."""
    num_actions = len(ACTION_LABELS)
    
    teacher = create_clip_teacher(
        model_size='base',
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device).eval()
    
    student = create_mobilevit_student(
        model_size='xxs',
        num_actions=num_actions,
        teacher_embed_dim=512,
    ).to(device).eval()
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading student checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Best val acc: {checkpoint.get('best_val_acc', 0):.2%}")
    else:
        print("Warning: No checkpoint provided, using random weights")
    
    return teacher, student


def collect_predictions(teacher, student, dataloader, device, num_samples=100):
    """Collect predictions from both models."""
    teacher_preds = []
    student_preds = []
    all_labels = []
    all_probs = []
    sample_images = []
    sample_indices = []
    
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Collecting predictions')):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            label_names = batch['label_name']
            
            # Get predictions
            teacher_out = teacher(frames, return_features=False)
            student_out = student(frames, return_features=False)
            
            teacher_pred = teacher_out['logits'].argmax(dim=-1)
            student_pred = student_out['logits'].argmax(dim=-1)
            student_probs = F.softmax(student_out['logits'], dim=-1)
            
            # Store predictions
            teacher_preds.extend(teacher_pred.cpu().tolist())
            student_preds.extend(student_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(student_probs.cpu().tolist())
            
            # Collect sample images (first batch, diverse predictions)
            if batch_idx == 0:
                for i in range(min(8, len(labels))):
                    sample_images.append({
                        'image': frames[i].cpu(),
                        'label': labels[i].item(),
                        'label_name': label_names[i],
                        'teacher_pred': teacher_pred[i].item(),
                        'student_pred': student_pred[i].item(),
                        'student_probs': student_probs[i].cpu().numpy(),
                    })
            
            count += len(labels)
            if count >= num_samples:
                break
    
    return {
        'teacher_preds': teacher_preds,
        'student_preds': student_preds,
        'labels': all_labels,
        'probs': all_probs,
        'samples': sample_images,
    }


def plot_sample_predictions(samples, output_path):
    """Plot sample images with predictions."""
    n_samples = len(samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Denormalize image
        img = sample['image'][0]  # (C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        ax.axis('off')
        
        # Get predictions
        true_label = ACTION_LABELS[sample['label']]
        teacher_pred = ACTION_LABELS[sample['teacher_pred']]
        student_pred = ACTION_LABELS[sample['student_pred']]
        student_conf = sample['student_probs'][sample['student_pred']] * 100
        
        # Color code: green=correct, red=wrong
        teacher_color = 'green' if sample['teacher_pred'] == sample['label'] else 'red'
        student_color = 'green' if sample['student_pred'] == sample['label'] else 'red'
        
        title = f"True: {true_label}\n"
        title += f"Teacher: {teacher_pred} ({'✓' if sample['teacher_pred'] == sample['label'] else '✗'})\n"
        title += f"Student: {student_pred} ({'✓' if sample['student_pred'] == sample['label'] else '✗'}) [{student_conf:.1f}%]"
        
        ax.set_title(title, fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions: Teacher vs Student', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Raw Counts', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Normalized (percentages)
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage'})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Normalized', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_class_accuracy(y_true, y_pred, labels, output_path):
    """Plot per-class accuracy."""
    class_acc = []
    for i in range(len(labels)):
        mask = np.array(y_true) == i
        if mask.sum() > 0:
            acc = (np.array(y_pred)[mask] == i).sum() / mask.sum() * 100
        else:
            acc = 0
        class_acc.append(acc)
    
    # Sort by accuracy
    sorted_idx = np.argsort(class_acc)
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_acc = [class_acc[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in sorted_acc]
    bars = ax.barh(range(len(sorted_labels)), sorted_acc, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axvline(x=100/len(labels), color='gray', linestyle='--', alpha=0.5, label='Random (6.7%)')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_prediction_distribution(y_true, y_pred, labels, output_path):
    """Plot distribution of predictions vs true labels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # True label distribution
    ax = axes[0]
    true_counts = [y_true.count(i) for i in range(len(labels))]
    ax.bar(range(len(labels)), true_counts, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('True Label Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Predicted label distribution
    ax = axes[1]
    pred_counts = [y_pred.count(i) for i in range(len(labels))]
    ax.bar(range(len(labels)), pred_counts, color='coral', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, 
                       default='results/human_action_v3/best_student.pt',
                       help='Path to student checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to evaluate')
    args = parser.parse_args()
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    teacher, student = load_models(device, args.checkpoint)
    
    # Load data
    print("\nLoading validation dataset...")
    _, val_loader = create_human_action_dataloaders(
        data_dir='data/human_action',
        batch_size=32,
        num_workers=0,
        image_size=224,
        frames_per_clip=1,
    )
    
    # Collect predictions
    print(f"\nCollecting predictions on {args.num_samples} samples...")
    results = collect_predictions(teacher, student, val_loader, device, args.num_samples)
    
    # Calculate metrics
    teacher_acc = sum(t == l for t, l in zip(results['teacher_preds'], results['labels'])) / len(results['labels'])
    student_acc = sum(s == l for s, l in zip(results['student_preds'], results['labels'])) / len(results['labels'])
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Teacher Accuracy: {teacher_acc:.2%}")
    print(f"Student Accuracy: {student_acc:.2%}")
    print(f"Agreement: {sum(t == s for t, s in zip(results['teacher_preds'], results['student_preds'])) / len(results['teacher_preds']):.2%}")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # 1. Sample predictions
    print("\n1. Sample predictions...")
    plot_sample_predictions(
        results['samples'],
        output_dir / 'sample_predictions.png'
    )
    
    # 2. Confusion matrix (student)
    print("\n2. Student confusion matrix...")
    plot_confusion_matrix(
        results['labels'],
        results['student_preds'],
        ACTION_LABELS,
        output_dir / 'student_confusion_matrix.png',
        title="Student Model Confusion Matrix"
    )
    
    # 3. Teacher confusion matrix
    print("\n3. Teacher confusion matrix...")
    plot_confusion_matrix(
        results['labels'],
        results['teacher_preds'],
        ACTION_LABELS,
        output_dir / 'teacher_confusion_matrix.png',
        title="Teacher Model Confusion Matrix"
    )
    
    # 4. Per-class accuracy
    print("\n4. Per-class accuracy...")
    plot_per_class_accuracy(
        results['labels'],
        results['student_preds'],
        ACTION_LABELS,
        output_dir / 'per_class_accuracy.png'
    )
    
    # 5. Prediction distribution
    print("\n5. Prediction distribution...")
    plot_prediction_distribution(
        results['labels'],
        results['student_preds'],
        ACTION_LABELS,
        output_dir / 'prediction_distribution.png'
    )
    
    # Save classification report
    print("\n6. Classification report...")
    report = classification_report(
        results['labels'],
        results['student_preds'],
        target_names=ACTION_LABELS,
        output_dict=True
    )
    
    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved: {output_dir / 'classification_report.json'}")
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - sample_predictions.png")
    print(f"  - student_confusion_matrix.png")
    print(f"  - teacher_confusion_matrix.png")
    print(f"  - per_class_accuracy.png")
    print(f"  - prediction_distribution.png")
    print(f"  - classification_report.json")


if __name__ == '__main__':
    main()

