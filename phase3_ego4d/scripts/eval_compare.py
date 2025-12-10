"""
Phase 3: Evaluation and Comparison Script

Compares CLIP teacher vs MobileViT student on:
- Accuracy (action prediction)
- Latency (inference time)
- Model size (parameters & disk)
- Feature similarity (how well student matches teacher)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from data.ego4d_subset import Ego4DSyntheticDataset


def count_parameters(model: torch.nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Measure inference latency."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize for accurate timing
    if input_tensor.device.type == 'cuda':
        torch.cuda.synchronize()
    elif input_tensor.device.type == 'mps':
        torch.mps.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            
            if input_tensor.device.type == 'cuda':
                torch.cuda.synchronize()
            elif input_tensor.device.type == 'mps':
                torch.mps.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
    }


def compute_feature_similarity(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
) -> Dict[str, float]:
    """Compute similarity metrics between teacher and student features."""
    # Normalize features
    teacher_norm = F.normalize(teacher_features, dim=-1)
    student_norm = F.normalize(student_features, dim=-1)
    
    # Cosine similarity
    cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1).mean().item()
    
    # MSE
    mse = F.mse_loss(student_norm, teacher_norm).item()
    
    # Correlation
    teacher_flat = teacher_features.flatten(1)
    student_flat = student_features.flatten(1)
    
    t_centered = teacher_flat - teacher_flat.mean(dim=1, keepdim=True)
    s_centered = student_flat - student_flat.mean(dim=1, keepdim=True)
    
    correlation = (t_centered * s_centered).sum(dim=1) / (
        t_centered.norm(dim=1) * s_centered.norm(dim=1) + 1e-8
    )
    correlation = correlation.mean().item()
    
    return {
        'cosine_similarity': cosine_sim,
        'mse': mse,
        'correlation': correlation,
    }


def evaluate_models(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate both models on the dataset."""
    teacher.eval()
    student.eval()
    
    teacher_correct = 0
    student_correct = 0
    total = 0
    
    all_teacher_features = []
    all_student_features = []
    
    teacher_preds = []
    student_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            # Teacher forward
            teacher_out = teacher(frames, return_features=True)
            teacher_pred = teacher_out['logits'].argmax(dim=-1)
            
            # Student forward
            student_out = student(frames, return_features=True)
            student_pred = student_out['logits'].argmax(dim=-1)
            
            # Accuracy
            teacher_correct += (teacher_pred == labels).sum().item()
            student_correct += (student_pred == labels).sum().item()
            total += labels.size(0)
            
            # Collect features and predictions
            all_teacher_features.append(teacher_out['pooled_features'].cpu())
            all_student_features.append(student_out['pooled_features'].cpu())
            teacher_preds.extend(teacher_pred.cpu().tolist())
            student_preds.extend(student_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Concatenate features
    all_teacher_features = torch.cat(all_teacher_features, dim=0)
    all_student_features = torch.cat(all_student_features, dim=0)
    
    # Compute feature similarity
    feature_sim = compute_feature_similarity(all_teacher_features, all_student_features)
    
    # Compute agreement (how often student agrees with teacher)
    agreement = sum(t == s for t, s in zip(teacher_preds, student_preds)) / len(teacher_preds)
    
    return {
        'teacher_accuracy': teacher_correct / total,
        'student_accuracy': student_correct / total,
        'agreement': agreement,
        'feature_similarity': feature_sim,
        'teacher_preds': teacher_preds,
        'student_preds': student_preds,
        'labels': all_labels,
    }


def create_comparison_plot(results: Dict, output_path: str):
    """Create visualization comparing teacher and student."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    metrics = ['Teacher', 'Student']
    accuracies = [results['teacher_accuracy'] * 100, results['student_accuracy'] * 100]
    colors = ['#2196F3', '#4CAF50']
    bars = ax.bar(metrics, accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Latency comparison
    ax = axes[0, 1]
    latencies = [results['teacher_latency']['mean_ms'], results['student_latency']['mean_ms']]
    bars = ax.bar(metrics, latencies, color=colors)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency')
    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{lat:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 3. Model size comparison
    ax = axes[1, 0]
    sizes = [results['teacher_params'] / 1e6, results['student_params'] / 1e6]
    bars = ax.bar(metrics, sizes, color=colors)
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Size')
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{size:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 4. Feature similarity metrics
    ax = axes[1, 1]
    sim_metrics = list(results['feature_similarity'].keys())
    sim_values = list(results['feature_similarity'].values())
    
    # Normalize for visualization
    display_values = []
    display_labels = []
    for m, v in zip(sim_metrics, sim_values):
        if 'mse' in m.lower():
            # Invert MSE for visualization (lower is better)
            display_values.append(1 - min(v, 1))
            display_labels.append(f'{m}\n(1-MSE)')
        else:
            display_values.append(v)
            display_labels.append(m)
    
    bars = ax.bar(display_labels, display_values, color='#9C27B0')
    ax.set_ylabel('Score')
    ax.set_title('Feature Similarity (higher = better)')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, display_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def print_summary(results: Dict):
    """Print summary of evaluation results."""
    print("\n" + "="*70)
    print("PHASE 3 EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 ACCURACY")
    print(f"  Teacher: {results['teacher_accuracy']:.2%}")
    print(f"  Student: {results['student_accuracy']:.2%}")
    print(f"  Agreement: {results['agreement']:.2%}")
    
    print("\n⚡ LATENCY")
    print(f"  Teacher: {results['teacher_latency']['mean_ms']:.2f}ms (±{results['teacher_latency']['std_ms']:.2f})")
    print(f"  Student: {results['student_latency']['mean_ms']:.2f}ms (±{results['student_latency']['std_ms']:.2f})")
    speedup = results['teacher_latency']['mean_ms'] / results['student_latency']['mean_ms']
    print(f"  Speedup: {speedup:.1f}x")
    
    print("\n📦 MODEL SIZE")
    print(f"  Teacher: {results['teacher_params']:,} params ({results['teacher_size_mb']:.1f} MB)")
    print(f"  Student: {results['student_params']:,} params ({results['student_size_mb']:.1f} MB)")
    compression = results['teacher_params'] / results['student_params']
    print(f"  Compression: {compression:.1f}x")
    
    print("\n🎯 FEATURE SIMILARITY")
    for k, v in results['feature_similarity'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    acc_drop = (results['teacher_accuracy'] - results['student_accuracy']) * 100
    print(f"  • Accuracy drop: {acc_drop:.1f}%")
    print(f"  • Speedup: {speedup:.1f}x faster")
    print(f"  • Compression: {compression:.1f}x smaller")
    print(f"  • Feature alignment: {results['feature_similarity']['cosine_similarity']:.2%}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Evaluation')
    
    parser.add_argument('--student_checkpoint', type=str, default=None,
                       help='Path to trained student checkpoint')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--teacher_size', type=str, default='base',
                       choices=['base', 'large'])
    parser.add_argument('--student_size', type=str, default='xxs',
                       choices=['tiny', 'xxs', 'xs', 's'])
    parser.add_argument('--num_actions', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create models
    print("\nLoading models...")
    
    teacher = create_clip_teacher(
        model_size=args.teacher_size,
        num_actions=args.num_actions,
        for_distillation=True,
        freeze_clip=True,
    ).to(device)
    
    student = create_mobilevit_student(
        model_size=args.student_size,
        num_actions=args.num_actions,
        teacher_embed_dim=512 if args.teacher_size == 'base' else 768,
    ).to(device)
    
    # Load student checkpoint if provided
    if args.student_checkpoint and os.path.exists(args.student_checkpoint):
        print(f"Loading student checkpoint from {args.student_checkpoint}")
        checkpoint = torch.load(args.student_checkpoint, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    eval_dataset = Ego4DSyntheticDataset(
        num_samples=args.num_samples,
        frames_per_clip=8,
        image_size=224,
        num_actions=args.num_actions,
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Run evaluation
    print("\nEvaluating models...")
    results = evaluate_models(teacher, student, eval_loader, device)
    
    # Measure latency
    print("\nMeasuring latency...")
    dummy_input = torch.randn(1, 8, 3, 224, 224).to(device)
    
    results['teacher_latency'] = measure_latency(teacher, dummy_input)
    results['student_latency'] = measure_latency(student, dummy_input)
    
    # Model sizes
    results['teacher_params'] = count_parameters(teacher)
    results['student_params'] = count_parameters(student)
    results['teacher_size_mb'] = get_model_size_mb(teacher)
    results['student_size_mb'] = get_model_size_mb(student)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON (without large arrays)
    results_json = {k: v for k, v in results.items() 
                   if k not in ['teacher_preds', 'student_preds', 'labels']}
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Create comparison plot
    create_comparison_plot(results, str(output_dir / 'comparison_plot.png'))
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()

