"""
Measure deployment metrics: Latency, Bandwidth, Model Size
For all three experiments (Human Action v2, v3, Ego4D)
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from data.human_action import create_human_action_dataloaders, ACTION_LABELS as HUMAN_ACTION_LABELS
from data.ego4d_loader import create_ego4d_dataloaders


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
            _ = model(input_tensor, return_features=False)
    
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
            _ = model(input_tensor, return_features=False)
            
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


def calculate_bandwidth(
    input_shape: tuple,
    frames_per_clip: int = 1,
    bits_per_pixel: int = 24,  # RGB = 3 channels × 8 bits
) -> Dict[str, float]:
    """
    Calculate bandwidth requirements.
    
    Teacher (cloud): Need to send image data
    Student (on-device): No data transfer needed
    
    Args:
        input_shape: (C, H, W) or (T, C, H, W)
        frames_per_clip: Number of frames
        bits_per_pixel: Bits per pixel (RGB = 24)
    """
    if len(input_shape) == 3:
        C, H, W = input_shape
        T = frames_per_clip
    else:
        T, C, H, W = input_shape
    
    # Calculate size per frame
    pixels_per_frame = H * W
    bits_per_frame = pixels_per_frame * bits_per_pixel
    bytes_per_frame = bits_per_frame / 8
    
    # Total for clip
    bytes_per_clip = bytes_per_frame * T
    
    # Teacher: Send full image data to cloud
    teacher_bandwidth_per_inference = bytes_per_clip
    
    # Student: Runs on-device, no transfer
    student_bandwidth_per_inference = 0.0
    
    # Bandwidth saved
    bandwidth_saved = teacher_bandwidth_per_inference
    bandwidth_reduction = 1.0  # 100% reduction
    
    return {
        'teacher_bytes_per_inference': teacher_bandwidth_per_inference,
        'student_bytes_per_inference': student_bandwidth_per_inference,
        'bandwidth_saved_bytes': bandwidth_saved,
        'bandwidth_saved_kb': bandwidth_saved / 1024,
        'bandwidth_saved_mb': bandwidth_saved / (1024 * 1024),
        'bandwidth_reduction_percent': bandwidth_reduction * 100,
        'frames_per_clip': T,
        'image_size': f"{H}×{W}",
    }


def evaluate_experiment(
    experiment_name: str,
    checkpoint_path: str,
    num_actions: int,
    input_shape: tuple,
    frames_per_clip: int = 1,
    device: torch.device = None,
) -> Dict:
    """Evaluate a single experiment."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*70}")
    
    # Create models
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
    
    # Load student checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        
        # Update num_actions from checkpoint if available
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            for k, v in state.items():
                if 'classifier' in k and 'weight' in k:
                    actual_num_actions = v.shape[0]
                    if actual_num_actions != num_actions:
                        print(f"  Note: Found {actual_num_actions} classes in checkpoint (expected {num_actions})")
                        # Recreate student with correct number of actions
                        student = create_mobilevit_student(
                            model_size='xxs',
                            num_actions=actual_num_actions,
                            teacher_embed_dim=512,
                        ).to(device).eval()
                        student.load_state_dict(checkpoint['model_state_dict'])
                        num_actions = actual_num_actions
                    break
    else:
        print(f"⚠ Warning: Checkpoint not found: {checkpoint_path}")
        print("  Using random weights (results will be invalid)")
    
    # Model sizes
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    teacher_size_mb = get_model_size_mb(teacher)
    student_size_mb = get_model_size_mb(student)
    
    print(f"\n📦 Model Sizes:")
    print(f"  Teacher: {teacher_params:,} params ({teacher_size_mb:.2f} MB)")
    print(f"  Student: {student_params:,} params ({student_size_mb:.2f} MB)")
    print(f"  Compression: {teacher_params / student_params:.1f}x")
    
    # Create dummy input
    if len(input_shape) == 3:
        C, H, W = input_shape
        dummy_input = torch.randn(1, frames_per_clip, C, H, W).to(device)
    else:
        dummy_input = torch.randn(1, *input_shape).to(device)
    
    print(f"\n⚡ Measuring Latency (100 runs)...")
    teacher_latency = measure_latency(teacher, dummy_input)
    student_latency = measure_latency(student, dummy_input)
    
    print(f"  Teacher: {teacher_latency['mean_ms']:.2f}ms (±{teacher_latency['std_ms']:.2f})")
    print(f"  Student: {student_latency['mean_ms']:.2f}ms (±{student_latency['std_ms']:.2f})")
    speedup = teacher_latency['mean_ms'] / student_latency['mean_ms']
    print(f"  Speedup: {speedup:.1f}x faster")
    
    print(f"\n📡 Calculating Bandwidth...")
    bandwidth = calculate_bandwidth(input_shape, frames_per_clip)
    print(f"  Teacher (cloud): {bandwidth['teacher_bytes_per_inference'] / 1024:.2f} KB per inference")
    print(f"  Student (on-device): {bandwidth['student_bytes_per_inference']:.2f} KB per inference")
    print(f"  Bandwidth saved: {bandwidth['bandwidth_saved_kb']:.2f} KB ({bandwidth['bandwidth_reduction_percent']:.1f}% reduction)")
    
    # Calculate bandwidth for real-time scenarios
    fps_30 = bandwidth['teacher_bytes_per_inference'] * 30  # 30 FPS
    fps_60 = bandwidth['teacher_bytes_per_inference'] * 60  # 60 FPS
    
    print(f"\n📊 Real-time Bandwidth (if running at 30 FPS):")
    print(f"  Teacher: {fps_30 / (1024 * 1024):.2f} MB/s")
    print(f"  Student: 0 MB/s (on-device)")
    print(f"  Saved: {fps_30 / (1024 * 1024):.2f} MB/s")
    
    return {
        'experiment': experiment_name,
        'checkpoint': checkpoint_path,
        'model_size': {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'teacher_size_mb': teacher_size_mb,
            'student_size_mb': student_size_mb,
            'compression_ratio': teacher_params / student_params,
        },
        'latency': {
            'teacher': teacher_latency,
            'student': student_latency,
            'speedup': speedup,
        },
        'bandwidth': bandwidth,
        'real_time_bandwidth': {
            'fps_30_teacher_mb_per_sec': fps_30 / (1024 * 1024),
            'fps_30_student_mb_per_sec': 0.0,
            'fps_60_teacher_mb_per_sec': fps_60 / (1024 * 1024),
            'fps_60_student_mb_per_sec': 0.0,
        },
    }


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    results = {}
    
    # Experiment 1: Human Action v2
    results['human_action_v2'] = evaluate_experiment(
        experiment_name="Human Action v2",
        checkpoint_path="results/human_action_v2/best_student.pt",
        num_actions=15,
        input_shape=(3, 224, 224),
        frames_per_clip=1,
        device=device,
    )
    
    # Experiment 2: Human Action v3
    results['human_action_v3'] = evaluate_experiment(
        experiment_name="Human Action v3",
        checkpoint_path="results/human_action_v3/best_student.pt",
        num_actions=15,
        input_shape=(3, 224, 224),
        frames_per_clip=1,
        device=device,
    )
    
    # Experiment 3: Ego4D
    results['ego4d'] = evaluate_experiment(
        experiment_name="Ego4D",
        checkpoint_path="results/ego4d_distill/best_student.pt",
        num_actions=8,
        input_shape=(3, 224, 224),
        frames_per_clip=8,
        device=device,
    )
    
    # Experiment 4: EgoHands
    results['egohands'] = evaluate_experiment(
        experiment_name="EgoHands",
        checkpoint_path="results/egohands_visualized/best_student.pt",
        num_actions=4,  # Will be determined from checkpoint if available
        input_shape=(3, 224, 224),
        frames_per_clip=8,
        device=device,
    )
    
    # Experiment 5: Unified (EgoHands + Ego4D)
    results['unified'] = evaluate_experiment(
        experiment_name="Unified (EgoHands + Ego4D)",
        checkpoint_path="results/unified_egocentric/best_student.pt",
        num_actions=12,  # Combined classes from both datasets
        input_shape=(3, 224, 224),
        frames_per_clip=8,
        device=device,
    )
    
    # Save results
    output_path = Path("results/deployment_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print("\n📊 Latency Comparison:")
    print(f"{'Experiment':<20} {'Teacher (ms)':<15} {'Student (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for exp_name, exp_data in results.items():
        t_lat = exp_data['latency']['teacher']['mean_ms']
        s_lat = exp_data['latency']['student']['mean_ms']
        speedup = exp_data['latency']['speedup']
        print(f"{exp_name:<20} {t_lat:<15.2f} {s_lat:<15.2f} {speedup:<10.1f}x")
    
    print("\n📡 Bandwidth Comparison (per inference):")
    print(f"{'Experiment':<20} {'Teacher (KB)':<15} {'Student (KB)':<15} {'Saved (KB)':<15}")
    print("-" * 65)
    for exp_name, exp_data in results.items():
        t_bw = exp_data['bandwidth']['teacher_bytes_per_inference'] / 1024
        s_bw = exp_data['bandwidth']['student_bytes_per_inference'] / 1024
        saved = exp_data['bandwidth']['bandwidth_saved_kb']
        print(f"{exp_name:<20} {t_bw:<15.2f} {s_bw:<15.2f} {saved:<15.2f}")
    
    print("\n💾 Model Size:")
    print(f"{'Experiment':<20} {'Teacher (MB)':<15} {'Student (MB)':<15} {'Compression':<15}")
    print("-" * 65)
    for exp_name, exp_data in results.items():
        t_size = exp_data['model_size']['teacher_size_mb']
        s_size = exp_data['model_size']['student_size_mb']
        comp = exp_data['model_size']['compression_ratio']
        print(f"{exp_name:<20} {t_size:<15.2f} {s_size:<15.2f} {comp:<15.1f}x")
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

