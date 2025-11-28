"""
Evaluate and Compare Vision Models
Measures latency, bandwidth, and performance for AR deployment scenarios
"""

import time
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ar_vision_env import ARVisionEnvSimple
from models.vision_teacher import VisionTeacherCNN, VisionTeacherPolicy
from models.vision_student import VisionStudentCNN, VisionStudentPolicy


def run_teacher_episode(teacher_policy, env, net_delay_ms=50, device='cpu'):
    """
    Simulate teacher running off-device (cloud/edge server)
    
    Simulates:
    - Network latency for sending images
    - Network latency for receiving actions
    - Bandwidth cost of transmitting images
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    total_ms = 0.0
    total_bytes = 0
    
    while not done:
        t0 = time.time()
        
        # Simulate sending image to server (image size in bytes)
        image_bytes = obs.nbytes  # 3 channels * 32 * 32 = 3,072 bytes
        time.sleep(net_delay_ms / 1000.0)
        
        # Teacher inference on server
        action, _ = teacher_policy.predict(obs, deterministic=True)
        
        # Simulate receiving action from server
        time.sleep(net_delay_ms / 1000.0)
        
        elapsed = (time.time() - t0) * 1000.0
        total_ms += elapsed
        total_bytes += image_bytes + 4  # image + action (4 bytes)
        
        obs, _, done, _, info = env.step(int(action))
        steps += 1
    
    success = int(info.get('success', False))
    avg_latency = total_ms / max(steps, 1)
    avg_bytes = total_bytes / max(steps, 1)
    
    return success, steps, avg_latency, avg_bytes


def run_student_episode(student_policy, env, device='cpu'):
    """
    Simulate student running on-device (mobile)
    
    No network latency, only inference time
    Zero bandwidth cost
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    total_ms = 0.0
    
    while not done:
        t0 = time.time()
        
        # Student inference on device
        with torch.no_grad():
            action, _ = student_policy.predict(obs, deterministic=True)
        
        elapsed = (time.time() - t0) * 1000.0
        total_ms += elapsed
        
        obs, _, done, _, info = env.step(int(action))
        steps += 1
    
    success = int(info.get('success', False))
    avg_latency = total_ms / max(steps, 1)
    
    return success, steps, avg_latency, 0  # Zero bandwidth


def run_hybrid_episode(student_policy, teacher_policy, env, 
                       hint_frequency=10, net_delay_ms=50, device='cpu'):
    """
    Simulate hybrid approach: student on-device with occasional teacher hints
    
    Student runs locally, but queries teacher every N steps for guidance
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    total_ms = 0.0
    total_bytes = 0
    
    while not done:
        t0 = time.time()
        
        # Check if we should ask teacher
        if steps > 0 and steps % hint_frequency == 0:
            # Ask teacher (with network latency)
            image_bytes = obs.nbytes
            time.sleep(net_delay_ms / 1000.0)
            action, _ = teacher_policy.predict(obs, deterministic=True)
            time.sleep(net_delay_ms / 1000.0)
            total_bytes += image_bytes + 4
        else:
            # Use student (local, fast)
            with torch.no_grad():
                action, _ = student_policy.predict(obs, deterministic=True)
        
        elapsed = (time.time() - t0) * 1000.0
        total_ms += elapsed
        
        obs, _, done, _, info = env.step(int(action))
        steps += 1
    
    success = int(info.get('success', False))
    avg_latency = total_ms / max(steps, 1)
    avg_bytes = total_bytes / max(steps, 1)
    
    return success, steps, avg_latency, avg_bytes


def evaluate_all_modes(num_episodes=200, device='cpu'):
    """Evaluate all three deployment modes"""
    
    print("="*60)
    print("AR Vision Model Evaluation")
    print("="*60)
    
    # Load models
    env = ARVisionEnvSimple()
    
    teacher_model = VisionTeacherCNN()
    teacher_model.load_state_dict(torch.load("results/vision_teacher.pt", map_location=device))
    teacher_policy = VisionTeacherPolicy(teacher_model, device=device)
    
    student_model = VisionStudentCNN()
    student_model.load_state_dict(torch.load("results/vision_student.pt", map_location=device))
    student_policy = VisionStudentPolicy(student_model, device=device)
    
    print(f"\nModels loaded:")
    print(f"  Teacher: {teacher_model.get_num_parameters():,} parameters")
    print(f"  Student: {student_model.get_num_parameters():,} parameters")
    print(f"  Compression: {teacher_model.get_num_parameters() / student_model.get_num_parameters():.1f}x")
    print(f"\nRunning {num_episodes} episodes per mode...")
    print("="*60)
    
    modes = {
        "teacher_offdevice": lambda: run_teacher_episode(
            teacher_policy, env, net_delay_ms=50, device=device
        ),
        "student_ondevice": lambda: run_student_episode(
            student_policy, env, device=device
        ),
        "student_plus_hints": lambda: run_hybrid_episode(
            student_policy, teacher_policy, env, 
            hint_frequency=10, net_delay_ms=50, device=device
        ),
    }
    
    results = {}
    
    for mode_name, mode_fn in modes.items():
        print(f"\n[{mode_name}]")
        successes = []
        steps = []
        latencies = []
        bandwidths = []
        
        for i in range(num_episodes):
            success, step_count, latency, bandwidth = mode_fn()
            successes.append(success)
            steps.append(step_count)
            latencies.append(latency)
            bandwidths.append(bandwidth)
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{num_episodes}")
        
        results[mode_name] = {
            'success_rate': np.mean(successes),
            'avg_steps': np.mean(steps),
            'avg_latency_ms': np.mean(latencies),
            'avg_bandwidth_bytes': np.mean(bandwidths),
        }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode_name, metrics in results.items():
        print(f"\n== {mode_name} ==")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Avg steps: {metrics['avg_steps']:.2f}")
        print(f"Avg per-step latency (ms): {metrics['avg_latency_ms']:.2f}")
        print(f"Avg per-step bandwidth (bytes): {metrics['avg_bandwidth_bytes']:.0f}")
    
    # Speedup analysis
    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS")
    print("="*60)
    
    teacher_lat = results['teacher_offdevice']['avg_latency_ms']
    student_lat = results['student_ondevice']['avg_latency_ms']
    hybrid_lat = results['student_plus_hints']['avg_latency_ms']
    
    print(f"\nStudent vs Teacher:")
    print(f"  Latency speedup: {teacher_lat / student_lat:.1f}x")
    print(f"  Bandwidth reduction: 100% (zero network)")
    
    print(f"\nHybrid vs Teacher:")
    print(f"  Latency speedup: {teacher_lat / hybrid_lat:.1f}x")
    teacher_bw = results['teacher_offdevice']['avg_bandwidth_bytes']
    hybrid_bw = results['student_plus_hints']['avg_bandwidth_bytes']
    print(f"  Bandwidth reduction: {(1 - hybrid_bw/teacher_bw)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    evaluate_all_modes(num_episodes=200, device=device)

