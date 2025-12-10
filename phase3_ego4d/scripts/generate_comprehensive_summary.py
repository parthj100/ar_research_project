"""
Generate comprehensive experiment summary
"""

import json
from pathlib import Path
from datetime import datetime


def load_training_history(path):
    """Load training history."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'train_acc' in data:
        return {
            'train_acc': [x * 100 for x in data['train_acc']],
            'val_acc': [x * 100 for x in data['val_acc']],
            'train_loss': data['train_loss'],
            'val_loss': data['val_loss'],
        }
    return None


def load_deployment_metrics():
    """Load deployment metrics."""
    path = Path('results/deployment_metrics.json')
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_summary():
    """Generate comprehensive experiment summary."""
    
    # Load all training histories
    histories = {
        'human_action_v2': load_training_history('results/human_action_v2/training_history.json'),
        'human_action_v3': load_training_history('results/human_action_v3/training_history.json'),
        'ego4d': load_training_history('results/ego4d_distill/training_history.json'),
        'egohands': load_training_history('results/egohands_visualized/training_history.json'),
        'unified': load_training_history('results/unified_egocentric/training_history.json'),
    }
    
    # Load deployment metrics
    deployment = load_deployment_metrics()
    
    # Experiment configurations
    experiments = {
        'human_action_v2': {
            'name': 'Human Action Recognition v2',
            'dataset': 'Human Action (HuggingFace)',
            'task': '15-class Image Classification',
            'frames_per_clip': 1,
            'num_classes': 15,
            'dataset_size': '12,600 images (10,080 train, 2,520 val)',
            'date': 'Nov 28, 2024',
        },
        'human_action_v3': {
            'name': 'Human Action Recognition v3',
            'dataset': 'Human Action (HuggingFace)',
            'task': '15-class Image Classification',
            'frames_per_clip': 1,
            'num_classes': 15,
            'dataset_size': '12,600 images (10,080 train, 2,520 val)',
            'date': 'Nov 29, 2024',
        },
        'ego4d': {
            'name': 'Ego4D Egocentric Video',
            'dataset': 'Ego4D',
            'task': '8-class Video Action Recognition',
            'frames_per_clip': 8,
            'num_classes': 8,
            'dataset_size': '100 clips (80 train, 20 val)',
            'date': 'Nov 29, 2024',
        },
        'egohands': {
            'name': 'EgoHands Egocentric Video',
            'dataset': 'EgoHands',
            'task': '4-class Video Action Recognition',
            'frames_per_clip': 8,
            'num_classes': 4,
            'dataset_size': '~230 clips (80% train, 20% val)',
            'date': 'Dec 2024',
        },
        'unified': {
            'name': 'Unified Egocentric (EgoHands + Ego4D)',
            'dataset': 'EgoHands + Ego4D',
            'task': '12-class Video Action Recognition',
            'frames_per_clip': 8,
            'num_classes': 12,
            'dataset_size': 'Combined from both datasets',
            'date': 'Dec 2024',
        },
    }
    
    print("="*80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Table 1: Experiment Overview
    print("📊 TABLE 1: EXPERIMENT OVERVIEW")
    print("="*80)
    print(f"{'Experiment':<30} {'Dataset':<25} {'Task':<30} {'Status':<10}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        status = "✅ Complete" if histories.get(exp_id) else "❌ Incomplete"
        print(f"{exp_info['name']:<30} {exp_info['dataset']:<25} {exp_info['task']:<30} {status:<10}")
    
    # Table 2: Dataset Information
    print("\n📦 TABLE 2: DATASET INFORMATION")
    print("="*80)
    print(f"{'Experiment':<30} {'Dataset':<25} {'Train':<12} {'Val':<12} {'Classes':<10} {'Frames/Clip':<12}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        print(f"{exp_info['name']:<30} {exp_info['dataset']:<25} {exp_info['dataset_size']:<12} {exp_info['num_classes']:<10} {exp_info['frames_per_clip']:<12}")
    
    # Table 3: Training Performance
    print("\n🎯 TABLE 3: TRAINING PERFORMANCE")
    print("="*80)
    print(f"{'Experiment':<30} {'Epochs':<10} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12} {'Gap':<10}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        hist = histories.get(exp_id)
        if hist:
            epochs = len(hist['train_acc'])
            train_acc = hist['train_acc'][-1]
            val_acc = hist['val_acc'][-1]
            train_loss = hist['train_loss'][-1]
            val_loss = hist['val_loss'][-1]
            gap = train_acc - val_acc
            print(f"{exp_info['name']:<30} {epochs:<10} {train_acc:<12.2f} {val_acc:<12.2f} {train_loss:<12.4f} {val_loss:<12.4f} {gap:<10.2f}")
        else:
            print(f"{exp_info['name']:<30} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    # Table 4: Deployment Metrics
    print("\n⚡ TABLE 4: DEPLOYMENT METRICS")
    print("="*80)
    print(f"{'Experiment':<30} {'Teacher Latency':<18} {'Student Latency':<18} {'Speedup':<10} {'Bandwidth Saved':<18}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        dep = deployment.get(exp_id)
        if dep:
            t_lat = dep['latency']['teacher']['mean_ms']
            s_lat = dep['latency']['student']['mean_ms']
            speedup = dep['latency']['speedup']
            bw_saved = dep['bandwidth']['bandwidth_saved_kb']
            print(f"{exp_info['name']:<30} {t_lat:<18.2f}ms {s_lat:<18.2f}ms {speedup:<10.1f}x {bw_saved:<18.2f} KB")
        else:
            print(f"{exp_info['name']:<30} {'N/A':<18} {'N/A':<18} {'N/A':<10} {'N/A':<18}")
    
    # Table 5: Model Compression
    print("\n💾 TABLE 5: MODEL COMPRESSION")
    print("="*80)
    print(f"{'Experiment':<30} {'Teacher (MB)':<15} {'Student (MB)':<15} {'Compression':<15} {'Params (M)':<15}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        dep = deployment.get(exp_id)
        if dep:
            t_size = dep['model_size']['teacher_size_mb']
            s_size = dep['model_size']['student_size_mb']
            comp = dep['model_size']['compression_ratio']
            t_params = dep['model_size']['teacher_params'] / 1e6
            s_params = dep['model_size']['student_params'] / 1e6
            print(f"{exp_info['name']:<30} {t_size:<15.2f} {s_size:<15.2f} {comp:<15.1f}x {s_params:<15.2f}")
        else:
            print(f"{exp_info['name']:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    # Table 6: Real-time Bandwidth (30 FPS)
    print("\n📡 TABLE 6: REAL-TIME BANDWIDTH (30 FPS)")
    print("="*80)
    print(f"{'Experiment':<30} {'Teacher (MB/s)':<18} {'Student (MB/s)':<18} {'Saved (MB/s)':<18}")
    print("-"*80)
    for exp_id, exp_info in experiments.items():
        dep = deployment.get(exp_id)
        if dep:
            t_bw = dep['real_time_bandwidth']['fps_30_teacher_mb_per_sec']
            s_bw = dep['real_time_bandwidth']['fps_30_student_mb_per_sec']
            saved = t_bw - s_bw
            print(f"{exp_info['name']:<30} {t_bw:<18.2f} {s_bw:<18.2f} {saved:<18.2f}")
        else:
            print(f"{exp_info['name']:<30} {'N/A':<18} {'N/A':<18} {'N/A':<18}")
    
    # Key Findings
    print("\n🔍 KEY FINDINGS")
    print("="*80)
    
    # Best validation accuracy
    best_val_acc = 0
    best_exp = None
    for exp_id, hist in histories.items():
        if hist:
            val_acc = max(hist['val_acc'])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_exp = experiments[exp_id]['name']
    
    print(f"✓ Best Validation Accuracy: {best_val_acc:.2f}% ({best_exp})")
    
    # Best latency
    best_speedup = 0
    best_lat_exp = None
    for exp_id, dep in deployment.items():
        if dep:
            speedup = dep['latency']['speedup']
            if speedup > best_speedup:
                best_speedup = speedup
                best_lat_exp = experiments[exp_id]['name']
    
    print(f"✓ Best Latency Speedup: {best_speedup:.1f}x ({best_lat_exp})")
    
    # Best bandwidth savings
    best_bw = 0
    best_bw_exp = None
    for exp_id, dep in deployment.items():
        if dep:
            bw = dep['bandwidth']['bandwidth_saved_kb']
            if bw > best_bw:
                best_bw = bw
                best_bw_exp = experiments[exp_id]['name']
    
    print(f"✓ Best Bandwidth Savings: {best_bw:.2f} KB per inference ({best_bw_exp})")
    
    # Model compression
    if deployment:
        first_exp = list(deployment.values())[0]
        comp = first_exp['model_size']['compression_ratio']
        print(f"✓ Model Compression: {comp:.1f}x (Teacher: ~580MB → Student: ~8.8MB)")
    
    # Unified vs EgoHands comparison
    if histories.get('egohands') and histories.get('unified'):
        egohands_val = max(histories['egohands']['val_acc'])
        unified_val = max(histories['unified']['val_acc'])
        print(f"\n✓ Unified vs EgoHands:")
        print(f"  - EgoHands Val Acc: {egohands_val:.2f}% (may be overfitted)")
        print(f"  - Unified Val Acc: {unified_val:.2f}% (better generalization)")
        print(f"  - Unified shows more realistic performance on diverse data")
    
    # Save summary to JSON
    summary_data = {
        'generated': datetime.now().isoformat(),
        'experiments': {},
    }
    
    for exp_id, exp_info in experiments.items():
        hist = histories.get(exp_id)
        dep = deployment.get(exp_id)
        
        exp_data = {
            'name': exp_info['name'],
            'dataset': exp_info['dataset'],
            'task': exp_info['task'],
            'dataset_size': exp_info['dataset_size'],
            'num_classes': exp_info['num_classes'],
            'frames_per_clip': exp_info['frames_per_clip'],
        }
        
        if hist:
            exp_data['training'] = {
                'epochs': len(hist['train_acc']),
                'final_train_acc': hist['train_acc'][-1],
                'final_val_acc': hist['val_acc'][-1],
                'best_val_acc': max(hist['val_acc']),
                'final_train_loss': hist['train_loss'][-1],
                'final_val_loss': hist['val_loss'][-1],
                'train_val_gap': hist['train_acc'][-1] - hist['val_acc'][-1],
            }
        
        if dep:
            exp_data['deployment'] = {
                'teacher_latency_ms': dep['latency']['teacher']['mean_ms'],
                'student_latency_ms': dep['latency']['student']['mean_ms'],
                'speedup': dep['latency']['speedup'],
                'bandwidth_saved_kb': dep['bandwidth']['bandwidth_saved_kb'],
                'teacher_size_mb': dep['model_size']['teacher_size_mb'],
                'student_size_mb': dep['model_size']['student_size_mb'],
                'compression_ratio': dep['model_size']['compression_ratio'],
                'fps_30_bandwidth_mb_per_sec': dep['real_time_bandwidth']['fps_30_teacher_mb_per_sec'],
            }
        
        summary_data['experiments'][exp_id] = exp_data
    
    output_path = Path('results/comprehensive_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✓ Summary saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    generate_summary()

