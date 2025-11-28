"""
Performance Analysis for Vision AR Experiment
Generates confusion matrices and classification metrics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ar_vision_env import ARVisionEnvSimple
from models.vision_teacher import VisionTeacherCNN, VisionTeacherPolicy
from models.vision_student import VisionStudentCNN, VisionStudentPolicy


def collect_predictions(teacher_policy, student_policy, env, num_episodes=200):
    """Collect teacher and student predictions for comparison"""
    teacher_actions = []
    student_actions = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Teacher prediction
            t_action, _ = teacher_policy.predict(obs, deterministic=True)
            
            # Student prediction
            s_action, _ = student_policy.predict(obs, deterministic=True)
            
            teacher_actions.append(int(t_action))
            student_actions.append(int(s_action))
            
            # Take teacher's action to continue episode
            obs, _, done, _, _ = env.step(int(t_action))
    
    return np.array(teacher_actions), np.array(student_actions)


def plot_confusion_matrix(y_true, y_pred, title, filename, action_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=action_names, yticklabels=action_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Action', fontsize=12)
    plt.ylabel('True Action (Teacher)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def calculate_metrics(y_true, y_pred, action_names):
    """Calculate classification metrics"""
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(action_names)), zero_division=0
    )
    
    print("\nPer-Action Metrics:")
    print("-" * 60)
    print(f"{'Action':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i, action in enumerate(action_names):
        print(f"{action:<15} {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}  {support[i]:>8d}")
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print("-" * 60)
    print(f"{'Macro Avg':<15} {precision_macro:>10.4f}  {recall_macro:>10.4f}  {f1_macro:>10.4f}")
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"{'Weighted Avg':<15} {precision_weighted:>10.4f}  {recall_weighted:>10.4f}  {f1_weighted:>10.4f}")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support
    }


def plot_action_distribution(teacher_actions, student_actions, action_names, filename):
    """Plot action distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Teacher distribution
    teacher_counts = [np.sum(teacher_actions == i) for i in range(len(action_names))]
    ax1.bar(action_names, teacher_counts, color='steelblue', alpha=0.8)
    ax1.set_title('Teacher Action Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_xlabel('Action', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Student distribution
    student_counts = [np.sum(student_actions == i) for i in range(len(action_names))]
    ax2.bar(action_names, student_counts, color='coral', alpha=0.8)
    ax2.set_title('Student Action Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_xlabel('Action', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    print("="*60)
    print("Vision AR Performance Analysis")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    env = ARVisionEnvSimple()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    teacher_model = VisionTeacherCNN()
    teacher_model.load_state_dict(torch.load("results/vision_teacher.pt", map_location=device))
    teacher_policy = VisionTeacherPolicy(teacher_model, device=device)
    
    student_model = VisionStudentCNN()
    student_model.load_state_dict(torch.load("results/vision_student.pt", map_location=device))
    student_policy = VisionStudentPolicy(student_model, device=device)
    
    print("✓ Models loaded")
    print(f"  Teacher: {teacher_model.get_num_parameters():,} parameters")
    print(f"  Student: {student_model.get_num_parameters():,} parameters")
    
    # Action names
    action_names = ['Forward', 'Back', 'Left', 'Right']
    
    # Collect predictions
    print("\nCollecting predictions (200 episodes)...")
    teacher_actions, student_actions = collect_predictions(
        teacher_policy, student_policy, env, num_episodes=200
    )
    
    print(f"✓ Collected {len(teacher_actions)} action pairs")
    
    # Create results directory
    os.makedirs("results/analysis", exist_ok=True)
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        teacher_actions, 
        student_actions,
        "Vision AR: Student vs Teacher Action Predictions",
        "results/analysis/vision_confusion_matrix.png",
        action_names
    )
    
    # Plot action distributions
    print("Generating action distribution plots...")
    plot_action_distribution(
        teacher_actions,
        student_actions,
        action_names,
        "results/analysis/vision_action_distribution.png"
    )
    
    # Calculate metrics
    metrics = calculate_metrics(teacher_actions, student_actions, action_names)
    
    # Action distribution
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)
    print("\nTeacher action distribution:")
    for i, action in enumerate(action_names):
        count = np.sum(teacher_actions == i)
        pct = count / len(teacher_actions) * 100
        print(f"  {action:<10}: {count:>5d} ({pct:>5.1f}%)")
    
    print("\nStudent action distribution:")
    for i, action in enumerate(action_names):
        count = np.sum(student_actions == i)
        pct = count / len(student_actions) * 100
        print(f"  {action:<10}: {count:>5d} ({pct:>5.1f}%)")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Results saved to: results/analysis/")
    print("="*60)


if __name__ == "__main__":
    main()

