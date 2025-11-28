"""
Visualize Vision Agents in AR Environment
Shows how teacher and student navigate visual scenes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ar_vision_env import ARVisionEnvSimple
from models.vision_teacher import VisionTeacherCNN, VisionTeacherPolicy
from models.vision_student import VisionStudentCNN, VisionStudentPolicy


class VisionAgentVisualizer:
    """Visualize agent's visual perception and navigation"""
    
    def __init__(self, env):
        self.env = env
        self.fig, (self.ax_scene, self.ax_view) = plt.subplots(1, 2, figsize=(12, 5))
    
    def visualize_episode(self, policy, title="Agent", delay=0.5, device='cpu'):
        """Run episode and visualize in real-time"""
        obs, _ = self.env.reset()
        done = False
        steps = 0
        
        # Initial state
        self._draw_frame(obs, title, steps)
        plt.pause(delay)
        
        # Run episode
        while not done and steps < 30:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, _, info = self.env.step(int(action))
            steps += 1
            
            # Visualize
            self._draw_frame(obs, title, steps, action=int(action))
            plt.pause(delay)
        
        # Show result
        success = info.get('success', False)
        result_text = "SUCCESS! 🎯" if success else "Failed ❌"
        self.ax_scene.text(0.5, -0.1, result_text, 
                          transform=self.ax_scene.transAxes,
                          fontsize=16, fontweight='bold',
                          color='green' if success else 'red',
                          ha='center')
        plt.pause(2)
        
        print(f"{title}: {'Success' if success else 'Failed'} in {steps} steps")
        return success, steps
    
    def _draw_frame(self, obs, title, step, action=None):
        """Draw current observation and scene"""
        self.ax_scene.clear()
        self.ax_view.clear()
        
        # Left: What the agent sees (camera view)
        obs_display = obs.transpose(1, 2, 0)  # CHW -> HWC for display
        self.ax_view.imshow(obs_display)
        self.ax_view.set_title(f"Camera View (32x32)", fontsize=12)
        self.ax_view.axis('off')
        
        # Right: Top-down scene view
        self.ax_scene.set_xlim(0, self.env.grid_size)
        self.ax_scene.set_ylim(0, self.env.grid_size)
        self.ax_scene.set_aspect('equal')
        self.ax_scene.set_title(f"{title} - Step {step}", fontsize=14, fontweight='bold')
        
        # Draw objects
        for obj in self.env.objects:
            if obj['is_target']:
                self.ax_scene.plot(obj['pos'][0], obj['pos'][1],
                                  marker='*', markersize=25, color='red',
                                  markeredgecolor='darkred', markeredgewidth=2,
                                  label='Target', zorder=2)
            else:
                self.ax_scene.plot(obj['pos'][0], obj['pos'][1],
                                  marker='o', markersize=15, color=obj['color'],
                                  markeredgecolor='black', markeredgewidth=1,
                                  zorder=1)
        
        # Draw agent
        self.ax_scene.plot(self.env.agent_pos[0], self.env.agent_pos[1],
                          marker='o', markersize=20, color='blue',
                          markeredgecolor='darkblue', markeredgewidth=2,
                          label='Agent', zorder=3)
        
        # Show action taken
        if action is not None:
            action_names = ['↑ Forward', '↓ Back', '← Left', '→ Right']
            self.ax_scene.text(0.02, 0.98, f"Action: {action_names[action]}",
                             transform=self.ax_scene.transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax_scene.legend(loc='upper right')
        self.ax_scene.grid(True, alpha=0.3)


def compare_models(device='cpu', seed=42):
    """Compare teacher and student side-by-side"""
    
    print("="*60)
    print("Vision Agent Visualization")
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
    print(f"  Teacher: {teacher_model.get_num_parameters():,} params")
    print(f"  Student: {student_model.get_num_parameters():,} params")
    print(f"  Compression: {teacher_model.get_num_parameters() / student_model.get_num_parameters():.1f}x")
    
    # Visualize teacher
    print("\n1. Running Teacher Model...")
    print("   (Close window to continue)")
    vis = VisionAgentVisualizer(env)
    env.reset(seed=seed)
    env_state = env._rng.bit_generator.state
    vis.visualize_episode(teacher_policy, title="Teacher (Off-Device)", delay=0.5, device=device)
    
    # Visualize student on same episode
    print("\n2. Running Student Model...")
    print("   (Close window to continue)")
    vis = VisionAgentVisualizer(env)
    env._rng.bit_generator.state = env_state
    vis.visualize_episode(student_policy, title="Student (On-Device)", delay=0.5, device=device)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    compare_models(device=device, seed=42)

