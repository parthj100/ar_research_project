"""
Visualize Phase 2 Agent Training in Real-Time
Watch the agent navigate the enhanced synthetic environment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

sys.path.insert(0, '.')

from envs.enhanced_vision_env import EnhancedVisionEnv
from models.phase2_teacher import Phase2TeacherResNet


def visualize_agent(model_path=None, num_episodes=3, delay=0.3):
    """
    Visualize agent navigating in real-time
    
    Args:
        model_path: Path to trained model (None = random agent)
        num_episodes: Number of episodes to show
        delay: Seconds between frames
    """
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    env = EnhancedVisionEnv()
    
    # Load model if provided
    if model_path and os.path.exists(model_path):
        model = Phase2TeacherResNet(num_actions=4, freeze_backbone=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        title_prefix = "Trained Agent"
        print(f"✓ Loaded model: {model_path}")
    else:
        model = None
        title_prefix = "Random Agent (No Training Yet)"
        print("⚠ No trained model found - showing random agent")
    
    print(f"\nVisualizing {num_episodes} episodes...")
    print("Close window to see next episode\n")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        print(f"Episode {ep+1}/{num_episodes}")
        
        while not done and steps < 50:
            # Get action
            if model is not None:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                    q_values = model(obs_tensor)
                    action = q_values.argmax().item()
            else:
                action = env.action_space.sample()
            
            # Take action
            obs, reward, done, _, info = env.step(action)
            steps += 1
            
            # Display current view
            ax1.clear()
            ax1.imshow(obs.transpose(1, 2, 0))
            ax1.set_title(f"Agent's View (64x64)\nStep {steps}", 
                         fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Show top-down representation
            ax2.clear()
            ax2.set_xlim(0, env.grid_size)
            ax2.set_ylim(0, env.grid_size)
            ax2.set_aspect('equal')
            ax2.set_title(f"Top-Down View\nAction: {['Forward','Back','Left','Right'][action]}", 
                         fontsize=12, fontweight='bold')
            
            # Draw background
            if env.background_style == 'gradient':
                ax2.set_facecolor('#e0e0e0')
            else:
                ax2.set_facecolor('#e8e8e8')
            
            # Draw objects
            for obj in env.objects:
                if obj['is_target']:
                    ax2.plot(obj['pos'][0], obj['pos'][1], 
                            marker='*', markersize=25, color='red',
                            markeredgecolor='darkred', markeredgewidth=2,
                            label='Target', zorder=10)
                else:
                    size_scale = obj['size'] * 15
                    ax2.scatter(obj['pos'][0], obj['pos'][1],
                              s=size_scale**2, c=obj['color'], 
                              edgecolors='black', linewidth=1.5,
                              alpha=0.7, zorder=5)
            
            # Draw agent
            ax2.plot(env.agent_pos[0], env.agent_pos[1],
                    marker='o', markersize=15, color='blue',
                    markeredgecolor='darkblue', markeredgewidth=2,
                    label='Agent', zorder=20)
            
            # Distance to target
            dist = np.linalg.norm(env.agent_pos - env.target['pos'])
            ax2.text(0.02, 0.98, f"Distance: {dist:.2f}",
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            
            # Overall title
            success_text = "🎯 SUCCESS!" if info.get('success') else ""
            fig.suptitle(f"{title_prefix} - Episode {ep+1}\n{success_text}", 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.pause(delay)
        
        # Show final result
        success = info.get('success', False)
        result = "✓ SUCCESS!" if success else "✗ Failed"
        color = 'green' if success else 'red'
        
        ax2.text(env.grid_size/2, -0.8, result, 
                ha='center', fontsize=18, fontweight='bold', color=color)
        
        print(f"  Result: {result} in {steps} steps")
        
        plt.pause(2)
        plt.close()
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/phase2_teacher_best.pt',
                       help='Path to model (default: results/phase2_teacher_best.pt)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to show')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between frames (seconds)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Phase 2 Agent Visualization")
    print("="*70)
    
    visualize_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        delay=args.delay
    )
