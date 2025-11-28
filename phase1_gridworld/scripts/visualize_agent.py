import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from envs.gridworld import GridworldEnv
from models.student import StudentPolicy

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

class GridworldVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.grid_size = env.grid_size
        
    def draw_grid(self, agent_pos, goal_pos, title="Gridworld", step=0, hint_text=None):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, linewidth=2, color='black')
        self.ax.set_title(f"{title} - Step {step}", fontsize=16, fontweight='bold')
        
        # Draw goal (green star)
        self.ax.plot(goal_pos[0], goal_pos[1], marker='*', markersize=40, 
                    color='green', markeredgecolor='darkgreen', markeredgewidth=2, 
                    label='Goal')
        
        # Draw agent (blue circle)
        self.ax.plot(agent_pos[0], agent_pos[1], marker='o', markersize=30, 
                    color='blue', markeredgecolor='darkblue', markeredgewidth=2,
                    label='Agent')
        
        # Add hint indicator if provided
        if hint_text:
            self.ax.text(self.grid_size/2, self.grid_size + 0.5, hint_text,
                        ha='center', fontsize=12, color='orange', 
                        fontweight='bold', bbox=dict(boxstyle='round', 
                        facecolor='lightyellow', edgecolor='orange', linewidth=2))
        
        self.ax.legend(loc='upper right', fontsize=12)
        self.ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left
        
    def visualize_episode(self, get_action_fn, title="Agent", delay=0.5, get_hint_fn=None):
        """
        Run one episode and visualize it step by step in REAL-TIME.
        get_action_fn: function that takes (observation, step) and returns (action, hint_text)
        get_hint_fn: optional function that returns hint text for current step
        """
        obs, _ = self.env.reset()
        done = False
        steps = 0
        
        # Show initial state
        hint_text = get_hint_fn(steps) if get_hint_fn else None
        self.draw_grid(self.env.agent.copy(), self.env.goal.copy(), title=title, step=0, hint_text=hint_text)
        plt.pause(delay)
        
        # Run and visualize in real-time
        while not done and steps < 50:
            action, hint_text = get_action_fn(obs, steps)
            obs, reward, done, _, _ = self.env.step(action)
            steps += 1
            
            # Draw immediately after each action
            self.draw_grid(self.env.agent.copy(), self.env.goal.copy(), title=title, step=steps, hint_text=hint_text)
            plt.pause(delay)
        
        success = np.array_equal(self.env.agent, self.env.goal)
        result = "SUCCESS! 🎉" if success else "Failed"
        self.ax.text(self.grid_size/2, -1, result, ha='center', 
                    fontsize=20, fontweight='bold', 
                    color='green' if success else 'red')
        plt.pause(2)
        
        return success, steps

def run_teacher(env, model):
    """Visualize teacher model"""
    vis = GridworldVisualizer(env)
    
    def get_action(obs, step):
        action, _ = model.predict(obs, deterministic=True)
        return int(action), None  # No hint text for teacher
    
    success, steps = vis.visualize_episode(get_action, title="Teacher Model (PPO)", delay=0.5)
    print(f"Teacher: {'Success' if success else 'Failed'} in {steps} steps")
    return vis.fig

def run_student(env, model):
    """Visualize student model"""
    vis = GridworldVisualizer(env)
    
    def get_action(obs, step):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = int(torch.argmax(logits, dim=1).item())
        return action, None  # No hint text for student
    
    success, steps = vis.visualize_episode(get_action, title="Student Model (Distilled)", delay=0.5)
    print(f"Student: {'Success' if success else 'Failed'} in {steps} steps")
    return vis.fig

def run_student_with_hints(env, student, teacher, k=10):
    """Visualize student with occasional teacher hints"""
    vis = GridworldVisualizer(env)
    
    def get_action(obs, step):
        # Student proposes
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = student(x)
            s_action = int(torch.argmax(logits, dim=1).item())
        
        # Occasional teacher hint
        hint_text = None
        if step % k == 0 and step > 0:
            t_action, _ = teacher.predict(obs, deterministic=True)
            action = int(t_action)
            hint_text = "🔍 Teacher Hint!"
        else:
            action = s_action
        
        return action, hint_text
    
    success, steps = vis.visualize_episode(get_action, 
                                          title=f"Student + Teacher Hints (every {k} steps)", 
                                          delay=0.5)
    print(f"Student+Hints: {'Success' if success else 'Failed'} in {steps} steps")
    return vis.fig

def compare_all_models(env, teacher, student, seed=None):
    """Compare all three models side by side"""
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
    
    print("\n" + "="*60)
    print("Visualizing all models on the same environment...")
    print("="*60 + "\n")
    
    # Run teacher
    print("1. Running Teacher Model...")
    env_state = env._rng.bit_generator.state
    run_teacher(env, teacher)
    
    # Reset to same state for student
    print("\n2. Running Student Model...")
    env._rng.bit_generator.state = env_state
    run_student(env, student)
    
    # Reset to same state for student+hints
    print("\n3. Running Student + Hints...")
    env._rng.bit_generator.state = env_state
    run_student_with_hints(env, student, teacher, k=10)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    print("Loading models...")
    env = GridworldEnv()
    teacher = PPO.load("results/teacher_ppo_gridworld")
    student = StudentPolicy().to(device)
    student.load_state_dict(torch.load("results/student_policy.pt", map_location=device))
    student.eval()
    
    print("\nStarting visualization...")
    print("Close each window to proceed to the next model.\n")
    
    # Compare all models on the same random environment
    compare_all_models(env, teacher, student, seed=42)
    
    plt.show()

