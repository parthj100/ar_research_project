"""
AR Vision Environment
Simulates a visual object localization task similar to real AR scenarios.
Agent receives camera-like images and must navigate to target objects.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import io
from PIL import Image


class ARVisionEnv(gym.Env):
    """
    Simulated AR environment with visual observations.
    
    Task: Agent sees a rendered scene and must navigate to a target object.
    Similar to AR wayfinding or object interaction tasks.
    
    Observations: 64x64 RGB images (simulating camera feed)
    Actions: 4 directions (forward, back, left, right)
    """
    
    metadata = {"render_modes": ["rgb_array", "human"]}
    
    def __init__(self, grid_size=10, image_size=64, max_steps=30, num_objects=3):
        super().__init__()
        
        self.grid_size = grid_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_objects = num_objects
        
        # Observation: RGB image (like camera feed)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, image_size, image_size), 
            dtype=np.uint8
        )
        
        # Actions: forward, back, left, right
        self.action_space = spaces.Discrete(4)
        
        self._rng = np.random.default_rng()
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng = np.random.default_rng(seed)
        
        self.steps = 0
        
        # Agent position (continuous for smoother visuals)
        self.agent_pos = self._rng.uniform(1, self.grid_size-1, size=2)
        
        # Object positions (different shapes/colors to simulate AR objects)
        self.objects = []
        for i in range(self.num_objects):
            pos = self._rng.uniform(1, self.grid_size-1, size=2)
            obj_type = i  # 0=target, 1,2=distractors
            color = ['red', 'blue', 'green'][i]
            self.objects.append({
                'pos': pos,
                'type': obj_type,
                'color': color,
                'is_target': (i == 0)
            })
        
        self.target = self.objects[0]  # First object is always target
        
        return self._get_observation(), {}
    
    def _render_scene(self):
        """Render the current scene as an image (simulates camera view)"""
        fig, ax = plt.subplots(figsize=(4, 4), dpi=32)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Background (simulates AR scene)
        ax.set_facecolor('#f0f0f0')
        
        # Draw objects
        for obj in self.objects:
            if obj['is_target']:
                # Target: Red star (like AR marker)
                ax.plot(obj['pos'][0], obj['pos'][1], 
                       marker='*', markersize=20, color='red', 
                       markeredgecolor='darkred', markeredgewidth=2,
                       zorder=2)
            else:
                # Distractor: Smaller circles
                ax.plot(obj['pos'][0], obj['pos'][1],
                       marker='o', markersize=10, color=obj['color'],
                       markeredgecolor='black', markeredgewidth=1,
                       zorder=1)
        
        # Draw agent (blue circle with direction indicator)
        ax.plot(self.agent_pos[0], self.agent_pos[1],
               marker='o', markersize=15, color='blue',
               markeredgecolor='darkblue', markeredgewidth=2,
               zorder=3)
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        # Get the RGBA buffer and convert to RGB
        buf = np.array(fig.canvas.buffer_rgba())
        img = buf[:, :, :3]  # Drop alpha channel
        plt.close(fig)
        
        # Resize to target size
        img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img)
        
        # Convert to CHW format (channels first) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def _get_observation(self):
        """Get visual observation (simulates camera feed)"""
        return self._render_scene()
    
    def step(self, action):
        self.steps += 1
        
        # Movement (continuous for smoother navigation)
        move_speed = 0.5
        if action == 0:    # forward (up)
            self.agent_pos[1] += move_speed
        elif action == 1:  # back (down)
            self.agent_pos[1] -= move_speed
        elif action == 2:  # left
            self.agent_pos[0] -= move_speed
        elif action == 3:  # right
            self.agent_pos[0] += move_speed
        
        # Keep agent in bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        
        # Check if reached target
        dist_to_target = np.linalg.norm(self.agent_pos - self.target['pos'])
        
        done = False
        reward = -0.01  # Small step penalty
        
        # Success: Close to target
        if dist_to_target < 0.8:
            reward = 1.0
            done = True
        
        # Timeout
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_observation(), reward, done, False, {
            'success': dist_to_target < 0.8,
            'distance': dist_to_target
        }
    
    def render(self):
        """Render for visualization"""
        return self._render_scene()


class ARVisionEnvSimple(ARVisionEnv):
    """
    Simpler version with smaller images for faster training.
    Use this for initial experiments.
    """
    def __init__(self):
        super().__init__(grid_size=8, image_size=32, max_steps=20, num_objects=2)

