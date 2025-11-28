"""
Phase 2-Lite: Enhanced Synthetic Vision Environment
More complex than Phase 1, but trainable
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge
import io
from PIL import Image


class EnhancedVisionEnv(gym.Env):
    """
    Enhanced synthetic environment for Phase 2
    
    Improvements over Phase 1:
    - 64x64 resolution (4x more pixels)
    - Textured backgrounds
    - 5-7 objects with occlusions
    - Multiple shapes (circles, squares, triangles, diamonds)
    - Lighting variation
    - Sensor noise
    """
    
    def __init__(self, grid_size=10, image_size=64, max_steps=40, num_objects=6):
        super().__init__()
        
        self.grid_size = grid_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_objects = num_objects
        
        # Observation: 64x64 RGB images
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
        
        # Agent position
        self.agent_pos = self._rng.uniform(1.5, self.grid_size-1.5, size=2)
        
        # Generate varied objects
        self.objects = []
        
        # Target (always first) - red star
        target_pos = self._rng.uniform(2, self.grid_size-2, size=2)
        self.objects.append({
            'pos': target_pos,
            'shape': 'star',
            'color': 'red',
            'size': self._rng.uniform(0.5, 0.7),
            'is_target': True,
            'z_order': 10,
            'angle': self._rng.uniform(0, 360)
        })
        
        # Distractors with variety
        colors = ['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        shapes = ['circle', 'square', 'triangle', 'diamond']
        
        for i in range(self.num_objects - 1):
            pos = self._rng.uniform(1.5, self.grid_size-1.5, size=2)
            self.objects.append({
                'pos': pos,
                'shape': self._rng.choice(shapes),
                'color': colors[i % len(colors)],
                'size': self._rng.uniform(0.3, 0.5),
                'is_target': False,
                'z_order': self._rng.integers(1, 6),
                'angle': self._rng.uniform(0, 360)
            })
        
        # Background style
        self.background_style = self._rng.choice(['gradient', 'textured', 'patterned'])
        self.lighting = self._rng.uniform(0.75, 1.0)
        
        self.target = self.objects[0]
        
        return self._get_observation(), {}
    
    def _render_scene(self):
        """Render enhanced scene"""
        fig, ax = plt.subplots(figsize=(4, 4), dpi=self.image_size//2)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Enhanced background
        if self.background_style == 'gradient':
            gradient = np.linspace(0.95, 0.75, 50).reshape(10, 5)
            ax.imshow(gradient, extent=[0, self.grid_size, 0, self.grid_size],
                     cmap='gray', alpha=0.6, zorder=0)
        elif self.background_style == 'textured':
            texture = self._rng.uniform(0.85, 0.95, (30, 30))
            ax.imshow(texture, extent=[0, self.grid_size, 0, self.grid_size],
                     cmap='gray', alpha=0.7, zorder=0)
        elif self.background_style == 'patterned':
            for i in range(0, self.grid_size, 2):
                color = 0.9 if i % 4 == 0 else 0.85
                ax.axhspan(i, i+2, facecolor=str(color), alpha=0.3, zorder=0)
        else:
            ax.set_facecolor('#e8e8e8')
        
        # Sort by z-order for occlusions
        sorted_objects = sorted(self.objects, key=lambda x: x['z_order'])
        
        # Draw objects with variety
        for obj in sorted_objects:
            color = obj['color']
            size = obj['size'] * 20
            alpha = self.lighting
            angle = obj['angle']
            
            if obj['shape'] == 'circle':
                circle = Circle(obj['pos'], size, color=color, alpha=alpha,
                              edgecolor='black', linewidth=1.8, zorder=obj['z_order'])
                ax.add_patch(circle)
            
            elif obj['shape'] == 'square':
                square = Rectangle((obj['pos'][0]-size, obj['pos'][1]-size),
                                  size*2, size*2, color=color, alpha=alpha,
                                  angle=angle,
                                  edgecolor='black', linewidth=1.8, zorder=obj['z_order'])
                ax.add_patch(square)
            
            elif obj['shape'] == 'triangle':
                points = np.array([
                    [obj['pos'][0], obj['pos'][1] + size],
                    [obj['pos'][0] - size*0.866, obj['pos'][1] - size*0.5],
                    [obj['pos'][0] + size*0.866, obj['pos'][1] - size*0.5]
                ])
                triangle = Polygon(points, color=color, alpha=alpha,
                                 edgecolor='black', linewidth=1.8, zorder=obj['z_order'])
                ax.add_patch(triangle)
            
            elif obj['shape'] == 'diamond':
                points = np.array([
                    [obj['pos'][0], obj['pos'][1] + size],
                    [obj['pos'][0] - size, obj['pos'][1]],
                    [obj['pos'][0], obj['pos'][1] - size],
                    [obj['pos'][0] + size, obj['pos'][1]]
                ])
                diamond = Polygon(points, color=color, alpha=alpha,
                                edgecolor='black', linewidth=1.8, zorder=obj['z_order'])
                ax.add_patch(diamond)
            
            elif obj['shape'] == 'star':
                # Red star for target (bigger, more visible)
                ax.plot(obj['pos'][0], obj['pos'][1],
                       marker='*', markersize=size*2.5, color='red',
                       markeredgecolor='darkred', markeredgewidth=2.5,
                       zorder=obj['z_order'])
        
        # Draw agent
        ax.plot(self.agent_pos[0], self.agent_pos[1],
               marker='o', markersize=18, color='blue',
               markeredgecolor='darkblue', markeredgewidth=2.5,
               zorder=20)
        
        # Convert to array
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        # Resize
        img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img)
        
        # Add sensor noise
        noise = self._rng.integers(-3, 3, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # CHW format
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def _get_observation(self):
        return self._render_scene()
    
    def step(self, action):
        self.steps += 1
        
        # Movement
        move_speed = 0.5
        if action == 0:    self.agent_pos[1] += move_speed  # forward
        elif action == 1:  self.agent_pos[1] -= move_speed  # back
        elif action == 2:  self.agent_pos[0] -= move_speed  # left
        elif action == 3:  self.agent_pos[0] += move_speed  # right
        
        # Bounds
        self.agent_pos = np.clip(self.agent_pos, 0.5, self.grid_size - 0.5)
        
        # Check target
        dist_to_target = np.linalg.norm(self.agent_pos - self.target['pos'])
        
        done = False
        reward = -0.01
        
        # Success
        if dist_to_target < 1.0:
            reward = 1.0
            done = True
        
        # Timeout
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_observation(), reward, done, False, {
            'success': dist_to_target < 1.0,
            'distance': dist_to_target,
            'steps': self.steps
        }


if __name__ == "__main__":
    print("Testing Enhanced Synthetic Environment...")
    env = EnhancedVisionEnv()
    
    obs, _ = env.reset(seed=42)
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Actions: {env.action_space.n}")
    print(f"✓ Objects: {env.num_objects}")
    print(f"✓ Resolution: {env.image_size}x{env.image_size}")
    
    # Quick test
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    print("\n✓ Phase 2-Lite environment ready!")
    print("  4x more pixels than Phase 1")
    print("  More complex visuals")
    print("  Still fast to train!")
