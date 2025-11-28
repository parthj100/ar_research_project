"""
Train Vision Teacher Model
Uses Deep Q-Learning (DQN) to train a CNN-based agent on visual AR task
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ar_vision_env import ARVisionEnvSimple
from models.vision_teacher import VisionTeacherCNN


class DQNAgent:
    """Simple DQN agent for training the teacher"""
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.target_model = VisionTeacherCNN().to(device)
        self.target_model.load_state_dict(model.state_dict())
        
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.from_numpy(np.array(states)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_teacher(num_episodes=1000, device='cpu'):
    """Train teacher model using DQN"""
    env = ARVisionEnvSimple()
    model = VisionTeacherCNN(num_actions=4, image_channels=3)
    agent = DQNAgent(model, device)
    
    print(f"Training Vision Teacher on {device}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Environment: {env.observation_space.shape}")
    print("="*60)
    
    episode_rewards = []
    episode_steps = []
    episode_successes = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_successes.append(int(info.get('success', False)))
        
        agent.decay_epsilon()
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target()
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps = np.mean(episode_steps[-50:])
            success_rate = np.mean(episode_successes[-50:])
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/vision_teacher.pt")
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to results/vision_teacher.pt")
    
    # Final evaluation
    print("\nFinal Evaluation (200 episodes):")
    eval_successes = []
    eval_steps = []
    
    for _ in range(200):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state, eval_mode=True)
            state, _, done, _, info = env.step(action)
            steps += 1
        
        eval_successes.append(int(info.get('success', False)))
        eval_steps.append(steps)
    
    print(f"Success Rate: {np.mean(eval_successes):.2%}")
    print(f"Avg Steps: {np.mean(eval_steps):.2f}")
    
    return model


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    train_teacher(num_episodes=1000, device=device)

