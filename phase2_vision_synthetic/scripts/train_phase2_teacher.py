"""
Train Phase 2 Teacher (Enhanced Synthetic with ResNet)
Should converge faster than THOR - it's synthetic but more complex than Phase 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.enhanced_vision_env import EnhancedVisionEnv
from models.phase2_teacher import Phase2TeacherResNet


class DQNAgent:
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.target_model = Phase2TeacherResNet(freeze_backbone=True).to(device)
        self.target_model.load_state_dict(model.state_dict())
        
        self.device = device
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        self.memory = deque(maxlen=15000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.gamma = 0.99
        self.batch_size = 64
        
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
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_teacher(num_episodes=800, device='cpu'):
    print("="*70)
    print("PHASE 2: Enhanced Synthetic (64x64) with Pre-trained ResNet")
    print("="*70)
    
    env = EnhancedVisionEnv()
    model = Phase2TeacherResNet(num_actions=4, freeze_backbone=True)
    agent = DQNAgent(model, device, lr=1e-4)
    
    print(f"\nSetup:")
    print(f"  Device: {device}")
    print(f"  Resolution: 64x64 (4x Phase 1)")
    print(f"  Objects: {env.num_objects} (vs 2-3 in Phase 1)")
    print(f"  Trainable params: {model.get_trainable_parameters():,}")
    print(f"  Pre-trained backbone: ResNet18")
    print(f"  Episodes: {num_episodes}")
    print("="*70)
    
    episode_rewards = []
    episode_steps = []
    episode_successes = []
    best_success_rate = 0.0
    
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
        
        if episode % 10 == 0:
            agent.update_target()
        
        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_steps = np.mean(episode_steps[-25:])
            success_rate = np.mean(episode_successes[-25:])
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  Reward: {avg_reward:.3f}  Steps: {avg_steps:.1f}")
            print(f"  Success: {success_rate:.1%}  Epsilon: {agent.epsilon:.3f}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                os.makedirs("results", exist_ok=True)
                torch.save(model.state_dict(), "results/phase2_teacher_best.pt")
                print(f"  ✓ New best! ({success_rate:.1%})")
    
    torch.save(model.state_dict(), "results/phase2_teacher.pt")
    print("\n" + "="*70)
    print(f"Training complete! Best: {best_success_rate:.1%}")
    
    # Final eval
    print("\nFinal Evaluation (200 episodes)...")
    eval_successes = []
    for _ in range(200):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, eval_mode=True)
            state, _, done, _, info = env.step(action)
        eval_successes.append(int(info['success']))
    
    final_success = np.mean(eval_successes)
    print(f"Final Success Rate: {final_success:.1%}")
    print("="*70)
    
    return model


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Using device: {device}\n")
    print("Expected: 60-80% success (synthetic but enhanced)")
    print("Time: ~1 hour for 800 episodes\n")
    
    try:
        train_teacher(num_episodes=800, device=device)
    except KeyboardInterrupt:
        print("\n\nInterrupted - model saved")
