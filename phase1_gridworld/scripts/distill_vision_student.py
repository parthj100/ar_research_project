"""
Distill Vision Student Model
Collect teacher demonstrations and train student via behavioral cloning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ar_vision_env import ARVisionEnvSimple
from models.vision_teacher import VisionTeacherCNN, VisionTeacherPolicy
from models.vision_student import VisionStudentCNN


def collect_teacher_data(teacher_policy, env, num_episodes=500):
    """Collect (observation, action) pairs from teacher"""
    print("Collecting teacher demonstrations...")
    
    states = []
    actions = []
    successes = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        
        while not done:
            action, _ = teacher_policy.predict(obs, deterministic=True)
            action = int(action)
            
            episode_states.append(obs.copy())
            episode_actions.append(action)
            
            obs, _, done, _, info = env.step(action)
        
        # Only keep successful episodes for better training
        if info.get('success', False):
            states.extend(episode_states)
            actions.extend(episode_actions)
            successes += 1
        
        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep+1}/{num_episodes} episodes, "
                  f"{successes} successful ({successes/(ep+1):.1%})")
    
    states = np.array(states)
    actions = np.array(actions)
    
    print(f"\nDataset collected:")
    print(f"  Total samples: {len(states):,}")
    print(f"  Successful episodes: {successes}/{num_episodes} ({successes/num_episodes:.1%})")
    print(f"  State shape: {states.shape}")
    
    return states, actions


def train_student(teacher_path, num_collection_episodes=500, 
                 num_epochs=30, batch_size=64, device='cpu'):
    """Train student model via knowledge distillation"""
    
    print("="*60)
    print("Vision Student Distillation")
    print("="*60)
    
    # Load teacher
    env = ARVisionEnvSimple()
    teacher_model = VisionTeacherCNN()
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher_policy = VisionTeacherPolicy(teacher_model, device=device)
    
    print(f"\nTeacher loaded: {teacher_model.get_num_parameters():,} parameters")
    
    # Collect data
    states, actions = collect_teacher_data(teacher_policy, env, num_collection_episodes)
    
    # Save dataset
    os.makedirs("results", exist_ok=True)
    np.savez("results/vision_teacher_dataset.npz", states=states, actions=actions)
    print(f"Dataset saved to results/vision_teacher_dataset.npz")
    
    # Create student model
    student_model = VisionStudentCNN()
    print(f"\nStudent model: {student_model.get_num_parameters():,} parameters")
    print(f"Compression ratio: {teacher_model.get_num_parameters() / student_model.get_num_parameters():.1f}x")
    
    # Prepare data
    states_tensor = torch.from_numpy(states)
    actions_tensor = torch.from_numpy(actions).long()
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    student_model = student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining student for {num_epochs} epochs...")
    print("="*60)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        student_model.train()
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            logits = student_model(batch_states)
            loss = criterion(logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_states.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_actions).sum().item()
            total += batch_actions.size(0)
        
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, accuracy={accuracy:.3f}")
    
    # Save student
    torch.save(student_model.state_dict(), "results/vision_student.pt")
    print("\n" + "="*60)
    print("Student saved to results/vision_student.pt")
    
    # Evaluate student
    print("\nEvaluating student on environment...")
    from models.vision_student import VisionStudentPolicy
    student_policy = VisionStudentPolicy(student_model, device=device)
    
    successes = 0
    steps_list = []
    
    for _ in range(200):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = student_policy.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
            steps += 1
        
        successes += int(info.get('success', False))
        steps_list.append(steps)
    
    print(f"Student Success Rate: {successes/200:.2%}")
    print(f"Student Avg Steps: {np.mean(steps_list):.2f}")
    
    return student_model


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    train_student(
        teacher_path="results/vision_teacher.pt",
        num_collection_episodes=500,
        num_epochs=30,
        device=device
    )

