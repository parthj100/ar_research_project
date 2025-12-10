"""
Phase 3: CLIP → MobileViT Distillation Training

Knowledge distillation from CLIP teacher to MobileViT student.
Supports multiple distillation strategies:
1. Feature distillation (match CLIP embeddings)
2. Response distillation (soft labels)
3. Combined distillation (both)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from data.ego4d_subset import Ego4DSyntheticDataset, create_ego4d_dataloader


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for knowledge transfer.
    
    Loss = α * L_feature + β * L_response + γ * L_task
    
    Where:
    - L_feature: MSE between student and teacher features
    - L_response: KL divergence on soft labels
    - L_task: Cross-entropy on hard labels
    """
    
    def __init__(
        self,
        alpha: float = 1.0,   # Feature distillation weight
        beta: float = 1.0,    # Response distillation weight
        gamma: float = 0.5,   # Task loss weight
        temperature: float = 4.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Compute combined distillation loss.
        
        Returns dict with total_loss and individual components.
        """
        losses = {}
        
        # Feature distillation (MSE on normalized features)
        if self.alpha > 0:
            student_feat_norm = F.normalize(student_features, dim=-1)
            teacher_feat_norm = F.normalize(teacher_features, dim=-1)
            losses['feature'] = self.mse(student_feat_norm, teacher_feat_norm)
        else:
            losses['feature'] = torch.tensor(0.0, device=student_logits.device)
        
        # Response distillation (KL on soft labels)
        if self.beta > 0:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            losses['response'] = self.kl(student_soft, teacher_soft) * (self.temperature ** 2)
        else:
            losses['response'] = torch.tensor(0.0, device=student_logits.device)
        
        # Task loss (CE on hard labels)
        if self.gamma > 0:
            losses['task'] = self.ce(student_logits, labels)
        else:
            losses['task'] = torch.tensor(0.0, device=student_logits.device)
        
        # Combined loss
        losses['total'] = (
            self.alpha * losses['feature'] +
            self.beta * losses['response'] +
            self.gamma * losses['task']
        )
        
        return losses


class DistillationTrainer:
    """
    Trainer for CLIP → MobileViT distillation.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = 'cpu',
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        temperature: float = 4.0,
        output_dir: str = 'results',
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        
        # Loss
        self.criterion = DistillationLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
        )
        
        # Metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.student.train()
        
        total_loss = 0.0
        loss_components = {'feature': 0, 'response': 0, 'task': 0}
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get teacher predictions (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(frames, return_features=True)
            
            # Get student predictions
            student_out = self.student(frames, return_features=True)
            
            # Compute loss
            losses = self.criterion(
                student_logits=student_out['logits'],
                student_features=student_out['pooled_features'],
                teacher_logits=teacher_out['logits'],
                teacher_features=teacher_out['pooled_features'],
                labels=labels,
            )
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += losses['total'].item() * frames.size(0)
            for k in loss_components:
                loss_components[k] += losses[k].item() * frames.size(0)
            
            preds = student_out['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc': f"{100*correct/total:.1f}%",
            })
        
        self.scheduler.step()
        
        metrics = {
            'loss': total_loss / total,
            'accuracy': correct / total,
        }
        for k, v in loss_components.items():
            metrics[f'loss_{k}'] = v / total
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate student model."""
        if self.val_loader is None:
            return {}
        
        self.student.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get predictions
            teacher_out = self.teacher(frames, return_features=True)
            student_out = self.student(frames, return_features=True)
            
            # Compute loss
            losses = self.criterion(
                student_logits=student_out['logits'],
                student_features=student_out['pooled_features'],
                teacher_logits=teacher_out['logits'],
                teacher_features=teacher_out['pooled_features'],
                labels=labels,
            )
            
            total_loss += losses['total'].item() * frames.size(0)
            preds = student_out['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
        }
    
    def train(self, num_epochs: int = 50):
        """Full training loop."""
        print("="*60)
        print("Starting CLIP → MobileViT Distillation")
        print("="*60)
        print(f"Teacher params: {sum(p.numel() for p in self.teacher.parameters()):,}")
        print(f"Student params: {sum(p.numel() for p in self.student.parameters()):,}")
        print(f"Compression: {sum(p.numel() for p in self.teacher.parameters()) / sum(p.numel() for p in self.student.parameters()):.1f}x")
        print(f"Device: {self.device}")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.2%}")
            print(f"    Feature: {train_metrics['loss_feature']:.4f}  "
                  f"Response: {train_metrics['loss_response']:.4f}  "
                  f"Task: {train_metrics['loss_task']:.4f}")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.2%}")
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint('best_student.pt')
                    print(f"  ✓ New best! ({val_metrics['accuracy']:.2%})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'student_epoch_{epoch}.pt')
        
        # Save final model
        self.save_checkpoint('final_student.pt')
        self.save_history()
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2%}")
        print("="*60)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def save_history(self):
        """Save training history."""
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: CLIP → MobileViT Distillation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/ego4d',
                       help='Path to Ego4D data directory')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data (for testing without Ego4D)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for synthetic data')
    
    # Model arguments
    parser.add_argument('--teacher_size', type=str, default='base',
                       choices=['base', 'large'],
                       help='CLIP teacher size')
    parser.add_argument('--student_size', type=str, default='xxs',
                       choices=['tiny', 'xxs', 'xs', 's'],
                       help='MobileViT student size')
    parser.add_argument('--num_actions', type=int, default=10,
                       help='Number of action classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Distillation arguments
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Feature distillation weight')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Response distillation weight')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Task loss weight')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Distillation temperature')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    
    train_dataset = Ego4DSyntheticDataset(
        num_samples=args.num_samples,
        frames_per_clip=8,
        image_size=224,
        num_actions=args.num_actions,
    ) if args.use_synthetic else None
    
    if train_dataset is None:
        print("Note: Real Ego4D data loading not yet implemented.")
        print("Using synthetic data for testing...")
        train_dataset = Ego4DSyntheticDataset(
            num_samples=args.num_samples,
            frames_per_clip=8,
            image_size=224,
            num_actions=args.num_actions,
        )
    
    # Split into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    
    # Create models
    print("\nCreating models...")
    
    teacher = create_clip_teacher(
        model_size=args.teacher_size,
        num_actions=args.num_actions,
        for_distillation=True,
        freeze_clip=True,
    )
    
    student = create_mobilevit_student(
        model_size=args.student_size,
        num_actions=args.num_actions,
        teacher_embed_dim=512 if args.teacher_size == 'base' else 768,
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()

