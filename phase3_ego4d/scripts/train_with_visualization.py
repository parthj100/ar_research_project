"""
Train with Real-Time Visualization

Shows:
- Sample frames from current batch
- Model predictions vs ground truth
- Training metrics updating in real-time
- Loss curves
- Confusion matrix (updating)
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque

from data.egohands_loader import create_egohands_dataloaders
from models.clip_teacher import create_clip_teacher
from models.mobilevit_student import create_mobilevit_student
from scripts.train_distill import DistillationTrainer, DistillationLoss


class VisualizedTrainer(DistillationTrainer):
    """Trainer with real-time visualization."""
    
    def __init__(self, *args, visualize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualize = visualize
        self.viz_data = {
            'epoch': 0,
            'batch': 0,
            'loss_history': deque(maxlen=100),
            'acc_history': deque(maxlen=100),
            'current_batch': None,
            'predictions': [],
            'ground_truth': [],
            'sample_frames': None,
        }
        self.viz_fig = None
        self.viz_gs = None
        self.viz_axes = {}
        
        if visualize:
            self._start_visualization()
    
    def _start_visualization(self):
        """Start visualization in main thread (required for macOS)."""
        # On macOS, matplotlib GUI must be in main thread
        # Use default backend (should work on macOS)
        self._run_visualization()
    
    def _run_visualization(self):
        """Initialize visualization (called in main thread)."""
        plt.ion()  # Interactive mode
        fig = plt.figure(figsize=(16, 10))
        self.viz_fig = fig
        self.viz_gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplot 1-4: Sample frames (2x2 grid in gs[0:2, 0:2])
        # Subplot 2: Loss curve
        ax_loss = fig.add_subplot(self.viz_gs[0, 2:4])
        ax_loss.set_title('Training Loss (Real-time)', fontsize=11, fontweight='bold')
        ax_loss.set_xlabel('Batch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        
        # Subplot 3: Accuracy curve
        ax_acc = fig.add_subplot(self.viz_gs[1, 2:4])
        ax_acc.set_title('Training Accuracy (Real-time)', fontsize=11, fontweight='bold')
        ax_acc.set_xlabel('Batch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_ylim(0, 100)
        ax_acc.grid(True, alpha=0.3)
        
        # Subplot 4: Confusion matrix (simplified)
        ax_conf = fig.add_subplot(self.viz_gs[2, 0:2])
        ax_conf.set_title('Recent Predictions vs Ground Truth', fontsize=11, fontweight='bold')
        ax_conf.axis('off')
        
        # Subplot 5: Training stats
        ax_stats = fig.add_subplot(self.viz_gs[2, 2:4])
        ax_stats.set_title('Training Statistics', fontsize=11, fontweight='bold')
        ax_stats.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Store axes for updates
        self.viz_axes = {
            'loss': ax_loss,
            'acc': ax_acc,
            'conf': ax_conf,
            'stats': ax_stats,
        }
    
    def _update_plots(self):
        """Update all plots with current data (called during training)."""
        if not self.visualize or self.viz_fig is None:
            return
        
        data = self.viz_data
        fig = self.viz_fig
        gs = self.viz_gs
        ax_loss = self.viz_axes['loss']
        ax_acc = self.viz_axes['acc']
        ax_conf = self.viz_axes['conf']
        ax_stats = self.viz_axes['stats']
        
        # 1. Sample frames with predictions
        # Clear existing frame subplots
        for i in range(4):
            row = i // 2
            col = i % 2
            try:
                # Remove existing subplot if it exists
                for ax in fig.get_axes():
                    if ax.get_subplotspec() and ax.get_subplotspec().get_gridspec() == gs:
                        pos = ax.get_subplotspec().get_geometry()
                        if pos[0] == row and pos[1] == col and pos[2] == 1 and pos[3] == 1:
                            ax.remove()
            except:
                pass
        
        # Show frames in a grid
        if data['sample_frames'] is not None and len(data['predictions']) > 0:
            frames = data['sample_frames']
            preds = data['predictions']
            labels = data['ground_truth']
            action_labels = getattr(self.train_loader.dataset, 'action_labels', 
                                   [f'class_{i}' for i in range(len(preds))])
            
            n_show = min(4, frames.shape[1])  # Show 4 frames
            for i in range(n_show):
                if i < frames.shape[1]:
                    row = i // 2
                    col = i % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Denormalize frame
                    frame = frames[0, i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    frame = frame * std + mean
                    frame = torch.clamp(frame, 0, 1)
                    frame = frame.permute(1, 2, 0).numpy()
                    
                    ax.imshow(frame)
                    ax.axis('off')
                    
                    if i == 0 and len(preds) > 0:
                        pred_label = action_labels[preds[0]] if preds[0] < len(action_labels) else 'unknown'
                        true_label = action_labels[labels[0]] if labels[0] < len(action_labels) else 'unknown'
                        color = 'green' if preds[0] == labels[0] else 'red'
                        ax.set_title(f'Frame {i+1}\nPred: {pred_label}\nTrue: {true_label}', 
                                   fontsize=9, color=color, fontweight='bold')
                    else:
                        ax.set_title(f'Frame {i+1}', fontsize=9)
        
        if data['sample_frames'] is None:
            # Show placeholder
            ax = fig.add_subplot(gs[0, 0])
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            n_show = min(4, frames.shape[1])  # Show 4 frames
            # Create a 2x2 grid for frames within the frames subplot area
            for i in range(n_show):
                if i < frames.shape[1]:
                    # Create subplot within the frames area
                    row = i // 2
                    col = i % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Denormalize frame
                    frame = frames[0, i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    frame = frame * std + mean
                    frame = torch.clamp(frame, 0, 1)
                    frame = frame.permute(1, 2, 0).numpy()
                    
                    ax.imshow(frame)
                    ax.axis('off')
                    
                    if i == 0 and len(preds) > 0:
                        pred_label = action_labels[preds[0]] if preds[0] < len(action_labels) else 'unknown'
                        true_label = action_labels[labels[0]] if labels[0] < len(action_labels) else 'unknown'
                        color = 'green' if preds[0] == labels[0] else 'red'
                        ax.set_title(f'Frame {i+1}\nPred: {pred_label}\nTrue: {true_label}', 
                                   fontsize=9, color=color, fontweight='bold')
                    else:
                        ax.set_title(f'Frame {i+1}', fontsize=9)
        
        # 2. Loss curve
        ax_loss.clear()
        if data['loss_history']:
            losses = list(data['loss_history'])
            ax_loss.plot(losses, 'b-', linewidth=2, label='Loss')
            ax_loss.set_title(f'Training Loss (Current: {losses[-1]:.4f})', fontsize=11, fontweight='bold')
            ax_loss.set_xlabel('Batch')
            ax_loss.set_ylabel('Loss')
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend()
        
        # 3. Accuracy curve
        ax_acc.clear()
        if data['acc_history']:
            accs = list(data['acc_history'])
            ax_acc.plot(accs, 'g-', linewidth=2, label='Accuracy')
            ax_acc.set_title(f'Training Accuracy (Current: {accs[-1]:.1f}%)', fontsize=11, fontweight='bold')
            ax_acc.set_xlabel('Batch')
            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.set_ylim(0, 100)
            ax_acc.grid(True, alpha=0.3)
            ax_acc.legend()
        
        # 4. Confusion matrix (recent predictions)
        ax_conf.clear()
        if len(data['predictions']) > 0 and len(data['ground_truth']) > 0:
            # Show recent predictions
            recent_preds = data['predictions'][-20:]
            recent_labels = data['ground_truth'][-20:]
            
            correct = sum(p == l for p, l in zip(recent_preds, recent_labels))
            total = len(recent_preds)
            acc = correct / total * 100 if total > 0 else 0
            
            ax_conf.text(0.5, 0.7, f'Recent Accuracy: {acc:.1f}%', 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        transform=ax_conf.transAxes)
            ax_conf.text(0.5, 0.5, f'Correct: {correct}/{total}', 
                        ha='center', va='center', fontsize=12,
                        transform=ax_conf.transAxes)
            
            # Show prediction distribution
            if len(recent_preds) > 0:
                pred_counts = {}
                for p in recent_preds:
                    pred_counts[p] = pred_counts.get(p, 0) + 1
                
                if pred_counts:
                    ax_conf.text(0.5, 0.3, 'Prediction Distribution:', 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               transform=ax_conf.transAxes)
                    dist_text = ', '.join([f'C{p}:{c}' for p, c in list(pred_counts.items())[:5]])
                    ax_conf.text(0.5, 0.2, dist_text, 
                               ha='center', va='center', fontsize=9,
                               transform=ax_conf.transAxes)
        
        ax_conf.axis('off')
        
        # 5. Training stats
        ax_stats.clear()
        stats_text = f"""Epoch: {data['epoch']}
Batch: {data['batch']}
Total Batches: {getattr(self, 'total_batches', '?')}

Loss Components:
  - Feature: {data.get('loss_feature', 0):.4f}
  - Response: {data.get('loss_response', 0):.4f}
  - Task: {data.get('loss_task', 0):.4f}
  - Total: {data.get('loss_total', 0):.4f}

Current Accuracy: {data['acc_history'][-1] if data['acc_history'] else 0:.1f}%
"""
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center', transform=ax_stats.transAxes)
        ax_stats.axis('off')
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch with visualization."""
        self.student.train()
        self.viz_data['epoch'] = epoch
        
        total_loss = 0.0
        loss_components = {'feature': 0, 'response': 0, 'task': 0}
        correct = 0
        total = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            self.viz_data['batch'] = batch_idx
            self.total_batches = len(self.train_loader)
            
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Store sample for visualization
            if batch_idx % 10 == 0:  # Update every 10 batches
                self.viz_data['sample_frames'] = frames[:1]  # First sample
            
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
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            # Update visualization data
            if batch_idx % 5 == 0:  # Update frequently
                self.viz_data['loss_history'].append(losses['total'].item())
                self.viz_data['acc_history'].append(100 * batch_correct / labels.size(0))
                self.viz_data['predictions'].extend(preds.cpu().tolist())
                self.viz_data['ground_truth'].extend(labels.cpu().tolist())
                self.viz_data['loss_feature'] = losses['feature'].item()
                self.viz_data['loss_response'] = losses['response'].item()
                self.viz_data['loss_task'] = losses['task'].item()
                self.viz_data['loss_total'] = losses['total'].item()
                
                # Update visualization
                try:
                    self._update_plots()
                    plt.pause(0.01)  # Non-blocking update
                except Exception as e:
                    if 'closed' not in str(e).lower():
                        print(f"Visualization update error: {e}")
                
                # Keep only recent predictions (for memory)
                if len(self.viz_data['predictions']) > 100:
                    self.viz_data['predictions'] = self.viz_data['predictions'][-100:]
                    self.viz_data['ground_truth'] = self.viz_data['ground_truth'][-100:]
            
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


def main():
    """Main training function with visualization."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*70)
    print("EgoHands Training with Real-Time Visualization")
    print("="*70)
    
    # Load EgoHands data
    print("\nLoading EgoHands dataset...")
    train_loader, val_loader, action_labels = create_egohands_dataloaders(
        data_dir='data/egohands',
        batch_size=16,
        num_workers=0,
        image_size=224,
        frames_per_clip=8,
        train_split=0.8,
    )
    
    num_actions = len(action_labels)
    print(f"\n✓ Loaded {num_actions} action classes:")
    for i, label in enumerate(action_labels):
        print(f"  {i}: {label}")
    
    # Create models
    print("\nCreating models...")
    teacher = create_clip_teacher(
        model_size='base',
        num_actions=num_actions,
        for_distillation=True,
        freeze_clip=True,
    )
    
    student = create_mobilevit_student(
        model_size='xxs',
        num_actions=num_actions,
        teacher_embed_dim=512,
    )
    
    # Create trainer with visualization
    print("\nInitializing trainer with visualization...")
    print("  A visualization window will open showing:")
    print("    - Sample frames from current batch")
    print("    - Model predictions vs ground truth")
    print("    - Real-time loss and accuracy curves")
    print("    - Training statistics")
    
    trainer = VisualizedTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-5,
        alpha=0.5,
        beta=1.0,
        gamma=1.0,
        temperature=3.0,
        output_dir='results/egohands_visualized',
        visualize=True,
    )
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Action classes: {num_actions}")
    print(f"Frames per clip: 8")
    print("\n⚠ Keep the visualization window open during training!")
    print("="*70)
    
    # Train
    try:
        trainer.train(num_epochs=30)
        print("\n" + "="*70)
        print("EgoHands Training Completed!")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Closing visualization...")
        if trainer.viz_fig:
            plt.close(trainer.viz_fig)


if __name__ == '__main__':
    main()

