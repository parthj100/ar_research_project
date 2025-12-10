"""
Experiment: ConvNeXt Teacher → MobileNetV3 Student on CIFAR-100

This experiment tests knowledge distillation for AR-relevant object recognition:
- Teacher: ConvNeXt-Base (88.6M params) - powerful modern CNN
- Student: MobileNetV3-Small (2.5M params) - mobile-optimized
- Dataset: CIFAR-100 (100 classes, 60K images)
- Compression: ~35x parameter reduction

AR Relevance: Object recognition is fundamental for AR - anchoring virtual 
content to real objects, contextual understanding, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
from tqdm import tqdm
import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'teacher_model': 'convnext_base',
    'student_model': 'mobilenetv3_small_100', 
    'dataset': 'cifar100',
    'num_classes': 100,
    'image_size': 224,  # Upscale CIFAR to 224 for pretrained models
    'batch_size': 64,
    'epochs': 15,  # Reduced for faster iteration
    'lr': 1e-3,
    'weight_decay': 0.01,
    # Distillation settings
    'temperature': 4.0,
    'alpha': 0.7,  # Weight for distillation loss
    'beta': 0.3,   # Weight for task loss
}

# ============================================================
# Dataset
# ============================================================

class CIFAR100Dataset(torch.utils.data.Dataset):
    """CIFAR-100 dataset wrapper with transforms."""
    
    def __init__(self, hf_dataset, transform, split='train'):
        self.dataset = hf_dataset
        self.transform = transform
        self.split = split
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']  # PIL Image
        label = item['fine_label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_dataloaders(batch_size=64, image_size=224):
    """Create train and validation dataloaders."""
    
    # Load CIFAR-100 from HuggingFace
    print("Loading CIFAR-100 dataset...")
    dataset = load_dataset('cifar100')
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = CIFAR100Dataset(dataset['train'], train_transform)
    val_dataset = CIFAR100Dataset(dataset['test'], val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    
    return train_loader, val_loader

# ============================================================
# Models
# ============================================================

class TeacherModel(nn.Module):
    """ConvNeXt-Base teacher with pretrained weights."""
    
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=pretrained, num_classes=0)
        self.embed_dim = 1024  # ConvNeXt-Base output dim
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())


class StudentModel(nn.Module):
    """MobileNetV3-Small student for mobile deployment."""
    
    def __init__(self, num_classes=100, teacher_embed_dim=1024, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=0)
        
        # Dynamically get the feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.embed_dim = self.backbone(dummy).shape[-1]
        
        # Feature projection to match teacher
        self.feature_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, teacher_embed_dim),
        )
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.feature_projector(features)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features, 'projected_features': projected}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# Distillation Loss
# ============================================================

class DistillationLoss(nn.Module):
    """Combined distillation loss: KL divergence + feature matching + CE."""
    
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Distillation weight
        self.beta = beta    # Task (CE) weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_output, teacher_output, labels):
        # Soft label distillation (KL divergence)
        student_logits = student_output['logits']
        teacher_logits = teacher_output['logits']
        
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature distillation
        student_features = student_output['projected_features']
        teacher_features = teacher_output['features']
        feature_loss = self.mse_loss(student_features, teacher_features.detach())
        
        # Task loss (cross-entropy)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * (kl_loss + 0.5 * feature_loss) + self.beta * ce_loss
        
        return {
            'total': total_loss,
            'kl': kl_loss,
            'feature': feature_loss,
            'ce': ce_loss,
        }

# ============================================================
# Training
# ============================================================

def train_epoch(teacher, student, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    student.train()
    teacher.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_output = teacher(images)
        
        # Get student predictions
        student_output = student(images)
        
        # Calculate loss
        loss_dict = criterion(student_output, teacher_output, labels)
        loss = loss_dict['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = student_output['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100*correct/total:.1f}%"
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        logits = output['logits']
        
        loss = ce_loss(logits, labels)
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total
    }

# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("EXPERIMENT: ConvNeXt → MobileNetV3 Distillation on CIFAR-100")
    print("="*70)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create output directory
    output_dir = Path('results/convnext_mobilenet')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size']
    )
    
    # Create models
    print("\nCreating models...")
    teacher = TeacherModel(num_classes=CONFIG['num_classes'], pretrained=True).to(device)
    student = StudentModel(num_classes=CONFIG['num_classes'], teacher_embed_dim=1024, pretrained=True).to(device)
    
    print(f"  Teacher (ConvNeXt-Base): {teacher.get_params()/1e6:.1f}M params")
    print(f"  Student (MobileNetV3-Small): {student.get_params()/1e6:.1f}M params")
    print(f"  Compression ratio: {teacher.get_params()/student.get_params():.1f}x")
    
    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Fine-tune teacher head first
    print("\n" + "="*60)
    print("STEP 1: Fine-tuning Teacher Head")
    print("="*60)
    
    teacher.classifier.requires_grad_(True)
    teacher_optimizer = optim.AdamW(teacher.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    teacher_criterion = nn.CrossEntropyLoss()
    
    best_teacher_acc = 0
    for epoch in range(5):  # Quick teacher fine-tuning
        teacher.train()
        for images, labels in tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/5'):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = teacher.backbone(images)
            logits = teacher.classifier(features)
            
            loss = teacher_criterion(logits, labels)
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
        
        # Evaluate
        teacher_metrics = evaluate(teacher, val_loader, device)
        print(f"  Epoch {epoch+1}: Val Acc = {100*teacher_metrics['accuracy']:.2f}%")
        
        if teacher_metrics['accuracy'] > best_teacher_acc:
            best_teacher_acc = teacher_metrics['accuracy']
            torch.save(teacher.state_dict(), output_dir / 'teacher_best.pt')
    
    print(f"\nBest Teacher Accuracy: {100*best_teacher_acc:.2f}%")
    
    # Load best teacher and freeze
    teacher.load_state_dict(torch.load(output_dir / 'teacher_best.pt', weights_only=True))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Train student with distillation
    print("\n" + "="*60)
    print("STEP 2: Training Student with Distillation")
    print("="*60)
    
    optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = DistillationLoss(
        temperature=CONFIG['temperature'],
        alpha=CONFIG['alpha'],
        beta=CONFIG['beta']
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        
        # Train
        train_metrics = train_epoch(teacher, student, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(student, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {100*train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}  Acc: {100*val_metrics['accuracy']:.2f}%")
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(student.state_dict(), output_dir / 'student_best.pt')
            print(f"  ✓ New best! ({100*best_val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(student.state_dict(), output_dir / f'student_epoch_{epoch}.pt')
    
    # Save final
    torch.save(student.state_dict(), output_dir / 'student_final.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"  Teacher (ConvNeXt-Base): {100*best_teacher_acc:.2f}% val accuracy")
    print(f"  Student (MobileNetV3-Small): {100*best_val_acc:.2f}% val accuracy")
    print(f"  Compression: {teacher.get_params()/student.get_params():.1f}x")
    print(f"  Knowledge Transfer: {100*best_val_acc/best_teacher_acc:.1f}% of teacher performance")
    print("="*70)


if __name__ == '__main__':
    main()

