"""
Experiment: DINOv2 Features → EfficientNet Student

This experiment uses self-supervised DINOv2 features for distillation:
- Teacher: DINOv2 ViT-B/14 (86M params) - self-supervised features
- Student: EfficientNet-B0 (5.3M params) - efficient CNN
- Dataset: CIFAR-100 (treat as mini scene/object dataset)
- Approach: Feature distillation from frozen DINOv2

AR Relevance: DINOv2 provides rich semantic features useful for 
scene understanding, object segmentation, and depth estimation in AR.
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
    'teacher_model': 'vit_base_patch14_dinov2',  # timm DINOv2 variant
    'student_model': 'efficientnet_b0',
    'dataset': 'cifar100',
    'num_classes': 100,
    'image_size': 224,
    'batch_size': 64,
    'epochs': 25,
    'lr': 5e-4,
    'weight_decay': 0.05,
    # Distillation
    'temperature': 3.0,
    'feature_weight': 2.0,  # Higher weight for DINOv2 features
    'task_weight': 1.0,
}

# ============================================================
# Dataset (shared with other experiment)
# ============================================================

class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        label = item['fine_label']
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(batch_size=64, image_size=224):
    print("Loading CIFAR-100...")
    dataset = load_dataset('cifar100')
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=8),
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    return train_loader, val_loader

# ============================================================
# Models
# ============================================================

class DINOv2Teacher(nn.Module):
    """DINOv2 as a frozen feature extractor with trainable classifier."""
    
    def __init__(self, num_classes=100):
        super().__init__()
        # Try to load DINOv2 from timm
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        self.embed_dim = 768  # DINOv2 ViT-B output
        
        # Trainable classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())


class EfficientNetStudent(nn.Module):
    """EfficientNet-B0 student with feature projection."""
    
    def __init__(self, num_classes=100, teacher_embed_dim=768):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Dynamically get the feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.embed_dim = self.backbone(dummy).shape[-1]
        
        # Project to match teacher features
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, teacher_embed_dim),
        )
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projector(features)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features, 'projected': projected}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# Distillation
# ============================================================

class DINODistillLoss(nn.Module):
    """Distillation loss optimized for DINOv2 features."""
    
    def __init__(self, temperature=3.0, feature_weight=2.0, task_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.task_weight = task_weight
        
    def forward(self, student_out, teacher_out, labels):
        # Feature matching (cosine similarity)
        student_proj = F.normalize(student_out['projected'], dim=-1)
        teacher_feat = F.normalize(teacher_out['features'], dim=-1)
        feature_loss = 1 - (student_proj * teacher_feat).sum(dim=-1).mean()
        
        # Soft label distillation
        s_logits = student_out['logits'] / self.temperature
        t_logits = teacher_out['logits'] / self.temperature
        kl_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss
        ce_loss = F.cross_entropy(student_out['logits'], labels)
        
        total = self.feature_weight * feature_loss + kl_loss + self.task_weight * ce_loss
        
        return {'total': total, 'feature': feature_loss, 'kl': kl_loss, 'ce': ce_loss}

# ============================================================
# Training
# ============================================================

def train_epoch(teacher, student, loader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            t_out = teacher(images)
        
        s_out = student(images)
        loss_dict = criterion(s_out, t_out, labels)
        
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        
        total_loss += loss_dict['total'].item()
        preds = s_out['logits'].argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f"{loss_dict['total'].item():.3f}", 'acc': f"{100*correct/total:.1f}%"})
    
    return {'loss': total_loss / len(loader), 'accuracy': correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)['logits']
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {'accuracy': correct / total}

# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("EXPERIMENT: DINOv2 → EfficientNet-B0 on CIFAR-100")
    print("="*70)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    output_dir = Path('results/dino_efficientnet')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_loader, val_loader = create_dataloaders(CONFIG['batch_size'], CONFIG['image_size'])
    
    # Models
    print("\nLoading models...")
    teacher = DINOv2Teacher(num_classes=CONFIG['num_classes']).to(device)
    student = EfficientNetStudent(num_classes=CONFIG['num_classes'], teacher_embed_dim=768).to(device)
    
    t_params = teacher.get_params() / 1e6
    s_params = student.get_params() / 1e6
    print(f"  Teacher (DINOv2 ViT-B): {t_params:.1f}M params")
    print(f"  Student (EfficientNet-B0): {s_params:.1f}M params")
    print(f"  Compression: {t_params/s_params:.1f}x")
    
    # Train teacher classifier first
    print("\n" + "="*60)
    print("STEP 1: Training Teacher Classifier")
    print("="*60)
    
    t_opt = optim.AdamW(teacher.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    best_t_acc = 0
    
    for epoch in range(8):
        teacher.classifier.train()
        for images, labels in tqdm(train_loader, desc=f'Teacher {epoch+1}/8'):
            images, labels = images.to(device), labels.to(device)
            out = teacher(images)
            loss = F.cross_entropy(out['logits'], labels)
            t_opt.zero_grad()
            loss.backward()
            t_opt.step()
        
        t_acc = evaluate(teacher, val_loader, device)['accuracy']
        print(f"  Epoch {epoch+1}: {100*t_acc:.2f}%")
        
        if t_acc > best_t_acc:
            best_t_acc = t_acc
            torch.save(teacher.state_dict(), output_dir / 'teacher_best.pt')
    
    print(f"\nBest Teacher: {100*best_t_acc:.2f}%")
    teacher.load_state_dict(torch.load(output_dir / 'teacher_best.pt', weights_only=True))
    teacher.eval()
    
    # Distill to student
    print("\n" + "="*60)
    print("STEP 2: Distilling to Student")
    print("="*60)
    
    optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = DINODistillLoss(
        temperature=CONFIG['temperature'],
        feature_weight=CONFIG['feature_weight'],
        task_weight=CONFIG['task_weight']
    )
    
    best_s_acc = 0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        
        train_metrics = train_epoch(teacher, student, train_loader, optimizer, criterion, device)
        val_acc = evaluate(student, val_loader, device)['accuracy']
        scheduler.step()
        
        print(f"  Train: {100*train_metrics['accuracy']:.2f}% | Val: {100*val_acc:.2f}%")
        
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_acc)
        
        if val_acc > best_s_acc:
            best_s_acc = val_acc
            torch.save(student.state_dict(), output_dir / 'student_best.pt')
            print(f"  ✓ Best! ({100*best_s_acc:.2f}%)")
    
    torch.save(student.state_dict(), output_dir / 'student_final.pt')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Teacher (DINOv2): {100*best_t_acc:.2f}%")
    print(f"  Student (EfficientNet): {100*best_s_acc:.2f}%")
    print(f"  Compression: {t_params/s_params:.1f}x")
    print(f"  Knowledge Transfer: {100*best_s_acc/best_t_acc:.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()

