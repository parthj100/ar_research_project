"""
Simple Distillation Experiment: ResNet18 → MobileNetV3-Small on CIFAR-100

A simpler, faster experiment to demonstrate knowledge distillation.
- Teacher: ResNet18 (11.7M params)  
- Student: MobileNetV3-Small (2.5M params)
- Compression: ~4.7x
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
# Configuration - Simplified for stability
# ============================================================

CONFIG = {
    'teacher_model': 'resnet18',
    'student_model': 'mobilenetv3_small_100', 
    'dataset': 'cifar100',
    'num_classes': 100,
    'image_size': 224,
    'batch_size': 32,  # Smaller batch size for stability
    'teacher_epochs': 3,  # Quick teacher tuning
    'student_epochs': 5,  # Fewer epochs
    'lr': 5e-4,
    'weight_decay': 0.01,
    'temperature': 3.0,
    'alpha': 0.5,
    'beta': 0.5,
}

# ============================================================
# Dataset
# ============================================================

class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img'].convert('RGB')
        label = item['fine_label']
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(batch_size=32, image_size=224):
    print("Loading CIFAR-100 dataset...")
    dataset = load_dataset('cifar100')
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CIFAR100Dataset(dataset['train'], train_transform)
    val_dataset = CIFAR100Dataset(dataset['test'], val_transform)
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

# ============================================================
# Models
# ============================================================

class TeacherModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, num_classes=0)
        self.embed_dim = 512  # ResNet18 output
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())

class StudentModel(nn.Module):
    def __init__(self, num_classes=100, teacher_embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=0)
        
        # Get actual feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.embed_dim = self.backbone(dummy).shape[-1]
        
        self.feature_projector = nn.Linear(self.embed_dim, teacher_embed_dim)
        self.classifier = nn.Linear(teacher_embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.feature_projector(features)
        logits = self.classifier(projected)
        return {'logits': logits, 'features': features, 'projected_features': projected}
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# Training
# ============================================================

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {'accuracy': correct / total}

def main():
    print("="*70)
    print("SIMPLE DISTILLATION: ResNet18 → MobileNetV3-Small on CIFAR-100")
    print("="*70)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    output_dir = Path('results/simple_distill')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size']
    )
    
    # Create models
    print("\nCreating models...")
    teacher = TeacherModel(num_classes=CONFIG['num_classes'], pretrained=True).to(device)
    student = StudentModel(num_classes=CONFIG['num_classes'], teacher_embed_dim=512, pretrained=True).to(device)
    
    print(f"  Teacher (ResNet18): {teacher.get_params()/1e6:.1f}M params")
    print(f"  Student (MobileNetV3-Small): {student.get_params()/1e6:.1f}M params")
    print(f"  Compression ratio: {teacher.get_params()/student.get_params():.1f}x")
    
    # ============================================================
    # STEP 1: Fine-tune teacher
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: Fine-tuning Teacher")
    print("="*60)
    
    teacher_optimizer = optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=0.01)
    teacher_criterion = nn.CrossEntropyLoss()
    
    best_teacher_acc = 0
    for epoch in range(CONFIG['teacher_epochs']):
        teacher.train()
        pbar = tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/{CONFIG["teacher_epochs"]}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = teacher(images)
            loss = teacher_criterion(outputs['logits'], labels)
            
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        # Evaluate
        metrics = evaluate(teacher, val_loader, device)
        print(f"  Epoch {epoch+1}: Val Acc = {100*metrics['accuracy']:.2f}%")
        
        if metrics['accuracy'] > best_teacher_acc:
            best_teacher_acc = metrics['accuracy']
            torch.save(teacher.state_dict(), output_dir / 'teacher_best.pt')
    
    print(f"\nBest Teacher Accuracy: {100*best_teacher_acc:.2f}%")
    
    # Load best teacher
    teacher.load_state_dict(torch.load(output_dir / 'teacher_best.pt', weights_only=True))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # ============================================================
    # STEP 2: Train student with distillation
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Training Student with Distillation")
    print("="*60)
    
    student_optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    best_student_acc = 0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, CONFIG['student_epochs'] + 1):
        student.train()
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Student Epoch {epoch}/{CONFIG["student_epochs"]}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_out = teacher(images)
            
            # Get student predictions
            student_out = student(images)
            
            # Distillation loss (KL divergence on soft labels)
            T = CONFIG['temperature']
            soft_teacher = F.softmax(teacher_out['logits'] / T, dim=-1)
            soft_student = F.log_softmax(student_out['logits'] / T, dim=-1)
            kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
            
            # Feature loss
            feature_loss = mse_criterion(student_out['projected_features'], teacher_out['features'].detach())
            
            # Task loss (CE with true labels)
            ce_loss = ce_criterion(student_out['logits'], labels)
            
            # Combined loss
            loss = CONFIG['alpha'] * (kl_loss + 0.5 * feature_loss) + CONFIG['beta'] * ce_loss
            
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            
            # Track accuracy
            preds = student_out['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100*correct/total:.1f}%')
        
        train_acc = correct / total
        
        # Evaluate
        val_metrics = evaluate(student, val_loader, device)
        val_acc = val_metrics['accuracy']
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch}: Train Acc = {100*train_acc:.2f}%, Val Acc = {100*val_acc:.2f}%")
        
        if val_acc > best_student_acc:
            best_student_acc = val_acc
            torch.save(student.state_dict(), output_dir / 'student_best.pt')
            print(f"  → New best! Saved checkpoint.")
    
    # ============================================================
    # Results Summary
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    # Also evaluate student without distillation baseline (random init)
    baseline_student = StudentModel(num_classes=CONFIG['num_classes'], teacher_embed_dim=512, pretrained=True).to(device)
    baseline_metrics = evaluate(baseline_student, val_loader, device)
    
    results = {
        'teacher': {
            'model': CONFIG['teacher_model'],
            'params': teacher.get_params(),
            'best_accuracy': best_teacher_acc,
        },
        'student': {
            'model': CONFIG['student_model'],
            'params': student.get_params(),
            'best_accuracy': best_student_acc,
        },
        'baseline_student': {
            'accuracy': baseline_metrics['accuracy'],
        },
        'compression_ratio': teacher.get_params() / student.get_params(),
        'knowledge_retention': best_student_acc / best_teacher_acc if best_teacher_acc > 0 else 0,
        'distillation_gain': best_student_acc - baseline_metrics['accuracy'],
        'history': history,
    }
    
    print(f"\nTeacher ({CONFIG['teacher_model']}):")
    print(f"  Parameters: {teacher.get_params()/1e6:.1f}M")
    print(f"  Best Accuracy: {100*best_teacher_acc:.2f}%")
    
    print(f"\nStudent ({CONFIG['student_model']}):")
    print(f"  Parameters: {student.get_params()/1e6:.1f}M")
    print(f"  Best Accuracy: {100*best_student_acc:.2f}%")
    print(f"  Baseline (no distillation): {100*baseline_metrics['accuracy']:.2f}%")
    
    print(f"\nCompression: {results['compression_ratio']:.1f}x")
    print(f"Knowledge Retention: {100*results['knowledge_retention']:.1f}%")
    print(f"Distillation Gain: +{100*results['distillation_gain']:.2f}%")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'history'}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return results

if __name__ == '__main__':
    main()


