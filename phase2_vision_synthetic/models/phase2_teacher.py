"""
Phase 2 Teacher with Pre-trained ResNet
For 64x64 enhanced synthetic environment
"""

import torch
import torch.nn as nn
from torchvision import models


class Phase2TeacherResNet(nn.Module):
    """Teacher using pre-trained ResNet18 for 64x64 images"""
    def __init__(self, num_actions=4, freeze_backbone=True):
        super().__init__()
        
        # Pre-trained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Q-network head
        self.q_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )
        
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x):
        # Normalize
        x = x.float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        # Features
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Q-values
        q_values = self.q_head(features)
        
        return q_values
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Phase2TeacherResNet()
    print(f"Teacher: {model.get_trainable_parameters():,} trainable params")
    print(f"Total: {model.get_num_parameters():,} params")
    
    # Test
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print("✓ Ready!")
