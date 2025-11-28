"""
Phase 2 Student - Compressed CNN for 64x64
"""

import torch
import torch.nn as nn


class Phase2StudentCNN(nn.Module):
    """
    Lightweight CNN for 64x64 images
    ~100K parameters (10x smaller than full ResNet)
    """
    def __init__(self, num_actions=4):
        super().__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # 3x64x64 -> 24x32x32
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            
            # 24x32x32 -> 48x16x16
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # 48x16x16 -> 96x8x8
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            # 96x8x8 -> 128x4x4
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        # Normalize
        x = x.float() / 255.0
        
        # Extract features
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Get logits
        logits = self.policy(x)
        
        return logits
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    student = Phase2StudentCNN()
    print(f"Student: {student.get_num_parameters():,} params")
    
    # Test
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    out = student(x)
    print(f"Output shape: {out.shape}")
    print("✓ Ready!")
