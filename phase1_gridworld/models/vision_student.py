"""
Vision-based Student Model
Lightweight CNN for on-device inference (mobile/edge).
Designed to be distilled from the teacher.
"""

import torch
import torch.nn as nn


class VisionStudentCNN(nn.Module):
    """
    Student network: Lightweight CNN for mobile deployment
    ~50K parameters (10x smaller than teacher)
    
    Architecture inspired by MobileNet-Nano / SqueezeNet
    Optimized for low latency and small model size
    """
    def __init__(self, num_actions=4, image_channels=3):
        super().__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 16x16x16
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2: 16x16x16 -> 32x8x8
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 32x8x8 -> 64x4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Compact policy head
        self.policy = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_actions)
        )
    
    def forward(self, x):
        """
        Args:
            x: Image tensor [batch, 3, H, W], values in [0, 255]
        Returns:
            logits: Action logits [batch, num_actions]
        """
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Extract features
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Get action logits
        logits = self.policy(x)
        
        return logits
    
    def get_num_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VisionStudentPolicy:
    """
    Wrapper for student model
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, obs, deterministic=True):
        """Predict action from observation"""
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs)
            
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
            
            obs = obs.to(self.device)
            logits = self.model(obs)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            
            return action.cpu().numpy(), None


if __name__ == "__main__":
    # Test the model
    model = VisionStudentCNN()
    print(f"Student Model: {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    dummy_input = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Compare with teacher
    from vision_teacher import VisionTeacherCNN
    teacher = VisionTeacherCNN()
    print(f"\nTeacher: {teacher.get_num_parameters():,} params")
    print(f"Student: {model.get_num_parameters():,} params")
    print(f"Compression ratio: {teacher.get_num_parameters() / model.get_num_parameters():.1f}x")

