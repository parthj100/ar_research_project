"""
Vision-based Teacher Model
Larger CNN architecture similar to models that would run on cloud/edge servers.
Similar to MobileNetV2 or small ResNet architectures.
"""

import torch
import torch.nn as nn


class VisionTeacherCNN(nn.Module):
    """
    Teacher network: Larger CNN for visual reasoning
    ~500K parameters (simulates cloud-based model)
    
    Architecture inspired by MobileNetV2 / small ResNets
    """
    def __init__(self, num_actions=4, image_channels=3):
        super().__init__()
        
        # Feature extractor (convolutional backbone)
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 128x4x4 -> 256x2x2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
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


class VisionTeacherPolicy:
    """
    Wrapper for teacher model compatible with Stable-Baselines3 style interface
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, obs, deterministic=True):
        """
        Predict action from observation
        Compatible with SB3 interface
        """
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
    model = VisionTeacherCNN()
    print(f"Teacher Model: {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    dummy_input = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

