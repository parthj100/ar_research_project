"""
CLIP Teacher Model for Phase 3

Uses CLIP (Contrastive Language-Image Pre-training) as the teacher.
CLIP provides rich visual representations that can be distilled to a smaller model.

Two modes:
1. Feature Distillation: Student learns to match CLIP's visual embeddings
2. Action Prediction: Fine-tune CLIP for action classification, then distill
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import warnings

# Try to import CLIP backends
try:
    import open_clip
    CLIP_BACKEND = 'open_clip'
except ImportError:
    try:
        from transformers import CLIPModel, CLIPProcessor
        CLIP_BACKEND = 'transformers'
    except ImportError:
        CLIP_BACKEND = None
        warnings.warn("No CLIP backend found. Install open-clip-torch or transformers.")


class CLIPTeacher(nn.Module):
    """
    CLIP-based teacher model for egocentric action recognition.
    
    Architecture:
    - CLIP Vision Encoder (ViT-B/32 by default): ~86M params
    - Optional action classification head
    - Outputs both features and action predictions
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        num_actions: int = 10,
        freeze_clip: bool = True,
        use_temporal: bool = True,
    ):
        """
        Args:
            model_name: CLIP model variant (ViT-B-32, ViT-L-14, etc.)
            pretrained: Pretrained weights source
            num_actions: Number of action classes
            freeze_clip: Whether to freeze CLIP weights
            use_temporal: Whether to aggregate temporal features
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_actions = num_actions
        self.use_temporal = use_temporal
        self.freeze_clip = freeze_clip
        
        # Load CLIP
        if CLIP_BACKEND == 'open_clip':
            self._init_open_clip(model_name, pretrained)
        elif CLIP_BACKEND == 'transformers':
            self._init_transformers_clip()
        else:
            raise RuntimeError("No CLIP backend available. Install open-clip-torch or transformers.")
        
        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Temporal aggregation (for multi-frame input)
        if use_temporal:
            self.temporal_pool = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        
        # Action classification head
        self.action_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_actions),
        )
        
        print(f"CLIP Teacher initialized ({CLIP_BACKEND})")
        print(f"  Model: {model_name}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Actions: {num_actions}")
        print(f"  Frozen: {freeze_clip}")
    
    def _init_open_clip(self, model_name: str, pretrained: str):
        """Initialize using open_clip library"""
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.embed_dim = self.clip_model.visual.output_dim
        self.backend = 'open_clip'
    
    def _init_transformers_clip(self):
        """Initialize using HuggingFace transformers"""
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embed_dim = 512  # ViT-B/32
        self.backend = 'transformers'
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP vision encoder.
        
        Args:
            images: (B, C, H, W) or (B, T, C, H, W) tensor
            
        Returns:
            features: (B, embed_dim) or (B, T, embed_dim) tensor
        """
        # Handle temporal dimension
        has_temporal = images.dim() == 5
        if has_temporal:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        
        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images.float() / 255.0
        
        # CLIP expects specific normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Get features
        if self.backend == 'open_clip':
            features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.get_image_features(pixel_values=images)
        
        # Normalize features (CLIP convention)
        features = F.normalize(features, dim=-1)
        
        # Reshape if temporal
        if has_temporal:
            features = features.view(B, T, -1)
        
        return features
    
    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for action prediction.
        
        Args:
            images: (B, T, C, H, W) video frames
            return_features: Whether to return intermediate features
            
        Returns:
            dict with 'logits', 'features', 'pooled_features'
        """
        # Get frame features
        features = self.encode_image(images)  # (B, T, embed_dim)
        
        # Temporal pooling
        if self.use_temporal and features.dim() == 3:
            # Mean pooling over time + learned transformation
            pooled = features.mean(dim=1)  # (B, embed_dim)
            pooled = self.temporal_pool(pooled)
        else:
            pooled = features.mean(dim=1) if features.dim() == 3 else features
        
        # Action prediction
        logits = self.action_head(pooled)  # (B, num_actions)
        
        output = {
            'logits': logits,
            'pooled_features': pooled,
        }
        
        if return_features:
            output['features'] = features
        
        return output
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CLIPTeacherForDistillation(CLIPTeacher):
    """
    CLIP teacher specifically designed for knowledge distillation.
    
    Provides:
    - Feature embeddings for feature-based distillation
    - Soft labels (logits with temperature) for response-based distillation
    - Attention maps for attention-based distillation (optional)
    """
    
    def __init__(self, temperature: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def get_soft_labels(self, images: torch.Tensor) -> torch.Tensor:
        """Get soft probability distribution for distillation"""
        with torch.no_grad():
            output = self.forward(images)
            soft_labels = F.softmax(output['logits'] / self.temperature, dim=-1)
        return soft_labels
    
    def get_distillation_targets(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all targets needed for distillation.
        
        Returns:
            dict with:
            - 'features': CLIP visual embeddings
            - 'soft_labels': Soft probability distribution
            - 'hard_labels': Argmax predictions
            - 'logits': Raw logits
        """
        with torch.no_grad():
            output = self.forward(images, return_features=True)
            
            return {
                'features': output['pooled_features'],
                'frame_features': output.get('features'),
                'soft_labels': F.softmax(output['logits'] / self.temperature, dim=-1),
                'hard_labels': output['logits'].argmax(dim=-1),
                'logits': output['logits'],
            }


def create_clip_teacher(
    model_size: str = "base",
    num_actions: int = 10,
    for_distillation: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CLIP teacher.
    
    Args:
        model_size: "base" (ViT-B/32), "large" (ViT-L/14), or "huge" (ViT-H/14)
        num_actions: Number of action classes
        for_distillation: Whether to use distillation-specific wrapper
    """
    model_configs = {
        "base": ("ViT-B-32", "openai"),
        "large": ("ViT-L-14", "openai"),
        "huge": ("ViT-H-14", "laion2b_s32b_b79k"),
    }
    
    model_name, pretrained = model_configs.get(model_size, model_configs["base"])
    
    TeacherClass = CLIPTeacherForDistillation if for_distillation else CLIPTeacher
    
    return TeacherClass(
        model_name=model_name,
        pretrained=pretrained,
        num_actions=num_actions,
        **kwargs
    )


if __name__ == "__main__":
    print("="*60)
    print("CLIP Teacher Model - Testing")
    print("="*60)
    
    # Check backend
    print(f"\nCLIP Backend: {CLIP_BACKEND}")
    
    if CLIP_BACKEND is None:
        print("\nNo CLIP backend found!")
        print("Install with: pip install open-clip-torch")
        print("Or: pip install transformers")
        exit(1)
    
    # Create teacher
    print("\nCreating CLIP teacher...")
    teacher = create_clip_teacher(
        model_size="base",
        num_actions=10,
        for_distillation=True,
        freeze_clip=True,
    )
    
    print(f"\nTotal parameters: {teacher.get_num_parameters():,}")
    print(f"Trainable parameters: {teacher.get_trainable_parameters():,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher = teacher.to(device)
    
    # Batch of video clips: (B, T, C, H, W)
    dummy_input = torch.randn(2, 8, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = teacher(dummy_input, return_features=True)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Features shape: {output['pooled_features'].shape}")
    
    # Test distillation targets
    print("\nTesting distillation targets...")
    targets = teacher.get_distillation_targets(dummy_input)
    
    print(f"  Features: {targets['features'].shape}")
    print(f"  Soft labels: {targets['soft_labels'].shape}")
    print(f"  Hard labels: {targets['hard_labels'].shape}")
    
    print("\n" + "="*60)
    print("✓ CLIP Teacher ready!")
    print("="*60)

