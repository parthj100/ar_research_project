"""
MobileViT Student Model for Phase 3

MobileViT combines the strengths of CNNs and Vision Transformers
in a lightweight architecture suitable for mobile/edge deployment.

Model sizes:
- MobileViT-XXS: ~1.3M params (our default)
- MobileViT-XS: ~2.3M params
- MobileViT-S: ~5.6M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import warnings

# Try to import MobileViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not found. Install with: pip install timm")


class MobileViTStudent(nn.Module):
    """
    MobileViT-based student model for egocentric action recognition.
    
    Architecture:
    - MobileViT backbone (XXS by default): ~1.3M params
    - Temporal aggregation module
    - Action classification head
    
    Designed to match CLIP teacher's output for distillation.
    """
    
    def __init__(
        self,
        model_name: str = "mobilevit_xxs",
        num_actions: int = 10,
        pretrained: bool = True,
        teacher_embed_dim: int = 512,
        use_temporal: bool = True,
        dropout: float = 0.2,
    ):
        """
        Args:
            model_name: MobileViT variant (mobilevit_xxs, mobilevit_xs, mobilevit_s)
            num_actions: Number of action classes
            pretrained: Whether to use ImageNet pretrained weights
            teacher_embed_dim: Dimension of teacher's feature space (for alignment)
            use_temporal: Whether to use temporal aggregation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_actions = num_actions
        self.teacher_embed_dim = teacher_embed_dim
        self.use_temporal = use_temporal
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for MobileViT. Install with: pip install timm")
        
        # Load MobileViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Feature projection to match teacher's embedding dimension
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, teacher_embed_dim),
            nn.LayerNorm(teacher_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Temporal aggregation for video input
        if use_temporal:
            self.temporal_attention = TemporalAttentionPool(
                embed_dim=teacher_embed_dim,
                num_heads=4,
            )
        
        # Action classification head
        self.action_head = nn.Sequential(
            nn.Linear(teacher_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions),
        )
        
        self._print_info()
    
    def _print_info(self):
        print(f"MobileViT Student initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Backbone features: {self.feature_dim}")
        print(f"  Projected features: {self.teacher_embed_dim}")
        print(f"  Actions: {self.num_actions}")
        print(f"  Total params: {self.get_num_parameters():,}")
    
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using MobileViT backbone.
        
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
        
        # ImageNet normalization (same as timm default)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Get backbone features
        features = self.backbone(images)  # (B*T, feature_dim)
        
        # Project to teacher's embedding space
        features = self.feature_proj(features)  # (B*T, teacher_embed_dim)
        
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
        features = self.encode_frames(images)  # (B, T, embed_dim)
        
        # Temporal aggregation
        if self.use_temporal and features.dim() == 3:
            pooled = self.temporal_attention(features)  # (B, embed_dim)
        else:
            pooled = features.mean(dim=1) if features.dim() == 3 else features
        
        # Normalize features (to match CLIP's normalized output)
        pooled_normalized = F.normalize(pooled, dim=-1)
        
        # Action prediction
        logits = self.action_head(pooled)  # (B, num_actions)
        
        output = {
            'logits': logits,
            'pooled_features': pooled_normalized,
        }
        
        if return_features:
            output['features'] = features
            output['pooled_features_unnorm'] = pooled
        
        return output
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_backbone_parameters(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())


class TemporalAttentionPool(nn.Module):
    """
    Learnable temporal pooling using self-attention.
    Aggregates frame features into a single video representation.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Learnable query for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) frame features
            
        Returns:
            pooled: (B, embed_dim) aggregated features
        """
        B, T, D = x.shape
        
        # Expand CLS token for batch
        cls_token = self.cls_token.expand(B, -1, -1)
        
        # Concatenate CLS token with frame features
        x = torch.cat([cls_token, x], dim=1)  # (B, T+1, D)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Take CLS token output
        pooled = attn_out[:, 0]  # (B, D)
        
        return self.norm(pooled)


class MobileViTStudentTiny(nn.Module):
    """
    Even smaller MobileViT variant for extreme compression.
    Custom architecture based on MobileViT design principles.
    ~500K parameters.
    """
    
    def __init__(
        self,
        num_actions: int = 10,
        teacher_embed_dim: int = 512,
        base_channels: int = 16,
    ):
        super().__init__()
        
        self.teacher_embed_dim = teacher_embed_dim
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        
        # Stages with progressive downsampling
        self.stage1 = self._make_mv_block(base_channels, base_channels * 2, stride=2)
        self.stage2 = self._make_mv_block(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = self._make_mv_block(base_channels * 4, base_channels * 8, stride=2)
        self.stage4 = self._make_mv_block(base_channels * 8, base_channels * 16, stride=2)
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        final_channels = base_channels * 16
        self.feature_proj = nn.Sequential(
            nn.Linear(final_channels, teacher_embed_dim),
            nn.LayerNorm(teacher_embed_dim),
            nn.GELU(),
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(teacher_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions),
        )
        
        print(f"MobileViT-Tiny Student: {self.get_num_parameters():,} params")
    
    def _make_mv_block(self, in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
        """Mobile inverted bottleneck block"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        # Handle temporal
        has_temporal = images.dim() == 5
        if has_temporal:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        
        # Normalize
        if images.max() > 1.0:
            images = images.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Forward through stages
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Pool and project
        x = self.pool(x).flatten(1)
        features = self.feature_proj(x)
        
        # Temporal mean pooling
        if has_temporal:
            features = features.view(B, T, -1).mean(dim=1)
        
        pooled = F.normalize(features, dim=-1)
        logits = self.action_head(features)
        
        output = {'logits': logits, 'pooled_features': pooled}
        if return_features:
            output['features'] = features
        
        return output
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_mobilevit_student(
    model_size: str = "xxs",
    num_actions: int = 10,
    teacher_embed_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to create MobileViT student.
    
    Args:
        model_size: "tiny" (~500K), "xxs" (~1.3M), "xs" (~2.3M), "s" (~5.6M)
        num_actions: Number of action classes
        teacher_embed_dim: CLIP teacher's embedding dimension
    """
    if model_size == "tiny":
        return MobileViTStudentTiny(
            num_actions=num_actions,
            teacher_embed_dim=teacher_embed_dim,
        )
    
    model_configs = {
        "xxs": "mobilevit_xxs",
        "xs": "mobilevit_xs", 
        "s": "mobilevit_s",
    }
    
    model_name = model_configs.get(model_size, model_configs["xxs"])
    
    return MobileViTStudent(
        model_name=model_name,
        num_actions=num_actions,
        teacher_embed_dim=teacher_embed_dim,
        **kwargs
    )


if __name__ == "__main__":
    print("="*60)
    print("MobileViT Student Model - Testing")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    
    # Test different sizes
    for size in ["tiny", "xxs"]:
        print(f"\n{'-'*40}")
        print(f"Testing MobileViT-{size.upper()}")
        print(f"{'-'*40}")
        
        try:
            student = create_mobilevit_student(
                model_size=size,
                num_actions=10,
                teacher_embed_dim=512,
            ).to(device)
            
            # Test with video input: (B, T, C, H, W)
            dummy_input = torch.randn(2, 8, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = student(dummy_input, return_features=True)
            
            print(f"  Input: {dummy_input.shape}")
            print(f"  Logits: {output['logits'].shape}")
            print(f"  Features: {output['pooled_features'].shape}")
            print(f"  Parameters: {student.get_num_parameters():,}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Compare with CLIP teacher size
    print(f"\n{'='*60}")
    print("Compression Ratios (vs CLIP ViT-B/32 ~86M params):")
    print(f"  MobileViT-Tiny: ~172x compression")
    print(f"  MobileViT-XXS:  ~66x compression")
    print(f"  MobileViT-XS:   ~37x compression")
    print(f"  MobileViT-S:    ~15x compression")
    print("="*60)

