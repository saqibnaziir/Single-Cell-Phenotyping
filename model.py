"""
Hybrid Multi-Task Model for Label-Free Single-Cell Phenotyping
==============================================================
Architecture for the paper:
  "Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
  Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
  ICPR 2026 | arXiv:2605.14717

The model jointly performs:
  - WBC classification (3 classes: Lymphocyte, Granulocyte, Monocyte)
  - Protein-expression regression (4 markers: CD45, CD3, CD19, CD14)

Architecture overview:
  Input (B, 4, 28, 28) — 4-channel DPC images
    └─ Shared ECA channel attention
        ├─ CNN Branch  (2 Inception modules + residual, feature_dim=192)
        └─ ViT Branch  (patch_size=4, depth=2, embed_dim=128)
              └─ Cross-modal fusion  (256-dim)
                    ├─ Task-specific refinement
                    └─ Task gating (learnable cross-task information exchange)
                          ├─ Classification head → (B, num_classes)
                          └─ Regression head    → (B, num_proteins)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== CHANNEL ATTENTION (ECA) ==================== #

class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) for multi-channel DPC
    Adaptively weights different DPC orientations
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use 1D conv for efficient channel attention
        self.conv = nn.Conv1d(
            1, 1, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Multi-channel input
        Returns:
            x * attention_weights: (B, C, H, W) - Weighted features
        """
        # Global average pooling
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        # 1D convolution for channel attention
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Sigmoid activation
        y = self.sigmoid(y)
        
        # Apply attention weights
        return x * y


# ==================== SIMPLIFIED INCEPTION MODULE (with Residual) ==================== #

class InceptionModule(nn.Module):
    """Simplified Inception module with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Calculate branch sizes (ensure they sum to out_channels)
        branch1_size = out_channels // 3
        branch2_size = out_channels // 3
        branch3_size = out_channels - branch1_size - branch2_size  # Remainder
        
        # Simplified: only 3 branches instead of 4
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1_size, 1, bias=False),
            nn.BatchNorm2d(branch1_size),
            nn.GELU()
        )
        
        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch2_size),
            nn.GELU()
        )
        
        # 5x5 conv branch (two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch3_size),
            nn.GELU(),
            nn.Conv2d(branch3_size, branch3_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch3_size),
            nn.GELU()
        )
        
        # Residual connection (if channel dimensions match)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        # Residual connection
        return out + self.shortcut(x)


# ==================== SIMPLIFIED CNN BACKBONE ==================== #

class CNNBackbone(nn.Module):
    """
    Simplified CNN backbone optimized for 28×28 images
    - Fewer Inception modules (2 instead of 3)
    - Residual connections
    - BatchNorm for better image feature learning
    """
    def __init__(self, in_channels=4):
        super().__init__()
        
        # Stem: 28×28 → 28×28
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        # Inception Block 1: 28×28 → 28×28
        self.inception1 = InceptionModule(64, 128)
        
        # Transition: 28×28 → 14×14
        self.transition = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        # Inception Block 2: 14×14 → 14×14
        self.inception2 = InceptionModule(128, 192)
        
        self.feature_dim = 192
    
    def forward(self, x):
        # Stem: 28×28
        x = self.stem(x)
        
        # Inception 1: 28×28
        x = self.inception1(x)
        
        # Transition: 28×28 → 14×14
        x = self.transition(x)
        
        # Inception 2: 14×14
        x = self.inception2(x)
        
        # Output: (B, 192, 14, 14)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, 196, 192)
        
        return x, x_flat


# ==================== LIGHTWEIGHT VIT BACKBONE ==================== #

class ViTBackbone(nn.Module):
    """
    Lightweight ViT optimized for 28×28 images
    - Larger patch size (patch_size=4 → 49 patches instead of 196)
    - Shallower depth (depth=2 instead of 4)
    - Smaller embed dimension (128 instead of 192)
    - NO channel attention (applied before branching)
    """
    def __init__(self, img_size=28, in_channels=4, patch_size=4,
                 embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 49 patches (7×7)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        # Transformer blocks (shallow)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 128, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (B, 49, 128)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 50, 128)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks (shallow)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x  # (B, 50, 128)


class TransformerBlock(nn.Module):
    """Standard Transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== SIMPLIFIED CROSS-MODAL FUSION ==================== #

class SimpleCrossModalFusion(nn.Module):
    """
    Simplified cross-modal fusion
    - Lightweight attention instead of full cross-attention
    - Direct feature concatenation
    """
    def __init__(self, cnn_dim, vit_dim, fusion_dim=256):
        super().__init__()
        
        # Project to common dimension
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        
        # Learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # Normalization
        self.norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, cnn_features, vit_features):
        """
        Args:
            cnn_features: (B, 196, 192) - CNN spatial features
            vit_features: (B, 50, 128) - ViT patch features (CLS + patches)
        Returns:
            fused: (B, fusion_dim) - Fused global features
        """
        # Global pooling
        cnn_global = cnn_features.mean(dim=1)  # (B, 192)
        vit_global = vit_features[:, 0]  # CLS token (B, 128)
        
        # Project to common dimension
        cnn_proj = self.cnn_proj(cnn_global)  # (B, 256)
        vit_proj = self.vit_proj(vit_global)  # (B, 256)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * cnn_proj + weights[1] * vit_proj
        
        # Normalize
        fused = self.norm(fused)
        
        return fused  # (B, 256)


# ==================== EFFICIENT TASK GATING ==================== #

class TaskGating(nn.Module):
    """
    Efficient task gating mechanism
    Replaces heavy cross-task attention with lightweight gating
    Much more efficient for single-token features
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # Gating networks
        self.cls_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.reg_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Feature mixing
        self.cls_mix = nn.Linear(feature_dim * 2, feature_dim)
        self.reg_mix = nn.Linear(feature_dim * 2, feature_dim)
        
        # Normalization (LayerNorm works better for 2D features)
        self.cls_norm = nn.LayerNorm(feature_dim)
        self.reg_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, cls_features, reg_features):
        """
        Args:
            cls_features: (B, feature_dim) - Classification features
            reg_features: (B, feature_dim) - Regression features
        Returns:
            cls_enhanced: (B, feature_dim) - Enhanced classification features
            reg_enhanced: (B, feature_dim) - Enhanced regression features
        """
        # Combine features
        combined = torch.cat([cls_features, reg_features], dim=1)  # (B, 2*D)
        
        # Generate gates
        cls_gate = self.cls_gate(combined)  # (B, D)
        reg_gate = self.reg_gate(combined)  # (B, D)
        
        # Mix features
        cls_mixed = self.cls_mix(combined)  # (B, D)
        reg_mixed = self.reg_mix(combined)  # (B, D)
        
        # Apply gates (residual connection)
        cls_enhanced = cls_features * cls_gate + cls_mixed * (1 - cls_gate)
        reg_enhanced = reg_features * reg_gate + reg_mixed * (1 - reg_gate)
        
        # Normalize (LayerNorm works on last dimension)
        cls_enhanced = self.cls_norm(cls_enhanced)
        reg_enhanced = self.reg_norm(reg_enhanced)
        
        return cls_enhanced, reg_enhanced


# ==================== TASK-SPECIFIC REFINEMENT ==================== #

class TaskSpecificRefinement(nn.Module):
    """
    Task-Specific Feature Refinement with residual connection
    Uses LayerNorm for both tasks (works better for 2D features)
    """
    def __init__(self, feature_dim, task='classification'):
        super().__init__()
        self.task = task
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, x):
        # Residual connection for better gradient flow
        return x + self.refinement(x)


# ==================== MULTI-TASK HEADS ==================== #

class ClassificationHead(nn.Module):
    """Classification head for cell type prediction"""
    def __init__(self, in_features, num_classes=3, dropout_rate=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class RegressionHead(nn.Module):
    """Regression head for protein expression prediction"""
    def __init__(self, in_features, num_proteins=4, dropout_rate=0.4):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_proteins)
        )
    
    def forward(self, x):
        return self.regressor(x)


# ==================== OPTIMIZED HYBRID MULTI-TASK MODEL ==================== #

class OptimizedHybridMultiTaskModel(nn.Module):
    """
    Optimized Hybrid Multi-Task Model for 28×28 images
    
    Key Improvements:
    1. Single shared ECA before branching (removes redundancy)
    2. Larger ViT patch size (patch_size=4 → 49 patches)
    3. Lighter architecture (2 Inception modules, depth=2 ViT)
    4. Efficient task gating (replaces heavy cross-task attention)
    5. Residual connections in CNN
    6. Better normalization (BatchNorm for CNN, LayerNorm for ViT)
    
    Architecture:
    Input (4 channels) 
      ↓
    Shared Channel Attention (ECA) ← SINGLE ECA BEFORE BRANCHING
      ↓
    ┌─────────────────┴─────────────────┐
    │                                    │
    CNN Branch                      ViT Branch (Light)
    (2 Inception + Res)             (patch_size=4, depth=2)
    ↓                                    ↓
    (B, 192, 14, 14)                  (B, 50, 128)
    │                                    │
    └─────────────────┬─────────────────┘
                      ↓
              Simple Fusion
                      ↓
              Shared Features (B, 256)
                      ↓
              Task-Specific Refinement
                      ↓
              Efficient Gating (not attention)
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
    Classification Head    Regression Head
    """
    def __init__(
        self,
        num_classes=3,
        num_proteins=4,
        img_size=28,
        in_channels=4,
        dropout_rate=0.4,
        use_task_gating=True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_proteins = num_proteins
        self.use_task_gating = use_task_gating
        
        # SINGLE shared channel attention (applied before branching)
        self.shared_channel_attention = EfficientChannelAttention(in_channels, kernel_size=3)
        
        # Dual backbones (NO internal channel attention)
        self.cnn = CNNBackbone(in_channels)
        self.vit = ViTBackbone(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=4,  # Larger patch size (7×7 = 49 patches)
            embed_dim=128,  # Smaller embed dim
            depth=2,  # Shallower (2 layers instead of 4)
            num_heads=4
        )
        
        # Simplified cross-modal fusion
        self.fusion = SimpleCrossModalFusion(
            cnn_dim=self.cnn.feature_dim,
            vit_dim=self.vit.embed_dim,
            fusion_dim=256
        )
        
        # Task-specific refinement
        self.cls_refinement = TaskSpecificRefinement(256, task='classification')
        self.reg_refinement = TaskSpecificRefinement(256, task='regression')
        
        # Efficient task gating (optional)
        if use_task_gating:
            self.task_gating = TaskGating(256)
        
        # Multi-task heads
        self.classifier = ClassificationHead(256, num_classes, dropout_rate)
        self.regressor = RegressionHead(256, num_proteins, dropout_rate)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 4, 28, 28) - Multi-channel DPC input
            return_features: If True, return intermediate features
        
        Returns:
            cls_logits: (B, num_classes) - Classification logits
            prot_preds: (B, num_proteins) - Protein predictions
            features_dict (optional): Dict with intermediate features
        """
        # Apply SINGLE shared channel attention
        x = self.shared_channel_attention(x)
        
        # Extract features (NO internal channel attention)
        cnn_feat, cnn_flat = self.cnn(x)  # (B, 192, 14, 14), (B, 196, 192)
        vit_feat = self.vit(x)            # (B, 50, 128)
        
        # Cross-modal fusion
        shared_features = self.fusion(cnn_flat, vit_feat)  # (B, 256)
        
        # Task-specific refinement
        cls_features = self.cls_refinement(shared_features)  # (B, 256)
        reg_features = self.reg_refinement(shared_features)  # (B, 256)
        
        # Efficient task gating (information sharing)
        if self.use_task_gating:
            cls_features, reg_features = self.task_gating(cls_features, reg_features)
        
        # Final predictions
        cls_logits = self.classifier(cls_features)
        prot_preds = self.regressor(reg_features)
        
        if return_features:
            features_dict = {
                'cnn_features': cnn_feat,
                'vit_features': vit_feat,
                'shared_features': shared_features,
                'cls_features': cls_features,
                'reg_features': reg_features
            }
            return cls_logits, prot_preds, features_dict
        
        return cls_logits, prot_preds


# ==================== MODEL FACTORY ==================== #

def create_model(num_classes=3, num_proteins=4, img_size=28, in_channels=4, 
                 use_task_gating=True):
    """
    Factory function to create the optimized hybrid multi-task model
    
    Args:
        num_classes: Number of cell type classes
        num_proteins: Number of protein markers
        img_size: Image size (assumed square)
        in_channels: Input channels (4 for multi-channel DPC)
        use_task_gating: Whether to use efficient task gating
    
    Returns:
        model: OptimizedHybridMultiTaskModel instance
    """
    model = OptimizedHybridMultiTaskModel(
        num_classes=num_classes,
        num_proteins=num_proteins,
        img_size=img_size,
        in_channels=in_channels,
        use_task_gating=use_task_gating
    )
    return model


# ==================== TEST ==================== #

if __name__ == "__main__":
    print("="*70)
    print("TESTING OPTIMIZED HYBRID MULTI-TASK MODEL")
    print("="*70)
    
    # Create model
    model = create_model(num_classes=3, num_proteins=4, img_size=28, in_channels=4)
    
    # Test input (4-channel DPC)
    batch_size = 4
    x = torch.randn(batch_size, 4, 28, 28)
    
    # Forward pass
    print("\n1. Forward pass:")
    cls_logits, prot_preds = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Classification logits: {cls_logits.shape}")
    print(f"   Protein predictions: {prot_preds.shape}")
    
    cls_probs = F.softmax(cls_logits, dim=1)
    print(f"\n   Sample predictions:")
    print(f"   Cell type probs: {cls_probs[0]}")
    print(f"   Protein expression: {prot_preds[0]}")
    
    # Model statistics
    print("\n2. Model statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Check forward pass with features
    print("\n3. Forward pass with features:")
    cls_logits, prot_preds, features_dict = model(x, return_features=True)
    print(f"   CNN features: {features_dict['cnn_features'].shape}")
    print(f"   ViT features: {features_dict['vit_features'].shape}")
    print(f"   Shared features: {features_dict['shared_features'].shape}")
    print(f"   Classification features: {features_dict['cls_features'].shape}")
    print(f"   Regression features: {features_dict['reg_features'].shape}")
    
    # Compare with old model (if available)
    print("\n4. Comparison with previous model:")
    print("   ✅ Single shared ECA (removed redundancy)")
    print("   ✅ Larger patch size (4 → 49 patches instead of 196)")
    print("   ✅ Lighter ViT (depth=2, embed_dim=128)")
    print("   ✅ Fewer Inception modules (2 instead of 3)")
    print("   ✅ Efficient task gating (replaces heavy attention)")
    print("   ✅ Residual connections in CNN")
    print("   ✅ Better normalization (BatchNorm for CNN)")
    
    print("\n" + "="*70)
    print("✅ OPTIMIZED MODEL TEST COMPLETE")
    print("="*70)
