"""
Loss Functions for Multi-Task Learning
=======================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Implements:
- ImprovedRegressionLoss: Smooth-L1 + Pearson correlation alignment
- AdaptiveMultiTaskLoss: Uncertainty-based task weighting (Kendall et al.)
- FocalLoss: Class-imbalance robust classification (Lin et al., ICCV 2017)
- LabelSmoothingCrossEntropy: Label smoothing regularization
- CombinedClassificationLoss: Focal + Label Smoothing hybrid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedRegressionLoss(nn.Module):
    """
    Improved regression loss combining MSE and correlation
    Better gradients for protein expression prediction
    """
    def __init__(self, mse_weight=0.7, corr_weight=0.3, huber_delta=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.huber_delta = huber_delta
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, num_proteins) - Predicted protein expression
            target: (B, num_proteins) - Ground truth protein expression
        
        Returns:
            loss: Combined MSE + Correlation loss
        """
        # Huber loss (robust to outliers)
        huber_loss = F.smooth_l1_loss(pred, target, beta=self.huber_delta)
        
        # Correlation loss (maximize correlation)
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute correlation (Pearson correlation)
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        denominator = pred_std * target_std + 1e-8
        
        correlation = numerator / denominator
        # Clamp correlation to [-1, 1] for numerical stability
        correlation = torch.clamp(correlation, -1.0, 1.0)
        corr_loss = 1 - correlation  # Minimize (1 - correlation) = maximize correlation
        
        # Combined loss
        total_loss = self.mse_weight * huber_loss + self.corr_weight * corr_loss
        
        return total_loss


class AdaptiveMultiTaskLoss(nn.Module):
    """
    Adaptive multi-task loss with uncertainty-based weighting
    Automatically balances classification and regression tasks
    """
    def __init__(
        self,
        cls_loss_fn,
        reg_loss_fn,
        initial_cls_weight=1.0,
        initial_reg_weight=0.5,
        learn_weights=True
    ):
        super().__init__()
        self.cls_loss_fn = cls_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.learn_weights = learn_weights
        
        if learn_weights:
            # Learnable task weights (log variance for numerical stability)
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))
        else:
            self.cls_weight = initial_cls_weight
            self.reg_weight = initial_reg_weight
    
    def forward(self, cls_logits, cls_targets, reg_preds, reg_targets):
        """
        Args:
            cls_logits: (B, num_classes) - Classification logits
            cls_targets: (B,) - Classification targets
            reg_preds: (B, num_proteins) - Regression predictions
            reg_targets: (B, num_proteins) - Regression targets
        
        Returns:
            loss_dict: Dictionary with individual and total losses
        """
        # Compute individual losses
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
        reg_loss = self.reg_loss_fn(reg_preds, reg_targets)
        
        # Compute task weights
        if self.learn_weights:
            # Uncertainty-based weighting (homoscedastic uncertainty)
            cls_weight = 1.0 / (2 * torch.exp(self.log_var_cls)) + 1e-8
            reg_weight = 1.0 / (2 * torch.exp(self.log_var_reg)) + 1e-8
            
            # Normalize weights
            total_weight = cls_weight + reg_weight
            cls_weight = cls_weight / total_weight
            reg_weight = reg_weight / total_weight
            
            # Add regularization term
            total_loss = cls_weight * cls_loss + reg_weight * reg_loss + \
                        (self.log_var_cls + self.log_var_reg) * 0.5
        else:
            # Fixed weights
            total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'regression': reg_loss,
            'cls_weight': cls_weight.item() if self.learn_weights else self.cls_weight,
            'reg_weight': reg_weight.item() if self.learn_weights else self.reg_weight
        }


class FocalLoss(nn.Module):
    """Focal Loss for classification"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CombinedClassificationLoss(nn.Module):
    """Combined Focal + Label Smoothing"""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, focal_weight=0.7):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ls_loss = LabelSmoothingCrossEntropy(smoothing)
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        ls = self.ls_loss(pred, target)
        return self.focal_weight * focal + (1 - self.focal_weight) * ls

