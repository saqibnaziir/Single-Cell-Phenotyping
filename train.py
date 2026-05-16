"""
Training Script — Label-Free Single-Cell Phenotyping
=====================================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Usage:
    python train.py --data_path /path/to/BSCCMNIST --save_dir checkpoints/run1

Optional Weights & Biases logging:
    python train.py --data_path /path/to/BSCCMNIST --use_wandb
"""

import os
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, cohen_kappa_score
)

from model import create_model
from losses import (
    ImprovedRegressionLoss,
    AdaptiveMultiTaskLoss,
    CombinedClassificationLoss
)
from data_loading import (
    load_bsccm_dataset,
    create_train_val_test_splits,
    create_data_loaders,
    PROTEIN_MARKERS
)


# ==================== DATA AUGMENTATION ==================== #

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# ==================== TRAINER ==================== #

class ImprovedMultiTaskTrainer:
    """Improved training pipeline with better strategies"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        save_dir,
        use_mixup=True,
        mixup_alpha=0.5,
        mixup_prob=0.3,
        use_wandb=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.use_wandb = use_wandb
        
        # History
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_reg_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_reg_loss': [],
            'val_acc': [],
            'val_reg_mse': [],
            'val_correlation': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        cls_correct = 0
        cls_total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, labels, proteins) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            proteins = proteins.to(self.device)
            
            # Mixup augmentation (reduced probability)
            if self.use_mixup and random.random() < self.mixup_prob:
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                
                self.optimizer.zero_grad()
                cls_logits, prot_preds = self.model(images)
                
                # Classification loss (mixed)
                cls_loss_a = self.loss_fn.cls_loss_fn(cls_logits, labels_a)
                cls_loss_b = self.loss_fn.cls_loss_fn(cls_logits, labels_b)
                cls_loss = lam * cls_loss_a + (1 - lam) * cls_loss_b
                
                # Regression loss (no mixup for regression)
                reg_loss = self.loss_fn.reg_loss_fn(prot_preds, proteins)
                
                # Total loss
                total_loss = self.loss_fn.cls_weight * cls_loss + self.loss_fn.reg_weight * reg_loss
            else:
                # Standard training
                self.optimizer.zero_grad()
                cls_logits, prot_preds = self.model(images)
                
                loss_dict = self.loss_fn(cls_logits, labels, prot_preds, proteins)
                total_loss = loss_dict['total']
                cls_loss = loss_dict['classification']
                reg_loss = loss_dict['regression']
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step scheduler (for OneCycleLR, step per batch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            running_total_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()
            
            _, predicted = cls_logits.max(1)
            cls_total += labels.size(0)
            cls_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*cls_correct/cls_total:.2f}%'
            })
        
        # Return metrics
        epoch_total_loss = running_total_loss / len(self.train_loader)
        epoch_cls_loss = running_cls_loss / len(self.train_loader)
        epoch_reg_loss = running_reg_loss / len(self.train_loader)
        epoch_acc = 100. * cls_correct / cls_total
        
        return epoch_total_loss, epoch_cls_loss, epoch_reg_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        running_reg_mse = 0.0
        
        all_cls_preds = []
        all_cls_labels = []
        all_prot_preds = []
        all_prot_labels = []
        all_cls_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for images, labels, proteins in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                proteins = proteins.to(self.device)
                
                cls_logits, prot_preds = self.model(images)
                
                # Loss
                loss_dict = self.loss_fn(cls_logits, labels, prot_preds, proteins)
                running_total_loss += loss_dict['total'].item()
                running_cls_loss += loss_dict['classification'].item()
                running_reg_loss += loss_dict['regression'].item()
                
                # Regression MSE
                reg_mse = F.mse_loss(prot_preds, proteins)
                running_reg_mse += reg_mse.item()
                
                # Predictions
                cls_probs = F.softmax(cls_logits, dim=1)
                _, predicted = cls_logits.max(1)
                
                all_cls_preds.extend(predicted.cpu().numpy())
                all_cls_labels.extend(labels.cpu().numpy())
                all_cls_probs.extend(cls_probs.cpu().numpy())
                all_prot_preds.extend(prot_preds.cpu().numpy())
                all_prot_labels.extend(proteins.cpu().numpy())
        
        # Calculate metrics
        val_metrics = {
            'total_loss': running_total_loss / len(self.val_loader),
            'cls_loss': running_cls_loss / len(self.val_loader),
            'reg_loss': running_reg_loss / len(self.val_loader),
            'reg_mse': running_reg_mse / len(self.val_loader),
            'accuracy': accuracy_score(all_cls_labels, all_cls_preds) * 100
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_cls_labels, all_cls_preds, average=None, zero_division=0
        )
        val_metrics['precision'] = precision
        val_metrics['recall'] = recall
        val_metrics['f1'] = f1
        val_metrics['support'] = support
        
        # Macro-averaged
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_cls_labels, all_cls_preds, average='macro', zero_division=0
        )
        val_metrics['macro_precision'] = macro_precision
        val_metrics['macro_recall'] = macro_recall
        val_metrics['macro_f1'] = macro_f1
        
        # Kappa
        val_metrics['kappa'] = cohen_kappa_score(all_cls_labels, all_cls_preds)
        
        # Confusion matrix
        val_metrics['confusion_matrix'] = confusion_matrix(all_cls_labels, all_cls_preds)
        
        # Regression correlation
        all_prot_preds = np.array(all_prot_preds)
        all_prot_labels = np.array(all_prot_labels)
        prot_corr = np.corrcoef(all_prot_preds.flatten(), all_prot_labels.flatten())[0, 1]
        val_metrics['prot_correlation'] = prot_corr
        
        return val_metrics
    
    def train(self, num_epochs, patience=30):
        """Complete training loop with early stopping"""
        print("="*80)
        print("STARTING IMPROVED MULTI-TASK TRAINING")
        print("="*80)
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_cls_loss, train_reg_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_reg_loss'].append(train_reg_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_cls_loss'].append(val_metrics['cls_loss'])
            self.history['val_reg_loss'].append(val_metrics['reg_loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_reg_mse'].append(val_metrics['reg_mse'])
            self.history['val_correlation'].append(val_metrics['prot_correlation'])
            
            # Learning rate scheduling (only if not OneCycleLR - it steps per batch)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
            print(f"  Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f} (Cls: {val_metrics['cls_loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f})")
            print(f"  Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Val Reg MSE: {val_metrics['reg_mse']:.4f}, Correlation: {val_metrics['prot_correlation']:.4f}")
            print(f"  Macro F1: {val_metrics['macro_f1']:.4f} | Kappa: {val_metrics['kappa']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Log task weights if using adaptive loss
            if hasattr(self.loss_fn, 'learn_weights') and self.loss_fn.learn_weights:
                # Get weights from loss function
                if hasattr(self.loss_fn, 'log_var_cls'):
                    cls_w = 1.0 / (2 * torch.exp(self.loss_fn.log_var_cls)) + 1e-8
                    reg_w = 1.0 / (2 * torch.exp(self.loss_fn.log_var_reg)) + 1e-8
                    total_w = cls_w + reg_w
                    print(f"  Task Weights - Cls: {(cls_w/total_w).item():.3f}, Reg: {(reg_w/total_w).item():.3f}")
            
            # Log to wandb
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/cls_loss': train_cls_loss,
                'train/reg_loss': train_reg_loss,
                'train/acc': train_acc,
                'val/loss': val_metrics['total_loss'],
                'val/cls_loss': val_metrics['cls_loss'],
                'val/reg_loss': val_metrics['reg_loss'],
                'val/acc': val_metrics['accuracy'],
                'val/reg_mse': val_metrics['reg_mse'],
                'val/correlation': val_metrics['prot_correlation'],
                'val/macro_f1': val_metrics['macro_f1'],
                'val/kappa': val_metrics['kappa'],
                'lr': current_lr,
                'epoch/improvement_count': no_improve_count
            }
            
            # Add per-class metrics
            for i, (p, r, f) in enumerate(zip(val_metrics['precision'], val_metrics['recall'], val_metrics['f1'])):
                log_dict[f'val/class_{i}_precision'] = p
                log_dict[f'val/class_{i}_recall'] = r
                log_dict[f'val/class_{i}_f1'] = f
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log(log_dict)

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                no_improve_count = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, self.save_dir / 'best_model.pth')
                print(f"  ✅ New best model saved! (Acc: {self.best_val_acc:.2f}%)")
                
                # Log confusion matrix to wandb
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(val_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({'confusion_matrix': wandb.Image(fig)})
                plt.close(fig)
            else:
                no_improve_count += 1
                print(f"  No improvement for {no_improve_count} epochs")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⚠️  Early stopping triggered after {patience} epochs")
                break
            
            print("-"*80)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.best_val_acc
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Total', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Total', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification loss
        axes[0, 1].plot(self.history['train_cls_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_cls_loss'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Classification Loss', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Regression loss
        axes[0, 2].plot(self.history['train_reg_loss'], label='Train', linewidth=2)
        axes[0, 2].plot(self.history['val_reg_loss'], label='Val', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Regression Loss', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 0].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[1, 0].axhline(y=self.best_val_acc, color='r', linestyle='--',
                          label=f'Best: {self.best_val_acc:.2f}%')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Accuracy', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Regression MSE
        axes[1, 1].plot(self.history['val_reg_mse'], label='Val MSE', linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Regression MSE', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation
        axes[1, 2].plot(self.history['val_correlation'], label='Val Correlation', linewidth=2, color='green')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].set_title('Protein Expression Correlation', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')

        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({'training_curves': wandb.Image(fig)})

        plt.close()
        print(f"✅ Training curves saved to {self.save_dir / 'training_curves.png'}")


# ==================== UTILITIES ==================== #

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(labels):
    """Calculate class weights for balanced loss"""
    labels_int = labels.astype(np.int64) if isinstance(labels, np.ndarray) else np.array(labels, dtype=np.int64)
    class_counts = np.bincount(labels_int)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return torch.FloatTensor(class_weights)


# ==================== MAIN ==================== #

def main():
    parser = argparse.ArgumentParser(
        description='Train Hybrid Multi-Task Model for Single-Cell Phenotyping',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to BSCCMNIST dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model
    parser.add_argument('--num_classes', type=int, default=3, help='Number of cell type classes')
    parser.add_argument('--num_proteins', type=int, default=4, help='Number of protein markers')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--in_channels', type=int, default=4, help='Input channels (4 for multi-channel DPC)')
    parser.add_argument('--multi_channel', action='store_true', default=True, help='Use multi-channel DPC')
    parser.add_argument('--balance_classes', action='store_true', default=True, help='Balance classes by oversampling')
    parser.add_argument('--use_cross_task_attention', action='store_true', default=True, help='Use cross-task attention (deprecated, use --use_task_gating)')
    parser.add_argument('--use_task_gating', action='store_true', default=True, help='Use efficient task gating')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (reduced)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # Loss
    parser.add_argument('--cls_weight', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--reg_weight', type=float, default=0.5, help='Regression loss weight')
    parser.add_argument('--use_adaptive_loss', action='store_true', default=False, help='Use adaptive multi-task loss')
    parser.add_argument('--use_improved_regression', action='store_true', default=True, help='Use improved regression loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    
    # Augmentation
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.5, help='Mixup alpha (reduced)')
    parser.add_argument('--mixup_prob', type=float, default=0.3, help='Mixup probability (reduced)')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='checkpoints/multitask_improved', help='Save directory')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging (requires: pip install wandb)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("IMPROVED MULTI-TASK TRAINING CONFIGURATION")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key:25s}: {value}")
    print("="*80 + "\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb (optional)
    if args.use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project="single-cell-phenotyping",
                name=f"multitask-{Path(args.data_path).name}",
                config=vars(args)
            )
            print("✓ Wandb initialized successfully")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}. Continuing without logging.")
            args.use_wandb = False
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb not installed. Run: pip install wandb. Continuing without logging.")
        args.use_wandb = False
    
    # Load dataset
    print("\nLoading dataset...")
    bsccm, indices, class_labels, sm_df, protein_markers = load_bsccm_dataset(
        args.data_path, 
        protein_markers=PROTEIN_MARKERS
    )
    
    # Create splits (with class balancing)
    splits = create_train_val_test_splits(
        indices, class_labels, 
        random_state=args.seed,
        balance_train=args.balance_classes
    )
    
    # Create dataloaders (with multi-channel DPC)
    loaders, datasets = create_data_loaders(
        args.data_path,
        splits,
        sm_df,
        protein_markers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        channel='DPC_Left',
        multi_channel=args.multi_channel,
        augment_train=True,
        use_weighted_sampler=True
    )
    
    # Create optimized model
    print(f"\nCreating optimized model...")
    # Use task_gating if available, fallback to cross_task_attention for backward compatibility
    use_gating = getattr(args, 'use_task_gating', args.use_cross_task_attention)
    model = create_model(
        num_classes=args.num_classes,
        num_proteins=len(protein_markers),
        img_size=args.img_size,
        in_channels=args.in_channels,
        use_task_gating=use_gating
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({"total_params": total_params})
    
    # Loss function
    class_weights = calculate_class_weights(splits['train'][1]).to(device)
    
    # Classification loss
    cls_loss_fn = CombinedClassificationLoss(
        alpha=class_weights,
        gamma=args.focal_gamma,
        smoothing=args.label_smoothing
    )
    
    # Regression loss
    if args.use_improved_regression:
        reg_loss_fn = ImprovedRegressionLoss(mse_weight=0.7, corr_weight=0.3)
        print("✓ Using improved regression loss (MSE + Correlation)")
    else:
        reg_loss_fn = nn.MSELoss()
        print("✓ Using standard MSE loss")
    
    # Multi-task loss
    if args.use_adaptive_loss:
        loss_fn = AdaptiveMultiTaskLoss(
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            learn_weights=True
        )
        print("✓ Using adaptive multi-task loss (learnable weights)")
    else:
        # Create a simple wrapper
        class SimpleMultiTaskLoss(nn.Module):
            def __init__(self, cls_loss_fn, reg_loss_fn, cls_weight, reg_weight):
                super().__init__()
                self.cls_loss_fn = cls_loss_fn
                self.reg_loss_fn = reg_loss_fn
                self.cls_weight = cls_weight
                self.reg_weight = reg_weight
            
            def forward(self, cls_logits, cls_targets, reg_preds, reg_targets):
                cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
                reg_loss = self.reg_loss_fn(reg_preds, reg_targets)
                total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
                return {
                    'total': total_loss,
                    'classification': cls_loss,
                    'regression': reg_loss
                }
        
        loss_fn = SimpleMultiTaskLoss(
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            cls_weight=args.cls_weight,
            reg_weight=args.reg_weight
        )
        print("✓ Using fixed-weight multi-task loss")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler (OneCycleLR for better convergence)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,  # Peak LR
        epochs=args.num_epochs,
        steps_per_epoch=len(loaders['train']),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Trainer
    trainer = ImprovedMultiTaskTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        use_wandb=args.use_wandb
    )

    # Train
    best_acc = trainer.train(
        num_epochs=args.num_epochs,
        patience=args.patience
    )

    print(f"\nTraining finished! Best accuracy: {best_acc:.2f}%")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

