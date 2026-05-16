"""
Data Loading Module for BSCCM Dataset
======================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Supports multi-channel DPC images and protein expression labels for
multi-task learning (WBC classification + protein regression).

Usage:
    from data_loading import load_bsccm_dataset, create_train_val_test_splits, create_data_loaders
    bsccm, indices, class_labels, sm_df, protein_markers = load_bsccm_dataset('/path/to/BSCCMNIST')

Dataset:
    BSCCM (Berkeley Single Cell Computational Microscopy) — https://github.com/Waller-Lab/BSCCM
    Download: python -c "from bsccm import download_dataset; download_dataset('./data', mnist=True)"
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from bsccm import BSCCM


# ==================== CONFIGURATION ==================== #

IMAGE_SIZE = 28
PROTEIN_MARKERS = [
    # Use ONLY full model unmixed columns which have 100% data in BSCCMNIST
    'CD123/HLA-DR/CD14_full_model_unmixed',
    'CD3/CD19/CD56_full_model_unmixed',
    'CD45_full_model_unmixed',
    'CD16_full_model_unmixed'
]


# ==================== TRANSFORMS ==================== #

def get_base_transform():
    """Base transform: to tensor and normalize"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1, 1]
    ])


def get_augmentation_transform():
    """Training augmentation (reduced intensity for better training accuracy)"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced from 0.5
        transforms.RandomVerticalFlip(p=0.3),    # Reduced from 0.5
        transforms.RandomRotation(degrees=5),     # Reduced from 15
        transforms.ColorJitter(brightness=0.1, contrast=0.1)  # Reduced from 0.2
    ])


def get_transform(train=True, augment=False):
    """Get complete transform pipeline"""
    transform_list = []
    
    if augment and train:
        transform_list.append(get_augmentation_transform())
    
    transform_list.append(get_base_transform())
    
    return transforms.Compose(transform_list)


# ==================== MULTI-TASK DATASET ==================== #

class BSCCMMultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning:
    - Classification: Cell type (3 classes: Lymphocyte, Granulocyte, Monocyte)
    - Regression: Protein expression (4 markers)
    """
    
    def __init__(
        self,
        dataset_path,
        indices,
        class_labels,
        surface_marker_df,
        protein_markers,
        channel='DPC_Left',
        multi_channel=False,
        transform=None,
        use_cache=True,
        protein_mean=None,
        protein_std=None
    ):
        """
        Args:
            dataset_path: Path to BSCCM dataset
            indices: Array of cell indices
            class_labels: Array of class labels (0, 1, or 2 for 3 classes)
            surface_marker_df: DataFrame with protein expression data
            protein_markers: List of protein marker column names
            channel: Which imaging channel to use ('DPC_Left', 'Brightfield', etc.)
            multi_channel: If True, stack 4 DPC angles (Left, Right, Top, Bottom)
            transform: Image transforms
            use_cache: Whether to cache loaded images in memory
        """
        self.dataset_path = dataset_path
        self.bsccm = BSCCM(dataset_path)
        self.indices = indices
        self.class_labels = class_labels
        self.surface_marker_df = surface_marker_df
        self.protein_markers = protein_markers
        self.channel = channel
        self.multi_channel = multi_channel
        self.transform = transform
        self.use_cache = use_cache
        
        # Validate that surface marker data exists
        missing_markers = [m for m in protein_markers if m not in surface_marker_df.columns]
        if missing_markers:
            print(f"Warning: Missing protein markers: {missing_markers}")
            self.protein_markers = [m for m in protein_markers if m in surface_marker_df.columns]
        
        # Filter to valid indices that exist in both datasets
        valid_mask = np.isin(self.indices, surface_marker_df.index)
        self.indices = self.indices[valid_mask]
        self.class_labels = self.class_labels[valid_mask]
        
        # Compute protein normalization statistics
        if protein_mean is not None and protein_std is not None:
            # Use provided statistics (e.g., from training set)
            self.protein_mean = protein_mean
            self.protein_std = protein_std
        else:
            # Compute from this dataset's indices
            valid_proteins = surface_marker_df.loc[self.indices, self.protein_markers].values.astype(np.float32)
            self.protein_mean = np.mean(valid_proteins, axis=0, keepdims=True)
            self.protein_std = np.std(valid_proteins, axis=0, keepdims=True) + 1e-8  # Add small epsilon
        
        print(f"Dataset initialized with {len(self.indices)} valid samples")
        print(f"Protein normalization: mean={self.protein_mean.squeeze()}, std={self.protein_std.squeeze()}")
        
        # Cache for images (optional, to speed up training)
        self.image_cache = {} if use_cache else None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (1, 28, 28) or (C, 28, 28) if multi_channel
            label: int, cell type class
            proteins: Tensor (N_markers,), protein expression values
        """
        index = self.indices[idx]
        label = int(self.class_labels[idx])
        
        # Load image from cache or disk
        if self.use_cache and index in self.image_cache:
            image_np = self.image_cache[index]
        else:
            # Load single channel
            try:
                image_np = self.bsccm.read_image(index, channel=self.channel)
            except Exception as e:
                print(f"Error loading image {index}: {e}")
                # Return dummy data
                image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
            if self.use_cache:
                self.image_cache[index] = image_np
        
        # Multi-channel stacking (4 DPC orientations: Left, Right, Top, Bottom)
        if self.multi_channel and self.channel == 'DPC_Left':
            try:
                image_right = self.bsccm.read_image(index, channel='DPC_Right')
                image_top = self.bsccm.read_image(index, channel='DPC_Top')
                image_bottom = self.bsccm.read_image(index, channel='DPC_Bottom')
                # Stack all 4 channels: [Left, Right, Top, Bottom]
                image_np = np.stack([image_np, image_right, image_top, image_bottom], axis=0)
            except Exception as e:
                # Fallback to single channel
                print(f"Warning: Could not load all DPC channels for index {index}: {e}")
                image_np = np.expand_dims(image_np, axis=0)
        else:
            image_np = np.expand_dims(image_np, axis=0)
        
        # Apply transforms
        if self.transform:
            # Handle multi-channel case
            if len(image_np.shape) == 3 and image_np.shape[0] > 1:
                # Apply same transform to all channels
                image_transformed = []
                for ch in range(image_np.shape[0]):
                    img = Image.fromarray(image_np[ch].astype(np.uint8))
                    img_t = self.transform(img)
                    image_transformed.append(img_t)
                image = torch.cat(image_transformed, dim=0)
            else:
                img = Image.fromarray(image_np.squeeze().astype(np.uint8))
                image = self.transform(img)
        else:
            image = torch.from_numpy(image_np).float()
            if image.dim() == 2:
                image = image.unsqueeze(0)
        
        # Load protein expression
        if index in self.surface_marker_df.index:
            proteins = self.surface_marker_df.loc[index, self.protein_markers].values
            
            # Convert to float array, handling any string or object types
            try:
                # Try direct conversion first (faster)
                proteins = proteins.astype(np.float32)
            except (ValueError, TypeError):
                # If that fails, convert via pandas which handles mixed types better
                proteins = pd.to_numeric(proteins, errors='coerce').values
                proteins = np.nan_to_num(proteins, nan=0.0).astype(np.float32)
        else:
            proteins = np.zeros(len(self.protein_markers), dtype=np.float32)
        
        # Normalize proteins using computed statistics
        proteins = (proteins - self.protein_mean.squeeze()) / self.protein_std.squeeze()
        proteins = torch.from_numpy(proteins).float()
        
        return image, label, proteins
    
    def get_class_distribution(self):
        """Return class distribution for 3 classes"""
        counts = Counter(self.class_labels)
        class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
        
        # Create distribution dict
        dist = {}
        for i, name in enumerate(class_names):
            dist[name] = counts.get(i, 0)
        dist['total'] = len(self.indices)
        dist['num_classes'] = 3
        return dist
    
    def visualize_samples(self, num_samples=9, save_path='dataset_samples.png'):
        """Visualize random samples from dataset"""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'Sample Images from Dataset (Channel: {self.channel})', 
                     fontsize=16, fontweight='bold')
        
        # Get number of classes from unique labels (3 classes)
        class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
        
        for ax in axes.flat:
            idx = np.random.randint(0, len(self))
            image, label, proteins = self[idx]
            
            # Convert image for display
            if image.dim() == 3:
                img_display = image[0].numpy()
            else:
                img_display = image.numpy()
            
            ax.imshow(img_display, cmap='gray')
            ax.set_title(f'{class_names[label]}', fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to {save_path}")
        plt.close()


# ==================== DATASET PREPARATION ==================== #

def load_bsccm_dataset(
    dataset_path,
    protein_markers=None,
    min_samples_per_class=None,
    random_state=42
):
    """
    Load and prepare BSCCM dataset for multi-task learning
    
    Args:
        dataset_path: Path to BSCCM dataset
        protein_markers: List of protein markers to extract (None = all)
        min_samples_per_class: Minimum samples per class (None = no filter)
        random_state: Random seed for reproducibility
    
    Returns:
        bsccm: BSCCM dataset object
        indices: Array of cell indices
        class_labels: Array of class labels
        surface_marker_df: DataFrame with protein data
        protein_markers: List of available protein marker names
    """
    print("="*70)
    print("LOADING BSCCM DATASET")
    print("="*70)
    
    # Load dataset
    bsccm = BSCCM(dataset_path)
    print(f"Dataset: {bsccm}")
    
    # Get classification data (3-class version)
    indices, class_labels = bsccm.get_cell_type_classification_data(ten_class_version=False)
    # Convert labels to integers (they may be float64 from dataset)
    class_labels = class_labels.astype(np.int64)
    print(f"Total samples: {len(indices)}")
    unique_labels, counts = np.unique(class_labels, return_counts=True)
    print(f"Class labels (3-class version): {len(unique_labels)} classes")
    class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
    for label, count in zip(unique_labels, counts):
        class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
        print(f"  {class_name} (Class {label}): {count} cells")
    
    # Load surface marker data
    sm_df = bsccm.surface_marker_dataframe
    print(f"Surface marker columns: {len(sm_df.columns)}")
    
    # Select protein markers
    if protein_markers is None:
        # Auto-detect: find all single antibody model columns
        protein_markers = [col for col in sm_df.columns 
                          if '_single_antibody_model_unmixed' in col]
    
    print(f"Selected protein markers ({len(protein_markers)}):")
    for i, marker in enumerate(protein_markers[:10]):
        print(f"  {i+1}. {marker}")
    if len(protein_markers) > 10:
        print(f"  ... and {len(protein_markers)-10} more")
    
    # Filter samples
    if min_samples_per_class is not None:
        unique_classes = np.unique(class_labels)
        valid_indices = []
        valid_labels = []
        for class_id in unique_classes:
            class_mask = class_labels == class_id
            class_indices = indices[class_mask]
            
            if len(class_indices) >= min_samples_per_class:
                valid_indices.extend(class_indices)
                valid_labels.extend([class_id] * len(class_indices))
        
        indices = np.array(valid_indices)
        class_labels = np.array(valid_labels)
        print(f"Filtered to {len(indices)} samples (min {min_samples_per_class}/class)")
    
    return bsccm, indices, class_labels, sm_df, protein_markers


def balance_classes_by_oversampling(indices, class_labels, random_state=42):
    """
    Balance classes by oversampling minority classes to match majority class count.
    
    Args:
        indices: Array of sample indices
        class_labels: Array of class labels
        random_state: Random seed for reproducibility
    
    Returns:
        balanced_indices: Array of indices with oversampling
        balanced_labels: Array of labels with oversampling
    """
    np.random.seed(random_state)
    
    # Count samples per class
    unique_labels, counts = np.unique(class_labels, return_counts=True)
    max_count = np.max(counts)
    
    print(f"\nClass distribution before balancing:")
    class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
    for label, count in zip(unique_labels, counts):
        class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
        print(f"  {class_name} (Class {label}): {count} samples")
    
    # Oversample minority classes
    balanced_indices = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get indices for this class
        class_mask = class_labels == label
        class_indices = indices[class_mask]
        class_count = len(class_indices)
        
        # If this is minority class, oversample
        if class_count < max_count:
            # Calculate how many samples to add
            samples_needed = max_count - class_count
            # Randomly sample with replacement
            additional_indices = np.random.choice(class_indices, size=samples_needed, replace=True)
            # Add original + additional samples
            balanced_indices.extend(class_indices)
            balanced_indices.extend(additional_indices)
            balanced_labels.extend([label] * class_count)
            balanced_labels.extend([label] * samples_needed)
            
            class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
            print(f"  → Oversampled {class_name}: {class_count} → {max_count} samples (+{samples_needed})")
        else:
            # Majority class: keep as is
            balanced_indices.extend(class_indices)
            balanced_labels.extend([label] * class_count)
    
    balanced_indices = np.array(balanced_indices)
    balanced_labels = np.array(balanced_labels)
    
    # Shuffle to mix classes
    shuffle_idx = np.random.permutation(len(balanced_indices))
    balanced_indices = balanced_indices[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    print(f"\nTotal samples after balancing: {len(balanced_indices)} (was {len(indices)})")
    
    return balanced_indices, balanced_labels


def create_train_val_test_splits(
    indices,
    class_labels,
    train_ratio=0.7,
    val_ratio=0.15,
    random_state=42,
    balance_train=True
):
    """
    Create stratified train/val/test splits with optional class balancing for training set
    
    Args:
        indices: Array of sample indices
        class_labels: Array of class labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (test = 1 - train - val)
        random_state: Random seed
        balance_train: If True, oversample minority classes in training set
    
    Returns:
        splits: Dict with 'train', 'val', 'test' keys containing index arrays
    """
    # Stratified split: train vs (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, class_labels,
        test_size=(1 - train_ratio),
        stratify=class_labels,
        random_state=random_state
    )
    
    # Split val + test
    val_size_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels,
        test_size=(1 - val_size_adjusted),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Balance training set if requested
    if balance_train:
        print("\n" + "="*70)
        print("BALANCING TRAINING SET")
        print("="*70)
        train_indices, train_labels = balance_classes_by_oversampling(
            train_indices, train_labels, random_state=random_state
        )
    
    splits = {
        'train': (train_indices, train_labels),
        'val': (val_indices, val_labels),
        'test': (test_indices, test_labels)
    }
    
    # Display splits with class distribution
    class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
    
    print("\n" + "="*70)
    print("DATASET SPLITS")
    print("="*70)
    print(f"{'Split':<10} {'Total':<10} {'Lymphocyte':<12} {'Granulocyte':<12} {'Monocyte':<12}")
    print("-"*70)
    
    for split_name, (split_indices, split_labels) in splits.items():
        counts = Counter(split_labels)
        total = len(split_indices)
        row = f"{split_name:<10} {total:<10}"
        for i in range(3):  # 3 classes
            count = counts.get(i, 0)
            pct = (count / total * 100) if total > 0 else 0
            row += f"{count} ({pct:.1f}%){'':<2}"
        print(row)
    
    print("-"*70)
    
    return splits


# ==================== DATALOADER CREATION ==================== #

def create_data_loaders(
    dataset_path,
    splits,
    surface_marker_df,
    protein_markers,
    batch_size=32,
    num_workers=4,
    channel='DPC_Left',
    multi_channel=True,  # Default to 4-channel DPC (Left, Right, Top, Bottom)
    augment_train=True,
    use_weighted_sampler=False,
    pin_memory=True
):
    """
    Create PyTorch DataLoaders for train/val/test
    
    Args:
        dataset_path: Path to BSCCM dataset
        splits: Dict with train/val/test splits
        surface_marker_df: Surface marker DataFrame
        protein_markers: List of protein marker names
        batch_size: Batch size
        num_workers: Number of data loading workers
        channel: Image channel to use
        multi_channel: Whether to use multi-channel images
        augment_train: Whether to augment training data
        use_weighted_sampler: Whether to use weighted sampling
        pin_memory: Pin memory for GPU
    
    Returns:
        loaders: Dict with 'train', 'val', 'test' DataLoaders
        datasets: Dict with Dataset objects
    """
    loaders = {}
    datasets = {}
    
    # First, compute normalization statistics from training set
    train_indices, _ = splits['train']
    train_proteins = surface_marker_df.loc[train_indices, protein_markers].values.astype(np.float32)
    protein_mean = np.mean(train_proteins, axis=0, keepdims=True)
    protein_std = np.std(train_proteins, axis=0, keepdims=True) + 1e-8
    
    for split_name, (split_indices, split_labels) in splits.items():
        is_train = (split_name == 'train')
        
        # Create dataset
        dataset = BSCCMMultiTaskDataset(
            dataset_path=dataset_path,
            indices=split_indices,
            class_labels=split_labels,
            surface_marker_df=surface_marker_df,
            protein_markers=protein_markers,
            channel=channel,
            multi_channel=multi_channel,
            transform=get_transform(train=is_train, augment=(is_train and augment_train)),
            use_cache=True if split_name == 'train' else False,  # Only cache training
            protein_mean=protein_mean,
            protein_std=protein_std
        )
        
        datasets[split_name] = dataset
        
        # Create sampler if weighted sampling is requested
        sampler = None
        if is_train and use_weighted_sampler:
            # Convert labels to integers (they may be float64 from dataset)
            split_labels_int = split_labels.astype(np.int64)
            class_counts = np.bincount(split_labels_int)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in split_labels_int]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(not sampler and is_train),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train  # Drop last incomplete batch in training
        )
        
        loaders[split_name] = loader
        print(f"\n{split_name.capitalize()} loader: {len(dataset)} samples, {len(loader)} batches")
    
    return loaders, datasets


# ==================== STATISTICS & VISUALIZATION ==================== #

def analyze_protein_distributions(surface_marker_df, protein_markers):
    """Analyze and visualize protein expression distributions"""
    print("\n" + "="*70)
    print("PROTEIN EXPRESSION STATISTICS")
    print("="*70)
    
    stats = {}
    for marker in protein_markers:
        if marker in surface_marker_df.columns:
            values = surface_marker_df[marker].dropna()
            stats[marker] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            }
    
    # Print summary
    print(f"\n{'Marker':<35} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*70)
    for marker, stat in stats.items():
        print(f"{marker:<35} {stat['mean']:<10.2f} {stat['std']:<10.2f} "
              f"{stat['min']:<10.2f} {stat['max']:<10.2f}")
    
    # Visualize distributions
    if len(protein_markers) > 0:
        n_plots = min(9, len(protein_markers))
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, marker in enumerate(protein_markers[:n_plots]):
            if marker in surface_marker_df.columns:
                values = surface_marker_df[marker].dropna()
                axes[idx].hist(values, bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(marker, fontweight='bold')
                axes[idx].set_xlabel('Expression Level')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('protein_distributions.png', dpi=150, bbox_inches='tight')
        print("\nProtein distribution plots saved to 'protein_distributions.png'")
        plt.close()
    
    return stats


# ==================== MAIN USAGE ==================== #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect BSCCM dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to BSCCMNIST dataset directory')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("DETAILED DATASET INSPECTION FOR PROTEIN MARKERS")
    print("="*80)

    # Load dataset
    bsccm, indices, class_labels, sm_df, protein_markers = load_bsccm_dataset(
        args.data_path,
        protein_markers=PROTEIN_MARKERS
    )
    
    # DETAILED INSPECTION: Check each protein marker column
    print("\n" + "="*80)
    print("DETAILED PROTEIN MARKER CHECK")
    print("="*80)
    print(f"\nChecking {len(protein_markers)} protein markers:")
    
    # Check for classification indices specifically
    sample_indices = indices[:20]
    
    for marker in protein_markers:
        if marker in sm_df.columns:
            # Overall stats
            overall_not_null = sm_df[marker].notna().sum()
            overall_pct = 100 * overall_not_null / len(sm_df)
            
            # Stats for classification indices only
            class_mask = sm_df.index.isin(sample_indices)
            class_values = sm_df.loc[class_mask, marker]
            class_not_null = class_values.notna().sum()
            class_pct = 100 * class_not_null / len(class_values) if len(class_values) > 0 else 0
            
            print(f"\n✅ {marker}")
            print(f"   Overall: {overall_not_null}/{len(sm_df)} ({overall_pct:.1f}% coverage)")
            print(f"   Classification indices (first 20): {class_not_null}/{len(class_values)} ({class_pct:.1f}% coverage)")
            
            if class_not_null > 0:
                sample_vals = class_values.dropna().head(3).tolist()
                print(f"   Sample values: {sample_vals}")
            else:
                print(f"   ⚠️  NO DATA in classification indices!")
        else:
            print(f"\n❌ {marker} - NOT FOUND in CSV!")
    
    # Check sample classification indices
    print("\n" + "="*80)
    print("CHECKING SAMPLE CLASSIFICATION INDICES")
    print("="*80)
    print(f"\nFirst 10 classification indices: {indices[:10]}")
    
    for idx in indices[:5]:
        if idx in sm_df.index:
            values = sm_df.loc[idx, protein_markers].values
            # Convert to numeric first
            values = pd.to_numeric(values, errors='coerce')
            has_nan = np.any(np.isnan(values))
            not_nan_count = np.sum(~np.isnan(values))
            print(f"  Index {idx}: has_nan={has_nan}, {not_nan_count}/{len(values)} values")
            if not_nan_count > 0:
                print(f"    Values: {values}")
        else:
            print(f"  Index {idx}: NOT IN surface_marker_df!")
    
    # Analyze protein distributions
    print("\n" + "="*80)
    print("PROTEIN DISTRIBUTION ANALYSIS")
    print("="*80)
    protein_stats = analyze_protein_distributions(sm_df, protein_markers)
    
    # Create splits
    splits = create_train_val_test_splits(
        indices, class_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        random_state=42
    )
    
    # Create dataloaders
    loaders, datasets = create_data_loaders(
        args.data_path,
        splits,
        sm_df,
        protein_markers,
        batch_size=32,
        num_workers=0,  # Set to 0 for debugging
        channel='DPC_Left',
        multi_channel=False,
        augment_train=True,
        use_weighted_sampler=True
    )
    
    # Test a batch
    print("\n" + "="*80)
    print("TESTING DATALOADER - ACTUAL VALUES")
    print("="*80)
    
    for batch_idx, (images, labels, proteins) in enumerate(loaders['train']):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images: shape={images.shape}, dtype={images.dtype}")
        print(f"  Labels: shape={labels.shape}")
        print(f"  Proteins: shape={proteins.shape}, dtype={proteins.dtype}")
        print(f"  Label distribution: {Counter(labels.numpy())}")
        
        if batch_idx == 0:  # Print first batch in detail
            print(f"\n  Sample image stats:")
            print(f"    Min: {images.min():.3f}, Max: {images.max():.3f}, Mean: {images.mean():.3f}")
            print(f"    Has NaN: {torch.any(torch.isnan(images))}, Has Inf: {torch.any(torch.isinf(images))}")
            
            print(f"\n  Protein expression DETAILED:")
            print(f"    Min: {proteins.min():.4f}, Max: {proteins.max():.4f}, Mean: {proteins.mean():.4f}")
            print(f"    Has NaN: {torch.any(torch.isnan(proteins))}, Has Inf: {torch.any(torch.isinf(proteins))}")
            
            print(f"\n    Per-marker stats:")
            for idx, marker in enumerate(protein_markers):
                col_data = proteins[:, idx]
                not_nan = (~torch.isnan(col_data)).sum().item()
                print(f"      {marker}:")
                print(f"        Coverage: {not_nan}/{len(col_data)} samples")
                if not_nan > 0:
                    print(f"        Mean: {col_data[~torch.isnan(col_data)].mean():.2f}")
                    print(f"        Std: {col_data[~torch.isnan(col_data)].std():.2f}")
                    print(f"        Range: [{col_data[~torch.isnan(col_data)].min():.2f}, {col_data[~torch.isnan(col_data)].max():.2f}]")
                else:
                    print(f"        ⚠️  ALL NaN!")
            
            # Show actual values for first sample
            print(f"\n  First sample actual values:")
            print(f"    Image: min={images[0].min():.3f}, max={images[0].max():.3f}")
            print(f"    Label: {labels[0].item()}")
            print(f"    Proteins: {proteins[0].tolist()}")
            print(f"    Proteins (rounded): {[round(x.item(), 2) for x in proteins[0]]}")
        
        break  # Only test first batch
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
