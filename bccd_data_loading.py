"""
BCCD Dataset Loading and Preprocessing
=======================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Data loader for the Blood Cell Images (BCCD) dataset used in Table 3.
Downloads: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
Maps 4-class BCCD (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL) to 3
classes matching BSCCM (Lymphocyte, Granulocyte, Monocyte).

Dataset structure (CORRECTED):
- TRAIN folder (9,957 images) → Training set
- TEST folder (2,487 images) → Validation set (for model selection)
- TEST_SIMPLE folder (73 images) → Final test set (for reporting)
- 4 classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
- RGB images (3 channels) at 128x128 resolution
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

# Class mapping: Kaggle (4 classes) → Our model (3 classes)
KAGGLE_TO_MODEL_MAPPING = {
    'EOSINOPHIL': 1,    # → Granulocyte
    'LYMPHOCYTE': 0,    # → Lymphocyte
    'MONOCYTE': 2,      # → Monocyte
    'NEUTROPHIL': 1,    # → Granulocyte (merge with EOSINOPHIL)
}

CLASS_NAMES = ['Lymphocyte', 'Granulocyte', 'Monocyte']


def load_kaggle_bccd_dataset(data_root, use_test_simple=True, seed=42):
    """
    Load Kaggle BCCD dataset from directory structure.
    
    **CORRECTED SPLIT STRATEGY**:
    - TRAIN folder → Training set (9,957 images)
    - TEST folder → Validation set (2,487 images) for model selection
    - TEST_SIMPLE folder → Final test set (73 images) for reporting
    
    Args:
        data_root: Root directory containing images/TRAIN, images/TEST, images/TEST_SIMPLE
        use_test_simple: Whether to use TEST_SIMPLE as final test set (default: True)
        seed: Random seed for reproducibility (for shuffling only)
    
    Returns:
        train_data, val_data, test_data: Lists of (image_path, label) tuples
        class_counts: Dictionary of class distributions
    """
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    
    print("="*70)
    print("LOADING KAGGLE BCCD DATASET")
    print("="*70)
    print("\n🔄 CORRECTED SPLIT STRATEGY:")
    print("  TRAIN → Training set")
    print("  TEST → Validation set (for model selection)")
    print("  TEST_SIMPLE → Final test set (for reporting)\n")
    
    # Load training data (TRAIN folder)
    train_dir = images_dir / 'TRAIN'
    train_samples = []
    
    for class_folder in train_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            if class_name in KAGGLE_TO_MODEL_MAPPING:
                mapped_label = KAGGLE_TO_MODEL_MAPPING[class_name]
                for img_file in class_folder.glob('*.jpeg'):
                    train_samples.append((str(img_file), mapped_label, class_name))
    
    # Load validation data (TEST folder - used during training for model selection)
    val_dir = images_dir / 'TEST'
    val_samples = []
    
    for class_folder in val_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            if class_name in KAGGLE_TO_MODEL_MAPPING:
                mapped_label = KAGGLE_TO_MODEL_MAPPING[class_name]
                for img_file in class_folder.glob('*.jpeg'):
                    val_samples.append((str(img_file), mapped_label, class_name))
    
    # Load final test data (TEST_SIMPLE folder - held-out for final evaluation)
    if use_test_simple:
        test_dir = images_dir / 'TEST_SIMPLE'
    else:
        # Fallback: use TEST folder if TEST_SIMPLE not desired
        test_dir = images_dir / 'TEST'
    
    test_samples = []
    
    for class_folder in test_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            if class_name in KAGGLE_TO_MODEL_MAPPING:
                mapped_label = KAGGLE_TO_MODEL_MAPPING[class_name]
                for img_file in class_folder.glob('*.jpeg'):
                    test_samples.append((str(img_file), mapped_label, class_name))
    
    # Compute class distributions
    train_counts = Counter([s[1] for s in train_samples])
    val_counts = Counter([s[1] for s in val_samples])
    test_counts = Counter([s[1] for s in test_samples])
    
    print(f"\nDataset loaded from: {data_root}")
    print(f"Total samples:")
    print(f"  Training (TRAIN): {len(train_samples)}")
    print(f"  Validation (TEST): {len(val_samples)}")
    print(f"  Test ({'TEST_SIMPLE' if use_test_simple else 'TEST'}): {len(test_samples)}")
    
    print(f"\nClass distribution (mapped to 3 classes):")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name} (Class {class_idx}):")
        print(f"    Train: {train_counts.get(class_idx, 0)}")
        print(f"    Val: {val_counts.get(class_idx, 0)}")
        print(f"    Test: {test_counts.get(class_idx, 0)}")
    
    # Show original Kaggle class mapping
    print(f"\nOriginal Kaggle classes → Model classes:")
    for kaggle_class, model_class in KAGGLE_TO_MODEL_MAPPING.items():
        print(f"  {kaggle_class} → {CLASS_NAMES[model_class]}")
    
    print("\n" + "="*70)
    
    return train_samples, val_samples, test_samples


def preprocess_rgb_image(rgb_image, img_size=256):
    """
    Preprocess RGB image for model input.
    Keep native 3-channel RGB (no conversion to 4-channel).
    
    Args:
        rgb_image: numpy array of shape (H, W, 3)
        img_size: Target image size
    
    Returns:
        Preprocessed RGB image: numpy array of shape (img_size, img_size, 3)
    """
    # Keep as RGB (3 channels) - native format
    return rgb_image.astype(np.float32)


class KaggleBCCDDataset(Dataset):
    """
    PyTorch Dataset for Kaggle BCCD images.
    Uses native RGB (3-channel) at high resolution.
    """
    
    def __init__(self, samples, transform=None, img_size=128, num_proteins=4):
        """
        Args:
            samples: List of (image_path, label, original_class) tuples
            transform: Optional torchvision transforms
            img_size: Target image size (default: 128 for memory efficiency)
            num_proteins: Number of protein outputs (dummy values)
        """
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.num_proteins = num_proteins
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        
        # Load RGB image
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # Normalize with ImageNet stats (standard for RGB images)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # RGB channels only
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        
        # Create dummy protein labels (all zeros)
        # Model expects protein outputs but we don't have ground truth
        proteins = torch.zeros(self.num_proteins, dtype=torch.float32)
        
        return image, label, proteins


def create_kaggle_data_loaders(data_root, batch_size=32, num_workers=4, 
                               img_size=128, augment_train=True, use_test_simple=True, seed=42):
    """
    Create DataLoaders for Kaggle BCCD dataset.
    
    **CORRECTED SPLIT STRATEGY**:
    - Training: TRAIN folder (9,957 images)
    - Validation: TEST folder (2,487 images) - used for model selection
    - Test: TEST_SIMPLE folder (73 images) - held-out for final evaluation
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size
        augment_train: Whether to apply data augmentation to training set
        use_test_simple: Whether to use TEST_SIMPLE as final test set
        seed: Random seed
    
    Returns:
        loaders: Dictionary with 'train', 'val', 'test' DataLoaders
        datasets: Dictionary with corresponding datasets
        class_names: List of class names
    """
    
    # Load dataset with corrected splits
    train_samples, val_samples, test_samples = load_kaggle_bccd_dataset(
        data_root, use_test_simple=use_test_simple, seed=seed
    )
    
    # Define transforms
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),  # Convert to tensor
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # No augmentation for val/test, just convert to tensor
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = KaggleBCCDDataset(
        train_samples, transform=train_transform, img_size=img_size
    )
    val_dataset = KaggleBCCDDataset(
        val_samples, transform=eval_transform, img_size=img_size
    )
    test_dataset = KaggleBCCDDataset(
        test_samples, transform=eval_transform, img_size=img_size
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    print("\n" + "="*70)
    print("DATA LOADERS CREATED")
    print("="*70)
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Augmentation: {'Enabled' if augment_train else 'Disabled'}")
    print("="*70 + "\n")
    
    return loaders, datasets, CLASS_NAMES


# Test the data loader
if __name__ == "__main__":
    # Test data loading
    data_root = r"D:\EdgeHill\Code\BSCCM\Data\Keggle_BCCD\dataset2-master\dataset2-master"
    
    loaders, datasets, class_names = create_kaggle_data_loaders(
        data_root=data_root,
        batch_size=32,  # Can use larger batch for 128x128 images
        img_size=128,   # Balanced resolution
        augment_train=True
    )
    
    # Test loading a batch
    print("Testing data loading...")
    for images, labels, proteins in loaders['train']:
        print(f"Batch shape: {images.shape}")  # Should be [32, 3, 128, 128]
        print(f"Labels shape: {labels.shape}")
        print(f"Proteins shape: {proteins.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique labels: {torch.unique(labels)}")
        print(f"Number of channels: {images.shape[1]} (should be 3 for RGB)")
        print(f"Image resolution: {images.shape[2]}x{images.shape[3]} (should be 128x128)")
        break
    
    print("\n✅ Data loader test successful!")
    print("✅ Using native RGB (3 channels) at 128x128 resolution")

