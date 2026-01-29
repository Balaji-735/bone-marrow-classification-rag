"""
Data preprocessing module for bone marrow cell classification.
Handles dataset loading, transforms, and data splits.
"""

import os
import json
import re
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from src.config import (
    RAW_DATA_DIR, SPLITS_DIR, CLASSES, CLASS_TO_IDX, 
    IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE
)
from src.utils import set_seed
from collections import Counter


class BoneMarrowDataset(Dataset):
    """
    PyTorch Dataset for bone marrow cell images.
    Assumes images are organized in folders named after classes (BLA, EOS, etc.).
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of integer labels corresponding to images
            transform: torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get an image and its label.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(mode='train'):
    """
    Get data augmentation transforms for training or validation/test.
    
    Args:
        mode: 'train', 'val', or 'test'
        
    Returns:
        torchvision transforms.Compose object
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_dataset_from_folders(data_dir):
    """
    Load dataset from folder structure where each class has its own folder.
    
    Args:
        data_dir: Path to directory containing class folders
        
    Returns:
        Tuple of (image_paths, labels) lists
    """
    image_paths = []
    labels = []
    
    data_dir = Path(data_dir)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Iterate through class folders
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class folder {class_dir} does not exist. Skipping.")
            continue
        
        class_idx = CLASS_TO_IDX[class_name]
        
        # Find all images in this class folder
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_paths.append(str(img_file))
                labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} images from {data_dir}")
    print(f"Class distribution:")
    for class_name in CLASSES:
        count = sum(1 for label in labels if label == CLASS_TO_IDX[class_name])
        print(f"  {class_name}: {count}")
    
    return image_paths, labels


def create_data_splits(image_paths, labels, save_splits=True):
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        save_splits: Whether to save split indices to file
        
    Returns:
        Tuple of (train_paths, val_paths, test_paths, train_labels, val_labels, test_labels)
    """
    set_seed(42)
    
    # First split: train vs (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=42
    )
    
    # Second split: val vs test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Val: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    # Save split indices
    if save_splits:
        splits = {
            'train_paths': train_paths,
            'val_paths': val_paths,
            'test_paths': test_paths,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }
        
        splits_file = SPLITS_DIR / "data_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Saved data splits to {splits_file}")
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def validate_data_splits(data_dir=None):
    """
    Validate that saved data splits match the current dataset.
    Compares class counts to detect if dataset has changed.
    
    Args:
        data_dir: Path to data directory (if None, uses RAW_DATA_DIR)
        
    Returns:
        bool: True if splits are valid, False if they need regeneration
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    splits_file = SPLITS_DIR / "data_splits.json"
    if not splits_file.exists():
        return False
    
    # Load current dataset
    current_paths, current_labels = load_dataset_from_folders(data_dir)
    current_counts = Counter(current_labels)
    
    # Load saved splits
    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Count images in saved splits
        saved_train_labels = splits.get('train_labels', [])
        saved_val_labels = splits.get('val_labels', [])
        saved_test_labels = splits.get('test_labels', [])
        saved_labels = saved_train_labels + saved_val_labels + saved_test_labels
        saved_counts = Counter(saved_labels)
        
        # Compare counts for each class
        for class_idx in range(len(CLASSES)):
            current_count = current_counts.get(class_idx, 0)
            saved_count = saved_counts.get(class_idx, 0)
            
            if current_count != saved_count:
                print(f"Class {CLASSES[class_idx]}: Current={current_count}, Saved={saved_count} - MISMATCH!")
                return False
        
        print("✓ Data splits validation passed - all class counts match")
        return True
    except Exception as e:
        print(f"Error validating splits: {e}")
        return False


def load_data_splits():
    """
    Load previously saved data splits.
    Automatically fixes paths if they point to old locations.
    
    Returns:
        Tuple of (train_paths, val_paths, test_paths, train_labels, val_labels, test_labels)
    """
    splits_file = SPLITS_DIR / "data_splits.json"
    
    if not splits_file.exists():
        raise FileNotFoundError(f"Data splits file not found: {splits_file}. Run create_data_splits first.")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Fix paths if they point to old location
    def fix_path(path):
        """Fix path if it points to old location."""
        path_str = str(path)
        # Check if path contains old location
        old_base = "rag-tutorial-v2-main (2)"
        if old_base in path_str:
            # Extract filename and class folder (e.g., "BLA/BLA_00675.jpg")
            # Look for pattern: data\raw\CLASS\FILENAME or data/raw/CLASS/FILENAME
            # Match class folder and filename
            match = re.search(r'[\\/]data[\\/]raw[\\/]([A-Z]+)[\\/]([A-Z]+_\d+\.(jpg|jpeg|png|bmp|tiff))', path_str, re.IGNORECASE)
            if match:
                class_name = match.group(1)
                filename = match.group(2)  # This already includes the extension
                # Reconstruct with current RAW_DATA_DIR
                new_path = Path(RAW_DATA_DIR) / class_name / filename
                return str(new_path)
        return path
    
    # Fix all paths
    train_paths = [fix_path(p) for p in splits['train_paths']]
    val_paths = [fix_path(p) for p in splits['val_paths']]
    test_paths = [fix_path(p) for p in splits['test_paths']]
    
    return (
        train_paths, val_paths, test_paths,
        splits['train_labels'], splits['val_labels'], splits['test_labels']
    )


def get_dataloaders(data_dir=None, use_saved_splits=True, batch_size=None, force_regenerate=False):
    """
    Get train/val/test dataloaders.
    
    Args:
        data_dir: Path to data directory (if None, uses RAW_DATA_DIR)
        use_saved_splits: Whether to use saved splits or create new ones
        batch_size: Batch size (if None, uses config BATCH_SIZE)
        force_regenerate: Force regeneration of data splits even if they exist
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    # Load or create splits
    if use_saved_splits and not force_regenerate:
        try:
            # Validate that splits match current dataset
            if not validate_data_splits(data_dir):
                print("\n⚠ Data splits are outdated or invalid. Regenerating...")
                force_regenerate = True
            else:
                train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_data_splits()
        except FileNotFoundError:
            print("Saved splits not found. Creating new splits...")
            force_regenerate = True
    
    if force_regenerate or not use_saved_splits:
        print("\nCreating new data splits from current dataset...")
        image_paths, labels = load_dataset_from_folders(data_dir)
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = create_data_splits(
            image_paths, labels, save_splits=True
        )
    
    # Create datasets
    train_dataset = BoneMarrowDataset(train_paths, train_labels, transform=get_transforms('train'))
    val_dataset = BoneMarrowDataset(val_paths, val_labels, transform=get_transforms('val'))
    test_dataset = BoneMarrowDataset(test_paths, test_labels, transform=get_transforms('test'))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data preprocessing...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")







