"""
Model training module for Vision Transformer (ViT) bone marrow classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import timm
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.config import (
    VIT_MODEL_NAME, NUM_CLASSES, PRETRAINED, VIT_MODEL_PATH,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA, DEVICE
)
from src.utils import get_device, count_parameters, format_time
import time


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier for bone marrow cell classification.
    """
    
    def __init__(self, model_name=VIT_MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout_rate=0.1):
        """
        Initialize ViT model.
        
        Args:
            model_name: Name of ViT model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for uncertainty estimation
        """
        super(ViTClassifier, self).__init__()
        
        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            drop_rate=dropout_rate
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor
            
        Returns:
            Logits for each class
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_vit(train_loader, val_loader, num_epochs=None, learning_rate=None, 
              weight_decay=None, scheduler_type='cosine', resume_from=None):
    """
    Train Vision Transformer model.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        scheduler_type: Type of LR scheduler ('cosine' or 'step')
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained model and training history
    """
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if learning_rate is None:
        learning_rate = LEARNING_RATE
    if weight_decay is None:
        weight_decay = WEIGHT_DECAY
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model
    model = ViTClassifier()
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
    
    # Training loop
    patience_counter = 0
    start_time = time.time()
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc + EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            torch.save(checkpoint, VIT_MODEL_PATH)
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Load best model
    if Path(VIT_MODEL_PATH).exists():
        print(f"\nLoading best model from {VIT_MODEL_PATH}")
        checkpoint = torch.load(VIT_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(elapsed_time)}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


