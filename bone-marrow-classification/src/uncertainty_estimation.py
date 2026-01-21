"""
Uncertainty estimation module using Monte Carlo Dropout.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.config import MC_DROPOUT_SAMPLES, DEVICE
from src.utils import get_device


def enable_dropout(model):
    """
    Enable dropout layers during inference for Monte Carlo sampling.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Keep dropout active


def disable_dropout(model):
    """
    Disable dropout layers (set to eval mode).
    
    Args:
        model: PyTorch model
    """
    model.eval()


def monte_carlo_dropout(model, images, num_samples=MC_DROPOUT_SAMPLES, device=None):
    """
    Perform Monte Carlo Dropout inference.
    
    Args:
        model: PyTorch model with dropout layers
        images: Batch of image tensors
        num_samples: Number of Monte Carlo samples
        device: Device to run on
        
    Returns:
        Tuple of (mean_probabilities, variance, entropy)
    """
    if device is None:
        device = get_device()
    
    images = images.to(device)
    model.eval()
    
    # Enable dropout
    enable_dropout(model)
    
    # Collect predictions
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())
    
    # Disable dropout
    disable_dropout(model)
    
    # Convert to numpy array
    predictions = np.array(predictions)  # Shape: (num_samples, batch_size, num_classes)
    
    # Compute statistics
    mean_probs = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    
    # Compute entropy (uncertainty measure)
    # Entropy = -sum(p * log(p))
    epsilon = 1e-10  # Avoid log(0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
    
    return mean_probs, variance, entropy


def estimate_uncertainty_for_batch(model, images, num_samples=None, device=None):
    """
    Estimate uncertainty for a batch of images using Monte Carlo Dropout.
    
    Args:
        model: Trained model
        images: Batch of image tensors
        num_samples: Number of MC samples (default from config)
        device: Device to run on
        
    Returns:
        Dictionary with predictions, confidence, and uncertainty metrics
    """
    if num_samples is None:
        num_samples = MC_DROPOUT_SAMPLES
    if device is None:
        device = get_device()
    
    # Get MC predictions
    mean_probs, variance, entropy = monte_carlo_dropout(model, images, num_samples, device)
    
    # Get predicted class and confidence
    predicted_classes = np.argmax(mean_probs, axis=1)
    confidence_scores = np.max(mean_probs, axis=1)
    
    # Compute epistemic uncertainty (variance of predictions)
    epistemic_uncertainty = np.mean(variance, axis=1)  # Average variance across classes
    
    # Aleatoric uncertainty (entropy of mean prediction)
    aleatoric_uncertainty = entropy
    
    # Total uncertainty (combination)
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    
    results = {
        'predicted_classes': predicted_classes.tolist(),
        'confidence': confidence_scores.tolist(),
        'mean_probabilities': mean_probs.tolist(),
        'epistemic_uncertainty': epistemic_uncertainty.tolist(),
        'aleatoric_uncertainty': aleatoric_uncertainty.tolist(),
        'total_uncertainty': total_uncertainty.tolist(),
        'variance': variance.tolist()
    }
    
    return results


def estimate_uncertainty_for_single(model, image_tensor, num_samples=None, device=None):
    """
    Estimate uncertainty for a single image.
    
    Args:
        model: Trained model
        image_tensor: Single image tensor (C, H, W) or (1, C, H, W)
        num_samples: Number of MC samples
        device: Device to run on
        
    Returns:
        Dictionary with prediction, confidence, and uncertainty
    """
    if num_samples is None:
        num_samples = MC_DROPOUT_SAMPLES
    if device is None:
        device = get_device()
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Get batch results
    batch_results = estimate_uncertainty_for_batch(model, image_tensor, num_samples, device)
    
    # Extract single result
    result = {
        'predicted_class_idx': batch_results['predicted_classes'][0],
        'confidence': batch_results['confidence'][0],
        'mean_probabilities': batch_results['mean_probabilities'][0],
        'epistemic_uncertainty': batch_results['epistemic_uncertainty'][0],
        'aleatoric_uncertainty': batch_results['aleatoric_uncertainty'][0],
        'total_uncertainty': batch_results['total_uncertainty'][0]
    }
    
    return result







