"""
Test inference module for evaluating the trained model.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.config import VIT_MODEL_PATH, NUM_CLASSES, IDX_TO_CLASS, METRICS_DIR, DEVICE
from src.model_training import ViTClassifier
from src.utils import get_device


def load_trained_model(model_path=None, device=None):
    """
    Load trained ViT model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = VIT_MODEL_PATH
    if device is None:
        device = get_device()
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    # Initialize model
    model = ViTClassifier()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    if 'best_val_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model


def predict_batch(model, images, device=None):
    """
    Predict classes and probabilities for a batch of images.
    
    Args:
        model: Trained model
        images: Batch of image tensors
        device: Device to run on
        
    Returns:
        Tuple of (predicted_classes, probabilities)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()


def evaluate_test_set(model, test_loader, device=None, save_predictions=True):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to run on
        save_predictions: Whether to save predictions to file
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    if device is None:
        device = get_device()
    
    model.eval()
    all_predicted = []
    all_probabilities = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            predicted, probabilities = predict_batch(model, images, device)
            
            all_predicted.extend(predicted)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_predicted = np.array(all_predicted)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = 100 * np.mean(all_predicted == all_labels)
    
    # Prepare results
    results = {
        'accuracy': float(accuracy),
        'predictions': all_predicted.tolist(),
        'probabilities': all_probabilities.tolist(),
        'true_labels': all_labels.tolist(),
        'num_samples': len(all_labels)
    }
    
    # Save predictions
    if save_predictions:
        predictions_file = METRICS_DIR / "test_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved predictions to {predictions_file}")
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Test Samples: {len(all_labels)}")
    
    return results


def predict_single_image(model, image_tensor, device=None):
    """
    Predict class and probability for a single image.
    
    Args:
        model: Trained model
        image_tensor: Single image tensor (C, H, W)
        device: Device to run on
        
    Returns:
        Dictionary with predicted class, confidence, and all probabilities
    """
    if device is None:
        device = get_device()
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class_idx = predicted.item()
    confidence_score = confidence.item()
    prob_dist = probabilities[0].cpu().numpy()
    
    result = {
        'predicted_class': IDX_TO_CLASS[predicted_class_idx],
        'predicted_class_idx': int(predicted_class_idx),
        'confidence': float(confidence_score),
        'probabilities': {
            IDX_TO_CLASS[i]: float(prob_dist[i]) for i in range(NUM_CLASSES)
        }
    }
    
    return result


if __name__ == "__main__":
    # Test inference
    from src.data_preprocessing import get_dataloaders
    
    print("Loading test data...")
    _, _, test_loader = get_dataloaders()
    
    print("Loading model...")
    model = load_trained_model()
    
    print("Evaluating...")
    results = evaluate_test_set(model, test_loader)


