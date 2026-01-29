"""
Utility functions for the bone marrow classification project.
Includes seeding, logging, and helper functions.
"""

import random
import numpy as np
import torch
import logging
from datetime import datetime
from pathlib import Path
import json

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name="bone_marrow_classifier", log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def save_dict_to_json(data, filepath):
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_dict_from_json(filepath):
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """
    Get the available device (CUDA or CPU) with explicit GPU verification.
    Includes detailed logging and assertion to ensure CUDA is available.
    
    Returns:
        torch.device object
        
    Raises:
        AssertionError: If CUDA is not available (training requires GPU)
    """
    # Explicit GPU verification with detailed logging
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"Selected device: {device}")
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Count: {gpu_count}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        return device
    else:
        print("Selected device: cpu")
        print("=" * 60)
        # Assertion: Training requires GPU for optimal performance
        assert False, "CUDA is not available! Training requires GPU. Please ensure CUDA is properly installed and accessible."
        return torch.device("cpu")

def format_time(seconds):
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"







