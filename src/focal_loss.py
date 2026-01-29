"""
Focal Loss implementation for addressing class imbalance and hard examples.
Focal Loss focuses learning on hard examples by down-weighting easy examples.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the predicted probability for the true class
    - alpha_t is the class weight (for class imbalance)
    - gamma is the focusing parameter (higher gamma = more focus on hard examples)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor (for class imbalance). If None, no class weighting.
                   Can be a float (single weight for all classes) or tensor of size [num_classes]
            gamma: Focusing parameter. Higher gamma focuses more on hard examples.
                   Default 2.0 as per original paper.
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Convert alpha to tensor if it's a single value
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                # Single value - will be handled in forward pass
                self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Model predictions (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Get predicted probabilities for the true class
        p_t = torch.exp(-ce_loss)  # p_t = exp(-CE_loss) = predicted prob for true class
        
        # Compute focal loss: (1 - p_t)^gamma * CE_loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self):
        return f'FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction})'


# Example usage:
# # With class weights
# class_weights = torch.tensor([1.0, 2.0, 1.5, ...])  # One weight per class
# focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
# 
# # Without class weights (only hard example focusing)
# focal_loss = FocalLoss(alpha=None, gamma=2.0)
# 
# # In training loop
# loss = focal_loss(outputs, labels)
