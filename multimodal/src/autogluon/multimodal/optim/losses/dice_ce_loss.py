"""
Dice + Cross-Entropy Loss for Multi-class Semantic Segmentation.

This combines Dice loss (which handles class imbalance well) with 
Cross-Entropy loss (which provides stable gradients) for multi-class
segmentation tasks like ACDC cardiac segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Combined Dice Loss and Cross-Entropy Loss for multi-class segmentation.
    
    Supports two input formats:
    1. Standard logits: [B, C, H, W] - will apply softmax internally
    2. Log probabilities: [B, C, H, W] - detected when values are all <= 0
    
    Parameters
    ----------
    num_classes : int
        Number of classes (including background).
    dice_weight : float
        Weight for the Dice loss component. Default: 1.0
    ce_weight : float
        Weight for the Cross-Entropy loss component. Default: 1.0
    smooth : float
        Smoothing factor for Dice loss to avoid division by zero. Default: 1.0
    ignore_index : int
        Label index to ignore (e.g., for padding). Default: -100
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        # Use NLLLoss for log probabilities, CrossEntropyLoss for logits
        self.nll_loss = nn.NLLLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Dice + CE loss.
        
        Parameters
        ----------
        input : torch.Tensor
            Model predictions of shape [B, C, H, W] where C is num_classes.
            Can be either raw logits or log probabilities (auto-detected).
        target : torch.Tensor
            Ground truth labels of shape [B, H, W] with integer class labels,
            or [B, 1, H, W] which will be squeezed.
            
        Returns
        -------
        torch.Tensor
            Combined loss value.
        """
        # Handle target with channel dimension
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        # Ensure target is long type for CE loss
        target = target.long()
        
        # Detect if input is log probabilities (all values <= 0 and sum to ~1 after exp)
        # Log probabilities: values are negative, exp(values).sum(dim=1) ≈ 1
        is_log_prob = (input.max() <= 0)
        
        if is_log_prob:
            # Input is log probabilities, use NLLLoss
            ce_loss = self.nll_loss(input, target)
            # For dice loss, convert log probs to probs
            probs = torch.exp(input)
        else:
            # Input is raw logits, use CrossEntropyLoss
            ce_loss = self.ce_loss(input, target)
            # For dice loss, apply softmax
            probs = F.softmax(input, dim=1)
        
        # Dice loss using probabilities
        dice_loss = self._dice_loss_from_probs(probs, target)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss
    
    def _dice_loss_from_probs(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class Dice loss from probability predictions.
        
        Parameters
        ----------
        probs : torch.Tensor
            Class probabilities of shape [B, C, H, W].
        target : torch.Tensor
            Ground truth labels of shape [B, H, W] with integer class labels.
            
        Returns
        -------
        torch.Tensor
            Dice loss value (1 - mean Dice score).
        """
        B, C, H, W = probs.shape
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=C)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Compute Dice score per class
        dims = (0, 2, 3)  # Reduce over batch, height, width
        intersection = (probs * target_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes
        dice_loss = 1.0 - dice_score.mean()
        
        return dice_loss
    

