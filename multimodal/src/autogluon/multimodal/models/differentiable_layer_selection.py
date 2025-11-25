"""
Differentiable Layer Selection for Joint Training with Conv-LoRA.

This module implements Gumbel-Softmax based layer selection that allows
gradients to flow through discrete layer selection decisions, enabling
joint training of both the layer selection policy and Conv-LoRA parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GumbelLayerSelector(nn.Module):
    """
    Differentiable layer selection using Gumbel-Softmax trick.
    
    This module learns to select which Conv-LoRA layers to activate for each
    input image. Unlike standard RL approaches, this uses Gumbel-Softmax with
    straight-through estimators to make the binary selection differentiable,
    allowing joint training with Conv-LoRA parameters.
    
    Architecture:
        Input: patch embeddings [B, H, W, C]
        -> Global Average Pooling -> [B, C]
        -> MLP(C -> 128 -> num_layers) -> logits [B, num_layers]
        -> Gumbel-Softmax -> binary masks [B, num_layers]
    
    Parameters
    ----------
    input_dim : int
        Dimension of input patch embeddings (default: 1280 for SAM-ViT-Huge)
    num_layers : int
        Number of transformer layers to select from (default: 32)
    temperature : float
        Temperature for Gumbel-Softmax (default: 1.0)
        Higher temperature -> softer, more exploratory
        Lower temperature -> harder, more deterministic
    init_bias : float
        Initial bias for output layer (default: 0.0 for 50% activation probability)
    """
    
    def __init__(
        self,
        input_dim: int = 1280,
        num_layers: int = 32,
        temperature: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.temperature = temperature
        
        # Lightweight policy network
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_layers),
        )
        
        # Initialize output layer
        # weight=0 ensures initial logits are close to 0
        # bias=init_bias controls initial activation probability
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.constant_(self.selector[-1].bias, init_bias)
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        hard: bool = True,
        force_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate layer selection masks.
        
        Parameters
        ----------
        patch_embeddings : torch.Tensor
            Patch embeddings from SAM, shape [B, H, W, C]
        hard : bool
            If True, use hard masks {0,1} with straight-through estimator
            If False, use soft masks [0,1] (for debugging/analysis)
        force_masks : torch.Tensor, optional
            If provided, use these masks instead of generating (for warmstart)
            Shape: [B, num_layers]
        
        Returns
        -------
        layer_masks : torch.Tensor
            Binary (or soft) layer selection masks, shape [B, num_layers]
        logits : torch.Tensor
            Raw logits before Gumbel-Softmax, shape [B, num_layers]
        """
        B, H, W, C = patch_embeddings.shape
        
        # If forced masks provided (warmstart phase), return them
        if force_masks is not None:
            # Still compute logits for monitoring
            features = patch_embeddings.permute(0, 3, 1, 2)  # [B, C, H, W]
            features = self.global_pool(features).view(B, -1)  # [B, C]
            logits = self.selector(features)  # [B, num_layers]
            return force_masks, logits
        
        # Global feature extraction
        features = patch_embeddings.permute(0, 3, 1, 2)  # [B, C, H, W]
        features = self.global_pool(features)  # [B, C, 1, 1]
        features = features.view(B, -1)  # [B, C]
        
        # Policy head
        logits = self.selector(features)  # [B, num_layers]
        
        if self.training:
            # Training: Gumbel-Softmax for differentiable sampling
            layer_masks = self.gumbel_sigmoid(logits, hard=hard)
        else:
            # Inference: Deterministic thresholding
            probs = torch.sigmoid(logits)
            layer_masks = (probs > 0.5).float()
        
        return layer_masks, logits
    
    def gumbel_sigmoid(
        self,
        logits: torch.Tensor,
        hard: bool = True,
    ) -> torch.Tensor:
        """
        Gumbel-Sigmoid sampling for binary layer selection.
        
        This implements the Gumbel-Softmax trick adapted for binary (Bernoulli)
        distributions, with straight-through estimator for hard sampling.
        
        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized layer selection scores, shape [B, num_layers]
        hard : bool
            If True, use straight-through estimator (forward: hard {0,1}, backward: soft gradients)
            If False, use soft masks [0,1]
        
        Returns
        -------
        masks : torch.Tensor
            Layer selection masks, shape [B, num_layers]
            Forward pass: {0, 1} if hard=True, [0, 1] if hard=False
            Backward pass: gradients flow through soft_masks
        """
        # Sample Gumbel noise
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
        
        # Add noise and apply temperature scaling
        gumbel_logits = (logits + gumbel_noise) / self.temperature
        
        # Sigmoid activation
        soft_masks = torch.sigmoid(gumbel_logits)
        
        if hard:
            # Straight-through estimator
            # Forward pass: use hard binary masks
            hard_masks = (soft_masks > 0.5).float()
            # Backward pass: gradients flow through soft_masks
            masks = hard_masks + (soft_masks - soft_masks.detach())
        else:
            # Soft masks (for analysis/debugging)
            masks = soft_masks
        
        return masks
    
    def get_selection_statistics(
        self,
        layer_masks: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict:
        """
        Compute statistics about layer selection for monitoring.
        
        Parameters
        ----------
        layer_masks : torch.Tensor
            Binary layer masks, shape [B, num_layers]
        logits : torch.Tensor
            Raw logits, shape [B, num_layers]
        
        Returns
        -------
        stats : dict
            Dictionary containing selection statistics
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            
            stats = {
                'mean_active_layers': layer_masks.sum(dim=1).mean().item(),
                'std_active_layers': layer_masks.sum(dim=1).std().item(),
                'prob_mean': probs.mean().item(),
                'prob_std': probs.std().item(),
                'prob_min': probs.min().item(),
                'prob_max': probs.max().item(),
                'layer_activation_freq': layer_masks.float().mean(dim=0).cpu().numpy(),
                'entropy': self.compute_entropy(probs).mean().item(),
            }
        
        return stats
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute binary entropy of the selection distribution.
        
        Parameters
        ----------
        probs : torch.Tensor
            Layer activation probabilities, shape [B, num_layers]
        
        Returns
        -------
        entropy : torch.Tensor
            Entropy per sample, shape [B]
        """
        # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
        entropy = -(
            probs * torch.log(probs + 1e-8) +
            (1 - probs) * torch.log(1 - probs + 1e-8)
        ).sum(dim=1)
        return entropy
    
    def set_temperature(self, temperature: float):
        """Update temperature for annealing."""
        self.temperature = temperature


class TemperatureScheduler:
    """
    Temperature annealing schedule for Gumbel-Softmax.
    
    Gradually decreases temperature during training to transition from
    soft (exploratory) to hard (deterministic) masks.
    
    Parameters
    ----------
    initial_temp : float
        Starting temperature (default: 1.0)
    final_temp : float
        Ending temperature (default: 0.1)
    anneal_start_epoch : int
        Epoch to start annealing (default: 4, after warmstart)
    anneal_end_epoch : int
        Epoch to finish annealing (default: 16)
    anneal_mode : str
        'linear' or 'exponential'
    """
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        anneal_start_epoch: int = 4,
        anneal_end_epoch: int = 16,
        anneal_mode: str = 'exponential',
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_start = anneal_start_epoch
        self.anneal_end = anneal_end_epoch
        self.mode = anneal_mode
    
    def get_temperature(self, epoch: int) -> float:
        """Get temperature for current epoch."""
        if epoch < self.anneal_start:
            return self.initial_temp
        elif epoch >= self.anneal_end:
            return self.final_temp
        else:
            # Compute progress
            progress = (epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)
            
            if self.mode == 'linear':
                temp = self.initial_temp + progress * (self.final_temp - self.initial_temp)
            elif self.mode == 'exponential':
                # Exponential decay
                temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
            else:
                raise ValueError(f"Unknown anneal_mode: {self.mode}")
            
            return temp
    
    def should_update_temperature(self, epoch: int) -> bool:
        """Check if temperature should be updated at this epoch."""
        return self.anneal_start <= epoch <= self.anneal_end

