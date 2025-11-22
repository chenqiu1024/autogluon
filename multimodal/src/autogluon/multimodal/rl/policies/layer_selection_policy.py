"""
Layer Selection Policy for RL-based Conv-LoRA Layer Selection.

This module implements a policy network that decides which Conv-LoRA layers
to activate for each input image in medical image segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LayerSelectionPolicy(nn.Module):
    """
    Policy network for selecting which Conv-LoRA layers to activate.
    
    The policy takes patch embeddings as input and outputs probabilities
    for activating each of the 32 transformer layers.
    
    Architecture:
        Input: patch embeddings [B, H, W, C]
        -> Global Average Pooling -> [B, C]
        -> MLP(C -> 256 -> 128 -> num_layers) -> [B, num_layers]
        -> Sigmoid -> probabilities [B, num_layers]
    
    Parameters
    ----------
    hidden_dim : int
        Dimension of the input patch embeddings (default: 1280 for SAM-ViT-Huge)
    num_layers : int
        Number of transformer layers (default: 32 for SAM-ViT-Huge)
    init_bias : float
        Initial bias value for the output layer. Positive values encourage
        activating all layers initially (default: 2.0)
    """
    
    def __init__(
        self,
        hidden_dim: int = 1280,
        num_layers: int = 32,
        init_bias: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head: MLP
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_layers),
        )
        
        # Initialize output layer with positive bias to encourage exploration
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.constant_(self.policy_head[-1].bias, init_bias)
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy network.
        
        Parameters
        ----------
        patch_embeddings : torch.Tensor
            Patch embeddings from SAM, shape [B, H, W, C]
        deterministic : bool
            If True, use deterministic policy (threshold at 0.5)
            If False, sample from Bernoulli distribution (default: False)
        temperature : float
            Temperature for Gumbel-Sigmoid sampling (default: 1.0)
            Lower temperature -> more deterministic
        
        Returns
        -------
        layer_mask : torch.Tensor
            Binary mask for layer selection, shape [B, num_layers]
        layer_probs : torch.Tensor
            Probabilities for each layer, shape [B, num_layers]
        log_probs : torch.Tensor
            Log probabilities of the sampled actions, shape [B]
        """
        B, H, W, C = patch_embeddings.shape
        
        # Global feature extraction: [B, H, W, C] -> [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        features = patch_embeddings.permute(0, 3, 1, 2)  # [B, C, H, W]
        features = self.global_pool(features)  # [B, C, 1, 1]
        features = features.view(B, -1)  # [B, C]
        
        # Policy head: [B, C] -> [B, num_layers]
        logits = self.policy_head(features)  # [B, num_layers]
        layer_probs = torch.sigmoid(logits)  # [B, num_layers]
        
        if deterministic:
            # Deterministic policy: threshold at 0.5
            layer_mask = (layer_probs > 0.5).float()
            # Log prob for deterministic action
            log_probs = torch.log(
                layer_probs * layer_mask + (1 - layer_probs) * (1 - layer_mask) + 1e-8
            ).sum(dim=1)
        else:
            # Stochastic policy: sample from Bernoulli
            if temperature != 1.0:
                # Apply temperature scaling
                layer_probs_temp = torch.sigmoid(logits / temperature)
            else:
                layer_probs_temp = layer_probs
            
            # Sample binary mask
            layer_mask = torch.bernoulli(layer_probs_temp)
            
            # Compute log probabilities for REINFORCE
            # log P(a) = sum_i [a_i * log(p_i) + (1-a_i) * log(1-p_i)]
            log_probs = (
                layer_mask * torch.log(layer_probs + 1e-8) +
                (1 - layer_mask) * torch.log(1 - layer_probs + 1e-8)
            ).sum(dim=1)  # [B]
        
        return layer_mask, layer_probs, log_probs
    
    def get_entropy(self, layer_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy for exploration bonus.
        
        Parameters
        ----------
        layer_probs : torch.Tensor
            Probabilities for each layer, shape [B, num_layers]
        
        Returns
        -------
        entropy : torch.Tensor
            Entropy of the policy, shape [B]
        """
        # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
        entropy = -(
            layer_probs * torch.log(layer_probs + 1e-8) +
            (1 - layer_probs) * torch.log(1 - layer_probs + 1e-8)
        ).sum(dim=1)  # [B]
        return entropy
    
    def get_layer_statistics(self, layer_probs: torch.Tensor) -> dict:
        """
        Get statistics about layer selection for monitoring.
        
        Parameters
        ----------
        layer_probs : torch.Tensor
            Probabilities for each layer, shape [B, num_layers]
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - mean_num_layers: average number of activated layers
            - layer_activation_freq: activation frequency per layer [num_layers]
        """
        with torch.no_grad():
            expected_num_layers = layer_probs.sum(dim=1).mean().item()
            layer_activation_freq = layer_probs.mean(dim=0).cpu().numpy()
        
        return {
            'mean_num_layers': expected_num_layers,
            'layer_activation_freq': layer_activation_freq,
        }

