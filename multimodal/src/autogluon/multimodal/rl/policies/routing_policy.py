"""
Routing Policy for RL-based expert selection in Conv-LoRA.

This policy network learns to route tokens to experts based on spatial features
and layer index, replacing the Noisy Top-K gating with a trainable policy.
"""

from typing import Optional

import torch
import torch.nn as nn


class RoutingPolicy(nn.Module):
    """
    Lightweight MLP-based routing policy for expert selection.
    
    Architecture:
    1. Global Average Pooling on spatial features
    2. Concatenate with layer index embedding
    3. 2-3 layer MLP to produce logits over M experts
    
    This is trained with GRPO to maximize: IoU - α·FLOPs - β·imbalance
    """
    
    def __init__(
        self,
        num_experts: int,
        feature_dim: int,
        hidden_dim: int = 256,
        layer_embed_dim: int = 32,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        num_experts
            Number of experts (M)
        feature_dim
            Input feature dimension (C from Conv-LoRA's lora_res)
        hidden_dim
            Hidden dimension of MLP
        layer_embed_dim
            Embedding dimension for layer index
        num_layers
            Total number of layers in the model (for embedding)
        dropout
            Dropout probability
        """
        super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Global average pooling (applied to spatial dims)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Layer index embedding
        self.layer_embedding = nn.Embedding(num_layers, layer_embed_dim)
        
        # MLP head
        input_dim = feature_dim + layer_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        feats: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute logits over experts for given features and layer.
        
        Parameters
        ----------
        feats
            Input features, shape (B, C, H, W)
        layer_idx
            Layer index (0 to num_layers-1). If None, uses 0.
            
        Returns
        -------
        logits
            Expert logits, shape (B, M)
        """
        batch_size = feats.size(0)
        
        # Global average pooling: (B, C, H, W) -> (B, C)
        pooled = self.gap(feats).view(batch_size, -1)
        
        # Layer embedding
        if layer_idx is None:
            layer_idx = 0
        layer_idx_tensor = torch.tensor(
            [layer_idx] * batch_size,
            dtype=torch.long,
            device=feats.device
        )
        layer_embed = self.layer_embedding(layer_idx_tensor)  # (B, layer_embed_dim)
        
        # Concatenate and pass through MLP
        combined = torch.cat([pooled, layer_embed], dim=1)  # (B, C + layer_embed_dim)
        logits = self.mlp(combined)  # (B, M)
        
        return logits
    
    def get_adapter_params(self):
        """
        Get adapter parameters for L2 regularization.
        
        Since the entire routing policy is considered an "adapter",
        this returns all parameters.
        """
        return list(self.parameters())
    
    def freeze_base(self):
        """
        Freeze base model (no-op for routing policy).
        
        This is here for interface compatibility. The routing policy
        has no "base" to freeze; it's entirely trainable.
        """
        pass

