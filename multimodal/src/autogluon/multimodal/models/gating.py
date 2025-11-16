"""
Gating mechanisms for Conv-LoRA MoE.

Provides a unified interface for different gating strategies:
- NoisyTopKGate: Wraps the existing MoEGate (Noisy Top-K from Conv-LoRA paper)
- RLGate: RL-based gating using a learned routing policy
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Protocol


class BaseGate(Protocol):
    """
    Protocol for gating mechanisms in Conv-LoRA MoE.
    
    All gating implementations should conform to this interface.
    """
    
    def forward(
        self,
        feats: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute gating values for expert selection.
        
        Parameters
        ----------
        feats
            Input features, shape (B, C, H, W)
        layer_idx
            Layer index (optional, used by RL gate)
            
        Returns
        -------
        gates
            Gating weights, shape (B, M) where M is number of experts
        aux_loss
            Auxiliary loss (e.g., load balancing)
        info_dict
            Additional info (logits, logprobs, actions, etc.)
        """
        ...


class NoisyTopKGate(nn.Module):
    """
    Wrapper around the existing MoEGate to conform to BaseGate protocol.
    
    This implements the Noisy Top-K gating from the Conv-LoRA paper (Sec. 3.2).
    It serves as the reference policy for KL regularization in RL-based routing.
    """
    
    def __init__(self, moe_gate):
        """
        Parameters
        ----------
        moe_gate
            An instance of MoEGate from adaptation_layers.py
        """
        super().__init__()
        self.moe_gate = moe_gate
        
    def forward(
        self,
        feats: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass using Noisy Top-K gating.
        
        Parameters
        ----------
        feats
            Input features, shape (B, C, H, W)
        layer_idx
            Unused, for interface compatibility
            
        Returns
        -------
        gates
            Gating weights, shape (B, M)
        aux_loss
            Load balancing loss
        info_dict
            Contains 'logits' for KL computation
        """
        gates, aux_loss = self.moe_gate(feats)
        
        # Reconstruct logits for KL computation (approximate from gates)
        # Note: This is an approximation since gates are already processed by softmax
        # For more accurate KL, we'd need to modify MoEGate to return pre-softmax logits
        info_dict = {
            'gates': gates,
            'aux_loss': aux_loss,
        }
        
        return gates, aux_loss, info_dict
    
    def get_ref_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Get reference logits for KL computation.
        
        This is used by RLGate to compute KL divergence to the reference policy.
        
        Parameters
        ----------
        feats
            Input features, shape (B, C, H, W)
            
        Returns
        -------
        logits
            Pre-softmax logits, shape (B, M)
        """
        batch_size = feats.shape[0]
        feats_S = self.moe_gate.gap(feats).view(batch_size, -1)
        clean_logits = feats_S @ self.moe_gate.w_gate
        return clean_logits


class RLGate(nn.Module):
    """
    RL-based gating using a learned routing policy.
    
    This replaces Noisy Top-K with a policy network that learns to route
    tokens to experts based on GRPO training with PERL regularizers.
    """
    
    def __init__(
        self,
        routing_policy,
        reference_gate: Optional[NoisyTopKGate] = None,
        k: int = 1,
        compute_kl: bool = True,
    ):
        """
        Parameters
        ----------
        routing_policy
            The routing policy network (RoutingPolicy instance)
        reference_gate
            Reference Noisy Top-K gate for KL regularization
        k
            Number of experts to select (Top-K)
        compute_kl
            Whether to compute KL to reference during training
        """
        super().__init__()
        self.routing_policy = routing_policy
        self.reference_gate = reference_gate
        self.k = k
        self.compute_kl = compute_kl
        
    def forward(
        self,
        feats: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass using RL-based routing.
        
        Parameters
        ----------
        feats
            Input features, shape (B, C, H, W)
        layer_idx
            Layer index, used by routing policy
            
        Returns
        -------
        gates
            Gating weights, shape (B, M)
        aux_loss
            Auxiliary loss (KL to reference + optional load balance)
        info_dict
            Contains logits, logprobs, actions for GRPO update
        """
        # Use stored layer_idx if not provided
        if layer_idx is None and hasattr(self, '_layer_idx'):
            layer_idx = self._layer_idx
        
        # Get policy logits
        logits = self.routing_policy(feats, layer_idx)  # (B, M)
        
        # Sample actions (Top-K)
        if self.training:
            # During training, sample from distribution for exploration
            if self.k == 1:
                # Categorical sampling for Top-1
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()  # (B,)
                logprobs = dist.log_prob(actions)  # (B,)
                
                # Create one-hot gates
                gates = torch.zeros_like(logits).scatter_(1, actions.unsqueeze(1), 1.0)
            else:
                # Gumbel-Softmax for Top-K (straight-through estimator)
                gates = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
                # For logprobs, use the actual selected experts
                _, top_k_indices = logits.topk(self.k, dim=1)
                log_probs_per_expert = torch.log_softmax(logits, dim=1)
                logprobs = log_probs_per_expert.gather(1, top_k_indices).sum(dim=1)  # (B,)
                actions = top_k_indices[:, 0]  # Use first expert as representative action
        else:
            # During inference, use greedy selection
            if self.k == 1:
                actions = logits.argmax(dim=1)  # (B,)
                gates = torch.zeros_like(logits).scatter_(1, actions.unsqueeze(1), 1.0)
                logprobs = torch.zeros(logits.size(0), device=logits.device)
            else:
                _, top_k_indices = logits.topk(self.k, dim=1)
                gates = torch.zeros_like(logits).scatter_(1, top_k_indices, 1.0 / self.k)
                actions = top_k_indices[:, 0]
                logprobs = torch.zeros(logits.size(0), device=logits.device)
        
        # Compute KL to reference policy
        kl_loss = torch.tensor(0.0, device=logits.device)
        if self.training and self.compute_kl and self.reference_gate is not None:
            with torch.no_grad():
                ref_logits = self.reference_gate.get_ref_logits(feats)
            
            # KL(π || π_ref)
            log_probs = torch.log_softmax(logits, dim=1)
            ref_log_probs = torch.log_softmax(ref_logits, dim=1)
            kl_loss = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=1).mean()
        
        # Auxiliary loss is KL loss (load balancing can be added to reward)
        aux_loss = kl_loss
        
        # Info dict for GRPO update
        info_dict = {
            'logits': logits,
            'logprobs': logprobs,
            'actions': actions,
            'gates': gates,
            'kl_loss': kl_loss,
        }
        
        return gates, aux_loss, info_dict

