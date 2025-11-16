"""
GRPO (Group Relative Policy Optimization) trainer with PERL enhancements.

Implements GRPO algorithm with PERL-style regularizers:
- KL divergence to reference policy
- Adapter L2 regularization
- Optional Lagrangian for compute budget
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class GRPOTrainer:
    """
    GRPO trainer for routing policy with PERL regularizers.
    
    Loss components:
    - Policy gradient: -E[A * log π(a|s)]
    - Entropy: -β * H(π)
    - KL to reference: λ_kl * KL(π || π_ref)
    - Adapter L2: λ_l2 * ||Δadapter||²
    - Optional Lagrangian for compute budget
    """
    
    def __init__(
        self,
        policy,
        optimizer,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.05,
        adapter_l2_coef: float = 0.001,
        grad_clip: float = 1.0,
        device: str = 'cuda',
    ):
        """
        Parameters
        ----------
        policy
            Routing policy network
        optimizer
            Optimizer for policy
        entropy_coef
            Coefficient for entropy regularization
        kl_coef
            Coefficient for KL to reference
        adapter_l2_coef
            Coefficient for adapter L2 regularization
        grad_clip
            Gradient clipping value
        device
            Device to run on
        """
        self.policy = policy
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.adapter_l2_coef = adapter_l2_coef
        self.grad_clip = grad_clip
        self.device = device
        
    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Parameters
        ----------
        logprobs
            Log probabilities of actions, shape (B,)
        advantages
            Normalized advantages, shape (B,)
            
        Returns
        -------
        loss
            Policy gradient loss
        """
        # GRPO: -E[A * log π]
        loss = -(advantages * logprobs).mean()
        return loss
    
    def compute_entropy_loss(
        self,
        logits_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Parameters
        ----------
        logits_list
            List of logits tensors from all layers
            
        Returns
        -------
        entropy_loss
            Negative entropy (to be minimized)
        """
        total_entropy = 0.0
        count = 0
        
        for logits in logits_list:
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            total_entropy += entropy
            count += 1
        
        if count > 0:
            avg_entropy = total_entropy / count
        else:
            avg_entropy = torch.tensor(0.0, device=self.device)
        
        # Return negative entropy (we want to maximize entropy, i.e., minimize -entropy)
        return -avg_entropy
    
    def compute_kl_loss(
        self,
        kl_losses: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate KL losses from all layers.
        
        Parameters
        ----------
        kl_losses
            List of KL loss tensors from all layers
            
        Returns
        -------
        kl_loss
            Mean KL loss
        """
        if len(kl_losses) == 0:
            return torch.tensor(0.0, device=self.device)
        
        kl_stack = torch.stack(kl_losses)
        return kl_stack.mean()
    
    def compute_adapter_l2_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization on adapter parameters.
        
        Returns
        -------
        l2_loss
            Sum of squared parameter norms
        """
        adapter_params = self.policy.get_adapter_params()
        l2_loss = sum(p.pow(2).sum() for p in adapter_params)
        return l2_loss
    
    def update(
        self,
        rollout_data: Dict,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one GRPO update step.
        
        Parameters
        ----------
        rollout_data
            Dict containing:
            - 'logprobs': list of logprob tensors (one per sample)
            - 'info_dicts': list of info dicts with 'logits', 'kl_loss', etc.
        advantages
            Normalized advantages, shape (n_samples,)
            
        Returns
        -------
        metrics
            Dict of training metrics
        """
        self.optimizer.zero_grad()
        
        # Concatenate logprobs from all samples
        logprobs = torch.cat(rollout_data['logprobs'])  # (n_samples,)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # Policy gradient loss
        pg_loss = self.compute_policy_loss(logprobs, advantages_tensor)
        
        # Entropy loss
        logits_list = [info['logits'] for info in rollout_data['info_dicts'] if 'logits' in info]
        entropy_loss = self.compute_entropy_loss(logits_list)
        
        # KL loss
        kl_losses = [info['kl_loss'] for info in rollout_data['info_dicts'] if 'kl_loss' in info]
        kl_loss = self.compute_kl_loss(kl_losses)
        
        # Adapter L2 loss
        adapter_l2_loss = self.compute_adapter_l2_loss()
        
        # Total loss
        total_loss = (
            pg_loss +
            self.entropy_coef * entropy_loss +
            self.kl_coef * kl_loss +
            self.adapter_l2_coef * adapter_l2_loss
        )
        
        # Backward and optimize
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # Return metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/policy_gradient': pg_loss.item(),
            'loss/entropy': entropy_loss.item(),
            'loss/kl': kl_loss.item(),
            'loss/adapter_l2': adapter_l2_loss.item(),
            'grad_norm': self._get_grad_norm(),
        }
        
        return metrics
    
    def _get_grad_norm(self) -> float:
        """Get total gradient norm."""
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def set_coefficients(
        self,
        entropy_coef: Optional[float] = None,
        kl_coef: Optional[float] = None,
        adapter_l2_coef: Optional[float] = None,
    ):
        """
        Update loss coefficients during training.
        
        Useful for annealing schedules.
        """
        if entropy_coef is not None:
            self.entropy_coef = entropy_coef
        if kl_coef is not None:
            self.kl_coef = kl_coef
        if adapter_l2_coef is not None:
            self.adapter_l2_coef = adapter_l2_coef

