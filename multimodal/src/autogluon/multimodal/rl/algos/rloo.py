"""
RLOO Algorithm for Policy Gradient Optimization.

This module implements the RLOO (Reinforcement Learning with Leave-One-Out) algorithm
for training the layer selection policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np

from .reinforce import REINFORCE


class RLOO(REINFORCE):
    """
    RLOO (Reinforcement Learning with Leave-One-Out) algorithm.
    
    RLOO improves upon REINFORCE by using a leave-one-out baseline to reduce
    variance in the gradient estimation. For each input, K samples are generated,
    and the baseline for the i-th sample is the average reward of the other K-1 samples.
    
    Parameters
    ----------
    policy : nn.Module
        Policy network (LayerSelectionPolicy)
    optimizer : torch.optim.Optimizer
        Optimizer for policy parameters
    k : int
        Number of samples per input (default: 4)
    entropy_coef : float
        Coefficient for entropy regularization (default: 0.01)
    max_grad_norm : float
        Maximum gradient norm for clipping (default: 1.0)
    device : str
        Device to run on (default: 'cuda')
    """
    
    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        k: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
    ):
        super().__init__(
            policy=policy,
            optimizer=optimizer,
            baseline_decay=0.0,  # Not used in RLOO
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.k = k
    
    def update(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        layer_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one RLOO update.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities of actions, shape [B * K]
            (Flattened from [B, K])
        rewards : torch.Tensor
            Rewards for each sample, shape [B * K]
            (Flattened from [B, K])
        layer_probs : torch.Tensor, optional
            Layer probabilities for entropy calculation, shape [B * K, num_layers]
        
        Returns
        -------
        info : dict
            Dictionary containing loss and statistics
        """
        self.num_updates += 1
        
        # Reshape to [B, K] to compute leave-one-out baseline
        # Assumes the input is flattened batch of K samples per image
        # B is inferred from total size and K
        total_samples = rewards.shape[0]
        batch_size = total_samples // self.k
        
        if total_samples % self.k != 0:
            raise ValueError(f"Total samples {total_samples} must be divisible by k={self.k}")
            
        rewards_reshaped = rewards.view(batch_size, self.k)
        log_probs_reshaped = log_probs.view(batch_size, self.k)
        
        # Compute leave-one-out baseline
        # baseline[i] = (sum(rewards) - rewards[i]) / (K - 1)
        sum_rewards = rewards_reshaped.sum(dim=1, keepdim=True)  # [B, 1]
        baselines = (sum_rewards - rewards_reshaped) / (self.k - 1)  # [B, K]
        
        # Compute advantages
        advantages = rewards_reshaped - baselines  # [B, K]
        
        # Flatten back for loss computation
        advantages = advantages.view(-1)
        
        # Normalize advantages (optional but recommended)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus
        entropy_loss = 0.0
        if self.entropy_coef > 0 and layer_probs is not None:
            entropy = -(
                layer_probs * torch.log(layer_probs + 1e-8) +
                (1 - layer_probs) * torch.log(1 - layer_probs + 1e-8)
            ).sum(dim=1).mean()
            entropy_loss = -self.entropy_coef * entropy
            
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item(),
            'mean_advantage': advantages.mean().item(),
            'baseline': baselines.mean().item(),
            'grad_norm': grad_norm.item(),
        }
