"""
REINFORCE Algorithm for Policy Gradient Optimization.

This module implements the REINFORCE (Williams, 1992) algorithm with
variance reduction techniques for training the layer selection policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np


class REINFORCE:
    """
    REINFORCE algorithm with baseline and entropy regularization.
    
    The algorithm uses policy gradients to optimize the layer selection policy
    based on segmentation performance rewards.
    
    Key features:
    - Moving average baseline for variance reduction
    - Entropy bonus for exploration
    - Gradient clipping for stability
    - Learning rate warmup
    
    Parameters
    ----------
    policy : nn.Module
        Policy network (LayerSelectionPolicy)
    optimizer : torch.optim.Optimizer
        Optimizer for policy parameters
    baseline_decay : float
        Decay rate for moving average baseline (default: 0.99)
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
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Moving average baseline
        self.baseline = None
        
        # Statistics
        self.num_updates = 0
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        baseline: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute returns (advantages) from rewards.
        
        Parameters
        ----------
        rewards : torch.Tensor
            Rewards for each sample, shape [B]
        baseline : float, optional
            Baseline value for variance reduction
        
        Returns
        -------
        advantages : torch.Tensor
            Advantage values (rewards - baseline), shape [B]
        """
        if baseline is None:
            baseline = self.baseline if self.baseline is not None else 0.0
        
        advantages = rewards - baseline
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update_baseline(self, rewards: torch.Tensor):
        """
        Update moving average baseline.
        
        Parameters
        ----------
        rewards : torch.Tensor
            Batch of rewards, shape [B]
        """
        mean_reward = rewards.mean().item()
        
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            self.baseline = (
                self.baseline_decay * self.baseline +
                (1 - self.baseline_decay) * mean_reward
            )
    
    def update(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        layer_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one policy gradient update.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities of actions, shape [B]
        rewards : torch.Tensor
            Rewards for each sample, shape [B]
        layer_probs : torch.Tensor, optional
            Layer probabilities for entropy calculation, shape [B, num_layers]
        
        Returns
        -------
        info : dict
            Dictionary containing loss and statistics
        """
        self.num_updates += 1
        
        # Compute advantages
        advantages = self.compute_returns(rewards)
        
        # Policy gradient loss: -E[log π(a|s) * A(s,a)]
        # Negative because we want to maximize reward (minimize negative reward)
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus for exploration
        entropy_loss = 0.0
        if self.entropy_coef > 0 and layer_probs is not None:
            # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
            entropy = -(
                layer_probs * torch.log(layer_probs + 1e-8) +
                (1 - layer_probs) * torch.log(1 - layer_probs + 1e-8)
            ).sum(dim=1).mean()
            entropy_loss = -self.entropy_coef * entropy  # Negative to maximize entropy
        
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
        
        # Update baseline
        self.update_baseline(rewards)
        
        # Return statistics
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item(),
            'mean_advantage': advantages.mean().item(),
            'baseline': self.baseline,
            'grad_norm': grad_norm.item(),
        }
    
    def get_lr_warmup_factor(self, warmup_steps: int = 100) -> float:
        """
        Compute learning rate warmup factor.
        
        Parameters
        ----------
        warmup_steps : int
            Number of warmup steps (default: 100)
        
        Returns
        -------
        warmup_factor : float
            Multiplicative factor for learning rate (0 to 1)
        """
        if self.num_updates < warmup_steps:
            return (self.num_updates + 1) / warmup_steps
        return 1.0
    
    def apply_lr_warmup(self, warmup_steps: int = 100):
        """
        Apply learning rate warmup.
        
        Parameters
        ----------
        warmup_steps : int
            Number of warmup steps (default: 100)
        """
        warmup_factor = self.get_lr_warmup_factor(warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= warmup_factor


class REINFORCEWithMultipleEvals:
    """
    REINFORCE variant that evaluates each action multiple times for variance reduction.
    
    This is useful when the reward signal is noisy. Each layer mask configuration
    is evaluated K times, and the mean reward is used for gradient computation.
    
    Parameters
    ----------
    policy : nn.Module
        Policy network (LayerSelectionPolicy)
    optimizer : torch.optim.Optimizer
        Optimizer for policy parameters
    num_eval_repeats : int
        Number of times to evaluate each action (default: 3)
    baseline_decay : float
        Decay rate for moving average baseline (default: 0.99)
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
        num_eval_repeats: int = 3,
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
    ):
        self.base_algo = REINFORCE(
            policy=policy,
            optimizer=optimizer,
            baseline_decay=baseline_decay,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.num_eval_repeats = num_eval_repeats
    
    def update(
        self,
        log_probs: torch.Tensor,
        rewards_list: List[torch.Tensor],
        layer_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one policy gradient update with multiple evaluations.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities of actions, shape [B]
        rewards_list : List[torch.Tensor]
            List of K reward tensors, each of shape [B]
        layer_probs : torch.Tensor, optional
            Layer probabilities for entropy calculation, shape [B, num_layers]
        
        Returns
        -------
        info : dict
            Dictionary containing loss and statistics
        """
        # Average rewards across evaluations
        mean_rewards = torch.stack(rewards_list).mean(dim=0)
        
        # Use base REINFORCE with mean rewards
        info = self.base_algo.update(log_probs, mean_rewards, layer_probs)
        
        # Add variance statistics
        if len(rewards_list) > 1:
            reward_variance = torch.stack(rewards_list).var(dim=0).mean().item()
            info['reward_variance'] = reward_variance
        
        return info
    
    @property
    def num_updates(self):
        return self.base_algo.num_updates
    
    @property
    def baseline(self):
        return self.base_algo.baseline

