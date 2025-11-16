"""
Rollout buffer and advantage computation for GRPO.

Implements group-wise advantage normalization as per GRPO algorithm.
"""

from typing import Dict, List

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer to store rollout data for GRPO update.
    
    Stores states, actions, rewards, logprobs, and auxiliary info
    for a batch of rollouts.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.ious = []
        self.flops = []
        self.gates = []
        self.info_dicts = []
    
    def add(
        self,
        state: Dict,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        iou: float,
        flops: float,
        gates: List[torch.Tensor],
        info_dict: Dict,
    ):
        """
        Add a single rollout to the buffer.
        
        Parameters
        ----------
        state
            State dict (not used in current GRPO but kept for extensibility)
        action
            Actions taken (per layer)
        logprob
            Log probabilities of actions
        reward
            Reward received
        iou
            IoU metric
        flops
            FLOPs consumed
        gates
            Gating decisions per layer
        info_dict
            Additional info
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.ious.append(iou)
        self.flops.append(flops)
        self.gates.append(gates)
        self.info_dicts.append(info_dict)
    
    def get(self):
        """Get all data from buffer."""
        return {
            'states': self.states,
            'actions': self.actions,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'ious': self.ious,
            'flops': self.flops,
            'gates': self.gates,
            'info_dicts': self.info_dicts,
        }
    
    def __len__(self):
        return len(self.rewards)


def compute_group_advantages(
    rewards: List[float],
    group_size: int = None,
) -> np.ndarray:
    """
    Compute group-wise normalized advantages (GRPO core).
    
    For each group, advantages are: A_i = (R_i - mean(R_group)) / std(R_group)
    
    Parameters
    ----------
    rewards
        List of rewards
    group_size
        Size of each group. If None, treats all rewards as one group.
        
    Returns
    -------
    advantages
        Normalized advantages, same length as rewards
    """
    rewards = np.array(rewards)
    n = len(rewards)
    
    if group_size is None or group_size >= n:
        # Single group
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        if std_r < 1e-8:
            std_r = 1.0
        advantages = (rewards - mean_r) / std_r
    else:
        # Multiple groups
        advantages = np.zeros_like(rewards)
        num_groups = (n + group_size - 1) // group_size
        
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, n)
            
            group_rewards = rewards[start_idx:end_idx]
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards)
            if std_r < 1e-8:
                std_r = 1.0
            
            advantages[start_idx:end_idx] = (group_rewards - mean_r) / std_r
    
    return advantages


def compute_imbalance(gates_list: List[torch.Tensor]) -> float:
    """
    Compute load imbalance across experts (coefficient of variation squared).
    
    Parameters
    ----------
    gates_list
        List of gate tensors from all layers, each shape (B, M)
        
    Returns
    -------
    imbalance
        CV^2 measure of imbalance (0 = perfectly balanced)
    """
    if len(gates_list) == 0:
        return 0.0
    
    # Concatenate all gates and sum over batch dimension
    all_gates = torch.cat(gates_list, dim=0)  # (B*L, M)
    expert_loads = all_gates.sum(dim=0)  # (M,)
    
    # CV^2 = var / (mean^2 + eps)
    mean_load = expert_loads.mean()
    var_load = expert_loads.var()
    eps = 1e-10
    
    cv_squared = (var_load / (mean_load ** 2 + eps)).item()
    
    return cv_squared

