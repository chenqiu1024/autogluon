
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

# Mock REINFORCE
class REINFORCE:
    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = 'cpu',
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.baseline = None
        self.num_updates = 0

# RLOO Implementation (Copied from rloo.py)
class RLOO(REINFORCE):
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
            baseline_decay=0.0,
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
        self.num_updates += 1
        
        total_samples = rewards.shape[0]
        batch_size = total_samples // self.k
        
        if total_samples % self.k != 0:
            raise ValueError(f"Total samples {total_samples} must be divisible by k={self.k}")
            
        rewards_reshaped = rewards.view(batch_size, self.k)
        log_probs_reshaped = log_probs.view(batch_size, self.k)
        
        # Compute leave-one-out baseline
        sum_rewards = rewards_reshaped.sum(dim=1, keepdim=True)
        baselines = (sum_rewards - rewards_reshaped) / (self.k - 1)
        
        # Compute advantages
        advantages = rewards_reshaped - baselines
        
        # Flatten back
        advantages = advantages.view(-1)
        
        # Normalize advantages
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
            'baseline': baselines.mean().item(),
        }

def test_rloo_update():
    print("Testing RLOO update...")
    
    # Setup
    device = 'cpu'
    policy = nn.Linear(10, 4) # Dummy policy
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    rloo = RLOO(policy, optimizer, k=2, entropy_coef=0.0, device=device)
    
    # Mock data
    # Batch size 2, K=2 -> Total 4 samples
    # Rewards: 
    # Image 1: Sample 1 (R=1.0), Sample 2 (R=0.0) -> Baseline 1 = 0.0, Baseline 2 = 1.0
    # Image 2: Sample 1 (R=0.5), Sample 2 (R=0.5) -> Baseline 1 = 0.5, Baseline 2 = 0.5
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
    
    # Log probs (dummy)
    log_probs = torch.tensor([-0.1, -0.2, -0.3, -0.4], requires_grad=True)
    
    # Update
    info = rloo.update(log_probs, rewards)
    
    print("Update info:", info)
    
    # Check baseline
    # Expected baselines: [0.0, 1.0, 0.5, 0.5]
    # Mean baseline: 0.5
    print(f"Computed baseline: {info['baseline']}")
    assert abs(info['baseline'] - 0.5) < 1e-5, f"Expected baseline 0.5, got {info['baseline']}"
    
    print("RLOO update test passed!")

if __name__ == "__main__":
    test_rloo_update()
