
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add autogluon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal.rl.algos.rloo import RLOO
from autogluon.multimodal.rl.policies import LayerSelectionPolicy

def test_rloo_update():
    print("Testing RLOO update...")
    
    # Setup
    device = 'cpu'
    policy = LayerSelectionPolicy(hidden_dim=128, num_layers=4).to(device)
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
