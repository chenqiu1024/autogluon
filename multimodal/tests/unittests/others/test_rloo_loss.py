import torch
import pytest
from autogluon.multimodal.optim.losses.rloo_loss import RLOOLoss

def test_rloo_loss_forward():
    # Batch size 2, 1 channel, 4x4 image
    B, C, H, W = 2, 1, 4, 4
    logits = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    
    # Initialize loss
    loss_func = RLOOLoss(k=4, rloo_weight=1.0, structure_weight=1.0)
    
    # Forward pass
    loss = loss_func(logits, target)
    
    # Check if loss is a scalar
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()

def test_rloo_loss_zero_weight():
    # If rloo_weight is 0, should be equal to structure loss
    B, C, H, W = 2, 1, 4, 4
    logits = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    
    loss_func = RLOOLoss(k=4, rloo_weight=0.0, structure_weight=1.0)
    loss = loss_func(logits, target)
    
    from autogluon.multimodal.optim.losses.structure_loss import StructureLoss
    struct_loss_func = StructureLoss()
    expected_loss = struct_loss_func(logits, target)
    
    assert torch.allclose(loss, expected_loss)

def test_rloo_loss_improvement():
    # Test if optimization reduces loss
    # This is a stochastic test, so we set seed
    torch.manual_seed(42)
    
    B, C, H, W = 1, 1, 4, 4
    # Logits far from target
    target = torch.ones(B, C, H, W).float()
    logits = torch.randn(B, C, H, W, requires_grad=True) # Random logits
    
    optimizer = torch.optim.SGD([logits], lr=1.0)
    loss_func = RLOOLoss(k=10, rloo_weight=1.0, structure_weight=0.0) # Only RLOO
    
    initial_loss = loss_func(logits, target).item()
    
    # Perform a few steps
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        
    final_loss = loss_func(logits, target).item()
    
    # We expect loss to decrease (maximize IoU -> minimize negative IoU)
    # Note: RLOO is noisy, but with high LR and simple target it should improve
    assert final_loss < initial_loss
