"""
Checkpoint utilities for RL training.

Supports saving/loading policy, optimizer, dual variables, RNG state, etc.
for resumable training.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state_dict: Dict[str, Any],
    output_path: str,
):
    """
    Save checkpoint to disk.
    
    Parameters
    ----------
    state_dict
        Dict containing:
        - 'policy': policy state dict
        - 'optimizer': optimizer state dict
        - 'scaler': AMP scaler state dict (if using mixed precision)
        - 'dual_alpha': dual variable for Lagrangian (if using)
        - 'global_step': current training step
        - 'rng_state': torch RNG state
        - 'config': training config
    output_path
        Path to save checkpoint
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load checkpoint from disk.
    
    Parameters
    ----------
    checkpoint_path
        Path to checkpoint file
    device
        Device to load tensors to
        
    Returns
    -------
    state_dict
        Loaded checkpoint dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict


def restore_training_state(
    policy,
    optimizer,
    checkpoint_path: str,
    scaler: Optional[Any] = None,
    device: str = 'cpu',
) -> tuple:
    """
    Restore full training state from checkpoint.
    
    Parameters
    ----------
    policy
        Policy network (will be loaded in-place)
    optimizer
        Optimizer (will be loaded in-place)
    checkpoint_path
        Path to checkpoint
    scaler
        AMP scaler (optional, will be loaded if provided)
    device
        Device to load to
        
    Returns
    -------
    global_step
        Restored global step
    dual_alpha
        Restored dual variable (or 0.0 if not in checkpoint)
    config
        Training config from checkpoint
    """
    ckpt = load_checkpoint(checkpoint_path, device)
    
    # Load policy
    policy.load_state_dict(ckpt['policy'])
    
    # Load optimizer
    optimizer.load_state_dict(ckpt['optimizer'])
    
    # Load scaler if available
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    
    # Restore RNG state
    if 'rng_state' in ckpt:
        torch.set_rng_state(ckpt['rng_state'].cpu())
    
    # Get training state
    global_step = ckpt.get('global_step', 0)
    dual_alpha = ckpt.get('dual_alpha', 0.0)
    config = ckpt.get('config', {})
    
    return global_step, dual_alpha, config

