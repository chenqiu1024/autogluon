"""
Forward hooks for capturing intermediate features.

Used for optional visualization of encoder outputs and expert activations
during inference or debugging.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class FeatureHook:
    """
    Context manager to attach forward hooks and capture intermediate features.
    
    Example:
    -------
    with FeatureHook(model, ['layer1', 'layer2']) as hook:
        output = model(input)
        features = hook.get_features()
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        model
            Model to attach hooks to
        target_layers
            List of layer names to capture (if None, captures all)
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []
        
    def __enter__(self):
        """Attach hooks when entering context."""
        self._attach_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks when exiting context."""
        self._remove_hooks()
    
    def _attach_hooks(self):
        """Attach forward hooks to target layers."""
        for name, module in self.model.named_modules():
            if self.target_layers is None or name in self.target_layers:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _make_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Store output (clone and detach to avoid gradient issues)
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach().clone()
            elif isinstance(output, tuple):
                self.features[name] = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output)
            else:
                self.features[name] = output
        return hook_fn
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get captured features."""
        return self.features
    
    def clear_features(self):
        """Clear stored features."""
        self.features = {}


def downsample_feature_map(
    feature_map: torch.Tensor,
    target_size: tuple = (64, 64),
) -> torch.Tensor:
    """
    Downsample a feature map for visualization.
    
    Parameters
    ----------
    feature_map
        Feature map, shape (B, C, H, W)
    target_size
        Target spatial size (H', W')
        
    Returns
    -------
    downsampled
        Downsampled feature map, shape (B, C, H', W')
    """
    if feature_map.size(2) <= target_size[0] and feature_map.size(3) <= target_size[1]:
        return feature_map
    
    return torch.nn.functional.interpolate(
        feature_map,
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )


def feature_map_to_heatmap(
    feature_map: torch.Tensor,
    reduce_channels: str = 'mean',
) -> torch.Tensor:
    """
    Convert multi-channel feature map to single-channel heatmap.
    
    Parameters
    ----------
    feature_map
        Feature map, shape (B, C, H, W) or (C, H, W)
    reduce_channels
        How to reduce channels: 'mean', 'max', or 'norm'
        
    Returns
    -------
    heatmap
        Single-channel heatmap, shape (B, H, W) or (H, W)
    """
    if feature_map.dim() == 3:
        # (C, H, W) -> (H, W)
        if reduce_channels == 'mean':
            heatmap = feature_map.mean(dim=0)
        elif reduce_channels == 'max':
            heatmap = feature_map.max(dim=0)[0]
        elif reduce_channels == 'norm':
            heatmap = feature_map.norm(dim=0)
        else:
            raise ValueError(f"Unknown reduce_channels: {reduce_channels}")
    elif feature_map.dim() == 4:
        # (B, C, H, W) -> (B, H, W)
        if reduce_channels == 'mean':
            heatmap = feature_map.mean(dim=1)
        elif reduce_channels == 'max':
            heatmap = feature_map.max(dim=1)[0]
        elif reduce_channels == 'norm':
            heatmap = feature_map.norm(dim=1)
        else:
            raise ValueError(f"Unknown reduce_channels: {reduce_channels}")
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {feature_map.dim()}D")
    
    return heatmap

