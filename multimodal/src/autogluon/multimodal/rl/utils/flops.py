"""
FLOPs estimation utilities for expert routing.

Used to compute the computational cost of different expert selections
for the reward function in RL-based routing.
"""

import torch


def compute_conv_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    height: int,
    width: int,
) -> float:
    """
    Compute theoretical FLOPs for a 2D convolution.
    
    FLOPs = 2 * in_channels * out_channels * kernel_size^2 * height * width
    
    Parameters
    ----------
    in_channels
        Input channels
    out_channels
        Output channels
    kernel_size
        Kernel size (assuming square kernel)
    height
        Output height
    width
        Output width
        
    Returns
    -------
    flops
        Estimated FLOPs
    """
    flops = 2 * in_channels * out_channels * (kernel_size ** 2) * height * width
    return float(flops)


def compute_expert_flops(
    gates: torch.Tensor,
    expert_configs: list,
    feature_shape: tuple,
) -> float:
    """
    Compute total FLOPs for a batch given gating decisions.
    
    Parameters
    ----------
    gates
        Gating weights, shape (B, M) where M is number of experts
    expert_configs
        List of dicts containing expert configurations:
        [{'in_c': int, 'out_c': int, 'kernel_size': int}, ...]
    feature_shape
        Shape of features (H, W)
        
    Returns
    -------
    total_flops
        Total FLOPs across batch
    """
    batch_size, num_experts = gates.shape
    height, width = feature_shape
    
    total_flops = 0.0
    
    for i in range(num_experts):
        # Count how many samples are routed to this expert
        active_samples = (gates[:, i] > 0).sum().item()
        
        if active_samples > 0 and i < len(expert_configs):
            config = expert_configs[i]
            conv_flops = compute_conv_flops(
                in_channels=config['in_c'],
                out_channels=config['out_c'],
                kernel_size=config.get('kernel_size', 3),
                height=height,
                width=width,
            )
            # Add GELU activation FLOPs (approximate as 8 ops per element)
            gelu_flops = 8 * config['out_c'] * height * width
            
            total_flops += active_samples * (conv_flops + gelu_flops)
    
    return total_flops


def compute_layer_flops_from_gates(
    all_layer_gates: list,
    expert_configs_per_layer: list,
    feature_shapes: list,
) -> float:
    """
    Compute total FLOPs across all Conv-LoRA layers.
    
    Parameters
    ----------
    all_layer_gates
        List of gates tensors, one per layer
    expert_configs_per_layer
        List of expert configs, one list per layer
    feature_shapes
        List of (H, W) tuples, one per layer
        
    Returns
    -------
    total_flops
        Total FLOPs across all layers
    """
    total_flops = 0.0
    
    for gates, configs, shape in zip(all_layer_gates, expert_configs_per_layer, feature_shapes):
        layer_flops = compute_expert_flops(gates, configs, shape)
        total_flops += layer_flops
    
    return total_flops

