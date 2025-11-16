"""
Visualization utilities for RL training.

Supports TensorBoard logging and local artifact saving (images, JSON stats).
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def log_scalars(
    writer: Optional['SummaryWriter'],
    scalars: Dict[str, float],
    global_step: int,
):
    """
    Log scalar values to TensorBoard.
    
    Parameters
    ----------
    writer
        TensorBoard SummaryWriter (can be None)
    scalars
        Dict of scalar name -> value
    global_step
        Global training step
    """
    if writer is None:
        return
    
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)


def log_histograms(
    writer: Optional['SummaryWriter'],
    histograms: Dict[str, torch.Tensor],
    global_step: int,
):
    """
    Log histograms to TensorBoard.
    
    Parameters
    ----------
    writer
        TensorBoard SummaryWriter (can be None)
    histograms
        Dict of histogram name -> tensor
    global_step
        Global training step
    """
    if writer is None:
        return
    
    for name, tensor in histograms.items():
        if tensor.numel() > 0:
            writer.add_histogram(name, tensor, global_step)


def save_expert_heatmap(
    gates: torch.Tensor,
    output_path: str,
    title: str = "Expert Usage",
):
    """
    Save expert usage as a bar chart or heatmap.
    
    Parameters
    ----------
    gates
        Gate tensor, shape (B, M) or list of such tensors
    output_path
        Path to save figure
    title
        Plot title
    """
    if isinstance(gates, list):
        gates = torch.cat(gates, dim=0)
    
    # Sum over batch to get per-expert usage
    if isinstance(gates, torch.Tensor):
        expert_usage = gates.sum(dim=0).cpu().numpy()
    else:
        expert_usage = np.array(gates).sum(axis=0)
    
    num_experts = len(expert_usage)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_experts), expert_usage)
    ax.set_xlabel('Expert Index')
    ax.set_ylabel('Total Usage')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json_stats(
    stats: Dict,
    output_path: str,
):
    """
    Save statistics as JSON file.
    
    Parameters
    ----------
    stats
        Dict of statistics (will be JSON serialized)
    output_path
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensors and numpy arrays to lists
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        else:
            return obj
    
    stats_serializable = convert(stats)
    
    with open(output_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)


def create_tensorboard_writer(logdir: str) -> Optional['SummaryWriter']:
    """
    Create TensorBoard writer if available.
    
    Parameters
    ----------
    logdir
        Log directory
        
    Returns
    -------
    writer
        SummaryWriter instance or None if tensorboard not available
    """
    if SummaryWriter is None:
        print("Warning: tensorboard not available, logging disabled")
        return None
    
    os.makedirs(logdir, exist_ok=True)
    return SummaryWriter(logdir)

