"""
Utility functions for RL training scripts.

Provides helpers for loading Conv-LoRA models, collecting data, etc.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear


def load_trained_conv_lora_model(task_name, model_path=None, device='cuda'):
    """
    Load a trained Conv-LoRA model for RL experiments.
    
    Parameters
    ----------
    task_name
        Dataset name (e.g., 'isic2017', 'polyp')
    model_path
        Path to trained model checkpoint. If None, trains a new one.
    device
        Device to load model on
        
    Returns
    -------
    predictor
        MultiModalPredictor with trained Conv-LoRA model
    sam_model
        The underlying SAM model
    conv_lora_layers
        List of ConvLoRALinear layers with their names
    """
    if model_path and os.path.exists(model_path):
        # Load existing model
        print(f"Loading trained Conv-LoRA model from {model_path}")
        predictor = MultiModalPredictor.load(model_path)
    else:
        # Need to train a baseline model first
        raise ValueError(
            f"Trained Conv-LoRA model not found at {model_path}. "
            "Please train a baseline Conv-LoRA model first using run_semantic_segmentation.py"
        )
    
    # Extract SAM model
    sam_model = predictor._model
    if hasattr(sam_model, 'model'):
        sam_model = sam_model.model  # Unwrap if needed
    
    # Find all Conv-LoRA layers
    conv_lora_layers = []
    for name, module in sam_model.named_modules():
        if isinstance(module, ConvLoRALinear):
            conv_lora_layers.append((name, module))
    
    print(f"Found {len(conv_lora_layers)} Conv-LoRA layers")
    
    # Move to device
    sam_model = sam_model.to(device)
    sam_model.eval()
    
    return predictor, sam_model, conv_lora_layers


def expand_path(df, dataset_dir):
    """Expand relative paths in dataframe."""
    for col in ["image", "label"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def prepare_dataset(task_name, split='train'):
    """
    Load dataset for a task.
    
    Parameters
    ----------
    task_name
        Dataset name
    split
        'train', 'val', or 'test'
        
    Returns
    -------
    df
        DataFrame with image and label paths
    dataset_dir
        Dataset directory
    """
    dataset_dir = os.path.join(f"datasets/{task_name}", task_name)
    csv_path = os.path.join(dataset_dir, f"{split}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            f"Please run: python prepare_semantic_segmentation_datasets.py"
        )
    
    df = pd.read_csv(csv_path)
    df = expand_path(df, dataset_dir)
    
    return df, dataset_dir


def get_default_training_setting(dataset_name):
    """Get default training settings for a dataset."""
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10
    elif dataset_name == "polyp":
        validation_metric = "sm"
    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20
    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4
    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def compute_iou(pred_masks, gt_masks, threshold=0.5):
    """
    Compute IoU between predicted and ground truth masks.
    
    Parameters
    ----------
    pred_masks
        Predicted masks, shape (B, H, W) or (B, 1, H, W)
    gt_masks
        Ground truth masks, shape (B, H, W) or (B, 1, H, W)
    threshold
        Threshold for binarization
        
    Returns
    -------
    iou
        IoU score per sample, shape (B,)
    """
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.dim() == 4 and gt_masks.size(1) == 1:
        gt_masks = gt_masks.squeeze(1)
    
    # Binarize
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    # IoU
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = (pred_binary + gt_binary).clamp(max=1).sum(dim=(1, 2))
    
    iou = intersection / (union + 1e-8)
    
    return iou

