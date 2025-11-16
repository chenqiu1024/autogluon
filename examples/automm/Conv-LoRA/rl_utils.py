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
from autogluon.multimodal.models.sam import SAMForSemanticSegmentation


def load_trained_conv_lora_model(task_name, model_path=None, device='cuda'):
    """
    Load a trained Conv-LoRA model for RL experiments.
    
    Supports loading from:
    1. Complete MultiModalPredictor directory (has assets.json)
    2. Training directory with .ckpt files (mid-training)
    3. Specific .ckpt file
    
    Parameters
    ----------
    task_name
        Dataset name (e.g., 'isic2017', 'polyp')
    model_path
        Path to model directory or .ckpt file
    device
        Device to load model on
        
    Returns
    -------
    predictor
        MultiModalPredictor (None if loaded from ckpt)
    sam_model
        The underlying SAM model  
    conv_lora_layers
        List of (name, ConvLoRALinear) tuples
    """
    if not model_path or not os.path.exists(model_path):
        raise ValueError(
            f"Model path not found: {model_path}. "
            "Please provide a valid model path."
        )
    
    # Case 1: Complete predictor with assets.json
    if os.path.exists(os.path.join(model_path, 'assets.json')):
        print(f"Loading complete MultiModalPredictor from {model_path}")
        predictor = MultiModalPredictor.load(model_path)
        sam_model = predictor._model
        if hasattr(sam_model, 'model'):
            sam_model = sam_model.model
        
        conv_lora_layers = []
        for name, module in sam_model.named_modules():
            if isinstance(module, ConvLoRALinear):
                conv_lora_layers.append((name, module))
        
        sam_model = sam_model.to(device)
        sam_model.eval()
        
        print(f"✓ Found {len(conv_lora_layers)} Conv-LoRA layers")
        return predictor, sam_model, conv_lora_layers
    
    # Case 2: Training directory or .ckpt file - load checkpoint directly
    ckpt_path = None
    
    if model_path.endswith('.ckpt'):
        ckpt_path = model_path
    elif os.path.isdir(model_path):
        # Find best checkpoint
        import glob
        import yaml
        
        # Try best_k_models.yaml first
        best_k_file = os.path.join(model_path, 'best_k_models.yaml')
        if os.path.exists(best_k_file):
            with open(best_k_file, 'r') as f:
                best_k_data = yaml.safe_load(f)
            if best_k_data:
                ckpt_path = list(best_k_data.keys())[0]
                print(f"Using best checkpoint from best_k_models.yaml: {ckpt_path}")
        
        # Otherwise find the latest epoch checkpoint
        if not ckpt_path:
            ckpt_files = glob.glob(os.path.join(model_path, 'epoch=*.ckpt'))
            if ckpt_files:
                # Sort by epoch number
                ckpt_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
                ckpt_path = ckpt_files[-1]  # Latest
                print(f"Using latest checkpoint: {os.path.basename(ckpt_path)}")
        
        # Last resort: last.ckpt
        if not ckpt_path and os.path.exists(os.path.join(model_path, 'last.ckpt')):
            ckpt_path = os.path.join(model_path, 'last.ckpt')
            print(f"Using last.ckpt")
    
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No usable checkpoint found in {model_path}. "
            f"Directory contents: {os.listdir(model_path) if os.path.isdir(model_path) else 'N/A'}"
        )
    
    # Load checkpoint and reconstruct model
    print(f"Loading from checkpoint: {ckpt_path}")
    print("Note: Reconstructing model from checkpoint (training was interrupted)")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    # Create a new SAM model with same config
    print("Reconstructing SAM model with Conv-LoRA...")
    from autogluon.multimodal.models.sam import SAMForSemanticSegmentation
    
    # Create model (will need to infer config from checkpoint)
    # For now, use default config for ISIC2017
    sam_model = SAMForSemanticSegmentation(
        prefix='sam',
        checkpoint_name='facebook/sam-vit-huge',
        num_classes=1,  # Binary segmentation for ISIC2017
        pretrained=True,  # Load pretrained SAM base
        frozen_layers=[],
        num_mask_tokens=1,
        image_norm='imagenet',  # Default normalization
    )
    
    # Apply Conv-LoRA to the model
    from autogluon.multimodal.models.utils import inject_adaptation_to_linear_layer
    sam_model.model = inject_adaptation_to_linear_layer(
        model=sam_model.model,
        peft='conv_lora',
        lora_r=3,  # Default for ISIC2017
        lora_alpha=32,
        module_filter=['.*vision_encoder.*attn'],
        filter=['q', 'v'],
        extra_trainable_params=['.*mask_decoder'],
        conv_lora_expert_num=8,
    )
    
    # Load weights from checkpoint
    print("Loading weights from checkpoint...")
    # Remove 'model.' prefix from keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    # Load state dict (strict=False to ignore missing keys)
    sam_model.model.load_state_dict(new_state_dict, strict=False)
    print("✓ Weights loaded successfully")
    
    # Find Conv-LoRA layers
    conv_lora_layers = []
    for name, module in sam_model.model.named_modules():
        if isinstance(module, ConvLoRALinear):
            conv_lora_layers.append((name, module))
    
    sam_model = sam_model.to(device)
    sam_model.eval()
    
    print(f"✓ Found {len(conv_lora_layers)} Conv-LoRA layers")
    
    # Return the full SAMForSemanticSegmentation (not just .model)
    return None, sam_model, conv_lora_layers


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

