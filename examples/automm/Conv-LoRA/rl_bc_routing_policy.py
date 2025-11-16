"""
Behavior Cloning (BC) warmstart for routing policy.

This script trains the routing policy to imitate Noisy-TopK gate decisions,
providing a strong initialization for subsequent RL fine-tuning.

Usage:
    python rl_bc_routing_policy.py --task polyp --output_dir bc_warmstart/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.utils.checkpoint import save_checkpoint
from autogluon.multimodal.rl.utils.visualization import create_tensorboard_writer, log_scalars

from rl_utils import load_trained_conv_lora_model, prepare_dataset


def collect_noisy_topk_actions(
    sam_model,
    conv_lora_layers,
    train_df,
    predictor,
    max_samples=1000,
    device='cuda',
):
    """
    Collect expert actions from Noisy-TopK gate by running real SAM forward passes.
    
    Parameters
    ----------
    sam_model
        SAM model with Conv-LoRA
    conv_lora_layers
        List of (name, ConvLoRALinear) tuples
    train_df
        Training dataframe
    predictor
        MultiModalPredictor for data processing (not used in direct loading)
    max_samples
        Maximum number of samples to collect
    device
        Device
        
    Returns
    -------
    bc_dataset
        List of (feats, layer_idx, expert_action) dicts
    """
    print(f"Collecting Noisy-TopK actions from {len(conv_lora_layers)} layers...")
    print(f"Target: {max_samples} samples from train set")
    
    # Storage for collected data
    bc_data = {i: [] for i in range(len(conv_lora_layers))}
    
    # Hook to capture features and gates AFTER they're saved in forward
    def make_hook(layer_idx, data_storage):
        def hook_fn(module, input, output):
            # After forward completes, _last_lora_res and _last_gates should be set
            if hasattr(module, '_last_lora_res') and hasattr(module, '_last_gates'):
                feats = module._last_lora_res  # Already detached in forward
                gates = module._last_gates      # Already detached in forward
                
                # Get expert action (argmax of gates)
                actions = gates.argmax(dim=1)  # (B,)
                
                # Store (feats, actions) pairs
                for i in range(feats.size(0)):
                    if feats[i].numel() > 0:  # Valid feature
                        data_storage[layer_idx].append({
                            'feats': feats[i].cpu(),  # (C, H, W)
                            'layer_idx': layer_idx,
                            'action': actions[i].item(),
                        })
        return hook_fn
    
    # Register hooks
    hooks = []
    for layer_idx, (name, module) in enumerate(conv_lora_layers):
        hook = module.register_forward_hook(make_hook(layer_idx, bc_data))
        hooks.append(hook)
    
    # Prepare data for SAM forward
    from PIL import Image
    import torchvision.transforms as transforms
    
    # SAM expects 1024x1024 images with ImageNet normalization
    # Using same normalization as in SAM training
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        ),
    ])
    
    print("Running forward passes to collect BC data...")
    sam_model.eval()
    
    num_collected = 0
    sample_limit = min(max_samples, len(train_df))
    batch_size = 2  # Small batch to save memory
    
    with torch.no_grad():
        # Process in small batches
        for idx in tqdm(range(0, sample_limit, batch_size)):
            batch_df = train_df.iloc[idx:idx+batch_size]
            
            try:
                # Load and preprocess images
                images = []
                for _, row in batch_df.iterrows():
                    img = Image.open(row['image']).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                
                if len(images) == 0:
                    continue
                
                # Stack into batch
                image_batch = torch.stack(images).to(device)  # (B, 3, 1024, 1024)
                
                # Set model to training mode to avoid label requirement
                was_training = sam_model.training
                sam_model.train()  # Training mode doesn't require labels
                
                # Forward through SAMForSemanticSegmentation
                # It expects batch dict with 'sam_image' key
                batch_dict = {
                    'sam_image': image_batch,
                }
                
                # Forward (this triggers hooks on Conv-LoRA layers!)
                output = sam_model(batch_dict)
                
                # Restore eval mode
                if not was_training:
                    sam_model.eval()
                
                num_collected += len(images)
                
            except Exception as e:
                print(f"\nWarning: Error processing batch at {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Flatten collected data
    bc_dataset = []
    for layer_idx in range(len(conv_lora_layers)):
        bc_dataset.extend(bc_data[layer_idx])
    
    print(f"\nCollected {len(bc_dataset)} BC samples from {len(conv_lora_layers)} layers")
    print(f"  Samples per layer: ~{len(bc_dataset) // max(len(conv_lora_layers), 1)}")
    
    # Verify data quality
    if len(bc_dataset) > 0:
        sample = bc_dataset[0]
        print(f"  Sample data shape: {sample['feats'].shape}")
        print(f"  Expert actions range: 0-{max([s['action'] for s in bc_dataset])}")
    
    # If still no data collected, there's a problem
    if len(bc_dataset) == 0:
        raise RuntimeError(
            "Failed to collect BC data even with real forward passes!\n"
            "This indicates hooks are not firing. Possible issues:\n"
            "1. Conv-LoRA layers not being called in forward\n"
            "2. _last_lora_res/_last_gates not being set\n"
            "3. Model structure mismatch"
        )
    
    return bc_dataset


def train_bc(
    routing_policy,
    bc_dataset,
    optimizer,
    device='cuda',
    num_epochs=10,
    batch_size=32,
    writer=None,
):
    """
    Train routing policy via behavior cloning.
    
    Parameters
    ----------
    routing_policy
        The routing policy to train
    bc_dataset
        List of dicts with 'feats', 'layer_idx', 'action'
    optimizer
        Optimizer
    device
        Device
    num_epochs
        Number of training epochs
    batch_size
        Batch size
    writer
        TensorBoard writer
    """
    routing_policy.train()
    
    # Prepare data tensors
    # Note: Features from different layers may have different spatial sizes
    # We need to handle this by training layer-by-layer or resizing
    
    feats_list = []
    layer_idx_list = []
    actions_list = []
    
    # Group by spatial size and resize to common size
    import torch.nn.functional as F
    target_size = 16  # Resize all features to 16x16
    
    for sample in bc_dataset:
        feat = sample['feats']  # (C, H, W)
        
        # Resize if needed
        if feat.size(1) != target_size or feat.size(2) != target_size:
            feat = F.interpolate(
                feat.unsqueeze(0),  # (1, C, H, W)
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (C, H, W)
        
        feats_list.append(feat)
        layer_idx_list.append(sample['layer_idx'])
        actions_list.append(sample['action'])
    
    # Stack into tensors
    feats_tensor = torch.stack(feats_list).to(device)  # (N, C, target_size, target_size)
    layer_idx_tensor = torch.tensor(layer_idx_list, dtype=torch.long, device=device)
    actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=device)
    
    # Create dataset
    dataset = TensorDataset(feats_tensor, layer_idx_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"BC training: {len(dataset)} samples, {num_epochs} epochs")
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for feats, layer_idxs, actions in tqdm(dataloader, desc=f"BC Epoch {epoch+1}/{num_epochs}"):
            # Forward
            logits_list = []
            for i in range(feats.size(0)):
                logits = routing_policy(feats[i:i+1], layer_idxs[i].item())
                logits_list.append(logits)
            
            logits = torch.cat(logits_list, dim=0)  # (B, M)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, actions)
            
            # Accuracy
            pred_actions = logits.argmax(dim=1)
            acc = (pred_actions == actions).float().mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1
            global_step += 1
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        # Log to tensorboard
        if writer:
            log_scalars(writer, {
                'bc/loss': avg_loss,
                'bc/accuracy': avg_acc,
            }, epoch)
    
    return routing_policy


def main():
    parser = argparse.ArgumentParser(description="BC warmstart for routing policy")
    parser.add_argument("--task", type=str, default="polyp")
    parser.add_argument("--output_dir", type=str, default="bc_warmstart")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained Conv-LoRA model (output from run_semantic_segmentation.py)")
    parser.add_argument("--bc_data", type=str, default=None,
                        help="Path to pre-collected BC data (from collect_bc_data.py)")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)  # Rank r
    parser.add_argument("--num_layers", type=int, default=32)  # Will be auto-detected
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max samples to collect for BC")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = create_tensorboard_writer(os.path.join(args.output_dir, "logs"))
    
    # Load BC data or collect from model
    if args.bc_data and os.path.exists(args.bc_data):
        # Load pre-collected BC data
        print(f"Loading pre-collected BC data from {args.bc_data}")
        bc_dataset = torch.load(args.bc_data)
        print(f"Loaded {len(bc_dataset)} BC samples")
        
        # Auto-detect parameters from data
        if len(bc_dataset) > 0:
            sample = bc_dataset[0]
            args.rank = sample['feats'].size(0)
            args.num_layers = max([s['layer_idx'] for s in bc_dataset]) + 1
            
            # Experts count needs to be provided or guessed
            print(f"Auto-detected from data: rank={args.rank}, layers={args.num_layers}")
    
    elif args.model_path:
        # Collect from trained model
        print(f"Loading Conv-LoRA model from {args.model_path}")
        try:
            predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
                args.task,
                args.model_path,
                device=args.device
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nPlease train a baseline Conv-LoRA model first:")
            print("  ./train_baseline_first.sh {}".format(args.task))
            sys.exit(1)
        
        # Auto-detect parameters from model
        if len(conv_lora_layers) > 0:
            sample_layer = conv_lora_layers[0][1]
            detected_rank = sample_layer.lora_A.size(0)
            detected_experts = sample_layer.num_experts
            
            if detected_rank != args.rank:
                print(f"Using detected rank: {detected_rank}")
                args.rank = detected_rank
            
            if detected_experts != args.num_experts:
                print(f"Using detected experts: {detected_experts}")
                args.num_experts = detected_experts
        
        args.num_layers = len(conv_lora_layers)
        print(f"Model config: rank={args.rank}, experts={args.num_experts}, layers={args.num_layers}")
        
        # Prepare training data
        train_df, dataset_dir = prepare_dataset(args.task, split='train')
        
        # Collect BC data (uses synthetic approach due to hook complexity)
        bc_dataset = collect_noisy_topk_actions(
            sam_model=sam_model,
            conv_lora_layers=conv_lora_layers,
            train_df=train_df,
            predictor=predictor,
            max_samples=args.max_samples,
            device=args.device,
        )
        
        if len(bc_dataset) == 0:
            print("ERROR: Failed to collect BC data.")
            sys.exit(1)
    
    else:
        print("ERROR: Must provide either --model_path or --bc_data")
        sys.exit(1)
    
    # Create routing policy
    routing_policy = RoutingPolicy(
        num_experts=args.num_experts,
        feature_dim=args.rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=args.num_layers,
    ).to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(routing_policy.parameters(), lr=args.lr)
    
    # Train BC
    routing_policy = train_bc(
        routing_policy,
        bc_dataset,
        optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        writer=writer,
    )
    
    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "routing_bc.pt")
    config_dict = vars(args).copy()
    # Ensure model_path is saved for RL training to reuse
    if args.model_path:
        config_dict['model_path'] = args.model_path
    elif args.bc_data:
        config_dict['model_path'] = None  # Can't infer from bc_data
    
    save_checkpoint({
        'policy': routing_policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config_dict,
        'num_experts': args.num_experts,
        'rank': args.rank,
        'num_layers': args.num_layers,
    }, ckpt_path)
    
    print(f"\n✓ BC warmstart checkpoint saved to {ckpt_path}")
    print(f"  - Trained on {len(bc_dataset)} samples")
    print(f"  - {args.num_layers} layers, {args.num_experts} experts, rank {args.rank}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

