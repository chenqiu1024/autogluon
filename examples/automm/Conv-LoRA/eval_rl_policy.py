#!/usr/bin/env python3
"""
Complete evaluation script for RL routing policy.
Evaluates on test/val set and compares with baseline.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.utils.checkpoint import load_checkpoint
from autogluon.multimodal.rl.utils.flops import compute_layer_flops_from_gates
from autogluon.multimodal.rl.utils.rollout import compute_imbalance
from autogluon.multimodal.models.gating import RLGate
from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear

from rl_utils import load_trained_conv_lora_model, prepare_dataset


def compute_iou(pred_masks, gt_masks, threshold=0.5):
    """Compute IoU between predicted and ground truth masks."""
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.dim() == 4 and gt_masks.size(1) == 1:
        gt_masks = gt_masks.squeeze(1)
    
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = (pred_binary + gt_binary).clamp(max=1).sum(dim=(1, 2))
    
    iou = intersection / (union + 1e-8)
    
    return iou


def compute_dice(pred_masks, gt_masks, threshold=0.5):
    """Compute DICE coefficient."""
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.dim() == 4 and gt_masks.size(1) == 1:
        gt_masks = gt_masks.squeeze(1)
    
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    pred_sum = pred_binary.sum(dim=(1, 2))
    gt_sum = gt_binary.sum(dim=(1, 2))
    
    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-8)
    
    return dice


def evaluate_with_policy(sam_model, routing_policy, test_df, device, use_rl=True):
    """Evaluate model on test set with or without RL policy."""
    
    # Find Conv-LoRA layers
    conv_lora_layers = []
    for name, module in sam_model.model.named_modules():
        if isinstance(module, ConvLoRALinear):
            conv_lora_layers.append((name, module))
    
    print(f"Found {len(conv_lora_layers)} Conv-LoRA layers")
    
    # Prepare RL gates if needed
    if use_rl:
        rl_gates = []
        for layer_idx, (name, module) in enumerate(conv_lora_layers):
            rl_gate = RLGate(
                routing_policy=routing_policy,
                reference_gate=None,
                k=1,
                compute_kl=False
            )
            rl_gate._layer_idx = layer_idx
            rl_gates.append(rl_gate)
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluation loop
    all_ious = []
    all_dices = []
    all_gates = []
    all_flops = []
    
    sam_model.eval()
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            try:
                # Load image and label
                image = Image.open(row['image']).convert('RGB')
                label = Image.open(row['label']).convert('L')
                
                image_tensor = transform(image).unsqueeze(0).to(device)
                label = label.resize((1024, 1024), Image.NEAREST)
                label_tensor = transforms.ToTensor()(label).to(device)
                label_tensor = (label_tensor > 0.5).float()
                
                # Replace gates if using RL
                if use_rl:
                    original_gates = {}
                    for (name, module), rl_gate in zip(conv_lora_layers, rl_gates):
                        original_gates[name] = module.lora_moe_gating
                        module.lora_moe_gating = rl_gate
                
                # Forward
                batch_dict = {'sam_image': image_tensor, 'sam_label': label_tensor}
                output = sam_model(batch_dict)
                pred_masks = output['sam']['logits']
                
                # Restore gates
                if use_rl:
                    for name, module in conv_lora_layers:
                        module.lora_moe_gating = original_gates[name]
                
                # Compute metrics
                iou = compute_iou(pred_masks, label_tensor).item()
                dice = compute_dice(pred_masks, label_tensor).item()
                
                all_ious.append(iou)
                all_dices.append(dice)
                
                # Collect gates if using RL
                if use_rl and hasattr(rl_gates[0], '_last_gates'):
                    gates_list = [g._last_gates for g in rl_gates if hasattr(g, '_last_gates')]
                    if gates_list:
                        all_gates.extend(gates_list)
                
            except Exception as e:
                print(f"\nWarning: Failed to process {row['image']}: {e}")
                continue
    
    return {
        'ious': all_ious,
        'dices': all_dices,
        'gates': all_gates,
        'flops': all_flops,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL routing policy")
    parser.add_argument("--task", type=str, required=True, help="Dataset name (e.g., isic2017)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Conv-LoRA model")
    parser.add_argument("--policy_checkpoint", type=str, required=True, help="Path to RL policy checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test or val)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🔍 Evaluation: RL Routing Policy vs Baseline")
    print("="*70)
    print()
    
    # Load model
    print("1. Loading Conv-LoRA model...")
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        args.task,
        args.model_path,
        device=args.device
    )
    print(f"✓ Model loaded with {len(conv_lora_layers)} Conv-LoRA layers")
    print()
    
    # Load routing policy
    print("2. Loading RL routing policy...")
    ckpt = load_checkpoint(args.policy_checkpoint, device=args.device)
    
    routing_policy = RoutingPolicy(
        num_experts=args.num_experts,
        feature_dim=args.rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=args.num_layers,
    ).to(args.device)
    
    routing_policy.load_state_dict(ckpt['policy'])
    routing_policy.eval()
    print("✓ RL policy loaded")
    print()
    
    # Load test dataset
    print(f"3. Loading {args.split} dataset...")
    test_df, dataset_dir = prepare_dataset(args.task, split=args.split)
    print(f"✓ {len(test_df)} images")
    print()
    
    # Evaluate with baseline (Noisy-TopK)
    print("4. Evaluating with Baseline (Noisy-TopK)...")
    baseline_results = evaluate_with_policy(sam_model, None, test_df, args.device, use_rl=False)
    print(f"✓ Baseline IoU: {sum(baseline_results['ious'])/len(baseline_results['ious']):.4f}")
    print(f"✓ Baseline DICE: {sum(baseline_results['dices'])/len(baseline_results['dices']):.4f}")
    print()
    
    # Evaluate with RL policy
    print("5. Evaluating with RL Policy...")
    rl_results = evaluate_with_policy(sam_model, routing_policy, test_df, args.device, use_rl=True)
    print(f"✓ RL IoU: {sum(rl_results['ious'])/len(rl_results['ious']):.4f}")
    print(f"✓ RL DICE: {sum(rl_results['dices'])/len(rl_results['dices']):.4f}")
    print()
    
    # Compute statistics
    baseline_iou = sum(baseline_results['ious']) / len(baseline_results['ious'])
    baseline_dice = sum(baseline_results['dices']) / len(baseline_results['dices'])
    rl_iou = sum(rl_results['ious']) / len(rl_results['ious'])
    rl_dice = sum(rl_results['dices']) / len(rl_results['dices'])
    
    print("="*70)
    print("📊 Results Summary")
    print("="*70)
    print()
    print(f"Dataset: {args.task} ({args.split} set, {len(test_df)} images)")
    print()
    print(f"{'Metric':<20} {'Baseline':<15} {'RL Policy':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'IoU':<20} {baseline_iou:<15.4f} {rl_iou:<15.4f} {rl_iou-baseline_iou:+.4f} ({(rl_iou/baseline_iou-1)*100:+.1f}%)")
    print(f"{'DICE':<20} {baseline_dice:<15.4f} {rl_dice:<15.4f} {rl_dice-baseline_dice:+.4f} ({(rl_dice/baseline_dice-1)*100:+.1f}%)")
    print()
    
    # Save detailed results
    results = {
        'task': args.task,
        'split': args.split,
        'num_images': len(test_df),
        'model_path': args.model_path,
        'policy_checkpoint': args.policy_checkpoint,
        'baseline': {
            'mean_iou': baseline_iou,
            'mean_dice': baseline_dice,
            'std_iou': float(torch.tensor(baseline_results['ious']).std()),
            'std_dice': float(torch.tensor(baseline_results['dices']).std()),
            'min_iou': min(baseline_results['ious']),
            'max_iou': max(baseline_results['ious']),
        },
        'rl_policy': {
            'mean_iou': rl_iou,
            'mean_dice': rl_dice,
            'std_iou': float(torch.tensor(rl_results['ious']).std()),
            'std_dice': float(torch.tensor(rl_results['dices']).std()),
            'min_iou': min(rl_results['ious']),
            'max_iou': max(rl_results['ious']),
        },
        'improvement': {
            'iou_absolute': rl_iou - baseline_iou,
            'iou_relative': (rl_iou / baseline_iou - 1) * 100,
            'dice_absolute': rl_dice - baseline_dice,
            'dice_relative': (rl_dice / baseline_dice - 1) * 100,
        }
    }
    
    # Save to JSON
    output_file = os.path.join(args.output_dir, f"eval_results_{args.task}_{args.split}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Detailed results saved to: {output_file}")
    print()
    print("="*70)
    print("✅ Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()


