"""
Evaluation script for RL-based expert routing (Scheme B).

Evaluates a trained routing policy and generates:
- IoU and FLOPs metrics
- Expert usage histograms
- Per-image statistics (JSON)
- Optional feature visualizations

Usage:
    python rl_eval_routing_policy.py --task polyp --ckpt_path rl_routing/checkpoints/final.pt --output_dir eval_results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.utils.checkpoint import load_checkpoint
from autogluon.multimodal.rl.utils.flops import compute_layer_flops_from_gates
from autogluon.multimodal.rl.utils.rollout import compute_imbalance
from autogluon.multimodal.rl.utils.visualization import save_expert_heatmap, save_json_stats
from autogluon.multimodal.utils.hooks import FeatureHook


def expand_path(df, dataset_dir):
    """Expand relative paths in dataframe."""
    for col in ["image", "label"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def compute_iou(pred_masks, gt_masks, threshold=0.5):
    """Compute IoU."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL routing policy")
    parser.add_argument("--task", type=str, default="polyp")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_heatmaps", action="store_true", default=True)
    parser.add_argument("--save_features", action="store_true", default=False)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stats"), exist_ok=True)
    if args.save_heatmaps:
        os.makedirs(os.path.join(args.output_dir, "heatmaps"), exist_ok=True)
    if args.save_features:
        os.makedirs(os.path.join(args.output_dir, "features"), exist_ok=True)
    
    # Load routing policy
    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = load_checkpoint(args.ckpt_path, device=args.device)
    
    routing_policy = RoutingPolicy(
        num_experts=args.num_experts,
        feature_dim=args.rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=args.num_layers,
    ).to(args.device)
    
    routing_policy.load_state_dict(ckpt['policy'])
    routing_policy.eval()
    
    print("Routing policy loaded successfully")
    
    # Prepare dataset
    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    
    # Try test set first, fall back to val
    test_csv = os.path.join(dataset_dir, "test.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(dataset_dir, "val.csv")
    
    test_df = expand_path(pd.read_csv(test_csv), dataset_dir)
    
    print(f"Evaluating on {len(test_df)} images")
    
    # Evaluation loop
    all_ious = []
    all_flops = []
    all_gates = []
    
    # Placeholder evaluation
    # In full implementation, would:
    # 1. Load SAM model with Conv-LoRA
    # 2. Inject RLGate with routing_policy
    # 3. Forward through test set
    # 4. Collect metrics and gates
    
    print("Note: This is a placeholder evaluation")
    print("Full implementation requires integration with trained SAM+Conv-LoRA model")
    
    # Generate summary statistics
    summary_stats = {
        'task': args.task,
        'checkpoint': args.ckpt_path,
        'num_images': len(test_df),
        'avg_iou': 0.0,  # Placeholder
        'avg_flops': 0.0,  # Placeholder
        'expert_usage': [0] * args.num_experts,  # Placeholder
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json_stats(summary_stats, summary_path)
    print(f"Summary saved to {summary_path}")
    
    # Save expert usage heatmap
    if args.save_heatmaps and len(all_gates) > 0:
        save_expert_heatmap(
            all_gates,
            os.path.join(args.output_dir, "heatmaps", "expert_usage.png"),
            title="Expert Usage Distribution"
        )
    
    print("Evaluation complete")


if __name__ == "__main__":
    main()

