"""
RL training for expert routing policy (Scheme B).

Trains a routing policy to replace Noisy-TopK gating using GRPO with PERL enhancements.

Reward: IoU - α·(FLOPs/budget) - β·imbalance
Loss: Policy gradient + entropy + KL-to-NoisyTopK + adapter L2 + optional Lagrangian

Usage:
    python rl_train_routing_policy.py --task polyp --output_dir rl_routing/ --warmstart bc_warmstart/routing_bc.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.gating import NoisyTopKGate, RLGate
from autogluon.multimodal.rl.algos.grpo import GRPOTrainer
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.utils.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from autogluon.multimodal.rl.utils.flops import compute_layer_flops_from_gates
from autogluon.multimodal.rl.utils.rollout import RolloutBuffer, compute_group_advantages, compute_imbalance
from autogluon.multimodal.rl.utils.visualization import (
    create_tensorboard_writer,
    log_histograms,
    log_scalars,
    save_expert_heatmap,
    save_json_stats,
)


def expand_path(df, dataset_dir):
    """Expand relative paths in dataframe."""
    for col in ["image", "label"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


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
    
    # Binarize predictions
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    # Compute IoU
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = (pred_binary + gt_binary).clamp(max=1).sum(dim=(1, 2))
    
    iou = intersection / (union + 1e-8)
    
    return iou


class RoutingRLTrainer:
    """
    Main trainer for RL-based expert routing.
    """
    
    def __init__(
        self,
        sam_model,
        routing_policy,
        reference_gate,
        grpo_trainer,
        device='cuda',
        compute_budget=1e10,
        imbalance_weight=0.01,
        use_lagrangian=True,
        lagrangian_lr=0.001,
        mixed_precision=True,
    ):
        """
        Parameters
        ----------
        sam_model
            Frozen SAM model with Conv-LoRA
        routing_policy
            Trainable routing policy
        reference_gate
            Reference Noisy-TopK gate for KL
        grpo_trainer
            GRPO trainer instance
        device
            Device
        compute_budget
            FLOPs budget for Lagrangian
        imbalance_weight
            Weight for load imbalance penalty
        use_lagrangian
            Whether to use Lagrangian for compute budget
        lagrangian_lr
            Learning rate for dual variable
        mixed_precision
            Whether to use AMP
        """
        self.sam_model = sam_model
        self.routing_policy = routing_policy
        self.reference_gate = reference_gate
        self.grpo_trainer = grpo_trainer
        self.device = device
        self.compute_budget = compute_budget
        self.imbalance_weight = imbalance_weight
        self.use_lagrangian = use_lagrangian
        self.lagrangian_lr = lagrangian_lr
        self.mixed_precision = mixed_precision
        
        # Dual variable for Lagrangian
        self.dual_alpha = 0.0
        
        # AMP scaler
        self.scaler = GradScaler() if mixed_precision else None
    
    def forward_with_rl_gates(self, batch):
        """
        Forward pass using RL gating.
        
        This is a placeholder - actual implementation would:
        1. Hook into Conv-LoRA layers
        2. Replace gates with policy outputs
        3. Collect logprobs, gates, FLOPs
        
        Returns rollout data and metrics.
        """
        # Placeholder implementation
        # In reality, we'd need to:
        # - Inject RLGate into ConvLoRALinear layers
        # - Forward through SAM
        # - Collect gates, logprobs, info_dicts from each layer
        
        images = batch['sam_image']
        labels = batch['sam_label']
        
        with torch.no_grad():
            # Use baseline SAM for now (placeholder)
            preds = self.sam_model(batch)
            pred_masks = preds['sam']['logits']
        
        # Compute IoU
        iou = compute_iou(pred_masks, labels).mean()
        
        # Placeholder: assume some gates and FLOPs
        batch_size = images.size(0)
        num_layers = 12  # SAM ViT-H has 32 layers, but only some have Conv-LoRA
        dummy_gates = [torch.zeros(batch_size, 8, device=self.device) for _ in range(num_layers)]
        dummy_logprobs = [torch.zeros(batch_size, device=self.device) for _ in range(num_layers)]
        dummy_info_dicts = [{'logits': torch.randn(batch_size, 8, device=self.device), 'kl_loss': torch.tensor(0.0)} for _ in range(num_layers)]
        
        flops = self.compute_budget * 0.5  # Placeholder
        imbalance = compute_imbalance(dummy_gates)
        
        # Compute reward
        reward = iou.item() - self.dual_alpha * (flops / self.compute_budget) - self.imbalance_weight * imbalance
        
        rollout_data = {
            'logprobs': dummy_logprobs,
            'info_dicts': dummy_info_dicts,
            'gates': dummy_gates,
        }
        
        metrics = {
            'iou': iou.item(),
            'flops': flops,
            'imbalance': imbalance,
            'reward': reward,
        }
        
        return rollout_data, metrics
    
    def train_step(self, batch):
        """
        Perform one training step.
        
        Returns metrics dict.
        """
        # Forward with RL gates
        rollout_data, metrics = self.forward_with_rl_gates(batch)
        
        # Compute group advantages (treat batch as one group)
        rewards = [metrics['reward']] * len(rollout_data['logprobs'])
        advantages = compute_group_advantages(rewards, group_size=None)
        
        # GRPO update
        if self.mixed_precision:
            with autocast():
                grpo_metrics = self.grpo_trainer.update(rollout_data, advantages)
        else:
            grpo_metrics = self.grpo_trainer.update(rollout_data, advantages)
        
        # Update dual variable (Lagrangian)
        if self.use_lagrangian:
            flops_violation = metrics['flops'] - self.compute_budget
            self.dual_alpha += self.lagrangian_lr * flops_violation
            self.dual_alpha = max(0.0, self.dual_alpha)
        
        # Merge metrics
        all_metrics = {**metrics, **grpo_metrics, 'dual_alpha': self.dual_alpha}
        
        return all_metrics


def main():
    parser = argparse.ArgumentParser(description="RL training for expert routing")
    parser.add_argument("--task", type=str, default="polyp")
    parser.add_argument("--output_dir", type=str, default="rl_routing")
    parser.add_argument("--warmstart", type=str, default=None, help="Path to BC checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume")
    
    # Model config
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=12)
    
    # RL config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--adapter_l2_coef", type=float, default=0.001)
    parser.add_argument("--compute_budget", type=float, default=1e10)
    parser.add_argument("--imbalance_weight", type=float, default=0.01)
    parser.add_argument("--use_lagrangian", action="store_true", default=True)
    parser.add_argument("--lagrangian_lr", type=float, default=0.001)
    
    # Logging
    parser.add_argument("--ckpt_interval", type=int, default=500)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "artifacts"), exist_ok=True)
    
    # Create TensorBoard writer
    writer = create_tensorboard_writer(os.path.join(args.output_dir, "logs"))
    
    # Create routing policy
    routing_policy = RoutingPolicy(
        num_experts=args.num_experts,
        feature_dim=args.rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=args.num_layers,
    ).to(args.device)
    
    # Load warmstart if provided
    if args.warmstart:
        print(f"Loading BC warmstart from {args.warmstart}")
        ckpt = load_checkpoint(args.warmstart, device=args.device)
        routing_policy.load_state_dict(ckpt['policy'])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(routing_policy.parameters(), lr=args.lr)
    
    # Create reference gate (placeholder - would wrap actual MoEGate)
    reference_gate = None  # Would create NoisyTopKGate wrapping actual MoEGate
    
    # Create GRPO trainer
    grpo_trainer = GRPOTrainer(
        policy=routing_policy,
        optimizer=optimizer,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        adapter_l2_coef=args.adapter_l2_coef,
        device=args.device,
    )
    
    # Create main trainer
    # Placeholder SAM model - would load actual Conv-LoRA model
    sam_model = None
    
    trainer = RoutingRLTrainer(
        sam_model=sam_model,
        routing_policy=routing_policy,
        reference_gate=reference_gate,
        grpo_trainer=grpo_trainer,
        device=args.device,
        compute_budget=args.compute_budget,
        imbalance_weight=args.imbalance_weight,
        use_lagrangian=args.use_lagrangian,
        lagrangian_lr=args.lagrangian_lr,
        mixed_precision=args.mixed_precision,
    )
    
    # Resume if needed
    global_step = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        global_step, dual_alpha, config = restore_training_state(
            routing_policy,
            optimizer,
            args.resume_from,
            scaler=trainer.scaler,
            device=args.device,
        )
        trainer.dual_alpha = dual_alpha
    
    # Prepare dataset
    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "train.csv")), dataset_dir)
    
    print(f"Starting RL training for {args.max_steps} steps")
    print(f"Output directory: {args.output_dir}")
    print(f"Compute budget: {args.compute_budget:.2e} FLOPs")
    print(f"KL coef: {args.kl_coef}, Entropy coef: {args.entropy_coef}")
    
    # Training loop
    # Note: This is a simplified placeholder
    # Full implementation would:
    # 1. Load batches from train_df
    # 2. Forward through SAM with RLGate
    # 3. Collect rollout data
    # 4. GRPO update
    # 5. Log and checkpoint
    
    for step in tqdm(range(global_step, args.max_steps)):
        # Placeholder batch
        batch = {}  # Would load from dataloader
        
        # Train step
        metrics = trainer.train_step(batch)
        
        # Log to TensorBoard
        if writer and step % 10 == 0:
            log_scalars(writer, {
                'train/reward': metrics['reward'],
                'train/iou': metrics['iou'],
                'train/flops': metrics['flops'],
                'train/imbalance': metrics['imbalance'],
                'train/dual_alpha': metrics['dual_alpha'],
                'loss/total': metrics.get('loss/total', 0),
                'loss/kl': metrics.get('loss/kl', 0),
                'loss/entropy': metrics.get('loss/entropy', 0),
                'loss/adapter_l2': metrics.get('loss/adapter_l2', 0),
            }, step)
        
        # Visualize gates
        if step % args.vis_interval == 0 and 'gates' in metrics:
            save_expert_heatmap(
                metrics['gates'],
                os.path.join(args.output_dir, "artifacts", f"gates_step_{step}.png"),
                title=f"Expert Usage at Step {step}"
            )
        
        # Save checkpoint
        if step % args.ckpt_interval == 0 and step > 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            save_checkpoint({
                'policy': routing_policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': trainer.scaler.state_dict() if trainer.scaler else None,
                'dual_alpha': trainer.dual_alpha,
                'global_step': step,
                'rng_state': torch.get_rng_state(),
                'config': vars(args),
            }, ckpt_path)
            print(f"\nCheckpoint saved: {ckpt_path}")
    
    # Save final checkpoint
    final_ckpt_path = os.path.join(args.output_dir, "checkpoints", "final.pt")
    save_checkpoint({
        'policy': routing_policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': trainer.scaler.state_dict() if trainer.scaler else None,
        'dual_alpha': trainer.dual_alpha,
        'global_step': args.max_steps,
        'rng_state': torch.get_rng_state(),
        'config': vars(args),
    }, final_ckpt_path)
    
    print(f"\nTraining complete. Final checkpoint: {final_ckpt_path}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

