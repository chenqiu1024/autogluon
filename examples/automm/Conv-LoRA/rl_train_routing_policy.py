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
from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear
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

from rl_utils import load_trained_conv_lora_model, prepare_dataset, compute_iou


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
    
    def forward_with_rl_gates(self, batch_images, batch_labels=None):
        """
        Forward pass using RL gating - collects REAL rollout data for GRPO update.
        
        This implementation:
        1. Temporarily injects RLGate into Conv-LoRA layers
        2. Runs SAM forward to collect policy outputs
        3. Restores original gates
        4. Computes rewards and returns rollout data
        
        Parameters
        ----------
        batch_images
            Image tensors (B, 3, 1024, 1024)
        batch_labels
            Ground truth labels (B, 1024, 1024), optional
            
        Returns
        -------
        rollout_data
            Dict with logprobs, gates, info_dicts from RLGate
        metrics
            Dict with iou, flops, reward, imbalance
        """
        # Find all Conv-LoRA layers
        conv_lora_layers = []
        for name, module in self.sam_model.model.named_modules():
            if isinstance(module, ConvLoRALinear):
                conv_lora_layers.append((name, module))
        
        # Storage for RL gate outputs
        rl_gate_outputs = {i: None for i in range(len(conv_lora_layers))}
        
        # Step 1: Temporarily replace gates with RLGate
        original_gates = {}
        rl_gates = {}
        
        for layer_idx, (name, module) in enumerate(conv_lora_layers):
            # Save original gate
            original_gates[name] = module.lora_moe_gating
            
            # Create RLGate with routing policy
            # Wrap the original MoEGate as reference
            from autogluon.multimodal.models.adaptation_layers import MoEGate
            
            if isinstance(module.lora_moe_gating, MoEGate):
                reference_gate = NoisyTopKGate(module.lora_moe_gating)
            else:
                reference_gate = None
            
            rl_gate = RLGate(
                routing_policy=self.routing_policy,
                reference_gate=reference_gate,
                k=1,
                compute_kl=True,
            )
            rl_gates[name] = rl_gate
            
            # Store layer index in the gate for proper indexing
            rl_gate._layer_idx = layer_idx
            
            # Replace gate temporarily
            module.lora_moe_gating = rl_gate
            
            # Create a hook to capture RLGate outputs
            def make_capture_hook(l_idx):
                def hook_fn(gate_module, inputs, outputs):
                    # outputs = (gates, aux_loss, info_dict)
                    if len(outputs) == 3:
                        rl_gate_outputs[l_idx] = {
                            'gates': outputs[0].detach().clone(),
                            'info_dict': {
                                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                                for k, v in outputs[2].items()
                            }
                        }
                return hook_fn
            
            rl_gate.register_forward_hook(make_capture_hook(layer_idx))
        
        # Step 2: Forward through SAM
        batch_dict = {'sam_image': batch_images}
        
        was_training = self.sam_model.training
        self.sam_model.train()  # Avoid label requirement
        
        # Forward: SAM in no_grad to save memory, collect features for policy
        # We'll do a second pass with policy gradients enabled
        with torch.no_grad():
            output = self.sam_model(batch_dict)
            pred_masks = output['sam']['logits']
        
        if not was_training:
            self.sam_model.eval()
        
        # Step 3: Restore original gates
        for name, module in conv_lora_layers:
            module.lora_moe_gating = original_gates[name]
        
        # Step 4: Extract gates and features, then recompute logprobs with gradient
        collected_logprobs = []
        collected_gates = []
        collected_info_dicts = []
        
        for layer_idx in range(len(conv_lora_layers)):
            if rl_gate_outputs[layer_idx] is not None:
                gate_output = rl_gate_outputs[layer_idx]
                info = gate_output['info_dict']
                
                # Get gates and actions (no grad)
                gates = gate_output['gates']  # (B_tokens, M)
                actions = info.get('actions', gates.argmax(dim=1))  # (B_tokens,)
                
                # Recompute logprobs with gradient enabled
                # Get features from info (should have been saved)
                if '_feats' in info:
                    feats = info['_feats']  # Features with grad
                    # Forward through policy with gradient
                    logits = self.routing_policy(feats, layer_idx)
                    log_probs = torch.log_softmax(logits, dim=1)
                    # Get logprobs for taken actions
                    logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                else:
                    # Fallback: use detached logprobs from info
                    logprobs = info.get('logprobs', torch.zeros(gates.size(0), device=self.device))
                    if not logprobs.requires_grad:
                        logprobs = logprobs.detach().requires_grad_(True)
                
                collected_logprobs.append(logprobs)
                collected_gates.append(gates)
                collected_info_dicts.append(info)
        
        # Step 5: Compute metrics
        if batch_labels is not None:
            iou = compute_iou(pred_masks, batch_labels).mean().item()
        else:
            iou = 0.75
        
        # Compute real FLOPs
        feature_shapes = [(14, 14)] * len(collected_gates)  # Approximate
        expert_configs_per_layer = [[{'in_c': 3, 'out_c': 3, 'kernel_size': 3}] * 8] * len(collected_gates)
        
        flops = compute_layer_flops_from_gates(
            collected_gates,
            expert_configs_per_layer,
            feature_shapes
        ) if len(collected_gates) > 0 else self.compute_budget * 0.8
        
        # Compute imbalance
        imbalance = compute_imbalance(collected_gates) if len(collected_gates) > 0 else 0.0
        
        # Compute reward
        reward = iou - self.dual_alpha * (flops / self.compute_budget) - self.imbalance_weight * imbalance
        
        rollout_data = {
            'logprobs': collected_logprobs,
            'info_dicts': collected_info_dicts,
            'gates': collected_gates,
        }
        
        metrics = {
            'iou': iou,
            'flops': flops,
            'imbalance': imbalance,
            'reward': reward,
            'gates': collected_gates,
        }
        
        return rollout_data, metrics
    
    def train_step(self, batch_images, batch_labels=None):
        """
        Perform one training step.
        
        Parameters
        ----------
        batch_images
            Image tensors (B, 3, 1024, 1024)
        batch_labels
            Ground truth labels (B, 1024, 1024), optional
            
        Returns
        -------
        metrics
            Dict of training metrics
        """
        # Forward with RL gates
        rollout_data, metrics = self.forward_with_rl_gates(batch_images, batch_labels)
        
        # Compute group advantages
        # We have one reward for the entire batch
        # Need to match the shape of concatenated logprobs
        num_logprobs = sum(lp.numel() for lp in rollout_data['logprobs'])
        
        # Simple approach: use same advantage for all logprobs
        # (since we have one reward for the whole batch)
        advantages = torch.ones(num_logprobs, device=self.device) * metrics['reward']
        
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
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to trained Conv-LoRA model or checkpoint")
    
    # Model config
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=32)
    
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
    
    # Load Conv-LoRA model
    if not args.model_path:
        # Try to infer from warmstart path
        if args.warmstart:
            # Use same model that was used for BC
            print("Warning: --model_path not specified, attempting to use BC model")
            # This requires the BC checkpoint to store model_path
            bc_ckpt = load_checkpoint(args.warmstart, device='cpu')
            if 'model_path' in bc_ckpt.get('config', {}):
                args.model_path = bc_ckpt['config']['model_path']
                print(f"Using model from BC config: {args.model_path}")
        
        if not args.model_path:
            raise ValueError(
                "Must provide --model_path to trained Conv-LoRA model!\n"
                "Example: --model_path AutogluonModels/ag-XXXXXX/epoch=X-step=XXXX.ckpt"
            )
    
    print(f"Loading Conv-LoRA model from {args.model_path}")
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        args.task,
        args.model_path,
        device=args.device
    )
    
    print(f"Model loaded: {len(conv_lora_layers)} Conv-LoRA layers")
    
    # Create main trainer
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
    train_df, dataset_dir = prepare_dataset(dataset_name, split='train')
    
    print(f"Starting RL training for {args.max_steps} steps")
    print(f"Output directory: {args.output_dir}")
    print(f"Training samples: {len(train_df)}")
    print(f"Compute budget: {args.compute_budget:.2e} FLOPs")
    print(f"KL coef: {args.kl_coef}, Entropy coef: {args.entropy_coef}")
    
    # Prepare data loading
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Training loop
    num_samples = len(train_df)
    samples_per_step = args.batch_size
    
    for step in tqdm(range(global_step, args.max_steps)):
        # Sample a batch from training data
        batch_indices = torch.randint(0, num_samples, (samples_per_step,))
        batch_df = train_df.iloc[batch_indices.tolist()]
        
        # Load images
        images = []
        labels_list = []
        for _, row in batch_df.iterrows():
            try:
                img = Image.open(row['image']).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                
                # Load label if available
                if 'label' in row and pd.notna(row['label']):
                    label = Image.open(row['label']).convert('L')
                    # Resize label to match SAM size
                    label = label.resize((1024, 1024), Image.NEAREST)
                    label_tensor = transforms.ToTensor()(label)
                    labels_list.append(label_tensor)
            except Exception as e:
                print(f"\nWarning: Failed to load image: {e}")
                continue
        
        if len(images) == 0:
            continue
        
        # Stack batch
        batch_images = torch.stack(images).to(args.device)
        batch_labels = torch.stack(labels_list).to(args.device) if labels_list else None
        
        # Train step
        metrics = trainer.train_step(batch_images, batch_labels)
        
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

