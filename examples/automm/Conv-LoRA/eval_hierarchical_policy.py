"""
Evaluation script for hierarchical RL policies (LayerPolicy + RoutingPolicy).

Evaluates the combined hierarchical RL model on validation/test set and reports:
- IoU, DICE metrics
- FLOPs usage
- Number of active layers
- Per-sample statistics

Usage:
    python eval_hierarchical_policy.py \
        --task isic2017 \
        --model_path AutogluonModels/ag-20251113_165105/model.ckpt \
        --hier_ckpt rl_hier_joint/checkpoints/step_3000.pt \
        --split val \
        --output_file hier_eval_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear
from autogluon.multimodal.models.gating import NoisyTopKGate, RLGate
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.policies.layer_policy import LayerPolicy
from autogluon.multimodal.rl.utils.checkpoint import load_checkpoint
from autogluon.multimodal.rl.utils.flops import compute_layer_flops_from_gates

from rl_utils import load_trained_conv_lora_model, prepare_dataset


def compute_iou_dice(pred_masks, gt_masks, threshold=0.5):
    """
    Compute IoU and DICE for predicted masks.
    
    Parameters
    ----------
    pred_masks : torch.Tensor
        Predicted masks, shape (B, H, W) or (B, 1, H, W)
    gt_masks : torch.Tensor
        Ground truth masks, shape (B, H, W) or (B, 1, H, W)
    threshold : float
        Threshold for binarization
        
    Returns
    -------
    iou : torch.Tensor
        IoU per sample, shape (B,)
    dice : torch.Tensor
        DICE per sample, shape (B,)
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
    
    # DICE
    dice = (2 * intersection) / (pred_binary.sum(dim=(1, 2)) + gt_binary.sum(dim=(1, 2)) + 1e-8)
    
    return iou, dice


def _extract_global_features(sam_model, batch_images: torch.Tensor) -> torch.Tensor:
    """Extract global features for LayerPolicy."""
    batch_dict = {"sam_image": batch_images}
    
    was_training = sam_model.training
    sam_model.train()  # Force train mode to avoid requiring sam_label
    with torch.no_grad():
        output = sam_model(batch_dict)
    if not was_training:
        sam_model.eval()

    sam_outputs = output["sam"]
    if "image_embeds" in sam_outputs:
        feats = sam_outputs["image_embeds"]  # (B, C, H, W)
        feats = feats.mean(dim=[2, 3])  # (B, C)
    else:
        logits = sam_outputs["logits"]  # (B, 1, H, W)
        feats = logits.mean(dim=[1, 2, 3], keepdim=False)  # (B,)
        feats = feats.unsqueeze(-1)  # (B, 1)
    return feats


def evaluate_hierarchical_policy(
    sam_model,
    layer_policy,
    routing_policy,
    test_loader,
    device="cuda",
):
    """
    Evaluate hierarchical RL policy on test set.
    
    Returns
    -------
    results : dict
        Evaluation metrics and per-sample statistics
    """
    sam_model.eval()
    layer_policy.eval()
    routing_policy.eval()
    
    # Collect Conv-LoRA layers
    conv_lora_layers = []
    for name, module in sam_model.model.named_modules():
        if isinstance(module, ConvLoRALinear):
            conv_lora_layers.append((name, module))
    
    num_layers = len(conv_lora_layers)
    
    all_ious = []
    all_dices = []
    all_flops = []
    all_active_layers = []
    
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(test_loader, desc="Evaluating"):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # 1) Get layer activation from LayerPolicy
            global_feats = _extract_global_features(sam_model, batch_images)
            layer_logits = layer_policy(global_feats)  # (B, L)
            layer_probs = torch.sigmoid(layer_logits)
            # Use batch-mean probability for layer mask (greedy)
            layer_mask = (layer_probs.mean(dim=0) > 0.5).float()  # (L,)
            
            # 2) Apply layer mask to ConvLoRALinear
            for layer_idx, (_, module) in enumerate(conv_lora_layers):
                active = layer_mask[layer_idx].item()
                module.set_lora_active(active)
            
            # 3) Inject RLGate for active layers (greedy mode)
            original_gates = {}
            rl_gate_outputs = {i: None for i in range(num_layers)}
            
            for layer_idx, (name, module) in enumerate(conv_lora_layers):
                original_gates[name] = module.lora_moe_gating
                
                if layer_mask[layer_idx] < 0.5:
                    continue  # Skip inactive layers
                
                from autogluon.multimodal.models.adaptation_layers import MoEGate
                
                if isinstance(module.lora_moe_gating, MoEGate):
                    reference_gate = NoisyTopKGate(module.lora_moe_gating)
                else:
                    reference_gate = None
                
                rl_gate = RLGate(
                    routing_policy=routing_policy,
                    reference_gate=reference_gate,
                    k=1,
                    compute_kl=False,  # No KL during eval
                )
                rl_gate._layer_idx = layer_idx
                module.lora_moe_gating = rl_gate
                
                # Hook to capture gates
                def make_capture_hook(l_idx):
                    def hook_fn(gate_module, inputs, outputs):
                        if len(outputs) == 3:
                            rl_gate_outputs[l_idx] = {
                                'gates': outputs[0].detach().clone(),
                            }
                    return hook_fn
                
                rl_gate.register_forward_hook(make_capture_hook(layer_idx))
            
            # 4) Forward through SAM (force train mode to avoid requiring sam_label)
            batch_dict = {"sam_image": batch_images}
            was_training = sam_model.training
            sam_model.train()
            output = sam_model(batch_dict)
            if not was_training:
                sam_model.eval()
            pred_masks = output["sam"]["logits"]
            
            # 5) Restore original gates
            for name, module in conv_lora_layers:
                module.lora_moe_gating = original_gates[name]
            
            # 6) Compute metrics
            iou, dice = compute_iou_dice(pred_masks, batch_labels)
            
            # Compute FLOPs
            collected_gates = []
            for layer_idx in range(num_layers):
                if rl_gate_outputs[layer_idx] is not None:
                    collected_gates.append(rl_gate_outputs[layer_idx]['gates'])
            
            feature_shapes = [(14, 14)] * len(collected_gates)
            expert_configs_per_layer = [
                [{'in_c': 3, 'out_c': 3, 'kernel_size': 3}] * 8
            ] * len(collected_gates)
            
            flops = compute_layer_flops_from_gates(
                collected_gates,
                expert_configs_per_layer,
                feature_shapes
            ) if len(collected_gates) > 0 else 0.0
            
            num_active = layer_mask.sum().item()
            
            # Store results
            all_ious.extend(iou.cpu().numpy().tolist())
            all_dices.extend(dice.cpu().numpy().tolist())
            all_flops.append(flops)
            all_active_layers.append(num_active)
    
    # Aggregate results
    results = {
        "mean_iou": float(np.mean(all_ious)),
        "std_iou": float(np.std(all_ious)),
        "mean_dice": float(np.mean(all_dices)),
        "std_dice": float(np.std(all_dices)),
        "mean_flops": float(np.mean(all_flops)),
        "mean_active_layers": float(np.mean(all_active_layers)),
        "num_samples": len(all_ious),
        "per_sample_iou": all_ious,
        "per_sample_dice": all_dices,
        "per_sample_flops": all_flops,
        "per_sample_active_layers": all_active_layers,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical RL policy")
    parser.add_argument("--task", type=str, default="isic2017")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Conv-LoRA model checkpoint")
    parser.add_argument("--hier_ckpt", type=str, required=True,
                        help="Hierarchical RL checkpoint (contains layer_policy + routing_policy)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default="hier_eval_results.json")
    
    args = parser.parse_args()
    
    # Load Conv-LoRA model
    print(f"Loading Conv-LoRA model from {args.model_path}")
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        args.task,
        args.model_path,
        device=args.device,
    )
    num_layers = len(conv_lora_layers)
    print(f"Found {num_layers} Conv-LoRA layers")
    
    # Infer feature_dim
    dummy_img = torch.zeros(1, 3, 1024, 1024, device=args.device)
    dummy_feats = _extract_global_features(sam_model, dummy_img)
    feature_dim = dummy_feats.size(1)
    
    # Get num_experts and rank from first layer
    num_experts = conv_lora_layers[0][1].num_experts if num_layers > 0 else 8
    rank = conv_lora_layers[0][1].r if num_layers > 0 else 3
    
    # Create policies
    routing_policy = RoutingPolicy(
        num_experts=num_experts,
        feature_dim=rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=num_layers,
    ).to(args.device)
    
    layer_policy = LayerPolicy(
        num_layers=num_layers,
        feature_dim=feature_dim,
        hidden_dim=256,
        dropout=0.1,
        use_patterns=False,
    ).to(args.device)
    
    # Load checkpoint
    print(f"Loading hierarchical RL checkpoint from {args.hier_ckpt}")
    ckpt = load_checkpoint(args.hier_ckpt, device=args.device)
    layer_policy.load_state_dict(ckpt["layer_policy"])
    routing_policy.load_state_dict(ckpt["routing_policy"])
    
    # Prepare dataset
    print(f"Loading {args.split} dataset...")
    df, dataset_dir = prepare_dataset(args.task, split=args.split)
    print(f"Dataset size: {len(df)}")
    
    # Create data loader
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    def collate_fn(batch_indices):
        images = []
        masks = []
        for idx in batch_indices:
            row = df.iloc[idx]
            img = Image.open(row["image"]).convert("RGB")
            msk = Image.open(row["label"]).convert("L")
            img_tensor = transform(img)
            mask_tensor = transforms.Resize((1024, 1024))(transforms.ToTensor()(msk))
            images.append(img_tensor)
            masks.append(mask_tensor)
        return torch.stack(images, dim=0), torch.stack(masks, dim=0)
    
    # Simple batch generator
    class SimpleDataLoader:
        def __init__(self, df, batch_size, collate_fn):
            self.df = df
            self.batch_size = batch_size
            self.collate_fn = collate_fn
            self.num_batches = (len(df) + batch_size - 1) // batch_size
        
        def __iter__(self):
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(self.df))
                batch_indices = list(range(start_idx, end_idx))
                yield self.collate_fn(batch_indices)
        
        def __len__(self):
            return self.num_batches
    
    test_loader = SimpleDataLoader(df, args.batch_size, collate_fn)
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_hierarchical_policy(
        sam_model,
        layer_policy,
        routing_policy,
        test_loader,
        device=args.device,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Hierarchical RL Evaluation Results")
    print("="*60)
    print(f"Dataset: {args.task} ({args.split})")
    print(f"Samples: {results['num_samples']}")
    print(f"\nMetrics:")
    print(f"  Mean IoU:  {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
    print(f"  Mean DICE: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print(f"\nCompute:")
    print(f"  Mean FLOPs: {results['mean_flops']:.2e}")
    print(f"  Mean Active Layers: {results['mean_active_layers']:.1f} / {num_layers}")
    print("="*60)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

