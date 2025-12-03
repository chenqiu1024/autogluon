"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou'
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, C, H, W] or [B, H, W]
            metric: 'iou', 'dice', or 'both'
            
        Returns:
            quality_scores: Tensor of shape [B] containing quality scores
        """
        # Ensure masks are in the right format
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        
        # Binarize if needed
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        if metric == 'iou':
            return iou
        
        # DICE
        dice = (2 * intersection + 1e-6) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        
        if metric == 'dice':
            return dice
        
        # Both: average of IoU and DICE
        return (iou + dice) / 2.0
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            # GSPO Exp3-style: Pass group_id for dynamic noise in MoEGate
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images, group_id=g, max_group_id=G-1)
            
            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(quality)
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss with dynamic temperature (Exp3-style)
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # GSPO Exp3-style: Dynamic temperature scaling
            # Progressive temperature: from conservative (2.0) to aggressive (10.0)
            # Lower temperature = more uniform weights, higher = more emphasis on high advantage
            temperature = 2.0 + 8.0 * (g / G)
            
            # Compute adaptive weights based on advantage with dynamic temperature
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()

