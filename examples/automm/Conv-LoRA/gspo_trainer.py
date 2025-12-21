"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA, Adapters, and Decoder LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support:
1. Encoder Adapters (Phase 1-3):
   - Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
   - Phase 2: Adapter scale adaptation (contribution score based on quality history)
   - Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

2. Decoder LoRA on Attention (NEW):
   - Phase 1: Quality history tracking for LoRA layers
   - Phase 2: Dynamic scaling adaptation based on quality feedback

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection, adapter quality, AND LoRA quality based on actual performance
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
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
        # GSPO-LoRA on Attention extension parameters (NEW)
        gspo_lora_attention_enabled: bool = False,
        gspo_lora_attention_momentum: float = 0.9,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # GSPO-LoRA on Attention extension (NEW)
        self.gspo_lora_attention_enabled = gspo_lora_attention_enabled
        self.gspo_lora_attention_momentum = gspo_lora_attention_momentum
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.lora_quality_log = []  # Track LoRA quality over time (NEW)
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

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel probability maps for
        boundary/smoothness computation.
        """
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            # Use the maximum class probability as the foreground probability
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
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
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
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
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
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
    
    def update_adapter_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-Adapter Extension: Update adapter quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all AdapterLayer modules in the model.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing AdapterLayer modules
        """
        if not self.gspo_adapter_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Collect all AdapterLayer modules from the model
        adapters = []
        for name, module in model.named_modules():
            # Check if this is an AdapterLayer with GSPO enabled
            if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
                if module.gspo_enabled:
                    adapters.append((name, module))
        
        if not adapters:
            return
        
        # Update each adapter with quality feedback
        for name, adapter in adapters:
            adapter.update_quality_feedback(avg_quality)
        
        # Log adapter quality for analysis
        self.adapter_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_adapters': len(adapters),
        })
    
    def get_adapter_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all adapters in the model.
        
        Args:
            model: The model containing AdapterLayer modules
            
        Returns:
            Dictionary with adapter statistics
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_gspo_stats'):
                adapter_stats = module.get_gspo_stats()
                if adapter_stats:
                    stats[name] = adapter_stats
        return stats
    
    def update_lora_attention_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-LoRA Extension: Update LoRA layer quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all LoRALinear modules (in Decoder Attention) that have GSPO enabled.
        
        The feedback mechanism allows LoRA layers to dynamically adjust their contribution
        based on the quality of predictions, following the GSPO principle of
        quality-aware optimization.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing LoRALinear modules (with gspo_enabled=True)
        """
        if not self.gspo_lora_attention_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Import LoRALinear to check module type
        # Note: We check for the method instead of type to avoid circular imports
        lora_layers = []
        for name, module in model.named_modules():
            # Check if this is a LoRALinear with GSPO enabled
            # LoRALinear has: update_quality_feedback, gspo_enabled, r > 0
            if (hasattr(module, 'update_quality_feedback') and 
                hasattr(module, 'gspo_enabled') and 
                hasattr(module, 'r') and
                module.gspo_enabled and
                module.r > 0):
                lora_layers.append((name, module))
        
        if not lora_layers:
            return
        
        # Update each LoRA layer with quality feedback
        for name, lora in lora_layers:
            lora.update_quality_feedback(avg_quality)
        
        # Log LoRA quality for analysis
        self.lora_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_lora_layers': len(lora_layers),
        })
    
    def get_lora_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all LoRA layers in the model.
        
        Args:
            model: The model containing LoRALinear modules
            
        Returns:
            Dictionary with LoRA statistics (quality_history, contribution_score, etc.)
        """
        stats = {}
        for name, module in model.named_modules():
            if (hasattr(module, 'get_gspo_stats') and 
                hasattr(module, 'r') and 
                hasattr(module, 'gspo_enabled')):
                if module.gspo_enabled and module.r > 0:
                    lora_stats = module.get_gspo_stats()
                    if lora_stats:
                        stats[name] = lora_stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
            'adapter_quality_log': self.adapter_quality_log,
            'lora_quality_log': self.lora_quality_log,  # NEW
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()
        self.adapter_quality_log.clear()
        self.lora_quality_log.clear()  # NEW

