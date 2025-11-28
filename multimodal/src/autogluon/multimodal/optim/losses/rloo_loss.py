import torch
import torch.nn as nn
import torch.nn.functional as F
from .structure_loss import StructureLoss

class RLOOLoss(nn.Module):
    """
    RLOO Loss for Segment Anything Model (SAM) fine-tuning.
    Combines a supervised Structure Loss with a Reinforcement Learning based loss
    using RLOO (REINFORCE Leave-One-Out) estimator to directly optimize IoU.
    
    The loss is defined as:
    L = L_structure + lambda * L_rloo
    
    where L_rloo is the policy gradient loss using RLOO baseline:
    Baseline(i) = (Sum(Rewards) - Reward(i)) / (k - 1)
    Advantage(i) = Reward(i) - Baseline(i)
    L_rloo = - Mean(Advantage(i) * log_prob(i))
    """
    def __init__(self, k: int = 4, rloo_weight: float = 1.0, structure_weight: float = 1.0):
        super().__init__()
        self.k = k
        self.rloo_weight = rloo_weight
        self.structure_weight = structure_weight
        self.structure_loss = StructureLoss()
        
    def _compute_iou(self, pred_mask, target_mask):
        """
        Compute IoU between predicted binary mask and target mask.
        Args:
            pred_mask: (B, 1, H, W) binary tensor
            target_mask: (B, 1, H, W) binary tensor
        Returns:
            iou: (B,) tensor
        """
        # Flatten to (B, -1)
        pred_flat = pred_mask.view(pred_mask.size(0), -1)
        target_flat = target_mask.view(target_mask.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        
        # Avoid division by zero
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (B, 1, H, W) logits from the model
            target: (B, 1, H, W) binary ground truth
        """
        # 1. Compute Supervised Loss (Structure Loss)
        # StructureLoss expects logits
        l_structure = self.structure_loss(input, target)
        
        if self.rloo_weight == 0:
            return l_structure

        # 2. Compute RLOO Loss
        
        # Input logits shape: (B, 1, H, W)
        # We treat each pixel as an independent Bernoulli distribution
        # Probabilities p = sigmoid(logits)
        probs = torch.sigmoid(input)
        
        # We need to sample k masks for each image in the batch
        # Expand probs to (B, k, 1, H, W)
        B, C, H, W = probs.shape
        probs_expanded = probs.unsqueeze(1).expand(B, self.k, C, H, W)
        
        # Sample k binary masks
        # We use the reparameterization trick? No, this is REINFORCE, so we sample discrete actions.
        # But we need to keep the graph connected to logits for log_prob computation.
        dist = torch.distributions.Bernoulli(probs_expanded)
        sampled_masks = dist.sample() # (B, k, 1, H, W), detached from graph
        
        # Compute Log Probabilities of the sampled masks
        # log_prob returns log(p) if sample=1, log(1-p) if sample=0
        # Sum over spatial dimensions to get log_prob of the whole mask
        log_probs = dist.log_prob(sampled_masks).sum(dim=(2, 3, 4)) # (B, k)
        
        # Compute Rewards (IoU) for each sample
        # Target needs to be expanded to match samples
        target_expanded = target.unsqueeze(1).expand(B, self.k, C, H, W)
        
        # Reshape for IoU computation: (B*k, 1, H, W)
        sampled_masks_flat = sampled_masks.view(B * self.k, C, H, W)
        target_expanded_flat = target_expanded.reshape(B * self.k, C, H, W)
        
        rewards_flat = self._compute_iou(sampled_masks_flat, target_expanded_flat)
        rewards = rewards_flat.view(B, self.k) # (B, k)
        
        # Compute RLOO Baseline and Advantage
        # Baseline for sample i is mean of rewards of all other samples j != i
        # Sum of all rewards for each batch item
        sum_rewards = rewards.sum(dim=1, keepdim=True) # (B, 1)
        
        # (Sum - Reward_i) / (k - 1)
        baselines = (sum_rewards - rewards) / (self.k - 1)
        
        advantages = rewards - baselines # (B, k)
        
        # Detach advantages to treat them as constants
        advantages = advantages.detach()
        
        # Policy Gradient Loss: - Mean(Advantage * log_prob)
        # We average over batch and k samples
        l_rloo = -(advantages * log_probs).mean()
        
        return self.structure_weight * l_structure + self.rloo_weight * l_rloo
