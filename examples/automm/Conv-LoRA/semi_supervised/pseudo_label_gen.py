"""
Pseudo Label Generator

基于 Teacher 预测和质量评估生成伪标签

核心功能：
1. 从 K 次采样生成平均伪标签
2. 质量过滤（q > q_min）
3. 质量加权
"""

import torch
from typing import List, Tuple, Optional


class PseudoLabelGenerator:
    """
    伪标签生成器
    
    结合 Teacher 预测和质量评估，生成可靠的伪标签用于 student 训练。
    
    Args:
        min_quality_threshold: 质量过滤阈值 q_min（默认 0.6）
        pseudo_label_threshold: 伪标签二值化阈值（默认 0.5）
        use_soft_label: 是否使用软标签（默认 False，使用硬标签）
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.6,
        pseudo_label_threshold: float = 0.5,
        use_soft_label: bool = False
    ):
        self.min_quality = min_quality_threshold
        self.threshold = pseudo_label_threshold
        self.use_soft_label = use_soft_label
        
        print(f"[Pseudo Label Generator] q_min={min_quality_threshold}, "
              f"use_soft={use_soft_label}")
    
    def generate_from_predictions(
        self,
        predictions: List[torch.Tensor],
        quality_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 K 次预测生成伪标签
        
        Args:
            predictions: K 个预测列表 [B, H, W]
            quality_scores: 质量分数 [B]
        
        Returns:
            pseudo_labels: 伪标签 [B, H, W]（软或硬标签）
            valid_mask: [B] bool，标记哪些样本通过质量过滤
        """
        # 平均 K 次预测
        avg_prediction = torch.stack(predictions, dim=0).mean(dim=0)
        
        # 生成伪标签
        if self.use_soft_label:
            # 软标签：直接使用平均概率
            pseudo_labels = avg_prediction
        else:
            # 硬标签：二值化
            pseudo_labels = (avg_prediction > self.threshold).float()
        
        # 质量过滤
        valid_mask = quality_scores >= self.min_quality
        
        return pseudo_labels, valid_mask
    
    def apply_quality_weight(
        self,
        loss: torch.Tensor,
        quality_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        beta: float = 10.0,
        q0: float = 0.5
    ) -> torch.Tensor:
        """
        对损失应用质量加权（对应公式 2.3）
        
        weighted_loss = w(q) * loss, where w(q) = sigmoid(β(q - q0))
        
        Args:
            loss: 原始损失 [B]
            quality_scores: 质量分数 [B]
            valid_mask: 有效样本 mask [B]
            beta: sigmoid 斜率
            q0: sigmoid 中心点
        
        Returns:
            weighted_loss: 质量加权后的损失（已过滤无效样本）
        """
        # 计算质量权重
        weights = torch.sigmoid(beta * (quality_scores - q0))
        
        # 应用权重
        weighted_loss = loss * weights
        
        # 过滤低质量样本
        weighted_loss = weighted_loss * valid_mask.float()
        
        return weighted_loss
    
    def get_statistics(
        self,
        quality_scores: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> dict:
        """
        获取伪标签生成统计信息（用于日志）
        
        Args:
            quality_scores: 质量分数 [B]
            valid_mask: 有效样本 mask [B]
        
        Returns:
            stats: 统计字典
        """
        stats = {
            'pseudo_label_ratio': valid_mask.float().mean().item(),
            'mean_quality': quality_scores.mean().item(),
            'mean_quality_valid': quality_scores[valid_mask].mean().item() if valid_mask.any() else 0.0,
            'min_quality': quality_scores.min().item(),
            'max_quality': quality_scores.max().item(),
        }
        return stats
