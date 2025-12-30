"""
Consistency Quality Estimator

通过多次采样一致性估计伪标签质量，无需真实标注。

核心功能：
1. 计算 K 个预测间的互 IoU（一致性质量）
2. 计算预测置信度（置信度质量）
3. 融合两种质量分数

对应设计文档公式 2.2
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class ConsistencyQualityEstimator:
    """
    基于多次采样一致性的质量评估器
    
    对应设计文档中的质量评估模块（公式 2.2）：
    - q_cons = (2/K(K-1)) * Σ IoU(p_i, p_j) for i<j
    - q_conf = (1/K) * Σ mean(max(p_k, 1-p_k))
    - q = α * q_cons + (1-α) * q_conf
    
    Args:
        consistency_weight: 一致性质量的权重 α（默认 0.7）
        threshold_binarize: 二值化阈值（用于 IoU 计算，默认 0.5）
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.7,
        threshold_binarize: float = 0.5
    ):
        self.consistency_weight = consistency_weight
        self.confidence_weight = 1.0 - consistency_weight
        self.threshold = threshold_binarize
        
        print(f"[Quality Estimator] α_cons={consistency_weight}, α_conf={self.confidence_weight}")
    
    def compute_iou(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        计算两个预测 mask 之间的 IoU
        
        Args:
            pred1, pred2: 预测概率 [B, H, W]
            eps: 数值稳定项
        
        Returns:
            iou: [B] 每个样本的 IoU
        """
        # 二值化
        pred1_bin = (pred1 > self.threshold).float()
        pred2_bin = (pred2 > self.threshold).float()
        
        # 展平空间维度
        pred1_flat = pred1_bin.view(pred1.shape[0], -1)
        pred2_flat = pred2_bin.view(pred2.shape[0], -1)
        
        # 计算交并集
        intersection = (pred1_flat * pred2_flat).sum(dim=1)
        union = pred1_flat.sum(dim=1) + pred2_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + eps) / (union + eps)
        
        return iou
    
    def compute_consistency_quality(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算一致性质量 q_cons（对应公式 2.2 第一部分）
        
        q_cons = (2/K(K-1)) * Σ IoU(p_i, p_j) for i<j
        
        Args:
            predictions: K 个预测的列表，每个形状 [B, H, W]
        
        Returns:
            q_cons: [B] 每个样本的一致性质量
        """
        K = len(predictions)
        B = predictions[0].shape[0]
        
        # 累计所有配对的 IoU
        total_iou = torch.zeros(B, device=predictions[0].device)
        num_pairs = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                iou = self.compute_iou(predictions[i], predictions[j])
                total_iou += iou
                num_pairs += 1
        
        # 平均 IoU（归一化因子 2/(K(K-1))）
        q_cons = total_iou / max(num_pairs, 1)
        
        return q_cons
    
    def compute_confidence_quality(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算置信度质量 q_conf（对应公式 2.2 第二部分）
        
        q_conf = (1/K) * Σ mean(max(p_k, 1-p_k))
        
        Args:
            predictions: K 个预测的列表，每个形状 [B, H, W]
        
        Returns:
            q_conf: [B] 每个样本的置信度质量
        """
        K = len(predictions)
        
        # 累计每个预测的平均最大概率
        total_conf = 0
        
        for pred in predictions:
            # max(p, 1-p) 表示预测的确定性
            max_prob = torch.maximum(pred, 1 - pred)
            # 空间平均
            mean_conf = max_prob.view(pred.shape[0], -1).mean(dim=1)
            total_conf += mean_conf
        
        # 在 K 次采样上平均
        q_conf = total_conf / K
        
        return q_conf
    
    def estimate_quality(
        self,
        predictions: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        估计伪标签质量（对应公式 2.2 完整版）
        
        q = α * q_cons + (1-α) * q_conf
        
        Args:
            predictions: Teacher 的 K 次预测列表
        
        Returns:
            q: [B] 融合质量分数
            q_cons: [B] 一致性质量（辅助输出）
            q_conf: [B] 置信度质量（辅助输出）
        """
        # 计算两种质量分数
        q_cons = self.compute_consistency_quality(predictions)
        q_conf = self.compute_confidence_quality(predictions)
        
        # 融合（加权平均）
        q = self.consistency_weight * q_cons + self.confidence_weight * q_conf
        
        return q, q_cons, q_conf
    
    def filter_by_quality(
        self,
        quality_scores: torch.Tensor,
        min_threshold: float = 0.6
    ) -> torch.Tensor:
        """
        根据质量阈值过滤样本
        
        Args:
            quality_scores: 质量分数 [B]
            min_threshold: 最低质量阈值 q_min
        
        Returns:
            mask: [B] bool tensor，True 表示通过过滤
        """
        return quality_scores >= min_threshold
    
    def compute_quality_weight(
        self,
        quality_scores: torch.Tensor,
        beta: float = 10.0,
        q0: float = 0.5
    ) -> torch.Tensor:
        """
        计算质量加权函数（对应公式 2.3）
        
        w(q) = sigmoid(β(q - q0))
        
        Args:
            quality_scores: 质量分数 [B]
            beta: sigmoid 斜率
            q0: sigmoid 中心点
        
        Returns:
            weights: [B] 质量权重
        """
        weights = torch.sigmoid(beta * (quality_scores - q0))
        return weights
