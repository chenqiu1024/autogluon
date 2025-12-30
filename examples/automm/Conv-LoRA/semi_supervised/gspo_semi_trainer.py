"""
GSPO Semi-Supervised Trainer Extension

扩展原有 GSPO训练器以支持半监督学习

核心功能：
1. 接受质量 proxy 作为 advantage 的输入
2. 对有标注和弱标注样本统一处理
3. 集成 EMA Teacher、质量评估器、伪标签生成器

对应设计文档公式 2.5
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os

# 添加父目录到路径以导入 gspo_trainer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gspo_trainer import GSPOConvLoRATrainer


class GSPOSemiSupervisedTrainer(GSPOConvLoRATrainer):
    """
    扩展 GSPO 训练器以支持半监督学习
    
    主要扩展：
    1. compute_segmentation_quality 支持 quality_proxy 参数
    2. 新增 compute_semi_supervised_loss 方法
    3. 新增 Box Jitter 一致性损失
    
    对应公式 2.5：
        q_quality = IoU(pred, gt) if labeled else q_cons
        A = q_quality - mean(q_g)
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        # 半监督特定参数
        pseudo_lambda_init: float = 0.0,
        pseudo_lambda_final: float = 1.0,
        pseudo_lambda_warmup_epochs: int = 5,
        consistency_lambda: float = 0.1,
        **kwargs
    ):
        # 调用父类初始化
        super().__init__(
            predictor=predictor,
            group_size=group_size,
            warmup_epochs=warmup_epochs,
            contrastive_weight=contrastive_weight,
            quality_metric=quality_metric,
            advantage_temperature=advantage_temperature,
            **kwargs
        )
        
        # 半监督特定参数
        self.pseudo_lambda_init = pseudo_lambda_init
        self.pseudo_lambda_final = pseudo_lambda_final
        self.pseudo_lambda_warmup_epochs = pseudo_lambda_warmup_epochs
        self.consistency_lambda = consistency_lambda
        
        print(f"[GSPO Semi] λ_u warmup: {pseudo_lambda_init} -> {pseudo_lambda_final} "
              f"over {pseudo_lambda_warmup_epochs} epochs")
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
        metric: str = 'iou',
        quality_proxy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        扩展质量计算以支持 quality_proxy（对应公式 2.5）
        
        q_quality = IoU(pred, gt) if gt is not None else quality_proxy
        
        Args:
            pred_masks: 预测 mask [B, H, W]
            gt_masks: 真实 mask [B, H, W]（可选）
            metric: 质量度量类型
            quality_proxy: 质量代理（用于弱标注样本）[B]
        
        Returns:
            quality_scores: [B] 质量分数
        """
        if gt_masks is not None:
            # 有标注：使用真实 IoU/Dice
            return super().compute_segmentation_quality(pred_masks, gt_masks, metric)
        elif quality_proxy is not None:
            # 弱标注：使用一致性质量 proxy
            return quality_proxy
        else:
            raise ValueError("必须提供 gt_masks 或 quality_proxy 之一")
    
    def get_pseudo_lambda(self, epoch: int) -> float:
        """
        获取当前 epoch 的伪监督权重 λ_u（对应公式 2.4）
        
        λ_u(t) = min(1.0, t / T_warmup) * (λ_final - λ_init) + λ_init
        
        Args:
            epoch: 当前 epoch
        
        Returns:
            lambda_u: 伪监督权重
        """
        if epoch < self.pseudo_lambda_warmup_epochs:
            progress = epoch / self.pseudo_lambda_warmup_epochs
        else:
            progress = 1.0
        
        lambda_u = self.pseudo_lambda_init + progress * (
            self.pseudo_lambda_final - self.pseudo_lambda_init
        )
        
        return lambda_u
    
    def compute_box_jitter_consistency_loss(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        loss_type: str = 'kl'
    ) -> torch.Tensor:
        """
        计算 Box Jitter 一致性损失（对应 Phase 4）
        
        L_cons = KL(p1 || p2) or MSE(p1, p2)
        
        Args:
            pred1, pred2: 两个不同 box prompt 的预测 [B, H, W]
            loss_type: 'kl' 或 'mse'
        
        Returns:
            loss: 一致性损失标量
        """
        if loss_type == 'kl':
            # KL 散度（需要 log-prob）
            pred1 = pred1.clamp(1e-7, 1 - 1e-7)
            pred2 = pred2.clamp(1e-7, 1 - 1e-7)
            
            kl_loss = pred1 * (torch.log(pred1) - torch.log(pred2))
            loss = kl_loss.mean()
        
        elif loss_type == 'mse':
            # MSE
            loss = F.mse_loss(pred1, pred2)
        
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        return loss
    
    def compute_semi_supervised_loss(
        self,
        student_pred: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
        pseudo_label: Optional[torch.Tensor],
        quality_weight: Optional[torch.Tensor],
        is_labeled: torch.Tensor,
        epoch: int,
        loss_fn,
        box_jitter_pred1: Optional[torch.Tensor] = None,
        box_jitter_pred2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算半监督损失（监督 + 伪监督 + 一致性）
        
        对应完整损失公式 2.4：
        L = L_s + λ_u(t) * Σ w(q_i) * L_u + λ_c * L_cons
        
        Args:
            student_pred: Student 预测 [B, H, W]
            gt_mask: 真实 mask [B, H, W]（仅对 labeled 样本有效）
            pseudo_label: 伪标签 [B, H, W]（仅对 weak 样本）
            quality_weight: 质量权重 [B]（仅对 weak 样本）
            is_labeled: 样本类型 mask [B]
            epoch: 当前 epoch
            loss_fn: 分割损失函数
            box_jitter_pred1: Box jitter 预测 1 [B, H, W]（可选）
            box_jitter_pred2: Box jitter 预测 2 [B, H, W]（可选）
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失分解字典
        """
        loss_dict = {}
        
        # 1. 监督损失（有标注样本）- L_s
        if is_labeled.any():
            labeled_pred = student_pred[is_labeled]
            labeled_gt = gt_mask[is_labeled] if gt_mask is not None else None
            
            if labeled_gt is not None:
                loss_supervised = loss_fn(labeled_pred, labeled_gt)
                loss_dict['loss_supervised'] = loss_supervised.item()
            else:
                loss_supervised = torch.tensor(0.0, device=student_pred.device)
                loss_dict['loss_supervised'] = 0.0
        else:
            loss_supervised = torch.tensor(0.0, device=student_pred.device)
            loss_dict['loss_supervised'] = 0.0
        
        # 2. 伪监督损失（弱标注样本）- λ_u * L_u
        weak_mask = ~is_labeled
        if weak_mask.any() and pseudo_label is not None:
            weak_pred = student_pred[weak_mask]
            weak_pseudo = pseudo_label[weak_mask]
            weak_weight = quality_weight[weak_mask] if quality_weight is not None else None
            
            # 计算未加权的伪监督损失
            loss_pseudo_raw = loss_fn(weak_pred, weak_pseudo)
            
            # 应用质量加权
            if weak_weight is not None:
                loss_pseudo = (loss_pseudo_raw * weak_weight).mean()
            else:
                loss_pseudo = loss_pseudo_raw.mean()
            
            # 应用 warmup 权重
            lambda_u = self.get_pseudo_lambda(epoch)
            loss_pseudo_weighted = lambda_u * loss_pseudo
            
            loss_dict['loss_pseudo'] = loss_pseudo.item()
            loss_dict['lambda_u'] = lambda_u
        else:
            loss_pseudo_weighted = torch.tensor(0.0, device=student_pred.device)
            loss_dict['loss_pseudo'] = 0.0
            loss_dict['lambda_u'] = 0.0
        
        # 3. Box Jitter 一致性损失 - λ_c * L_cons
        if box_jitter_pred1 is not None and box_jitter_pred2 is not None:
            loss_consistency = self.compute_box_jitter_consistency_loss(
                box_jitter_pred1, box_jitter_pred2, loss_type='kl'
            )
            loss_consistency_weighted = self.consistency_lambda * loss_consistency
            loss_dict['loss_consistency'] = loss_consistency.item()
            loss_dict['lambda_c'] = self.consistency_lambda
        else:
            loss_consistency_weighted = torch.tensor(0.0, device=student_pred.device)
            loss_dict['loss_consistency'] = 0.0
            loss_dict['lambda_c'] = 0.0
        
        # 总损失（对应公式 2.4）
        # L = L_s + λ_u * L_u + λ_c * L_cons
        total_loss = loss_supervised + loss_pseudo_weighted + loss_consistency_weighted
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
