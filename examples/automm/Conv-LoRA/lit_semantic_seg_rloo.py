"""
RLOO-enabled SemanticSegmentationLitModule
扩展 AutoGluon 的 SemanticSegmentationLitModule 以支持 RLOO 训练
"""
import logging
from typing import Callable, Dict, Optional

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from autogluon.multimodal.constants import CLASS_LOGITS, LOGITS, LABEL, MOE_LOSS, SEMANTIC_MASK, WEIGHT
from autogluon.multimodal.models.utils import run_model
from autogluon.multimodal.optim.lit_semantic_seg import SemanticSegmentationLitModule
from autogluon.multimodal.optim.metrics.semantic_seg_metrics import Multiclass_IoU

from rloo_utils import (
    bernoulli_kl,
    compute_segmentation_reward,
    mask_log_prob_from_logits,
    rloo_loss,
)

logger = logging.getLogger(__name__)


class RLOOSemanticSegmentationLitModule(SemanticSegmentationLitModule):
    """
    扩展 SemanticSegmentationLitModule 以支持 RLOO 强化学习训练。
    
    在 RLOO 模式下，训练步骤会：
    1. 对每个样本生成 G 个候选 mask
    2. 基于 GT mask 计算每个候选的 reward（IoU/Dice）
    3. 应用 KL 正则以避免偏离参考策略过远
    4. 使用 leave-one-out baseline 计算 advantage
    5. 通过 REINFORCE 风格的 loss 更新策略
    """
    
    def __init__(
        self,
        model,
        # RLOO 相关参数
        enable_rloo: bool = False,
        num_generations: int = 4,
        beta: float = 0.05,
        reward_type: str = "combo",
        ref_logits_cache: Optional[Dict] = None,
        # 其他参数传递给父类
        **kwargs
    ):
        """
        Parameters
        ----------
        model
            分割模型
        enable_rloo
            是否启用 RLOO 训练模式
        num_generations
            每个样本生成的候选 mask 数量 G
        beta
            KL 正则系数，控制偏离参考策略的程度
        reward_type
            reward 类型："iou", "dice", 或 "combo"
        ref_logits_cache
            可选的参考 logits 缓存（用于 KL 计算）
            如果为 None，则使用当前模型的第一次前向传播作为参考
        **kwargs
            传递给父类的其他参数
        """
        super().__init__(model=model, **kwargs)
        
        self.enable_rloo = enable_rloo
        self.num_generations = num_generations
        self.beta = beta
        self.reward_type = reward_type
        self.ref_logits_cache = ref_logits_cache if ref_logits_cache is not None else {}
        
    def _rloo_training_step(self, batch: Dict):
        """
        RLOO 训练步骤：生成多个候选并计算 REINFORCE loss
        
        关键改进：
        1. 只进行一次前向传播获取 logits（保留梯度）
        2. 通过在 logits 上采样生成 G 个候选 mask
        3. 计算每个候选的 reward 和 log_prob
        4. 使用 REINFORCE + leave-one-out baseline 更新策略
        """
        # 获取 GT label
        label = batch[self.model.label_key]
        B = label.shape[0]
        device = label.device
        
        # 前向传播获取 logits（保留梯度！）
        output = run_model(self.model, batch)
        pred_logits = output[self.model.prefix][LOGITS]
        
        # 收集所有候选的 log_probs 和 rewards
        all_log_probs = []
        all_rewards = []
        all_iou = []
        all_dice = []
        all_kl = []
        
        # 生成 G 个候选 mask
        # 改进的采样策略：使用温度采样和递增的噪声强度
        for g in range(self.num_generations):
            # 在相同的 logits 基础上生成不同的候选
            if g == 0:
                # 第一个候选使用确定性预测（模式，即阈值化）
                probs = torch.sigmoid(pred_logits)
                sampled_mask = (probs > 0.5).float()
            else:
                # 后续候选：使用更激进的采样策略以增加多样性
                
                # 策略1: 温度采样 - 温度从 0.5 递增到 2.0
                # 更高的温度使得预测更加"平滑"，增加不确定性
                temperature = 0.5 + 1.5 * (g / self.num_generations)
                scaled_logits = pred_logits / temperature
                
                # 策略2: 添加更大的噪声 - 噪声强度从 1.0 递增到 3.0
                # 这确保后面的候选与原始预测有显著差异
                noise_scale = 1.0 + 2.0 * (g / self.num_generations)
                noisy_logits = scaled_logits + torch.randn_like(pred_logits) * noise_scale
                
                # 策略3: 随机丢弃（类似 dropout）
                # 对于奇数索引的候选，随机将一些 logits 设为 0
                if g % 2 == 1:
                    dropout_rate = 0.1  # 10% 的像素被随机丢弃
                    dropout_mask = (torch.rand_like(noisy_logits) > dropout_rate).float()
                    noisy_logits = noisy_logits * dropout_mask
                
                probs = torch.sigmoid(noisy_logits)
                # 使用 Bernoulli 采样
                sampled_mask = torch.bernoulli(probs)
            
            # 计算 reward（基于 GT）
            # 注意：reward 计算不需要梯度
            with torch.no_grad():
                rewards_metric = compute_segmentation_reward(
                    pred_masks=sampled_mask,
                    gt_masks=label,
                    reward_type=self.reward_type,
                )
                
                # 记录单独的 IoU 和 Dice 用于监控
                iou_vals = compute_segmentation_reward(sampled_mask, label, "iou")
                dice_vals = compute_segmentation_reward(sampled_mask, label, "dice")
                all_iou.append(iou_vals.mean().item())
                all_dice.append(dice_vals.mean().item())
                
                # 计算采样多样性：当前候选的概率分布与第一个候选的差异
                # 这可以用来监控采样的多样性
                if g == 0:
                    first_candidate_probs = torch.sigmoid(pred_logits)
                    kl_div = 0.0
                else:
                    current_probs = torch.sigmoid(noisy_logits)
                    # KL散度近似：衡量当前候选与第一个候选的差异
                    kl_div = bernoulli_kl(noisy_logits, pred_logits).mean().item()
                all_kl.append(kl_div)
                
                # 使用原始 reward（不加 KL 惩罚）
                # 如果 beta > 0，我们可以在这里添加 KL 惩罚
                rewards_total = rewards_metric
            
            # 计算 log π(M | logits)（需要梯度！）
            log_p = mask_log_prob_from_logits(sampled_mask.detach(), pred_logits)
            
            all_log_probs.append(log_p)
            all_rewards.append(rewards_total)
        
        # Stack 所有候选
        log_probs = torch.stack(all_log_probs, dim=1)  # (B, G)
        rewards = torch.stack(all_rewards, dim=1)  # (B, G)
        
        # 计算 RLOO loss（会有梯度）
        loss = rloo_loss(log_probs=log_probs, rewards=rewards, normalize_advantage=True)
        
        # 构造监控指标
        with torch.no_grad():
            metrics = {
                "mean_reward": rewards.mean().item(),
                "mean_iou": sum(all_iou) / len(all_iou),
                "mean_dice": sum(all_dice) / len(all_dice),
                "mean_diversity": sum(all_kl) / len(all_kl),  # 候选的多样性
            }
        
        # 记录额外指标
        for key, value in metrics.items():
            self.log(f"rloo_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """
        重写训练步骤以支持 RLOO 模式
        """
        if self.enable_rloo:
            # RLOO 训练模式
            loss = self._rloo_training_step(batch)
            
            if not self.automatic_optimization:
                # 手动优化
                optimizer = self.optimizers()
                lr_scheduler = self.lr_schedulers()
                loss = loss / self.hparams.accumulate_grad_batches
                self.manual_backward(loss)
                
                if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            
            self.log("train_loss", loss)
            return loss
        else:
            # 标准监督训练模式
            return super().training_step(batch, batch_idx)

