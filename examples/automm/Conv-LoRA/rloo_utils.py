import math
from typing import Literal, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def compute_binary_iou(pred_mask: Tensor, gt_mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    计算二值分割的 IoU。

    参数
    ----
    pred_mask: (N, 1, H, W) 或 (N, H, W) 的 0/1 tensor
    gt_mask:   与 pred_mask 形状一致的 0/1 tensor
    """
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(1, 2, 3))
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        inter = (pred_mask * gt_mask).sum(dim=(1, 2))
        union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported mask shape: {pred_mask.shape}")

    iou = inter / (union + eps)
    return iou


def compute_binary_dice(pred_mask: Tensor, gt_mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    计算二值分割的 Dice 系数。

    参数
    ----
    pred_mask: (N, 1, H, W) 或 (N, H, W) 的 0/1 tensor
    gt_mask:   与 pred_mask 形状一致的 0/1 tensor
    """
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        size_sum = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        inter = (pred_mask * gt_mask).sum(dim=(1, 2))
        size_sum = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported mask shape: {pred_mask.shape}")

    dice = 2.0 * inter / (size_sum + eps)
    return dice


def bernoulli_kl(p_logits: Tensor, q_logits: Tensor, eps: float = 1e-6) -> Tensor:
    """
    计算两个 Bernoulli 分布之间的逐像素 KL(p || q)，并在空间维度上求和。

    参数
    ----
    p_logits: 当前策略的 logits，形状 (N, 1, H, W)
    q_logits: 参考策略的 logits，形状同上
    """
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)

    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)

    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    # 在通道与空间维度上求和，得到每个样本的 KL
    while kl.ndim > 1:
        kl = kl.sum(dim=-1)
    return kl  # 形状 (N,)


def mask_log_prob_from_logits(mask: Tensor, logits: Tensor, eps: float = 1e-6) -> Tensor:
    """
    根据给定 logits 与二值 mask，计算 Bernoulli 独立像素假设下的 log 概率。

    参数
    ----
    mask:   (N, 1, H, W) 或 (N, H, W) 的 0/1 tensor
    logits: 与 mask 形状兼容的 logits
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)

    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, eps, 1.0 - eps)

    log_p = mask * torch.log(probs) + (1.0 - mask) * torch.log(1.0 - probs)
    # 在通道与空间维度上求和，得到每个样本的 log_prob
    log_p = log_p.flatten(start_dim=1).sum(dim=1)
    return log_p  # 形状 (N,)


def compute_rloo_advantages(rewards: Tensor, normalize: bool = True) -> Tensor:
    """
    根据 RLOO 思想，计算 leave-one-out baseline 下的 advantage。

    参数
    ----
    rewards: 形状为 (B, G) 的 tensor，其中 B 为 batch size，G 为每张图像的候选数。
    normalize: 是否在 B×G 维度上对 advantage 做标准化。
    """
    if rewards.ndim != 2:
        raise ValueError(f"RLOO rewards should have shape (B, G), but got {rewards.shape}")

    B, G = rewards.shape
    # sum_j r_j
    sums = rewards.sum(dim=1, keepdim=True)  # (B, 1)
    # 对第 i 个样本：b_i = (sum_j r_j - r_i) / (G - 1)
    baselines = (sums - rewards) / max(G - 1, 1)
    advantages = rewards - baselines  # (B, G)

    if normalize:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        advantages = (advantages - mean) / (std + 1e-8)

    return advantages


def rloo_loss(
    log_probs: Tensor,
    rewards: Tensor,
    normalize_advantage: bool = True,
) -> Tensor:
    """
    计算单步在线设置下的 RLOO / REINFORCE 损失。

    参数
    ----
    log_probs: 形状为 (B, G) 的 tensor，log π(o_i | q)
    rewards:   形状为 (B, G) 的 tensor，对应每个候选的标量 reward
    normalize_advantage: 是否在 (B, G) 上对 advantage 标准化
    """
    if log_probs.shape != rewards.shape:
        raise ValueError(f"log_probs shape {log_probs.shape} != rewards shape {rewards.shape}")

    advantages = compute_rloo_advantages(rewards, normalize=normalize_advantage)
    loss = -(advantages * log_probs).mean()
    return loss


def combine_rewards(
    iou: Tensor,
    dice: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor:
    """
    一个简单的 reward 组合函数：r = α * IoU + β * Dice。
    若 dice 为 None，则只使用 IoU。
    """
    if dice is None:
        return alpha * iou
    return alpha * iou + beta * dice


RewardType = Literal["iou", "dice", "combo"]


def compute_segmentation_reward(
    pred_masks: Tensor,
    gt_masks: Tensor,
    reward_type: RewardType = "combo",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    """
    根据预测 mask 与 GT mask 计算分割 reward。

    参数
    ----
    pred_masks: (N, 1, H, W) 或 (N, H, W) 的 0/1 tensor
    gt_masks:   与 pred_masks 形状一致的 0/1 tensor
    reward_type: \"iou\" / \"dice\" / \"combo\"
    alpha, beta: 组合系数
    """
    if reward_type not in {"iou", "dice", "combo"}:
        raise ValueError(f"Unsupported reward_type={reward_type}")

    iou = compute_binary_iou(pred_masks, gt_masks)
    dice = compute_binary_dice(pred_masks, gt_masks)

    if reward_type == "iou":
        return iou
    elif reward_type == "dice":
        return dice
    else:
        return combine_rewards(iou=iou, dice=dice, alpha=alpha, beta=beta)


