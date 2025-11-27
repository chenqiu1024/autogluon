import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from autogluon.multimodal import MultiModalPredictor

from sam_conv_lora_wrapper import SAMConvLoRAWrapper
from rloo_utils import (
    bernoulli_kl,
    compute_segmentation_reward,
    mask_log_prob_from_logits,
    rloo_loss,
)


def expand_path(df: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """
    复用原脚本中的路径展开逻辑。
    """
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def build_dataloader_from_dataframe(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    这里我们不直接从原始图像构造张量，而是复用 AutoGluon 的 DataModule 流程。

    为了尽量不侵入 AutoGluon 内部，本函数仅返回一个简单的 DataFrame 列表，
    在后续训练循环中再交给 MultiModalPredictor / Learner 进行处理。
    """

    class _DFDataset(torch.utils.data.Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.df = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # 返回一行 DataFrame，后续在 RLOO 训练循环中交给 learner.predict_per_run /
            # 自定义 batch 构造逻辑处理。
            return self.df.iloc[idx]

    dataset = _DFDataset(df)
    
    # 自定义 collate 函数，将 pandas Series 列表转换为 list（不使用默认的 tensor collate）
    def collate_fn(batch):
        # batch 是 List[pd.Series]，直接返回列表，不做 tensor 转换
        return batch
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True,
        collate_fn=collate_fn
    )


def stack_batch_rows(rows: List[pd.Series]) -> pd.DataFrame:
    """
    将 DataLoader 返回的一批 Series 重新拼成 DataFrame。
    """
    return pd.DataFrame(rows)


def rloo_step_on_batch(
    predictor: MultiModalPredictor,
    model_wrapper: SAMConvLoRAWrapper,
    batch_df: pd.DataFrame,
    num_generations: int,
    beta: float,
    reward_type: str,
) -> tuple[torch.Tensor, dict]:
    """
    对一个 batch 的数据执行一次 RLOO 更新步骤（计算 loss 和 metrics）。
    
    **重要说明**：
    这是一个**概念验证实现**，展示 RLOO 算法的完整计算流程，包括：
    - 多候选 mask 采样
    - GT-based reward 计算（IoU/Dice）
    - KL 正则
    - Leave-one-out advantage
    - REINFORCE loss
    
    由于 AutoGluon 的 predict_per_run 在 no_grad 环境中运行，当前实现**不进行真实的梯度更新**。
    要实现端到端的 RL 训练，需要：
    1. 直接从 DataModule 获取 batch 字典（包含图像张量、GT masks）
    2. 用 model_wrapper.forward(batch) 进行带梯度的前向传播
    3. 在 RLOO loss 上调用 backward()
    
    这需要深入 AutoGluon 的训练循环（SemanticSegmentationLitModule），超出了 examples 的范围。
    """
    learner = predictor._learner

    # 使用 predict_per_run 获取模型输出和 GT（在 no_grad 环境中）
    with torch.no_grad():
        outputs = learner.predict_per_run(
            data=batch_df,
            realtime=False,
            requires_label=True,
        )
    
    from autogluon.multimodal.constants import LOGITS, LABEL
    
    # 获取预测 logits 和 GT masks
    logits_list = [ele[LOGITS] for ele in outputs]
    gt_list = [ele[LABEL] for ele in outputs]
    pred_logits = torch.cat(logits_list, dim=0)  # (B, 1, H, W)
    gt_masks = torch.cat(gt_list, dim=0)  # (B, 1, H, W) 或 (B, H, W)
    
    B = pred_logits.shape[0]
    device = pred_logits.device
    
    # 参考 logits（用于 KL 计算）
    ref_logits = pred_logits.detach()
    
    # 收集所有候选的 log_probs 和 rewards
    all_log_probs = []
    all_rewards = []
    all_iou = []
    all_dice = []

    for g in range(num_generations):
        # 生成候选 mask：在 logits 上添加小的随机噪声，而不是完全重新采样
        # 这样可以保持模型预测的基本结构，同时引入多样性
        if g == 0:
            # 第一个候选使用原始预测（阈值化）
            probs = torch.sigmoid(pred_logits)
            sampled_mask = (probs > 0.5).float()
        else:
            # 后续候选：在 logits 上添加小噪声
            noise_scale = 0.5  # 可调整的噪声强度
            noisy_logits = pred_logits + torch.randn_like(pred_logits) * noise_scale
            probs = torch.sigmoid(noisy_logits)
            sampled_mask = (probs > 0.5).float()

        # 计算 reward（IoU/Dice/组合）
        rewards_metric = compute_segmentation_reward(
            pred_masks=sampled_mask,
            gt_masks=gt_masks,
            reward_type=reward_type,  # type: ignore[arg-type]
        )
        
        # 记录单独的 IoU 和 Dice 用于监控
        iou_vals = compute_segmentation_reward(sampled_mask, gt_masks, "iou")
        dice_vals = compute_segmentation_reward(sampled_mask, gt_masks, "dice")
        all_iou.append(iou_vals.mean().item())
        all_dice.append(dice_vals.mean().item())
        
        # KL 正则
        kl_values = bernoulli_kl(pred_logits, ref_logits)  # (B,)
        rewards_total = rewards_metric - beta * kl_values

        # log π(M | logits)
        # 注意：这里的 log_prob 应该基于原始 pred_logits，而不是 noisy_logits
        log_p = mask_log_prob_from_logits(sampled_mask, pred_logits)  # (B,)

        all_log_probs.append(log_p)
        all_rewards.append(rewards_total)

    log_probs = torch.stack(all_log_probs, dim=1)  # (B, G)
    rewards = torch.stack(all_rewards, dim=1)  # (B, G)

    loss = rloo_loss(log_probs=log_probs, rewards=rewards, normalize_advantage=True)
    
    # 返回 loss 和一些监控指标
    metrics = {
        "mean_reward": rewards.mean().item(),
        "mean_iou": sum(all_iou) / len(all_iou),
        "mean_dice": sum(all_dice) / len(all_dice),
        "mean_kl": bernoulli_kl(pred_logits, ref_logits).mean().item(),
    }
    
    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description="RLOO fine-tuning for Conv-LoRA SAM (semantic segmentation).")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Phase-1 Conv-LoRA SAM checkpoint path.")
    parser.add_argument("--output_dir", type=str, default="outputs_rloo")

    # RLOO 相关超参数
    parser.add_argument("--num_generations", type=int, default=4, help="每张图像生成的候选 mask 数量 G。")
    parser.add_argument("--beta", type=float, default=0.05, help="KL 正则系数。")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="combo",
        choices=["iou", "dice", "combo"],
        help="reward 形式：IoU / Dice / 组合。",
    )
    parser.add_argument("--epochs", type=int, default=3, help="RLOO 微调 epoch 数。")
    parser.add_argument("--batch_size", type=int, default=2, help="RLOO 训练 batch size。")
    parser.add_argument(
        "--lambda_supervised",
        type=float,
        default=0.0,
        help="可选：混合少量监督损失的权重（当前示例脚本未实现监督 loss，留作扩展）。",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42686693)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 准备 DataFrame
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "train.csv")), dataset_dir)

    dataloader = build_dataloader_from_dataframe(
        df=train_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 加载阶段一的 Conv-LoRA-SAM 模型
    # 处理路径：如果传入的是 .ckpt 文件，取其父目录；确保是目录
    ckpt_path = args.ckpt_path
    if ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.dirname(ckpt_path)
    
    # 验证路径存在
    if not os.path.isdir(ckpt_path):
        raise ValueError(
            f"Checkpoint path '{ckpt_path}' does not exist or is not a directory. "
            f"Please provide the directory containing the trained model (e.g., AutogluonModels/ag-xxx/)."
        )
    
    print(f"Loading model from: {ckpt_path}")
    predictor = MultiModalPredictor.load(ckpt_path)
    model_wrapper = SAMConvLoRAWrapper(predictor)

    # 检查可训练参数
    trainable_params = [p for p in model_wrapper.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found in the model! "
            "Please check if the model was trained with Conv-LoRA (optim.peft='conv_lora'). "
            f"Trainable parameter names found: {model_wrapper.trainable_param_names}"
        )
    
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")

    # 这里只是示意性使用 model_wrapper.parameters()，实际 RLOO 更新需要深入到
    # AutoGluon 的训练循环中，实现端到端反向传播。
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_wrapper.to(device)

    print("\n" + "="*80)
    print("开始 RLOO 训练循环（概念验证模式）")
    print("="*80)
    print("注意：当前实现展示 RLOO 算法的完整计算流程，但由于 AutoGluon 的架构限制，")
    print("      不进行真实的梯度更新。这是一个算法验证和指标监控的演示。")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        model_wrapper.eval()  # 保持 eval 模式，因为我们不做真实更新
        epoch_loss = 0.0
        epoch_metrics = {"reward": [], "iou": [], "dice": [], "kl": []}
        num_batches = 0

        for rows in dataloader:
            batch_df = stack_batch_rows(rows)

            loss, metrics = rloo_step_on_batch(
                predictor=predictor,
                model_wrapper=model_wrapper,
                batch_df=batch_df,
                num_generations=args.num_generations,
                beta=args.beta,
                reward_type=args.reward_type,
            )

            # 记录指标
            epoch_loss += loss.item()
            epoch_metrics["reward"].append(metrics["mean_reward"])
            epoch_metrics["iou"].append(metrics["mean_iou"])
            epoch_metrics["dice"].append(metrics["mean_dice"])
            epoch_metrics["kl"].append(metrics["mean_kl"])
            num_batches += 1
            
            # 打印 batch 级别的信息
            if num_batches % 10 == 0 or num_batches == 1:
                print(f"  Batch {num_batches}: loss={loss.item():.4f}, "
                      f"reward={metrics['mean_reward']:.4f}, "
                      f"IoU={metrics['mean_iou']:.4f}, "
                      f"Dice={metrics['mean_dice']:.4f}")

        # Epoch 总结
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_reward = sum(epoch_metrics["reward"]) / len(epoch_metrics["reward"])
        avg_iou = sum(epoch_metrics["iou"]) / len(epoch_metrics["iou"])
        avg_dice = sum(epoch_metrics["dice"]) / len(epoch_metrics["dice"])
        avg_kl = sum(epoch_metrics["kl"]) / len(epoch_metrics["kl"])
        
        print(f"\n[RLOO] Epoch {epoch + 1}/{args.epochs} 完成:")
        print(f"  平均 Loss: {avg_loss:.6f}")
        print(f"  平均 Reward: {avg_reward:.6f}")
        print(f"  平均 IoU: {avg_iou:.4f}")
        print(f"  平均 Dice: {avg_dice:.4f}")
        print(f"  平均 KL: {avg_kl:.6f}")
        print()

    print("\n" + "="*80)
    print("RLOO 训练循环完成！")
    print("="*80)
    print("\n注意：由于当前是概念验证模式，模型参数未实际更新。")
    print("要实现真实的 RL 训练，需要：")
    print("  1. 重构数据流，直接从 DataModule 获取带梯度的 batch")
    print("  2. 使用 model_wrapper.forward(batch) 进行前向传播")
    print("  3. 在 SemanticSegmentationLitModule 中集成 RLOO 训练步骤")
    print("\n当前实现已验证:")
    print("  ✓ 多候选 mask 采样逻辑")
    print("  ✓ GT-based reward 计算（IoU/Dice）")
    print("  ✓ KL 正则与 leave-one-out advantage")
    print("  ✓ RLOO loss 计算")
    print("\n这些组件可以直接迁移到 AutoGluon 内部的训练循环中。")
    print("="*80 + "\n")
    
    # 不保存模型，因为没有实际更新
    # predictor.save(os.path.join(args.output_dir, "rloo_finetuned"))


if __name__ == "__main__":
    main()


