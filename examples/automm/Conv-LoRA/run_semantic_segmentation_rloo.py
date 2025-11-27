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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


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
) -> torch.Tensor:
    """
    对一个 batch 的数据执行一次 RLOO 更新步骤（计算 loss，不做 optimizer.step）。

    注意：为了与 AutoGluon 当前的数据流保持一致，这里简化实现为：
    - 使用 `predictor._learner.predict_per_run` 的内部数据处理和 DataModule 构造逻辑，
      生成包含 logits 的输出；
    - 由于 AutoGluon 的预测流程默认在 no_grad 环境下执行，真实场景中需要
      进一步深入到 model / datamodule 级别改造，才能实现真正的端到端梯度回传。
    - 因此，本示例脚本主要提供整体算法结构与接口设计，便于后续在 AutoGluon
      内部做更紧密的集成。
    """
    learner = predictor._learner

    # 这里为了示意，仅演示从 learner 获得模型输出并计算 reward / KL 的结构，
    # 不强行打破 AutoGluon 的 no_grad 推理逻辑。
    with torch.no_grad():
        # 先做一次预测，拿到 logits（或 semantic masks）用于 reward 计算。
        outputs = learner.predict_per_run(
            data=batch_df,
            realtime=False,
            requires_label=True,
        )

    # 从 outputs 中取出预测 logits 与 GT label
    from autogluon.multimodal.constants import LOGITS, LABEL

    logits_list = [ele[LOGITS] for ele in outputs]
    gt_list = [ele[LABEL] for ele in outputs]
    pred_logits = torch.cat(logits_list, dim=0)  # (B, 1, H, W)
    gt_masks = torch.cat(gt_list, dim=0)  # (B, 1, H, W) 或 (B, H, W)

    B = pred_logits.shape[0]
    device = pred_logits.device

    # 简化版：在无梯度环境下，仍然演示 RLOO 的 reward / KL / advantage / loss 计算流程。
    # 实际想要端到端更新 Conv-LoRA 参数，需要将下述逻辑迁移到 model + datamodule 的
    # 显式训练循环中，并移除 no_grad。

    all_log_probs = []
    all_rewards = []

    # 参考策略 logits（此处简单使用当前 predictor 的输出作为参考，可在实际实现中
    # 另行拷贝一份 reference 模型）
    ref_logits = pred_logits.detach()

    for _ in range(num_generations):
        # 重新采样 Bernoulli mask
        probs = torch.sigmoid(pred_logits)
        sampled_mask = torch.bernoulli(probs).to(device)

        # reward: IoU / Dice / 组合
        rewards_metric = compute_segmentation_reward(
            pred_masks=sampled_mask,
            gt_masks=gt_masks,
            reward_type=reward_type,  # type: ignore[arg-type]
        )

        # KL 正则：当前 logits vs 参考 logits
        kl_values = bernoulli_kl(pred_logits, ref_logits)  # (B,)
        rewards_total = rewards_metric - beta * kl_values

        # log π(M | logits)
        log_p = mask_log_prob_from_logits(sampled_mask, pred_logits)  # (B,)

        all_log_probs.append(log_p)
        all_rewards.append(rewards_total)

    log_probs = torch.stack(all_log_probs, dim=1)  # (B, G)
    rewards = torch.stack(all_rewards, dim=1)  # (B, G)

    loss = rloo_loss(log_probs=log_probs, rewards=rewards, normalize_advantage=True)
    return loss


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
    predictor = MultiModalPredictor.load(args.ckpt_path)
    model_wrapper = SAMConvLoRAWrapper(predictor)

    # 这里只是示意性使用 model_wrapper.parameters()，实际 RLOO 更新需要深入到
    # AutoGluon 的训练循环中，实现端到端反向传播。
    optimizer = torch.optim.AdamW(
        [p for p in model_wrapper.parameters() if p.requires_grad],
        lr=1e-5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_wrapper.to(device)

    for epoch in range(args.epochs):
        model_wrapper.train()
        epoch_loss = 0.0
        num_batches = 0

        for rows in dataloader:
            batch_df = stack_batch_rows(rows)

            loss = rloo_step_on_batch(
                predictor=predictor,
                model_wrapper=model_wrapper,
                batch_df=batch_df,
                num_generations=args.num_generations,
                beta=args.beta,
                reward_type=args.reward_type,
            )

            # 由于上面的 rloo_step_on_batch 当前在 no_grad 环境下计算，
            # 这里的 loss 不会产生真实梯度，示例中仅展示优化步骤结构：
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[RLOO] Epoch {epoch + 1}/{args.epochs}, avg loss = {avg_loss:.6f}")

    # 保存更新后的模型（示意）
    predictor.save(os.path.join(args.output_dir, "rloo_finetuned"))


if __name__ == "__main__":
    main()


