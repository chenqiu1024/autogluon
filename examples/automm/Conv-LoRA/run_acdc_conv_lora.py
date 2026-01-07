#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 ACDC (来自 ABD 预处理数据) 上运行 Conv-LoRA + SAM + GSPO + Adapter 的
训练 / 推理脚本。数据准备需先运行 prepare_acdc_conv_lora.py。
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor


def expand_path(df: pd.DataFrame, dataset_dir: str):
    df = df.copy()
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda p: os.path.join(dataset_dir, p))
    return df


def visualize_samples(predictor: MultiModalPredictor, df: pd.DataFrame, vis_dir: str, max_samples: int = 8):
    """
    仅保存前 max_samples 个验证样本的预测掩码，避免磁盘爆炸。
    会覆盖同名文件，文件数量固定。
    """
    from PIL import Image
    os.makedirs(vis_dir, exist_ok=True)
    subset = df.head(max_samples).copy()
    try:
        preds = predictor.predict(subset)
    except Exception as e:
        print(f"[VIS] predictor.predict failed: {e}")
        return
    
    # preds 对于语义分割是一个 list of numpy arrays
    if isinstance(preds, list):
        for idx, (_, row) in enumerate(subset.iterrows()):
            if idx >= len(preds):
                break
            mask = preds[idx]
            if mask is None:
                continue
            # 保存 mask 为 PNG
            mask_arr = np.array(mask, dtype=np.uint8)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze()  # 移除多余维度
            img_name = os.path.splitext(os.path.basename(row["image"]))[0]
            out_path = os.path.join(vis_dir, f"{img_name}_pred.png")
            Image.fromarray(mask_arr).save(out_path)
        print(f"[VIS] Saved up to {len(subset)} predicted masks to {vis_dir}")
    elif hasattr(preds, 'columns') and "semantic_mask" in preds.columns:
        for i, row in subset.iterrows():
            mask = preds.loc[i, "semantic_mask"]
            if mask is None:
                continue
            mask_arr = np.array(mask, dtype=np.uint8)
            img_name = os.path.splitext(os.path.basename(row["image"]))[0]
            out_path = os.path.join(vis_dir, f"{img_name}_pred.png")
            Image.fromarray(mask_arr).save(out_path)
        print(f"[VIS] Saved up to {len(subset)} predicted masks to {vis_dir}")
    else:
        print(f"[VIS] Unexpected prediction format: {type(preds)}; skip visualization.")


def main():
    parser = argparse.ArgumentParser(description="ACDC semantic segmentation with Conv-LoRA + GSPO + Adapter")
    parser.add_argument("--dataset_dir", type=str, default="datasets/acdc_conv_lora/acdc_conv_lora",
                        help="由 prepare_acdc_conv_lora.py 生成的数据集根目录")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/acdc_conv_lora")
    parser.add_argument("--ckpt_path", type=str, default=None, help="若只做评估，指定已训练模型路径")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="有效 batch size；若大于 per_gpu_batch_size*num_gpus，则会做累积")
    parser.add_argument("--eval_only", action="store_true", help="只做评估，不训练")
    # GSPO
    parser.add_argument("--gspo_enable", action="store_true")
    parser.add_argument("--gspo_group_size", type=int, default=4)
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5)
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9)
    # GSPO-Adapter 扩展
    parser.add_argument("--gspo_adapter_enable", action="store_true",
                        help="启用 GSPO 对 Encoder Adapter 的质量反馈")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9,
                        help="GSPO Adapter 质量历史的动量项")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="启用基于质量历史的自适应缩放")
    # Adapter
    parser.add_argument("--adapter_enable", action="store_true")
    parser.add_argument("--adapter_dim", type=int, default=64)
    # 可视化配置
    parser.add_argument("--vis_samples", type=int, default=8,
                        help="训练结束后从验证集可视化的样本数（固定覆盖，不额外增长）")
    parser.add_argument("--vis_output_dir", type=str, default="outputs/acdc_vis",
                        help="可视化输出目录")
    # Training epochs control
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (set large to disable early stopping)")
    # Loss function
    parser.add_argument("--loss", type=str, default="mask2former_loss",
                        choices=["dice_ce_loss", "mask2former_loss"],
                        help="Loss function: mask2former_loss (recommended for SAM multi-class) or dice_ce_loss")
    # Number of mask tokens (critical for Mask2Former-style multi-class segmentation)
    parser.add_argument("--num_mask_tokens", type=int, default=10,
                        help="Number of mask tokens/queries (should be >= num_classes for Mask2Former, recommended 10+)")
    # Quick / Debug
    parser.add_argument("--debug", action="store_true", help="仅处理少量样本以快速验证")
    parser.add_argument("--quick_test", type=int, default=None, help="仅处理前 N 个测试样本")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 CSV
    train_df = expand_path(pd.read_csv(os.path.join(args.dataset_dir, "train.csv")), args.dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(args.dataset_dir, "val.csv")), args.dataset_dir)
    test_df = expand_path(pd.read_csv(os.path.join(args.dataset_dir, "test.csv")), args.dataset_dir)

    # 快速模式
    if args.debug:
        train_df = train_df.head(20)
        val_df = val_df.head(10)
        test_df = test_df.head(5)
    if args.quick_test is not None:
        test_df = test_df.head(args.quick_test)

    # 默认训练设置（ACDC 多类别分割：0 背景 + 3 前景）
    validation_metric = "dice"
    lr = 1e-4

    print(f"\n{'='*60}")
    print(f"Loss function: {args.loss}")
    print(f"Max epochs: {args.max_epochs}, Patience: {args.patience}")
    print(f"Num mask tokens: {args.num_mask_tokens} (critical for Mask2Former multi-class)")
    print(f"{'='*60}\n")

    hyperparameters = {
        "optim.peft": "conv_lora",
        "optim.lora.r": args.rank,
        "optim.lora.conv_lora_expert_num": args.expert_num,
        "env.num_gpus": args.num_gpus,
        "optim.loss_func": args.loss,
        "optim.max_epochs": args.max_epochs,
        "optim.patience": args.patience,
        "optim.lr": lr,
        "env.per_gpu_batch_size": args.per_gpu_batch_size,
        "env.batch_size": args.batch_size,
        # Critical for Mask2Former-style multi-class segmentation: need enough mask tokens
        "model.sam.num_mask_tokens": args.num_mask_tokens,
    }

    if args.gspo_enable:
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
        })

    if args.adapter_enable:
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        # GSPO-Adapter 扩展
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
                hyperparameters.update({
                    "optim.lora.gspo_enabled": True,
                    "optim.lora.gspo_group_size": args.gspo_group_size,
                    "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
                    "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
                    "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
                })
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            if args.gspo_adapter_scale_adaptation:
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })

    if args.eval_only:
        if not args.ckpt_path:
            raise ValueError("--eval_only 需要指定 --ckpt_path")
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # 评估（IoU + Dice）
    res = predictor.evaluate(test_df, metrics=["iou", "dice"])
    print(f"Test results on ACDC: {res}")
    with open(os.path.join(args.output_dir, "metrics_acdc.txt"), "a") as f:
        f.write(f"{res}\n")

    # 可视化一小部分验证样本的预测（覆盖写，数量固定）
    visualize_samples(predictor, val_df, args.vis_output_dir, max_samples=args.vis_samples)


if __name__ == "__main__":
    main()

