#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 ACDC (来自 ABD 预处理数据) 上运行 Conv-LoRA + SAM + GSPO + Adapter 的
半/弱监督训练与推理脚本。数据准备需先运行 prepare_acdc_conv_lora.py。

相较 run_acdc_conv_lora.py，这个脚本额外支持：
- semi_labeled_fraction / semi_labeled_seed：只用一部分样本的精确掩码，剩余样本用弱监督盒约束
- weak_box_* / weak_loss_*：弱监督盒生成与约束超参
- gspo_allow_semisup：在半监督时允许 GSPO 仅对有标签子集启用
"""
import argparse
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor
from PIL import Image


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
            mask_arr = np.array(mask, dtype=np.uint8)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze()
            img_name = os.path.splitext(os.path.basename(row["image"]))[0]
            out_path = os.path.join(vis_dir, f"{img_name}_pred.png")
            Image.fromarray(mask_arr).save(out_path)
        print(f"[VIS] Saved up to {len(subset)} predicted masks to {vis_dir}")
    elif hasattr(preds, "columns") and "semantic_mask" in preds.columns:
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


def compute_fg_macro_dice(predictor: MultiModalPredictor, df: pd.DataFrame, num_classes: int = 4):
    """
    前景宏平均 Dice：对每个前景类 (1..num_classes-1) 分别计算二值 Dice，再取平均。
    ABD/ACDC 的 test 脚本口径：若某类预测像素全为 0，则该类 Dice 直接记为 0；
    否则按标准 Dice 计算（即便 GT 该类为空，Dice 也会是 0）。
    """
    eps = 1e-6
    dices = []
    preds = predictor.predict(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        gt = np.array(Image.open(row["label"]))
        pred = np.array(preds[idx])
        for c in range(1, num_classes):
            gt_c = (gt == c)
            pred_c = (pred == c)
            # 与 ABD `test_ACDC.py` 一致：只要 pred 该类为空，就直接记 0
            if pred_c.sum() == 0:
                dice_c = 0.0
            else:
                inter = np.logical_and(gt_c, pred_c).sum()
                union = gt_c.sum() + pred_c.sum()
                dice_c = (2 * inter + eps) / (union + eps)
            dices.append(dice_c)
    if not dices:
        return 0.0
    return float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description="ACDC semi/weak-supervised semantic segmentation with Conv-LoRA + GSPO + Adapter")
    parser.add_argument("--dataset_dir", type=str, default="datasets/acdc_conv_lora/acdc_conv_lora",
                        help="由 prepare_acdc_conv_lora.py 生成的数据集根目录")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/acdc_conv_lora_semi")
    parser.add_argument("--ckpt_path", type=str, default=None, help="若只做评估，指定已训练模型路径")
    parser.add_argument(
        "--eval_ckpt",
        type=str,
        default=None,
        help="纯评估入口：直接加载 Lightning .ckpt 做评估（不再训练，不依赖 output_dir 下的 last.ckpt）",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="严格意义断点续训：指定 Lightning .ckpt 路径（会恢复 optimizer/scheduler/step/epoch）。",
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="有效 batch size；若大于 per_gpu_batch_size*num_gpus，则会做累积")
    parser.add_argument("--eval_only", action="store_true", help="只做评估，不训练")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="eval_only 时评估的数据划分：train / val / test（train 表示在训练集上做评估）",
    )
    # GSPO
    parser.add_argument("--gspo_enable", action="store_true")
    parser.add_argument("--gspo_group_size", type=int, default=4)
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5)
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9)
    parser.add_argument("--gspo_allow_semisup", action="store_true",
                        help="半监督时允许 GSPO 只在有标签子集上启用，弱标子集跳过 GSPO。")
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
    parser.add_argument("--vis_output_dir", type=str, default="outputs/acdc_vis_semi",
                        help="可视化输出目录")
    # Training epochs control
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (set large to disable early stopping)")
    # Loss function
    parser.add_argument("--loss", type=str, default="mask2former_loss",
                        choices=["dice_ce_loss", "mask2former_loss"],
                        help="Loss function: mask2former_loss (recommended for SAM multi-class) or dice_ce_loss")
    # Mask tokens
    parser.add_argument("--num_mask_tokens", type=int, default=10,
                        help="Mask tokens/queries (建议 >= 类别数，Mask2Former 风格推荐 10+)")
    # Semi / weak supervision
    parser.add_argument("--semi_labeled_fraction", type=float, default=0.1,
                        help="使用精确掩码的样本比例，其余仅用弱监督盒约束")
    parser.add_argument("--semi_labeled_seed", type=int, default=123,
                        help="划分有/无精确掩码子集的随机种子")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box", choices=["box", "image", "pixel"],
                        help="弱监督盒的抖动模式")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.1,
                        help="弱监督盒的抖动幅度")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="弱监督盒仅向外扩张，不收缩")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="盒外背景约束权重")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="盒内熵最小化权重")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="盒内平滑 (TV) 约束权重")
    # Quick / Debug
    parser.add_argument("--debug", action="store_true", help="仅处理少量样本以快速验证")
    parser.add_argument("--quick_test", type=int, default=None, help="仅处理前 N 个测试样本")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 wandb（确保已登录）
    # 说明：
    # - AutoGluon/Lightning 默认用 TensorBoardLogger 记录训练与验证指标；
    # - 用 wandb 的 sync_tensorboard 能稳定同步曲线，无需触碰 learner._trainer 这种内部字段。
    #
    # 重要：请避免在脚本目录（如 Conv-LoRA/）下生成名为 `wandb/` 的目录，
    # 否则 Python 会优先 import 本地 `wandb` 目录，导致 `import wandb` 变成 namespace package，
    # 出现 "No module named 'wandb.xxx'" 或缺少 `wandb.init` 等问题。
    wandb = None
    try:
        import sys

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 避免本地 ./wandb/ 遮蔽 pip 安装的 wandb 包
        # 备注：当从 Conv-LoRA/ 目录运行脚本时，如果存在 Conv-LoRA/wandb/（wandb 默认运行目录），
        # Python 会优先 import 到这个本地目录，导致 wandb 变成 “namespace package”，缺少 wandb.init 等 API。
        # 因此在 import 前把脚本目录和空路径从 sys.path 中移除，并清掉已缓存模块。
        script_dir_norm = os.path.normpath(script_dir)
        cleaned = []
        for p in sys.path:
            if p == "":
                continue
            try:
                if os.path.normpath(p) == script_dir_norm:
                    continue
            except Exception:
                pass
            cleaned.append(p)
        sys.path = cleaned
        sys.modules.pop("wandb", None)
        import importlib

        wandb = importlib.import_module("wandb")
        # 额外保护：如果仍然被本地目录遮蔽（namespace package），则禁用 wandb，避免运行时崩溃
        if not (hasattr(wandb, "init") and hasattr(wandb, "util")):
            print(
                "[Warn] Imported 'wandb' does not look like the official package "
                f"(file={getattr(wandb, '__file__', None)}). "
                "This is usually caused by a local './wandb/' directory shadowing the pip package. "
                "Will skip wandb logging."
            )
            wandb = None
    except Exception as e:
        print(f"[Warn] wandb import failed (will skip wandb logging): {e}")

    exp_name = f"ACDC-{args.loss}-gspo{int(args.gspo_enable)}-semifrac{args.semi_labeled_fraction}-seed{args.seed}-{int(time.time())}"
    if wandb is not None:
        wandb.init(
            project="GSPOConvLoRA",
            name=exp_name,
            config=vars(args),
            dir=args.output_dir,  # wandb 输出目录放到 output_dir，避免污染工作目录
            sync_tensorboard=True,  # 同步 AutoGluon/Lightning 的 TensorBoard 曲线（train/val loss, val_dice 等）
            tags=[
                "ACDC",
                "Mask2Former",
                "Conv-LoRA",
                f"gspo={args.gspo_enable}",
                f"adapter={args.adapter_enable}",
                f"semi_frac={args.semi_labeled_fraction}",
            ],
        )

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
    print(f"Semi labeled fraction: {args.semi_labeled_fraction} (seed={args.semi_labeled_seed})")
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
        # Mask2Former-style 多类分割需要足够的 mask tokens
        "model.sam.num_mask_tokens": args.num_mask_tokens,
    }

    if args.gspo_enable:
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            "optim.gspo.allow_semisup": bool(args.gspo_allow_semisup),
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
        # 纯评估模式
        if args.ckpt_path:
            predictor = MultiModalPredictor.load(args.ckpt_path)
        elif args.eval_ckpt:
            # 构建模型，并直接从 Lightning ckpt 加载 state_dict（不进入训练循环）
            predictor = MultiModalPredictor(
                problem_type="semantic_segmentation",
                validation_metric="dice",
                eval_metric="dice",
                hyperparameters=hyperparameters,
                label="label",
                path=args.output_dir,
                warn_if_exist=True,
            )
            # 触发内部构建（无需训练），先准备数据与列类型
            predictor._learner.prepare_train_tuning_data(
                train_data=train_df,
                tuning_data=val_df,
                holdout_frac=None,
                seed=args.seed,
            )
            predictor._learner.infer_column_types(column_types=None)
            predictor._learner.infer_output_shape()
            predictor._learner.infer_validation_metric()
            predictor._learner.prepare_fit_args(
                time_limit=0,  # prepare_fit_args 内部转 timedelta；0 表示不真正训练
                seed=args.seed,
                standalone=True,
                clean_ckpts=False,
            )
            predictor._learner.get_df_preprocessor_per_run(
                df_preprocessor=None,
                config=predictor._learner._config,
            )
            predictor._learner.get_model_per_run(
                model=None,
                config=predictor._learner._config,
                df_preprocessor=predictor._learner._df_preprocessor,
            )
            # 从 ckpt 加载权重
            ckpt = torch.load(args.eval_ckpt, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = predictor._learner._model.load_state_dict(state_dict, strict=False)
            print(f"[Eval-only] Loaded weights from {args.eval_ckpt}")
            if missing:
                print(f"[Eval-only][Missing keys]: {missing}")
            if unexpected:
                print(f"[Eval-only][Unexpected keys]: {unexpected}")
        else:
            raise ValueError("--eval_only 需要指定 --ckpt_path 或 --eval_ckpt")
    else:
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
            path=args.output_dir,
            warn_if_exist=True,
        )
        if wandb is not None:
            try:
                wandb.config.update({"autogluon_save_path": predictor.path}, allow_val_change=True)
            except Exception as e:
                print(f"[Warn] wandb.config.update failed: {e}")

        # 配置半/弱监督与训练时的 box prompt（弱监督盒来自 GT box 抖动）
        predictor._learner._train_box_prompt_cfg = {
            "mode": "off",  # 对 ACDC 默认不注入显式 box prompt，但弱监督需要盒约束
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.0,
            # Semi/weak supervision
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # 严格意义断点续训（Lightning resume）：恢复模型 + optimizer/scheduler/step/epoch
        # 原理：BaseLearner.prepare_fit_args() 会读取 self._ckpt_path 并传给 trainer.fit(ckpt_path=...)
        # 注意：如果只是想用 ckpt 做评估而不要求保存目录中必须已有 last.ckpt，可将 _resume 设为 False 避免 process_save_path 的断言。
        if args.resume_ckpt:
            predictor._learner._ckpt_path = args.resume_ckpt
            predictor._learner._resume = False  # 避免要求 output_dir 下已有 last.ckpt
            print(f"[Resume] Will resume training from Lightning checkpoint: {args.resume_ckpt}")

        # 打印模型保存目录
        try:
            model_path = predictor.path
        except AttributeError:
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor (ACDC semi/weak)")
            print(f"  Save path   : {model_path}")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # 评估（IoU + Dice）
    eval_df = {"train": train_df, "val": val_df, "test": test_df}[args.eval_split]
    res = predictor.evaluate(eval_df, metrics=["iou", "dice"])
    fg_macro_dice = compute_fg_macro_dice(predictor, eval_df, num_classes=4)
    print(f"Eval split: {args.eval_split}")
    print(f"Eval results on ACDC ({args.eval_split}): {res}")
    print(f"Foreground macro Dice (classes 1..3, ABD-style): {fg_macro_dice:.6f}")
    with open(os.path.join(args.output_dir, f"metrics_acdc_{args.eval_split}.txt"), "a") as f:
        f.write(f"{res}\n")
        f.write(f"foreground_macro_dice_abd: {fg_macro_dice}\n")
    if wandb is not None:
        wandb.log(
            {**{f"{args.eval_split}/{k}": v for k, v in res.items()}, f"{args.eval_split}/fg_macro_dice_abd": fg_macro_dice}
        )

    # 可视化一小部分验证样本的预测（覆盖写，数量固定）
    visualize_samples(predictor, val_df, args.vis_output_dir, max_samples=args.vis_samples)


if __name__ == "__main__":
    main()

