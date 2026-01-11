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

import numpy as np
import pandas as pd
import torch
import wandb
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import LOGITS, SEMANTIC_MASK
from PIL import Image
from lightning.pytorch.callbacks import Callback
import torch.nn.functional as F


def ensure_unique_output_dir(output_dir: str) -> str:
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        suffix = time.strftime("%Y%m%d_%H%M%S")
        new_dir = f"{output_dir}_{suffix}"
        print(f"[Info] Output directory '{output_dir}' already exists; using '{new_dir}' instead to avoid overwriting.")
        return new_dir
    return output_dir


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


def compute_fg_macro_dice(
    predictor: MultiModalPredictor,
    df: pd.DataFrame,
    num_classes: int = 4,
    preds: list = None,
):
    """
    前景宏平均 Dice：对每个前景类 (1..num_classes-1) 分别计算二值 Dice，再取平均。
    ABD/ACDC 的 test 脚本口径：若某类预测像素全为 0，则该类 Dice 直接记为 0；
    否则按标准 Dice 计算（即便 GT 该类为空，Dice 也会是 0）。
    """
    eps = 1e-6
    dices = []
    if preds is None:
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


def compute_simple_seg_metrics(preds: list, df: pd.DataFrame, num_classes: int = 4):
    """
    简单多类分割指标：返回包含背景的宏平均 Dice/IoU，以及仅前景的宏平均。
    """
    eps = 1e-6
    dice_all, iou_all = [], []
    dice_fg, iou_fg = [], []
    for idx, (_, row) in enumerate(df.iterrows()):
        gt = np.array(Image.open(row["label"]))
        pred = np.array(preds[idx])
        for c in range(num_classes):
            gt_c = gt == c
            pred_c = pred == c
            inter = np.logical_and(gt_c, pred_c).sum()
            union = gt_c.sum() + pred_c.sum()
            denom_iou = gt_c.sum() + pred_c.sum() - inter
            # 若该类在 GT 与预测中都不存在，视为完美匹配
            if union == 0:
                dice_c = 1.0
            else:
                dice_c = (2 * inter + eps) / (union + eps)
            if denom_iou == 0:
                iou_c = 1.0
            else:
                iou_c = (inter + eps) / (denom_iou + eps)
            dice_all.append(dice_c)
            iou_all.append(iou_c)
            if c > 0:
                dice_fg.append(dice_c)
                iou_fg.append(iou_c)
    return {
        "dice_macro_all": float(np.mean(dice_all)) if dice_all else 0.0,
        "iou_macro_all": float(np.mean(iou_all)) if iou_all else 0.0,
        "dice_macro_fg": float(np.mean(dice_fg)) if dice_fg else 0.0,
        "iou_macro_fg": float(np.mean(iou_fg)) if iou_fg else 0.0,
    }


def manual_predict_semantic_masks(predictor: MultiModalPredictor, df: pd.DataFrame) -> list:
    """
    避开 AutoGluon 对 ret_type=SEMANTIC_MASK 的依赖，直接取 logits/semantic_mask。
    返回 np.ndarray list，每个元素形状 (H,W)。
    """
    learner = predictor._learner
    data = learner.on_predict_start(df)
    outputs = learner.predict_per_run(data=data, realtime=False, requires_label=False)
    image_col = learner.get_image_column_name(data)
    preds = []
    for idx, out in enumerate(outputs):
        if SEMANTIC_MASK in out:
            mask_logits = out[SEMANTIC_MASK]
            if isinstance(mask_logits, torch.Tensor) and mask_logits.ndim == 4:
                mask_logits = mask_logits[0]
        elif LOGITS in out:
            mask_logits = out[LOGITS]
        elif "logits" in out:
            mask_logits = out["logits"]
        else:
            # 兜底：取第一个 tensor-like
            mask_logits = next(iter(out.values()))
        if isinstance(mask_logits, torch.Tensor):
            # 统一为 (1,C,H,W)
            if mask_logits.ndim == 3:
                mask_logits = mask_logits.unsqueeze(0)
            elif mask_logits.ndim == 2:
                mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
            ori_size = Image.open(data[image_col][idx]).size  # (W,H)
            mask_logits = F.interpolate(
                mask_logits.float(), (ori_size[1], ori_size[0]), mode="bilinear", align_corners=False
            )
            mask = mask_logits.squeeze(0).argmax(dim=0).cpu().numpy()
        else:
            mask = np.array(mask_logits)
            if mask.ndim == 3:
                mask = mask.squeeze()
        preds.append(mask.astype(np.uint8))
    return preds


class WandbMetricsCallback(Callback):
    """将 Lightning 的 callback_metrics 持续推送到 wandb。"""

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            if hasattr(v, "item"):
                metrics[f"train/{k}"] = v.item()
        if metrics:
            wandb.log(metrics, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            if hasattr(v, "item"):
                metrics[f"val/{k}"] = v.item()
        if metrics:
            wandb.log(metrics, step=trainer.global_step)


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
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="有效 batch size；若大于 per_gpu_batch_size*num_gpus，则会做累积")
    parser.add_argument("--eval_only", action="store_true", help="只做评估，不训练")
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
    parser.add_argument("--disable_full_ckpt", action="store_true",
                        help="默认训练结束会额外保存完整 Lightning ckpt（含优化器状态）；加此参数可关闭。")
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

    save_full_ckpt = not args.disable_full_ckpt

    args.output_dir = ensure_unique_output_dir(args.output_dir)
    parent_dir = os.path.dirname(args.output_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    wandb_dir = os.path.join(parent_dir, "wandb_logs", os.path.basename(args.output_dir))
    os.makedirs(wandb_dir, exist_ok=True)

    # 初始化 wandb（确保已登录）
    exp_name = f"ACDC-{args.loss}-gspo{int(args.gspo_enable)}-semifrac{args.semi_labeled_fraction}-seed{args.seed}-{int(time.time())}"
    wandb.init(
        project="GSPOConvLoRA",
        name=exp_name,
        config=vars(args),
        dir=wandb_dir,                 # wandb 只在 wandb_logs/ 写文件，避免污染 output_dir
        sync_tensorboard=True,         # 关键：把 TensorBoard 的标量（含 val_dice）同步到 wandb
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
        if not args.ckpt_path:
            raise ValueError("--eval_only 需要指定 --ckpt_path")

        def _find_ckpt_file(path: str) -> str:
            if os.path.isfile(path) and path.endswith(".ckpt"):
                return path
            if os.path.isdir(path):
                preferred = os.path.join(path, "last.ckpt")
                if os.path.isfile(preferred):
                    return preferred
                ckpts = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".ckpt") and os.path.isfile(os.path.join(path, f))
                ]
                if ckpts:
                    ckpts.sort(key=os.path.getmtime, reverse=True)
                    return ckpts[0]
            raise ValueError(f"未找到可用的 .ckpt 文件，路径: {path}")

        assets_path = os.path.join(args.ckpt_path, "assets.json") if os.path.isdir(args.ckpt_path) else None
        if assets_path and os.path.isfile(assets_path):
            predictor = MultiModalPredictor.load(args.ckpt_path)
        else:
            ckpt_file = _find_ckpt_file(args.ckpt_path)
            print(f"[Eval-only] assets.json 不存在，改为直接加载 ckpt 权重: {ckpt_file}")
            predictor = MultiModalPredictor(
                problem_type="semantic_segmentation",
                validation_metric=validation_metric,
                eval_metric=validation_metric,
                hyperparameters=hyperparameters,
                label="label",
                path=args.output_dir,
                warn_if_exist=False,
            )
            # 手动构建 learner 所需的内部状态，确保 _model 已初始化
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
                time_limit=0,
                seed=args.seed,
                standalone=True,
                clean_ckpts=False,
            )
            predictor._learner.init_pretrained()
            # 确保 _train_data 可用
            if predictor._learner._train_data is None:
                predictor._learner._train_data = train_df
            df_proc = predictor._learner.get_df_preprocessor_per_run(
                df_preprocessor=None,
                config=predictor._learner._config,
            )
            if df_proc is None:
                df_proc = predictor._learner.get_df_preprocessor_per_run(
                    df_preprocessor=None,
                    data=train_df,
                    config=predictor._learner._config,
                    is_train=False,
                )
            predictor._learner._df_preprocessor = df_proc
            predictor._learner._config = predictor._learner.update_config_by_data_per_run(
                config=predictor._learner._config,
                df_preprocessor=df_proc,
            )
            model_built = predictor._learner.get_model_per_run(
                model=None,
                config=predictor._learner._config,
                df_preprocessor=df_proc,
            )
            predictor._learner._model = model_built
            state = torch.load(ckpt_file, map_location="cpu")
            state_dict = state.get("state_dict", state)
            # 兼容 Lightning ckpt 前缀：去掉 "model." 或 "model.model." 以匹配当前模型
            if isinstance(state_dict, dict):
                keys = list(state_dict.keys())
                if keys and all(k.startswith("model.model.") for k in keys):
                    # ckpt 带双重前缀，仅去掉一层，保留 'model.' 以匹配当前模型
                    state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
            missing, unexpected = predictor._learner._model.load_state_dict(state_dict, strict=False)
            print(f"[Eval-only] Loaded weights from {ckpt_file}")
            if missing:
                print(f"[Eval-only][Missing keys]: {missing}")
            if unexpected:
                print(f"[Eval-only][Unexpected keys]: {unexpected}")
            # 生成可复用的 predictor 目录，避免下次还要手动加载 ckpt
            try:
                predictor.save(args.output_dir, standalone=False)
                print(f"[Eval-only] 已导出可复用的预测器到: {args.output_dir}")
            except Exception as e:
                print(f"[Eval-only] 导出预测器失败（可忽略，仅影响复用）: {e}")
    else:
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
            path=args.output_dir,          # 建议加：保证 TB 日志在 output_dir 下，方便 wandb 同步
            warn_if_exist=True,            # 若 output_dir 已存在，则警告
        )
        # 记录模型/梯度（SemanticSegmentationLearner 使用 _model 作为实际模型句柄）
        model_to_watch = getattr(predictor._learner, "_model", None)
        if isinstance(model_to_watch, torch.nn.Module):
            wandb.watch(model_to_watch, log="all", log_freq=50)
        else:
            print("[Info] Skip wandb.watch: model not available yet.")
        # 挂载 wandb 回调，把训练/验证指标同步到 wandb
        try:
            predictor._learner._trainer.callbacks.append(WandbMetricsCallback())
        except Exception as e:
            print(f"[Warn] Failed to attach WandbMetricsCallback: {e}")

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
        # 保存可复用的 predictor 目录（含 assets.json），便于后续 eval_only 直接加载
        try:
            predictor.save(args.output_dir, standalone=False)
            print(f"[Info] Predictor saved to {args.output_dir}")
        except Exception as e:
            print(f"[Warn] Predictor save failed (non-fatal): {e}")
        # 额外保存完整 Lightning ckpt，便于续训/完整评估（含优化器/调度器状态）
        if save_full_ckpt:
            try:
                ckpt_path = os.path.join(args.output_dir, "last_full.ckpt")
                trainer = getattr(predictor._learner, "_trainer", None)
                if trainer is not None:
                    trainer.save_checkpoint(ckpt_path, weights_only=False)
                    print(f"[Info] Full Lightning ckpt saved: {ckpt_path}")
                else:
                    print("[Warn] Trainer not available, skip full ckpt saving.")
            except Exception as e:
                print(f"[Warn] Saving full Lightning ckpt failed: {e}")

    # 评估（IoU + Dice）
    preds_cache = None
    try:
        res = predictor.evaluate(test_df, metrics=["iou", "dice"])
    except Exception as e:
        print(f"[Warn] predictor.evaluate 失败，回退到手动评估: {e}")
        preds_cache = manual_predict_semantic_masks(predictor, test_df)
        manual = compute_simple_seg_metrics(preds_cache, test_df, num_classes=4)
        res = {
            "dice_manual_macro_all": manual["dice_macro_all"],
            "dice_manual_macro_fg": manual["dice_macro_fg"],
            "iou_manual_macro_all": manual["iou_macro_all"],
            "iou_manual_macro_fg": manual["iou_macro_fg"],
        }

    if preds_cache is None:
        preds_cache = manual_predict_semantic_masks(predictor, test_df)
    fg_macro_dice = compute_fg_macro_dice(predictor, test_df, num_classes=4, preds=preds_cache)
    print(f"Test results on ACDC (semi): {res}")
    print(f"Foreground macro Dice (classes 1..3): {fg_macro_dice:.6f}")
    with open(os.path.join(args.output_dir, "metrics_acdc_semi.txt"), "a") as f:
        f.write(f"{res}\n")
        f.write(f"foreground_macro_dice: {fg_macro_dice}\n")
    wandb.log({**{f"test/{k}": v for k, v in res.items()}, "test/fg_macro_dice": fg_macro_dice})

    # 可视化一小部分验证样本的预测（覆盖写，数量固定）
    visualize_samples(predictor, val_df, args.vis_output_dir, max_samples=args.vis_samples)


if __name__ == "__main__":
    main()



