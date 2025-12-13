"""
使用已训练的 bbox 预测模型生成 box prompt，并用原始 SAM（无 Adapter / 无 Conv-LoRA）
在 ISIC2017 测试集上做分割评估（Dice）。

前置要求：
- 已有 bbox 预测模型权重（例如通过 run_bbox_predictor.py 训练得到的 best_bbox.pth）
- 本地有原始 SAM 权重文件（如 sam_vit_h_4b8939.pth），无需任何适配层
- 已准备好 ISIC2017 数据集 CSV（含 image、label 列），路径默认为 datasets/isic2017/isic2017/test.csv

示例：
1) 先训练或已获得 bbox 模型：
   nohup python3 run_bbox_predictor.py \
     --data_dir datasets/isic2017/isic2017 \
     --output_dir outputs/bbox_predict-251212 \
     --epochs 60 --pretrained_backbone --export_preds \
     --image_size 320 --num_workers 4 \
     > outputs/bbox_predict-251212.log 2>&1 &

2) 再用原始 SAM 做分割评估（默认使用 bbox 模型在线预测框）：
   python3 eval_sam_with_pred_bbox.py \
     --data_dir datasets/isic2017/isic2017 \
     --sam_checkpoint path/to/sam_vit_h_4b8939.pth \
     --sam_model_type vit_h \
     --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
     --bbox_image_size 320 \
     --save_pred_csv outputs/bbox_predict-251212/test_bbox_preds.csv \
     --save_mask_dir outputs/bbox_predict-251212/sam_masks

若已有离线生成的 bbox CSV，可用：
   python3 eval_sam_with_pred_bbox.py \
     --data_dir datasets/isic2017/isic2017 \
     --sam_checkpoint path/to/sam_vit_h_4b8939.pth \
     --sam_model_type vit_h \
     --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from bbox_prompt_model import BBoxPromptPredictor

_orig_torch_load = torch.load

def _safe_torch_load(*args, **kwargs):
    # 如果调用者没显式传 weights_only，就强制设为 False
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _safe_torch_load

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as e:
    raise ImportError(
        "未找到 segment_anything 依赖，请先安装：pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


def load_binary_mask(path: str) -> np.ndarray:
    """加载二值掩码并转为 0/1 np.ndarray。"""
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """计算 Dice 系数，输入为 0/1 数组。"""
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return (2.0 * inter) / (denom + eps)


def attach_pred_boxes(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    为 DataFrame 附加 bbox_x1..bbox_y2 列。
    优先使用外部 CSV，否则使用 bbox 模型在线预测。
    """
    df = df.copy()
    if args.bbox_preds_csv:
        csv = pd.read_csv(args.bbox_preds_csv)
        # 优先精确匹配完整路径，若失败则按文件名匹配
        merged = df.merge(csv, on="image", how="left")
        if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
            csv["image_basename"] = csv["image"].apply(os.path.basename)
            merged["image_basename"] = merged["image"].apply(os.path.basename)
            merged = merged.merge(
                csv.drop(columns=["image"]),
                on="image_basename",
                how="left",
                suffixes=("", "_pred"),
            ).drop(columns=["image_basename"])
            for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                pred_col = f"{col}_pred"
                if pred_col in merged.columns:
                    merged[col] = merged[col].fillna(merged[pred_col])
                    merged.drop(columns=[pred_col], inplace=True)
        df = merged
    else:
        if not args.bbox_model_ckpt:
            raise ValueError("未提供 bbox_preds_csv，必须指定 --bbox_model_ckpt 才能在线预测框。")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bbox_predictor = BBoxPromptPredictor(
            ckpt_path=args.bbox_model_ckpt,
            device=device,
            image_size=args.bbox_image_size,
        )
        boxes = []
        for img_path in tqdm(df["image"], desc="Predicting bboxes"):
            box = bbox_predictor.predict(img_path)
            boxes.append(box)
        boxes = np.stack(boxes, axis=0)
        df["bbox_x1"] = boxes[:, 0]
        df["bbox_y1"] = boxes[:, 1]
        df["bbox_x2"] = boxes[:, 2]
        df["bbox_y2"] = boxes[:, 3]
    # 校验
    if df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().any().any():
        raise ValueError("存在缺失的 bbox，检查 bbox CSV 或模型预测是否正常。")
    return df


def main(args: argparse.Namespace):
    test_csv = os.path.join(args.data_dir, "test.csv")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"未找到测试集 CSV: {test_csv}")

    df = pd.read_csv(test_csv)
    if not {"image", "label"}.issubset(df.columns):
        raise ValueError("test.csv 需包含 image 和 label 列。")

    # 将相对路径补全为 data_dir 下的绝对路径
    df["image"] = df["image"].apply(lambda p: os.path.join(args.data_dir, p))
    df["label"] = df["label"].apply(lambda p: os.path.join(args.data_dir, p))

    if args.max_samples:
        df = df.head(args.max_samples)

    df = attach_pred_boxes(df, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    sam.eval()
    sam_predictor = SamPredictor(sam)

    dices = []
    os.makedirs(args.save_mask_dir, exist_ok=True) if args.save_mask_dir else None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SAM inference"):
        image_path = row["image"]
        label_path = row["label"]
        box = np.array(
            [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
            dtype=np.float32,
        )

        image = np.array(Image.open(image_path).convert("RGB"))
        gt_mask = load_binary_mask(label_path)

        sam_predictor.set_image(image)
        masks, scores, _ = sam_predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_idx = int(scores.argmax())
        pred_mask = masks[best_idx].astype(np.uint8)

        dice = dice_score(pred_mask, gt_mask)
        dices.append(dice)

        if args.save_mask_dir:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(args.save_mask_dir, f"{base}_sam.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(out_path)

    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    print(f"\n测试集平均 Dice: {mean_dice:.4f} （样本数 {len(dices)}）")

    if args.save_pred_csv:
        df.to_csv(args.save_pred_csv, index=False)
        print(f"已保存 bbox 预测到 {args.save_pred_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="用 bbox 预测 + 原始 SAM 计算 ISIC2017 测试集 Dice"
    )
    parser.add_argument("--data_dir", type=str, default="datasets/isic2017/isic2017")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="原始 SAM 权重路径 (.pth)")
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 模型类型，需与权重匹配",
    )
    parser.add_argument("--bbox_model_ckpt", type=str, default=None, help="bbox 模型权重，用于在线预测框")
    parser.add_argument("--bbox_preds_csv", type=str, default=None, help="已有 bbox 预测 CSV，包含 image,bbox_x1..bbox_y2")
    parser.add_argument("--bbox_image_size", type=int, default=320, help="bbox 模型输入尺寸（训练时的 image_size）")
    parser.add_argument("--save_pred_csv", type=str, default=None, help="可选，保存附带 bbox 列的 CSV")
    parser.add_argument("--save_mask_dir", type=str, default=None, help="可选，保存 SAM 预测掩码 PNG")
    parser.add_argument("--max_samples", type=int, default=None, help="仅跑前 N 张，用于快速验证")

    args = parser.parse_args()
    main(args)
