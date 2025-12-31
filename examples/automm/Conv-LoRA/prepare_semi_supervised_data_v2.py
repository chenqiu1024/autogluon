#!/usr/bin/env python3
"""
数据拆分脚本 v2：改进的 box 扰动策略

改进点：
1. 支持三种抖动模式：
   - none: 无抖动，使用精确的 GT bbox（作为基线）
   - expand_only: 只外扩，不内缩（符合实际使用场景）
   - mixed: 主要外扩，允许少量内缩
2. 支持多种度量单位：
   - bbox_relative: 相对于 bbox 尺寸的比例
   - image_relative: 相对于图片尺寸的比例
   - absolute_pixels: 绝对像素值
3. 可配置的扰动参数
"""
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


def compute_tight_box(mask_array):
    """计算 GT mask 的最小包围框"""
    coords = np.argwhere(mask_array > 127)
    if len(coords) == 0:
        return None
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    return [int(x_coords.min()), int(y_coords.min()), 
            int(x_coords.max()), int(y_coords.max())]


def add_box_jitter_v2(
    box, 
    mode='expand_only',
    expand_ratio=0.1,
    contract_ratio=0.05,
    contract_prob=0.1,
    jitter_unit='bbox_relative',
    img_w=None, 
    img_h=None
):
    """
    改进的 box 扰动函数
    
    Args:
        box: [x1, y1, x2, y2] 原始 bbox
        mode: 扰动模式
            - 'none': 无扰动，返回原始 bbox
            - 'expand_only': 只外扩（符合实际场景）
            - 'mixed': 主要外扩，允许少量内缩
        expand_ratio: 外扩比例
        contract_ratio: 内缩比例（仅在 mixed 模式下使用）
        contract_prob: 内缩的概率（仅在 mixed 模式下使用）
        jitter_unit: 度量单位
            - 'bbox_relative': 相对于 bbox 尺寸
            - 'image_relative': 相对于图片尺寸
            - 'absolute_pixels': 绝对像素值
        img_w, img_h: 图片宽高
    
    Returns:
        [x1_n, y1_n, x2_n, y2_n]: 扰动后的 bbox
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # 模式 1: 无扰动
    if mode == 'none':
        return box
    
    # 计算扰动的基准尺度
    if jitter_unit == 'bbox_relative':
        scale_x, scale_y = w, h
    elif jitter_unit == 'image_relative':
        scale_x, scale_y = img_w, img_h
    elif jitter_unit == 'absolute_pixels':
        scale_x, scale_y = 1.0, 1.0
    else:
        raise ValueError(f"Unknown jitter_unit: {jitter_unit}")
    
    # 模式 2: 只外扩
    if mode == 'expand_only':
        # 使用均匀分布在 [0, expand_ratio] 范围内生成外扩量
        expand_amount = np.random.uniform(0, expand_ratio, size=4)
        
        # 四个边界独立外扩
        dx1 = expand_amount[0] * scale_x  # 左边界向左扩
        dy1 = expand_amount[1] * scale_y  # 上边界向上扩
        dx2 = expand_amount[2] * scale_x  # 右边界向右扩
        dy2 = expand_amount[3] * scale_y  # 下边界向下扩
        
        x1_n = x1 - dx1
        y1_n = y1 - dy1
        x2_n = x2 + dx2
        y2_n = y2 + dy2
    
    # 模式 3: 主要外扩，允许少量内缩
    elif mode == 'mixed':
        jitter_amount = []
        for i in range(4):
            if np.random.rand() < contract_prob:
                # 小概率内缩
                jitter_amount.append(-np.random.uniform(0, contract_ratio))
            else:
                # 大概率外扩
                jitter_amount.append(np.random.uniform(0, expand_ratio))
        
        jitter_amount = np.array(jitter_amount)
        
        # 应用扰动
        dx1 = jitter_amount[0] * scale_x
        dy1 = jitter_amount[1] * scale_y
        dx2 = jitter_amount[2] * scale_x
        dy2 = jitter_amount[3] * scale_y
        
        x1_n = x1 - dx1  # 注意：正值表示外扩（向左），负值表示内缩（向右）
        y1_n = y1 - dy1
        x2_n = x2 + dx2  # 注意：正值表示外扩（向右），负值表示内缩（向左）
        y2_n = y2 + dy2
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # 裁剪到图片范围内
    x1_n = np.clip(x1_n, 0, img_w - 2)
    y1_n = np.clip(y1_n, 0, img_h - 2)
    x2_n = np.clip(x2_n, x1_n + 1, img_w)
    y2_n = np.clip(y2_n, y1_n + 1, img_h)
    
    return [int(x1_n), int(y1_n), int(x2_n), int(y2_n)]


def main():
    parser = argparse.ArgumentParser(description="半监督数据准备 v2 - 改进的 box 扰动")
    
    # 基础参数
    parser.add_argument("--task", default="isic2017", help="任务名称")
    parser.add_argument("--data_dir", default="datasets/isic2017/isic2017", help="数据目录")
    parser.add_argument("--labeled_ratio", type=float, default=0.1, help="有标注数据比例")
    parser.add_argument("--random_seed", type=int, default=42, help="数据划分随机种子")
    parser.add_argument("--box_seed", type=int, default=123, help="Box 扰动随机种子")
    
    # Box 扰动参数
    parser.add_argument("--box_jitter_mode", type=str, default="expand_only",
                       choices=["none", "expand_only", "mixed"],
                       help="Box 扰动模式：none（无扰动，精确bbox），expand_only（只外扩），mixed（主要外扩+少量内缩）")
    parser.add_argument("--box_expand_ratio", type=float, default=0.15,
                       help="外扩比例（默认0.15，即15%%）")
    parser.add_argument("--box_contract_ratio", type=float, default=0.05,
                       help="内缩比例（仅在mixed模式下使用，默认0.05，即5%%）")
    parser.add_argument("--box_contract_prob", type=float, default=0.1,
                       help="内缩概率（仅在mixed模式下使用，默认0.1，即10%%）")
    parser.add_argument("--box_jitter_unit", type=str, default="bbox_relative",
                       choices=["bbox_relative", "image_relative", "absolute_pixels"],
                       help="扰动度量单位：bbox_relative（相对bbox尺寸），image_relative（相对图片尺寸），absolute_pixels（绝对像素）")
    
    args = parser.parse_args()
    
    print("="*60)
    print("半监督数据准备 v2")
    print("="*60)
    print(f"任务: {args.task}")
    print(f"数据目录: {args.data_dir}")
    print(f"有标注比例: {args.labeled_ratio}")
    print(f"\nBox 扰动配置:")
    print(f"  模式: {args.box_jitter_mode}")
    print(f"  外扩比例: {args.box_expand_ratio}")
    if args.box_jitter_mode == "mixed":
        print(f"  内缩比例: {args.box_contract_ratio}")
        print(f"  内缩概率: {args.box_contract_prob}")
    print(f"  度量单位: {args.box_jitter_unit}")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    
    # 读取训练数据
    train_csv = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"❌ 错误: 找不到 {train_csv}")
        return
    
    df = pd.read_csv(train_csv)
    print(f"\n读取训练数据: {len(df)} 个样本")
    
    # 随机划分 labeled 和 weak
    n_labeled = int(len(df) * args.labeled_ratio)
    labeled_idx = np.random.choice(len(df), n_labeled, replace=False)
    labeled_df = df.iloc[labeled_idx].copy()
    weak_df = df.iloc[[i for i in range(len(df)) if i not in labeled_idx]].copy()
    
    print(f"划分结果:")
    print(f"  Labeled: {len(labeled_df)} 个样本")
    print(f"  Weak: {len(weak_df)} 个样本")
    
    # 为 weak 数据生成 noisy box
    print(f"\n生成 noisy box...")
    np.random.seed(args.box_seed)
    
    boxes = []
    iou_list = []  # 用于统计 box 与 GT bbox 的关系
    
    for _, row in tqdm(weak_df.iterrows(), total=len(weak_df), desc="Processing"):
        mask_path = os.path.join(args.data_dir, row['label'])
        img_path = os.path.join(args.data_dir, row['image'])
        
        # 读取 mask 和图片
        mask = np.array(Image.open(mask_path).convert('L'))
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # 计算 GT bbox
        tight_box = compute_tight_box(mask)
        
        if tight_box:
            # 应用扰动
            noisy_box = add_box_jitter_v2(
                tight_box,
                mode=args.box_jitter_mode,
                expand_ratio=args.box_expand_ratio,
                contract_ratio=args.box_contract_ratio,
                contract_prob=args.box_contract_prob,
                jitter_unit=args.box_jitter_unit,
                img_w=img_w,
                img_h=img_h
            )
            
            boxes.append(f"{noisy_box[0]},{noisy_box[1]},{noisy_box[2]},{noisy_box[3]}")
            
            # 计算统计信息（noisy box 相对于 tight box 的扩张比例）
            gt_w = tight_box[2] - tight_box[0]
            gt_h = tight_box[3] - tight_box[1]
            noisy_w = noisy_box[2] - noisy_box[0]
            noisy_h = noisy_box[3] - noisy_box[1]
            
            expand_ratio_x = (noisy_w - gt_w) / gt_w
            expand_ratio_y = (noisy_h - gt_h) / gt_h
            iou_list.append((expand_ratio_x, expand_ratio_y))
        else:
            # 如果没有 GT mask，使用占位符
            boxes.append("0,0,100,100")
    
    weak_df['box'] = boxes
    
    # 统计信息
    if iou_list:
        iou_array = np.array(iou_list)
        print(f"\nBox 扰动统计:")
        print(f"  平均宽度扩张: {iou_array[:, 0].mean():.3f} (±{iou_array[:, 0].std():.3f})")
        print(f"  平均高度扩张: {iou_array[:, 1].mean():.3f} (±{iou_array[:, 1].std():.3f})")
        print(f"  内缩样本数: {(iou_array < 0).any(axis=1).sum()} / {len(iou_array)}")
    
    # 保存
    labeled_csv = os.path.join(args.data_dir, f"train_labeled_{int(args.labeled_ratio*100)}pct.csv")
    weak_csv = os.path.join(args.data_dir, f"train_weak_{int((1-args.labeled_ratio)*100)}pct.csv")
    
    labeled_df[['image', 'label']].to_csv(labeled_csv, index=False)
    weak_df[['image', 'box']].to_csv(weak_csv, index=False)
    
    print(f"\n✅ 数据准备完成！")
    print(f"  Labeled: {labeled_csv}")
    print(f"  Weak: {weak_csv}")
    print("="*60)


if __name__ == "__main__":
    main()
