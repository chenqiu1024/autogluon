"""
Semi-Supervised Data Preparation Script

将 ISIC2017 数据集拆分为：
- 10% 有标注集（full mask）
- 90% 弱标注集（noisy box）

使用分层抽样确保样本分布均衡
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mask_area(mask_path):
    """计算 mask 的面积（像素数）"""
    try:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        area = (mask_array > 127).sum()
        return area
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return 0


def compute_tight_box(mask_array):
    """
    从二值 mask 计算 tight bounding box
    
    Returns:
        box: [x_min, y_min, x_max, y_max] 或 None（如果 mask 为空）
    """
    coords = np.argwhere(mask_array > 127)
    
    if len(coords) == 0:
        return None
    
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def add_box_noise(box, noise_std=0.10, img_width=None, img_height=None):
    """
    给 bounding box 添加随机噪声（对应实验协议中的 box 噪声注入）
    
    公式：box_noisy = box_gt * (1 + σ * randn(4))
    
    Args:
        box: [x_min, y_min, x_max, y_max]
        noise_std: 噪声标准差 σ（相对 box 宽高）
        img_width, img_height: 图像尺寸（用于裁剪）
    
    Returns:
        noisy_box: 添加噪声后的 box
    """
    x_min, y_min, x_max, y_max = box
    w, h = x_max - x_min, y_max - y_min
    
    # 生成相对噪声
    noise = np.random.randn(4) * noise_std
    
    # 应用噪声
    x_min_noisy = x_min + w * noise[0]
    y_min_noisy = y_min + h * noise[1]
    x_max_noisy = x_max + w * noise[2]
    y_max_noisy = y_max + h * noise[3]
    
    # 确保 box 有效（min < max）
    if x_min_noisy >= x_max_noisy:
        x_min_noisy, x_max_noisy = x_min, x_max
    if y_min_noisy >= y_max_noisy:
        y_min_noisy, y_max_noisy = y_min, y_max
    
    # 裁剪到图像边界内
    if img_width is not None and img_height is not None:
        x_min_noisy = np.clip(x_min_noisy, 0, img_width - 1)
        x_max_noisy = np.clip(x_max_noisy, 1, img_width)
        y_min_noisy = np.clip(y_min_noisy, 0, img_height - 1)
        y_max_noisy = np.clip(y_max_noisy, 1, img_height)
    
    return [int(x_min_noisy), int(y_min_noisy), int(x_max_noisy), int(y_max_noisy)]


def stratified_split(df, data_dir, labeled_ratio=0.1, random_seed=42):
    """
    分层抽样（按 mask 面积四分位数）
    
    Args:
        df: 原始 dataframe
        mask_dir: mask 目录
        labeled_ratio: 有标注比例（默认 0.1）
        random_seed: 随机种子
    
    Returns:
        labeled_df: 有标注集
        weak_df: 弱标注集
    """
    np.random.seed(random_seed)
    
    # 计算每个样本的 mask 面积
    print("计算 mask 面积用于分层...")
    areas = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # label 列已经是相对路径：train/ISIC-2017_Training_Part1_GroundTruth/xxx.png
        mask_path = os.path.join(data_dir, row['label'])
        area = compute_mask_area(mask_path)
        areas.append(area)
    
    df['mask_area'] = areas
    
    # 按面积四分位数分层
    df['area_quartile'] = pd.qcut(df['mask_area'], q=4, labels=False, duplicates='drop')
    
    # 在每层中抽样
    labeled_indices = []
    weak_indices = []
    
    for quartile in range(4):
        quartile_df = df[df['area_quartile'] == quartile]
        n_labeled = max(1, int(len(quartile_df) * labeled_ratio))
        
        # 随机抽取
        sampled_indices = np.random.choice(
            quartile_df.index,
            size=n_labeled,
            replace=False
        )
        
        labeled_indices.extend(sampled_indices)
        weak_indices.extend([idx for idx in quartile_df.index if idx not in sampled_indices])
    
    labeled_df = df.loc[labeled_indices].copy()
    weak_df = df.loc[weak_indices].copy()
    
    print(f"\n分层抽样完成：")
    print(f"  有标注集: {len(labeled_df)} 样本 ({labeled_ratio*100:.1f}%)")
    print(f"  弱标注集: {len(weak_df)} 样本 ({(1-labeled_ratio)*100:.1f}%)")
    
    return labeled_df, weak_df


def generate_weak_labels(weak_df, data_dir, noise_std=0.10, box_seed=123):
    """
    为弱标注集生成 noisy box
    
    Args:
        weak_df: 弱标注 dataframe
        mask_dir: mask 目录
        image_dir: image 目录
        noise_std: box 噪声强度
        box_seed: box 噪声种子
    
    Returns:
        weak_df_with_box: 添加了 'box' 列的 dataframe
    """
    np.random.seed(box_seed)
    
    print(f"\n生成弱标注 (noisy box, σ={noise_std})...")
    boxes = []
    
    for idx, row in tqdm(weak_df.iterrows(), total=len(weak_df)):
        # 直接使用 csv 中的相对路径拼 data_dir
        mask_path = os.path.join(data_dir, row['label'])
        image_path = os.path.join(data_dir, row['image'])
        
        # 读取 mask 计算 tight box
        mask = np.array(Image.open(mask_path).convert('L'))
        tight_box = compute_tight_box(mask)
        
        if tight_box is None:
            # 如果 mask 为空，使用默认 box
            img = Image.open(image_path)
            w, h = img.size
            tight_box = [w//4, h//4, 3*w//4, 3*h//4]
        
        # 添加噪声
        img = Image.open(image_path)
        w, h = img.size
        noisy_box = add_box_noise(tight_box, noise_std=noise_std, img_width=w, img_height=h)
        
        # 格式：x_min,y_min,x_max,y_max
        boxes.append(f"{noisy_box[0]},{noisy_box[1]},{noisy_box[2]},{noisy_box[3]}")
    
    weak_df['box'] = boxes
    return weak_df


def main():
    parser = argparse.ArgumentParser(description="准备半监督数据（10% mask + 90% noisy box）")
    parser.add_argument("--task", type=str, default="isic2017", help="任务名称")
    parser.add_argument("--data_dir", type=str, default="datasets/isic2017/isic2017", help="数据集根目录")
    parser.add_argument("--labeled_ratio", type=float, default=0.1, help="有标注比例")
    parser.add_argument("--box_noise_std", type=float, default=0.10, help="Box 噪声标准差")
    parser.add_argument("--random_seed", type=int, default=42, help="数据拆分随机种子")
    parser.add_argument("--box_seed", type=int, default=123, help="Box 噪声随机种子")
    
    args = parser.parse_args()
    
    # 路径
    train_csv = os.path.join(args.data_dir, "train.csv")
    
    # 读取原始数据
    print(f"读取数据：{train_csv}")
    df = pd.read_csv(train_csv)
    print(f"总样本数: {len(df)}")
    
    # 分层抽样
    labeled_df, weak_df = stratified_split(
        df, args.data_dir,
        labeled_ratio=args.labeled_ratio,
        random_seed=args.random_seed
    )
    
    # 生成弱标注 box
    weak_df = generate_weak_labels(
        weak_df, args.data_dir,
        noise_std=args.box_noise_std,
        box_seed=args.box_seed
    )

    # 为 weak 样本生成一个全零的占位 mask，避免被当成有标注样本
    zero_mask_rel = os.path.join("train_masks", "zero_mask.png")
    zero_mask_path = os.path.join(args.data_dir, zero_mask_rel)
    if not os.path.exists(zero_mask_path):
        # 用第一张训练图的尺寸创建零掩码
        first_img_path = os.path.join(args.data_dir, labeled_df['image'].iloc[0])
        img = Image.open(first_img_path).convert("RGB")
        w, h = img.size
        zero_mask = Image.new("L", (w, h), 0)
        os.makedirs(os.path.dirname(zero_mask_path), exist_ok=True)
        zero_mask.save(zero_mask_path)
        print(f"已创建弱标注占位 mask: {zero_mask_path} (size={w}x{h})")
    # 为所有 weak 样本指定占位 mask 路径
    weak_df["label"] = zero_mask_rel
    
    # 保存
    labeled_csv = os.path.join(args.data_dir, f"train_labeled_{int(args.labeled_ratio*100)}pct.csv")
    weak_csv = os.path.join(args.data_dir, f"train_weak_{int((1-args.labeled_ratio)*100)}pct.csv")
    
    # 有标注集保留 image 和 label 列
    labeled_df[['image', 'label']].to_csv(labeled_csv, index=False)
    
    # 弱标注集保留 image、label(全零mask)、box 列
    weak_df[['image', 'label', 'box']].to_csv(weak_csv, index=False)
    
    print(f"\n保存完成：")
    print(f"  有标注集: {labeled_csv}")
    print(f"  弱标注集: {weak_csv}")


if __name__ == "__main__":
    main()
