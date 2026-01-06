#!/usr/bin/env python3
"""
整理 ISIC2018 数据集，使其与 ISIC2017 格式兼容。

由于 ISIC2018 官方只提供训练集的 Ground Truth（验证/测试集标注不公开），
我们从训练集（2594张）中按照 80%/10%/10% 的比例划分 train/val/test 子集。

输入目录结构：
    datasets/ISIC2018/
    ├── ISIC2018_Task1-2_Training_Input/
    │   ├── ISIC_XXXXXXX.jpg
    │   └── ...
    └── ISIC2018_Task1_Training_GroundTruth/
        ├── ISIC_XXXXXXX_segmentation.png
        └── ...

输出目录结构（与 ISIC2017 对齐）：
    datasets/isic2018/isic2018/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── train/
    │   ├── images/
    │   │   └── ISIC_XXXXXXX.jpg (symbolic links)
    │   └── masks/
    │       └── ISIC_XXXXXXX_segmentation.png (symbolic links)
    ├── val/
    │   ├── images/
    │   │   └── ...
    │   └── masks/
    │       └── ...
    └── test/
        ├── images/
        │   └── ...
        └── masks/
            └── ...
"""

import os
import random
from pathlib import Path
import pandas as pd
import shutil

def main():
    random.seed(42)  # 固定随机种子以保证可复现性
    
    # 源数据目录
    base_dir = Path(__file__).parent / "datasets" / "ISIC2018"
    src_images_dir = base_dir / "ISIC2018_Task1-2_Training_Input"
    src_masks_dir = base_dir / "ISIC2018_Task1_Training_GroundTruth"
    
    # 目标目录
    out_dir = Path(__file__).parent / "datasets" / "isic2018" / "isic2018"
    
    # 收集所有有效的图像-标注对
    all_images = sorted(src_images_dir.glob("ISIC_*.jpg"))
    pairs = []
    
    for img_path in all_images:
        img_name = img_path.stem  # e.g., ISIC_0000000
        mask_name = f"{img_name}_segmentation.png"
        mask_path = src_masks_dir / mask_name
        
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"[Warning] No mask for {img_name}, skipping.")
    
    print(f"Found {len(pairs)} valid image-mask pairs.")
    
    # 随机打乱
    random.shuffle(pairs)
    
    # 划分数据集 (80% train, 10% val, 10% test)
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # 创建目录结构
    for split in ["train", "val", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    def process_split(pairs, split_name):
        """处理一个数据集划分，创建符号链接并返回 DataFrame 数据"""
        records = []
        images_dir = out_dir / split_name / "images"
        masks_dir = out_dir / split_name / "masks"
        
        for idx, (img_src, mask_src) in enumerate(pairs):
            img_dst = images_dir / img_src.name
            mask_dst = masks_dir / mask_src.name
            
            # 使用符号链接而非复制，节省磁盘空间
            if not img_dst.exists():
                # 使用绝对路径创建符号链接
                img_dst.symlink_to(img_src.resolve())
            if not mask_dst.exists():
                mask_dst.symlink_to(mask_src.resolve())
            
            # CSV 中使用相对路径（相对于 isic2018/isic2018/）
            records.append({
                "image": f"{split_name}/images/{img_src.name}",
                "label": f"{split_name}/masks/{mask_src.name}",
            })
        
        return records
    
    # 处理各个划分
    train_records = process_split(train_pairs, "train")
    val_records = process_split(val_pairs, "val")
    test_records = process_split(test_pairs, "test")
    
    # 生成 CSV 文件
    pd.DataFrame(train_records).to_csv(out_dir / "train.csv", index=True)
    pd.DataFrame(val_records).to_csv(out_dir / "val.csv", index=True)
    pd.DataFrame(test_records).to_csv(out_dir / "test.csv", index=True)
    
    print(f"\nDataset prepared successfully!")
    print(f"Output directory: {out_dir}")
    print(f"  train.csv: {len(train_records)} samples")
    print(f"  val.csv:   {len(val_records)} samples")
    print(f"  test.csv:  {len(test_records)} samples")

if __name__ == "__main__":
    main()

