#!/usr/bin/env python3
"""数据拆分脚本：10% mask + 90% noisy box"""
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

def compute_tight_box(mask_array):
    coords = np.argwhere(mask_array > 127)
    if len(coords) == 0:
        return None
    y, x = coords[:, 0], coords[:, 1]
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]

def add_box_noise(box, noise_std=0.10, img_w=None, img_h=None):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    noise = np.random.randn(4) * noise_std
    x1_n = np.clip(x1+w*noise[0], 0, img_w-1)
    y1_n = np.clip(y1+h*noise[1], 0, img_h-1)
    x2_n = np.clip(x2+w*noise[2], 1, img_w)
    y2_n = np.clip(y2+h*noise[3], 1, img_h)
    if x1_n >= x2_n: x1_n, x2_n = x1, x2
    if y1_n >= y2_n: y1_n, y2_n = y1, y2
    return [int(x1_n), int(y1_n), int(x2_n), int(y2_n)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="isic2017")
    parser.add_argument("--data_dir", default="datasets/isic2017")
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--box_noise_std", type=float, default=0.10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--box_seed", type=int, default=123)
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    train_csv = os.path.join(args.data_dir, "train.csv")
    df = pd.read_csv(train_csv)
    
    # 分层抽样
    n_labeled = int(len(df) * args.labeled_ratio)
    labeled_idx = np.random.choice(len(df), n_labeled, replace=False)
    labeled_df = df.iloc[labeled_idx].copy()
    weak_df = df.iloc[[i for i in range(len(df)) if i not in labeled_idx]].copy()
    
    # 生成 noisy box
    np.random.seed(args.box_seed)
    boxes = []
    for _, row in tqdm(weak_df.iterrows(), total=len(weak_df)):
        mask_path = os.path.join(args.data_dir, os.path.basename(row['label']))
        img_path = os.path.join(args.data_dir, os.path.basename(row['image']))
        mask = np.array(Image.open(mask_path).convert('L'))
        tight_box = compute_tight_box(mask)
        if tight_box:
            img = Image.open(img_path)
            w, h = img.size
            noisy_box = add_box_noise(tight_box, args.box_noise_std, w, h)
            boxes.append(f"{noisy_box[0]},{noisy_box[1]},{noisy_box[2]},{noisy_box[3]}")
        else:
            boxes.append("0,0,100,100")
    weak_df['box'] = boxes
    
    # 保存
    labeled_csv = os.path.join(args.data_dir, f"train_labeled_{int(args.labeled_ratio*100)}pct.csv")
    weak_csv = os.path.join(args.data_dir, f"train_weak_{int((1-args.labeled_ratio)*100)}pct.csv")
    labeled_df[['image', 'label']].to_csv(labeled_csv, index=False)
    weak_df[['image', 'box']].to_csv(weak_csv, index=False)
    print(f"✓ Labeled: {labeled_csv}")
    print(f"✓ Weak: {weak_csv}")

if __name__ == "__main__":
    main()
