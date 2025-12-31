#!/usr/bin/env python3
"""半监督数据可视化工具 - 修复版"""
import os, argparse, pandas as pd, numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

def get_gt_mask_path(image_path):
    """从图像路径推导GT mask路径 - 修复版"""
    basename = os.path.basename(image_path)
    image_id = basename.replace('.jpg', '')
    # 修复：直接使用固定的相对路径
    gt_path = os.path.join('train/ISIC-2017_Training_Part1_GroundTruth', f'{image_id}_segmentation.png')
    return gt_path

def visualize_labeled_sample(image_path, label_path, data_dir, output_path):
    img_full = os.path.join(data_dir, image_path)
    mask_full = os.path.join(data_dir, label_path)
    if not os.path.exists(img_full) or not os.path.exists(mask_full):
        return False
    img = cv2.imread(img_full)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
    overlay = img.copy()
    mask_colored = np.zeros_like(img)
    mask_colored[mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    result = np.hstack([img, overlay, mask_rgb])
    
    result_pil = Image.fromarray(result)
    draw = ImageDraw.Draw(result_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    basename = os.path.basename(image_path)
    draw.text((10, 10), f"Labeled: {basename}", fill=(255, 255, 0), font=font)
    
    result_pil.save(output_path)
    return True

def visualize_weak_sample(image_path, box_str, data_dir, output_path):
    img_full = os.path.join(data_dir, image_path)
    if not os.path.exists(img_full):
        return False
    img = cv2.imread(img_full)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    gt_mask_path = get_gt_mask_path(image_path)
    gt_mask_full = os.path.join(data_dir, gt_mask_path)
    if not os.path.exists(gt_mask_full):
        return False
    
    gt_mask = cv2.imread(gt_mask_full, cv2.IMREAD_GRAYSCALE)
    try:
        x1, y1, x2, y2 = map(int, box_str.split(','))
    except:
        return False
    
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 3)
    mask_colored = np.zeros_like(img)
    mask_colored[gt_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
    mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(mask_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
    result = np.hstack([img_with_box, overlay, mask_rgb])
    
    result_pil = Image.fromarray(result)
    draw = ImageDraw.Draw(result_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    basename = os.path.basename(image_path)
    draw.text((10, 10), f"Weak: {basename}", fill=(255, 255, 0), font=font)
    
    box_mask = np.zeros_like(gt_mask)
    box_mask[y1:y2, x1:x2] = 255
    intersection = np.logical_and(gt_mask > 0, box_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, box_mask > 0).sum()
    iou = intersection / union if union > 0 else 0
    draw.text((10, 40), f"Box-GT IoU: {iou:.3f}", fill=(255, 255, 0), font=font)
    
    result_pil.save(output_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="可视化半监督数据")
    parser.add_argument("--data_dir", type=str, default="datasets/isic2017/isic2017")
    parser.add_argument("--output_dir", type=str, default="datasets/isic2017/visualizations")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    labeled_output = os.path.join(args.output_dir, "labeled_samples")
    weak_output = os.path.join(args.output_dir, "weak_samples")
    os.makedirs(labeled_output, exist_ok=True)
    os.makedirs(weak_output, exist_ok=True)
    
    print("="*60)
    print("半监督数据可视化（修复版）")
    print("="*60)
    
    weak_csv = os.path.join(args.data_dir, "train_weak_90pct.csv")
    if os.path.exists(weak_csv):
        df_weak = pd.read_csv(weak_csv)
        print(f"\nWeak: {len(df_weak)} 个样本")
        if args.max_samples:
            df_weak = df_weak.head(args.max_samples)
        success_count = 0
        for idx, row in tqdm(df_weak.iterrows(), total=len(df_weak), desc="Weak"):
            basename = os.path.basename(row['image']).replace('.jpg', '.png')
            output_path = os.path.join(weak_output, basename)
            if visualize_weak_sample(row['image'], row['box'], args.data_dir, output_path):
                success_count += 1
        print(f"✅ Weak 完成: {success_count}/{len(df_weak)} 个样本")
    
    labeled_csv = os.path.join(args.data_dir, "train_labeled_10pct.csv")
    if os.path.exists(labeled_csv):
        df_labeled = pd.read_csv(labeled_csv)
        print(f"\nLabeled: {len(df_labeled)} 个样本")
        if args.max_samples:
            df_labeled = df_labeled.head(args.max_samples)
        success_count = 0
        for idx, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Labeled"):
            basename = os.path.basename(row['image']).replace('.jpg', '.png')
            output_path = os.path.join(labeled_output, basename)
            if visualize_labeled_sample(row['image'], row['label'], args.data_dir, output_path):
                success_count += 1
        print(f"✅ Labeled 完成: {success_count}/{len(df_labeled)} 个样本")

    print("\n" + "="*60)
    print("✅ 可视化完成！")
    print(f"输出目录: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
