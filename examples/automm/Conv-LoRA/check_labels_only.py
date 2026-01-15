import numpy as np
import pandas as pd
from PIL import Image
import os

dataset_dir = "datasets/acdc_conv_lora/acdc_conv_lora"
for split in ["train.csv", "val.csv", "test.csv"]:
    csv_path = os.path.join(dataset_dir, split)
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    print(f"\nChecking {split} ({len(df)} samples)...")
    
    fg_counts = []
    unique_vals = set()
    
    # 检查前 100 个样本（或者全部，如果样本少）
    check_num = min(len(df), 100)
    for i in range(check_num):
        lbl_path = os.path.join(dataset_dir, df.iloc[i]['label'])
        lbl = np.array(Image.open(lbl_path))
        unique_vals.update(np.unique(lbl).tolist())
        fg_counts.append((lbl > 0).sum())
    
    print(f"  Unique values in labels: {sorted(list(unique_vals))}")
    print(f"  Avg foreground pixels: {np.mean(fg_counts):.2f}")
    print(f"  Samples with 0 foreground: {sum(1 for c in fg_counts if c == 0)} / {check_num}")

