import torch
import numpy as np
import pandas as pd
import os
from autogluon.multimodal import MultiModalPredictor
from PIL import Image

def run_diagnostic(dataset_dir):
    print(f"Starting diagnostic check...")
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv")).head(5)
    
    # 扩展路径
    for col in ["image", "label"]:
        train_df[col] = train_df[col].apply(lambda p: os.path.join(dataset_dir, p))
    
    # 检查数据
    print("\n--- Data Check ---")
    for i, row in train_df.iterrows():
        lbl = np.array(Image.open(row['label']))
        unique = np.unique(lbl)
        fg_ratio = (lbl > 0).sum() / lbl.size
        print(f"Sample {i}: unique labels={unique}, foreground ratio={fg_ratio:.4f}")

    # 检查模型预测（使用一个未训练的模型看看初始状态）
    print("\n--- Model Initial State Check ---")
    try:
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            label="label",
            hyperparameters={
                "model.sam.num_mask_tokens": 10,
                "env.num_gpus": 1 if torch.cuda.is_available() else 0,
            }
        )
        
        # 强制初始化模型
        # 我们只预测一张图
        sample_img = train_df.iloc[0:1]
        print(f"Running predictor.predict on one sample...")
        preds = predictor.predict(sample_img)
        
        pred_mask = np.array(preds[0])
        print(f"Prediction unique values: {np.unique(pred_mask)}")
        print(f"Prediction foreground pixels: {(pred_mask > 0).sum()} / {pred_mask.size}")
        
    except Exception as e:
        print(f"Model check failed: {e}")

if __name__ == "__main__":
    dataset_path = "datasets/acdc_conv_lora/acdc_conv_lora"
    run_diagnostic(dataset_path)
