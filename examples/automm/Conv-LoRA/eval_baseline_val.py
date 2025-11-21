import sys
sys.path.insert(0, '/root/autodl-tmp/works/autogluon/multimodal/src')
sys.path.insert(0, '/root/autodl-tmp/works/autogluon/common/src')
sys.path.insert(0, '/root/autodl-tmp/works/autogluon/core/src')

import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# Load baseline model
model_path = "AutogluonModels/ag-20251113_165105"
print(f"Loading baseline model from: {model_path}")
predictor = MultiModalPredictor.load(model_path)

# Prepare validation data
dataset_dir = "datasets/isic2017/isic2017"
val_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))

# Expand paths
for col in ["image", "label"]:
    val_df[col] = val_df[col].apply(lambda x: os.path.join(dataset_dir, x))

print(f"\nValidation set size: {len(val_df)}")

# Evaluate
print("\n" + "="*80)
print("Evaluating Baseline on Validation Set")
print("="*80)
results = predictor.evaluate(val_df, metrics=["iou", "dice"])
print(f"\nResults:")
print(f"  Val IoU:  {results['iou']:.4f}")
print(f"  Val DICE: {results['dice']:.4f}")
print("="*80)
