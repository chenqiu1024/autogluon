#!/usr/bin/env python3
"""
Debug script to understand why IoU is always 0 during RL training.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent.parent / 'multimodal' / 'src'))

from rl_utils import load_trained_conv_lora_model, prepare_dataset
from rl_train_routing_policy import compute_iou
from PIL import Image
import torchvision.transforms as T
import pandas as pd

def main():
    print('🔍 诊断IoU计算问题')
    print('='*70)
    
    # 1. Load model
    print('\n1. 加载模型...')
    model_path = Path('AutogluonModels/ag-20251116_011442/epoch=3-step=2000.ckpt')
    if not model_path.exists():
        print(f'❌ Model path not found: {model_path}')
        return
    
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model('isic2017', model_path=str(model_path), device='cuda')
    sam_model.eval()
    print(f'✓ 模型加载成功，{len(conv_lora_layers)}层')
    
    # 2. Load a sample
    print('\n2. 加载样本数据...')
    train_df = pd.read_csv('isic2017/train.csv')
    
    # Get first sample
    row = train_df.iloc[0]
    print(f'样本：{row["image"]}')
    
    # Load and preprocess image
    image = Image.open(row['image']).convert('RGB')
    label = Image.open(row['label']).convert('L')
    
    print(f'原始大小：image {image.size}, label {label.size}')
    
    # Resize to 1024x1024
    image = image.resize((1024, 1024), Image.BILINEAR)
    label = label.resize((1024, 1024), Image.NEAREST)
    
    # Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).cuda()  # (1, 3, 1024, 1024)
    label_tensor = torch.tensor([[list(label.getdata())]], dtype=torch.float32).view(1, 1024, 1024).cuda()
    label_tensor = (label_tensor > 0).float()  # Binarize
    
    print(f'Tensor shapes: image {image_tensor.shape}, label {label_tensor.shape}')
    print(f'Label stats: min={label_tensor.min()}, max={label_tensor.max()}, mean={label_tensor.mean()}')
    
    # 3. Forward through SAM
    print('\n3. SAM前向传播...')
    batch_dict = {'sam_image': image_tensor}
    
    with torch.no_grad():
        output = sam_model(batch_dict)
    
    print(f'输出keys：{output.keys()}')
    
    if 'sam' in output:
        print(f'output["sam"] keys：{output["sam"].keys()}')
        
        if 'logits' in output['sam']:
            pred_masks = output['sam']['logits']
            print(f'pred_masks shape：{pred_masks.shape}')
            print(f'pred_masks stats：min={pred_masks.min():.4f}, max={pred_masks.max():.4f}, mean={pred_masks.mean():.4f}')
            
            # Check for NaN/Inf
            if torch.isnan(pred_masks).any():
                print('⚠️ pred_masks contains NaN!')
            if torch.isinf(pred_masks).any():
                print('⚠️ pred_masks contains Inf!')
            
            # 4. Compute IoU
            print('\n4. 计算IoU...')
            
            # Adjust shapes if needed
            if pred_masks.dim() == 4:
                print(f'Squeezing pred_masks from {pred_masks.shape} to {pred_masks.squeeze(1).shape}')
                pred_masks_2d = pred_masks.squeeze(1)  # (B, H, W)
            else:
                pred_masks_2d = pred_masks
            
            if label_tensor.dim() == 4:
                label_tensor_2d = label_tensor.squeeze(1)
            else:
                label_tensor_2d = label_tensor
            
            print(f'Final shapes: pred {pred_masks_2d.shape}, label {label_tensor_2d.shape}')
            
            # Binarize and check
            pred_binary = (torch.sigmoid(pred_masks_2d) > 0.5).float()
            gt_binary = label_tensor_2d
            
            print(f'pred_binary: {pred_binary.sum()} pixels = 1 (out of {pred_binary.numel()})')
            print(f'gt_binary: {gt_binary.sum()} pixels = 1 (out of {gt_binary.numel()})')
            
            intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
            union = (pred_binary + gt_binary).clamp(max=1).sum(dim=(1, 2))
            
            print(f'intersection: {intersection}')
            print(f'union: {union}')
            
            iou = intersection / (union + 1e-8)
            print(f'\n✓ IoU = {iou.item():.4f}')
            
            if iou.item() < 0.01:
                print('\n❌ IoU接近0！可能原因：')
                print('   1. SAM输出全是0或极小值')
                print('   2. Label全是0（无目标）')
                print('   3. SAM没有正确训练')
                print('   4. 缺少prompt（SAM需要prompt才能分割！）')
        else:
            print('❌ output["sam"]中没有"logits"')
    else:
        print('❌ output中没有"sam"')
        print(f'完整output：{output}')

if __name__ == '__main__':
    main()

