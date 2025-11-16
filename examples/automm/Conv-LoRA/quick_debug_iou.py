#!/usr/bin/env python3
"""
Quick diagnostic: Why is IoU always 0?
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent.parent / 'multimodal' / 'src'))

from rl_utils import load_trained_conv_lora_model, prepare_dataset

def main():
    print('🔍 诊断IoU=0问题')
    print('='*70)
    
    # 1. Load model
    print('\n1. 加载模型...')
    model_path = 'AutogluonModels/ag-20251116_011442/epoch=3-step=2000.ckpt'
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model('isic2017', model_path=model_path, device='cuda')
    sam_model.train()  # RL train mode
    print(f'✓ 模型：{len(conv_lora_layers)}层')
    
    # 2. Load data
    print('\n2. 加载数据...')
    train_df, dataset_dir = prepare_dataset('isic2017', split='train')
    print(f'✓ 数据集：{len(train_df)}样本')
    
    #3. Get first sample
    from PIL import Image
    import torchvision.transforms as T
    
    row = train_df.iloc[0]
    image = Image.open(row['image']).convert('RGB').resize((1024, 1024), Image.BILINEAR)
    label = Image.open(row['label']).convert('L').resize((1024, 1024), Image.NEAREST)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).cuda()
    label_tensor = torch.tensor([[list(label.getdata())]], dtype=torch.float32).view(1, 1024, 1024).cuda()
    label_tensor = (label_tensor > 0).float()
    
    print(f'✓ 样本：image {image_tensor.shape}, label {label_tensor.shape}')
    print(f'  Label: {label_tensor.sum().item()} foreground pixels / {label_tensor.numel()} total')
    
    # 4. Forward
    print('\n3. SAM前向...')
    batch_dict = {'sam_image': image_tensor, 'sam_label': label_tensor}
    
    print(f'模型状态：training={sam_model.training}')
    sam_model.eval()  # Use eval mode
    print(f'切换到eval模式')
    
    with torch.no_grad():
        output = sam_model(batch_dict)
    
    print(f'输出类型：{type(output)}')
    print(f'输出keys：{output.keys() if isinstance(output, dict) else "NOT A DICT!"}')
    
    if isinstance(output, dict) and 'sam' in output:
        print(f'\\noutput["sam"] keys：{output["sam"].keys()}')
        
        if 'logits' in output['sam']:
            pred = output['sam']['logits']
            print(f'\\npred_masks shape：{pred.shape}')
            print(f'pred_masks stats：min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}')
            
            # Check sigmoid
            pred_sigmoid = torch.sigmoid(pred)
            print(f'sigmoid(pred) stats：min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}')
            
            # Check binary
            pred_binary = (pred_sigmoid > 0.5).float()
            print(f'binary pred：{pred_binary.sum()} pixels = 1')
            
            # IoU
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            if label_tensor.dim() == 4:
                label_tensor = label_tensor.squeeze(1)
            
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            gt_bin = label_tensor
            
            intersection = (pred_bin * gt_bin).sum()
            union = (pred_bin + gt_bin).clamp(max=1).sum()
            iou = intersection / (union + 1e-8)
            
            print(f'\\n📊 IoU计算：')
            print(f'  Intersection：{intersection.item()}')
            print(f'  Union：{union.item()}')
            print(f'  IoU：{iou.item():.6f}')
            
            if iou.item() < 0.01:
                print(f'\\n❌ IoU接近0！')
                print(f'可能原因：')
                print(f'1. SAM没有prompt → 无法分割')
                print(f'2. SAM输出全是背景')
                print(f'3. Model没有正确训练')
                
                print(f'\\n🔎 检查SAM是否需要prompt：')
                print(f'  SAM模型要求：point_coords, point_labels, boxes, or masks')
                print(f'  当前只提供了：{list(batch_dict.keys())}')
                print(f'  → 这可能是问题所在！')
        else:
            print('❌ No "logits" in output["sam"]')
    else:
        print('❌ output不是dict或没有"sam" key')
        print(f'output: {output}')

if __name__ == '__main__':
    main()

