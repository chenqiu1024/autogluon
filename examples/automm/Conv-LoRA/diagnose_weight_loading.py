#!/usr/bin/env python3
"""
Diagnose weight loading from checkpoint
"""

import torch
from pathlib import Path

ckpt_path = 'AutogluonModels/ag-20251116_011442/epoch=3-step=2000.ckpt'

print('🔍 诊断Checkpoint权重加载')
print('='*70)

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f'\n1. Checkpoint结构：')
print(f'  keys: {list(ckpt.keys())}')

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f'  state_dict中有{len(state_dict)}个keys')
    
    # 分析key的结构
    sample_keys = list(state_dict.keys())[:10]
    print(f'\n2. 前10个keys：')
    for k in sample_keys:
        print(f'    {k}')
    
    # 检查key的前缀
    prefixes = {}
    for k in state_dict.keys():
        prefix = k.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print(f'\n3. Key前缀统计：')
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f'    {prefix}: {count}个')
    
    # 检查Conv-LoRA相关的keys
    conv_lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower() or 'expert' in k.lower() or 'gate' in k.lower()]
    print(f'\n4. Conv-LoRA相关keys：{len(conv_lora_keys)}个')
    if conv_lora_keys:
        print(f'  示例：')
        for k in conv_lora_keys[:5]:
            print(f'    {k}')
    
    # 检查是否有'model.'前缀
    has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
    print(f'\n5. 是否有"model."前缀：{has_model_prefix}')
    
    if has_model_prefix:
        print(f'   → 需要strip掉"model."前缀再加载！')
    else:
        print(f'   → 直接加载应该OK')
    
    # 检查vision_encoder keys
    vision_keys = [k for k in state_dict.keys() if 'vision_encoder' in k]
    print(f'\n6. vision_encoder keys：{len(vision_keys)}个')
    if vision_keys:
        print(f'  示例：')
        for k in vision_keys[:3]:
            print(f'    {k}')

print('\n' + '='*70)
print('✅ 诊断完成')

