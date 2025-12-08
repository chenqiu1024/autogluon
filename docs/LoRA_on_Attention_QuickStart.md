# LoRA on Attention 快速开始指南

## 1分钟快速上手

### 训练（推荐配置）
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir outputs/lora_attn_test
```

### 评估
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path outputs/lora_attn_test
```

### 评估 + TTA（最佳性能）
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path outputs/lora_attn_test \
  --tta_enable
```

## 主要参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--decoder_attn_lora_enable` | 启用功能 | 必须 |
| `--decoder_attn_lora_r` | LoRA 秩 | 8 (默认) |
| `--decoder_attn_lora_alpha` | 缩放因子 | 8 (默认) |

## 常见组合

### 与 Conv-LoRA 组合
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 --expert_num 8 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --output_dir outputs/combined
```

### 与 Adapter 组合
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --adapter_enable --adapter_dim 64 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --output_dir outputs/adapter_lora
```

## 检查是否生效

训练时查看日志，应该看到：
```
Decoder Attention LoRA enabled: r=8, alpha=8, dropout=0.0
```

## 更多详情

查看完整文档：`LoRA_on_Attention_Design.md`

