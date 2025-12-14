# LoRA on Attention 配置修复

## 问题描述

在初次实现 LoRA on Attention 时，训练报错：
```
KeyError: '"model.sam.decoder_attention_lora_r" is not found in the config.
```

## 原因分析

AutoGluon 的配置系统使用 OmegaConf，在应用 hyperparameters 覆盖时会检查键是否存在（`check_key_exist=True`）。由于 `decoder_attention_lora_*` 参数是新增的，但没有在默认配置文件中预定义，导致配置验证失败。

## 解决方案

在 `multimodal/src/autogluon/multimodal/configs/model/default.yaml` 的 `sam` 配置节中添加默认值：

```yaml
sam:
  checkpoint_name: "facebook/sam-vit-huge"
  # ... 其他配置 ...
  adapter_enabled: False
  adapter_dim: 64
  decoder_attention_lora_r: 0          # 新增：默认禁用
  decoder_attention_lora_alpha: 1      # 新增：默认缩放因子
  decoder_attention_lora_dropout: 0.0  # 新增：默认无 dropout
```

## 验证方法

重新运行训练命令：
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir outputs/lora_attn_r8
```

应该看到日志输出：
```
Enabling Decoder Attention LoRA: r=8, alpha=8, dropout=0.0
```
并且训练正常开始，不再报 KeyError。

## 向后兼容性确认

- ✅ 默认值 `decoder_attention_lora_r: 0` 确保 LoRA 默认禁用
- ✅ 旧训练脚本（不指定这些参数）将继续使用默认值
- ✅ 不影响现有功能

## 修改文件

- `multimodal/src/autogluon/multimodal/configs/model/default.yaml`

## 更新日期

2024-12-08

