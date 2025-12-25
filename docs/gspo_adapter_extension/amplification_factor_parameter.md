# 放大系数 k 命令行参数支持

## 概述

本次更新为 GSPO Adapter 和 GSPO-LoRA 的放大系数 k 添加了命令行参数支持，使得用户可以在运行时灵活调整贡献分数的计算方式，而无需修改代码。

## 背景

在 GSPO 的阶段2实现中，贡献分数（contribution score）通过以下公式计算：

$$
\text{contribution\_score} = \sigma\left(k \cdot (H_{adapter} - 0.5)\right)
$$

其中：
- $\sigma(\cdot)$ = sigmoid 函数
- $H_{adapter}$ = Adapter/LoRA 质量历史 (范围 [0, 1])
- $k$ = 放大系数（之前硬编码为 4.0）
- $0.5$ = 中心点（中性质量）

放大系数 k 控制了质量历史对贡献分数的影响程度：
- k 越大，贡献分数对质量历史的变化越敏感（更激进）
- k 越小，贡献分数对质量历史的变化越不敏感（更平滑）

## 修改内容

### 1. 新增命令行参数

#### 1.1 Adapter 放大系数

```bash
--gspo_adapter_amplification_factor FLOAT
```

- **类型**: float
- **默认值**: 4.0
- **说明**: 用于 Encoder Adapter 的贡献分数计算

#### 1.2 LoRA 放大系数

```bash
--gspo_lora_attention_amplification_factor FLOAT
```

- **类型**: float
- **默认值**: 4.0
- **说明**: 用于 Decoder Attention LoRA 的贡献分数计算

### 2. 代码修改

#### 2.1 AdapterLayer 类

**文件**: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`

- 添加 `gspo_amplification_factor` 参数到 `__init__` 方法
- 在 `update_quality_feedback` 方法中使用该参数替代硬编码的 4.0

```python
self.contribution_score = torch.sigmoid(
    (self.quality_history - 0.5) * self.gspo_amplification_factor
)
```

#### 2.2 LoRALinear 类

**文件**: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`

- 添加 `gspo_amplification_factor` 参数到 `__init__` 方法
- 在 `update_quality_feedback` 方法中使用该参数替代硬编码的 4.0

#### 2.3 SAM 模型类

**文件**: `multimodal/src/autogluon/multimodal/models/sam.py`

- 添加 `adapter_gspo_amplification_factor` 参数
- 添加 `gspo_lora_attention_amplification_factor` 参数
- 将参数传递到 config 和 AdapterLayer 实例化

#### 2.4 自定义 SAM 模型

**文件**: `multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`

- 在所有相关类中添加 `gspo_lora_amplification_factor` 参数：
  - `SamAttention`
  - `SamTwoWayAttentionBlock`
  - `SamTwoWayTransformer`
  - `SamMaskDecoder`
- 从 config 中提取并传递该参数

#### 2.5 训练脚本

**文件**: `examples/automm/Conv-LoRA/run_semantic_segmentation.py`

- 添加命令行参数解析
- 将参数传递到 hyperparameters 配置

## 使用示例

### 基本使用

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --gspo_adapter_amplification_factor 4.0
```

### 保守配置（稳定性优先）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --gspo_adapter_momentum 0.95 \
    --gspo_adapter_amplification_factor 2.0
```

**效果**：
- 贡献分数变化更平滑
- 对质量历史的波动不敏感
- 适合训练不稳定的场景

### 激进配置（自适应性优先）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --gspo_adapter_momentum 0.85 \
    --gspo_adapter_amplification_factor 8.0
```

**效果**：
- 贡献分数变化更激进
- 对质量历史的变化高度敏感
- 快速抑制低质量 Adapter，快速增强高质量 Adapter

### 同时使用 Adapter 和 LoRA

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --gspo_adapter_amplification_factor 4.0 \
    --decoder_attn_lora_enable \
    --decoder_attn_lora_r 8 \
    --gspo_lora_attention_enable \
    --gspo_lora_attention_scale_adaptation \
    --gspo_lora_attention_amplification_factor 6.0
```

## 参数调优指南

### k 值的影响

| k 值 | 效果 | 适用场景 |
|------|------|----------|
| 2.0 | 非常平滑，变化缓慢 | 训练极不稳定，需要保守策略 |
| 4.0 | 适中（默认） | 大多数场景 |
| 6.0 | 较激进，变化明显 | 质量信号清晰，希望快速响应 |
| 8.0 | 非常激进，变化剧烈 | 需要快速区分好坏 Adapter |

### 与 momentum 的配合

| momentum | amplification_factor | 特点 |
|----------|---------------------|------|
| 0.95 | 2.0 | 极度保守，长期稳定 |
| 0.9 | 4.0 | 平衡（默认） |
| 0.85 | 6.0 | 较激进，快速适应 |
| 0.8 | 8.0 | 极度激进，可能不稳定 |

### 调参建议

1. **从默认值开始**：k=4.0, momentum=0.9
2. **观察训练曲线**：
   - 如果贡献分数震荡 → 降低 k 或增加 momentum
   - 如果贡献分数变化太慢 → 增加 k 或降低 momentum
3. **根据任务特点调整**：
   - 数据噪声大 → 降低 k
   - 质量信号清晰 → 增加 k

## 数学分析

### 不同 k 值的 sigmoid 响应

当 quality_history 在 [0, 1] 范围内变化时：

**k = 2.0**:
- quality_history = 0.2 → contribution_score = 0.38
- quality_history = 0.8 → contribution_score = 0.62
- 变化范围：0.24

**k = 4.0** (默认):
- quality_history = 0.2 → contribution_score = 0.23
- quality_history = 0.8 → contribution_score = 0.77
- 变化范围：0.54

**k = 8.0**:
- quality_history = 0.2 → contribution_score = 0.07
- quality_history = 0.8 → contribution_score = 0.93
- 变化范围：0.86

可以看出，k 越大，贡献分数的动态范围越大，对质量历史的响应越敏感。

## 向后兼容性

- 所有新参数都有默认值 4.0，与之前的硬编码值一致
- 不使用新参数时，行为与之前完全相同
- 完全向后兼容

## 总结

本次更新通过添加命令行参数，使得 GSPO 的放大系数 k 可以在运行时灵活配置，为用户提供了更多的调优空间。用户可以根据具体任务和训练情况，选择合适的 k 值来平衡稳定性和自适应性。

## 相关文档

- [04_phase2_implementation.md](./04_phase2_implementation.md) - 阶段2实现细节
- [06_configuration_guide.md](./06_configuration_guide.md) - 完整配置指南

