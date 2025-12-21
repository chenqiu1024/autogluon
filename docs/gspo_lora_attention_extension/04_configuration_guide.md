# 配置参数说明

## 1. 完整参数列表

### 1.1 基础 LoRA on Attention 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--decoder_attn_lora_enable` | flag | False | 启用 Decoder LoRA |
| `--decoder_attn_lora_r` | int | 8 | LoRA 秩 |
| `--decoder_attn_lora_alpha` | int | 8 | LoRA 缩放因子 |
| `--decoder_attn_lora_dropout` | float | 0.0 | LoRA Dropout |

### 1.2 GSPO-LoRA 扩展参数 (NEW)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_lora_attention_enable` | flag | False | 启用 GSPO 对 LoRA 的优化 |
| `--gspo_lora_attention_momentum` | float | 0.9 | 质量历史动量 |
| `--gspo_lora_attention_scale_adaptation` | flag | False | 启用动态缩放自适应 |

---

## 2. 配置组合示例

### 2.1 基础 LoRA（无 GSPO）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --decoder_attn_lora_enable \
    --decoder_attn_lora_r 8 \
    --output_dir outputs/lora_baseline
```

**行为**：使用标准固定 scaling 的 LoRA。

### 2.2 LoRA + GSPO 质量追踪

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --decoder_attn_lora_enable \
    --gspo_lora_attention_enable \
    --output_dir outputs/lora_gspo_phase1
```

**行为**：
- GSPO 训练策略激活
- LoRA 层追踪质量历史
- **不**使用动态缩放

### 2.3 LoRA + GSPO 动态缩放（完整功能）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --decoder_attn_lora_enable \
    --gspo_lora_attention_enable \
    --gspo_lora_attention_scale_adaptation \
    --output_dir outputs/lora_gspo_phase2
```

**行为**：
- GSPO 训练策略激活
- LoRA 层追踪质量历史
- 基于质量历史动态调整 LoRA scaling

### 2.4 完整混合架构（所有 PEFT 模块 + GSPO）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --decoder_attn_lora_enable \
    --gspo_lora_attention_enable \
    --gspo_lora_attention_scale_adaptation \
    --output_dir outputs/full_hybrid_gspo
```

**行为**：
- Conv-LoRA MoE：GSPO 优化专家选择
- Encoder Adapter：GSPO 动态缩放
- Decoder LoRA：GSPO 动态缩放
- 所有模块共享统一的质量反馈

---

## 3. 配置依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         配置依赖关系图                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  --gspo_enable                                                              │
│      │                                                                       │
│      ├── 前提条件：必须启用，否则以下配置无效                                  │
│      │                                                                       │
│      ├── --gspo_lora_attention_enable                                       │
│      │       │                                                               │
│      │       └── --gspo_lora_attention_scale_adaptation                     │
│      │           (需要先启用 gspo_lora_attention_enable)                     │
│      │                                                                       │
│      └── --gspo_adapter_enable                                              │
│              │                                                               │
│              └── --gspo_adapter_scale_adaptation                            │
│                  (需要先启用 gspo_adapter_enable)                            │
│                                                                              │
│  --decoder_attn_lora_enable                                                 │
│      │                                                                       │
│      └── 前提条件：必须启用，否则 GSPO-LoRA 无效                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 配置文件方式

除了 CLI 参数，也可以通过配置文件设置：

### 4.1 YAML 配置

```yaml
model:
  sam:
    # 基础 LoRA
    decoder_attention_lora_r: 8
    decoder_attention_lora_alpha: 8
    decoder_attention_lora_dropout: 0.0
    
    # GSPO-LoRA (NEW)
    gspo_lora_attention_enabled: True
    gspo_lora_attention_momentum: 0.9
    gspo_lora_attention_scale_adaptation: True

optim:
  lora:
    gspo_enabled: True
    gspo_group_size: 4
    gspo_warmup_epochs: 5
    gspo_contrastive_weight: 0.1
```

### 4.2 Python 代码

```python
from autogluon.multimodal import MultiModalPredictor

hyperparameters = {
    # 基础 LoRA
    "model.sam.decoder_attention_lora_r": 8,
    "model.sam.decoder_attention_lora_alpha": 8,
    
    # GSPO-LoRA (NEW)
    "model.sam.gspo_lora_attention_enabled": True,
    "model.sam.gspo_lora_attention_momentum": 0.9,
    "model.sam.gspo_lora_attention_scale_adaptation": True,
    
    # GSPO 训练
    "optim.lora.gspo_enabled": True,
    "optim.lora.gspo_group_size": 4,
    "optim.lora.gspo_warmup_epochs": 5,
}

predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    hyperparameters=hyperparameters,
)
```

---

## 5. 推荐配置

### 5.1 快速实验

```bash
# 快速验证 GSPO-LoRA 效果
--gspo_enable \
--decoder_attn_lora_enable \
--gspo_lora_attention_enable
```

### 5.2 最佳性能

```bash
# 完整 GSPO 混合架构
--gspo_enable \
--adapter_enable \
--gspo_adapter_enable \
--gspo_adapter_scale_adaptation \
--decoder_attn_lora_enable \
--gspo_lora_attention_enable \
--gspo_lora_attention_scale_adaptation
```

### 5.3 调参建议

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `decoder_attn_lora_r` | 4-16 | 越大参数越多，推荐 8 |
| `gspo_lora_attention_momentum` | 0.8-0.95 | 越大越平滑，推荐 0.9 |
| `gspo_warmup_epochs` | 3-10 | 越大预热越长，推荐 5 |

---

## 6. 常见问题

### Q1: 为什么 `--gspo_lora_attention_enable` 没有效果？

**检查**：
1. 是否启用了 `--gspo_enable`
2. 是否启用了 `--decoder_attn_lora_enable`

### Q2: 如何查看 LoRA 的质量历史？

**方法**：在训练后，通过 trainer 获取统计信息：

```python
stats = gspo_trainer.get_lora_stats(model)
print(stats)
# 输出：{'mask_decoder.transformer.layers.0.self_attn.q_proj': {'quality_history': 0.72, ...}, ...}
```

### Q3: 动态缩放会影响推理速度吗？

**回答**：不会。`contribution_score` 在推理时固定为 1.0（因为 `self.training=False`）。

---

*返回：[README.md](./README.md)*

