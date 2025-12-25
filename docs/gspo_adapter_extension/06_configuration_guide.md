# 配置参数完整说明

## 1. 命令行参数一览

### 1.1 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | leaf_disease_segmentation | 数据集名称 |
| `--rank` | int | 3 | Conv-LoRA 低秩维度 |
| `--expert_num` | int | 8 | Conv-LoRA MoE 专家数 |
| `--seed` | int | 42686693 | 随机种子 |
| `--num_gpus` | int | 1 | GPU 数量 |
| `--output_dir` | str | outputs | 输出目录 |

### 1.2 GSPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_enable` | flag | False | 启用 GSPO 训练 |
| `--gspo_group_size` | int | 4 | 每个样本生成的预测数 |
| `--gspo_warmup_epochs` | int | 5 | GSPO 预热轮数 |
| `--gspo_contrastive_weight` | float | 0.1 | 对比损失权重 |
| `--gspo_quality_momentum` | float | 0.9 | 专家质量历史动量 |

### 1.3 Adapter 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--adapter_enable` | flag | False | 启用 Encoder Adapter |
| `--adapter_dim` | int | 64 | Adapter 瓶颈维度 |

### 1.4 GSPO-Adapter 扩展参数 (🆕)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_adapter_enable` | flag | False | 启用 GSPO-Adapter 质量追踪 (阶段1) |
| `--gspo_adapter_momentum` | float | 0.9 | Adapter 质量历史动量 |
| `--gspo_adapter_scale_adaptation` | flag | False | 启用 Scale 自适应调节 (阶段2) |

### 1.5 奖励/损失形状参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lambda_smooth` | float | 0.1 | 平滑损失权重 |
| `--lambda_boundary` | float | 0.3 | 边界损失权重 |
| `--w_boundary` | float | 0.3 | 边界奖励权重 |
| `--w_smooth` | float | 0.1 | 平滑奖励权重 |
| `--w_thin` | float | 0.05 | 细结构奖励权重 |

---

## 2. 配置文件结构

### 2.1 optim/default.yaml

```yaml
optim:
  # ... 其他参数 ...
  
  lora:
    r: 8
    alpha: 8
    conv_lora_expert_num: 8
    gspo_enabled: False
    gspo_group_size: 3
    gspo_quality_momentum: 0.9
    gspo_warmup_epochs: 5
    gspo_contrastive_weight: 0.1
  
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
    # GSPO-Adapter 扩展参数
    adapter_enabled: False        # 阶段1
    adapter_momentum: 0.9         # 阶段1-2
    adapter_scale_adaptation: False  # 阶段2
```

### 2.2 model/default.yaml

```yaml
model:
  sam:
    checkpoint_name: "facebook/sam-vit-huge"  # 可改为 MedSAM 的 HuggingFace repo 或本地转换目录
    # ... 其他参数 ...
    
    # Encoder Adapter 配置
    adapter_enabled: False
    adapter_dim: 64
    adapter_gspo_enabled: False        # 阶段1
    adapter_gspo_momentum: 0.9         # 阶段1-2
    adapter_gspo_scale_adaptation: False  # 阶段2
```

---

## 3. 常用配置组合

### 3.1 基线：仅 Conv-LoRA

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs/baseline_convlora
```

### 3.2 Conv-LoRA + GSPO

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --output_dir outputs/convlora_gspo
```

### 3.3 使用 MedSAM 主干（其余配置保持不变）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --sam_checkpoint <medsam_hf_repo_or_local_dir> \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --output_dir outputs/medsam_gspo_adapter
```

> 说明：`--sam_checkpoint` 需指向 HuggingFace 兼容的 MedSAM 权重（或已转换的本地目录）；GSPO、Adapter、Conv-LoRA、Decoder-LoRA 等逻辑无需修改。

### 3.4 Conv-LoRA + GSPO + Adapter（无 GSPO-Adapter）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --output_dir outputs/convlora_gspo_adapter
```

### 3.5 完整混合架构（阶段1）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --output_dir outputs/hybrid_phase1
```

### 3.6 完整混合架构（阶段2）

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
    --output_dir outputs/hybrid_phase2
```

---

## 4. 参数依赖关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      参数依赖关系图                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 0: 基础                                                           │
│  ┌──────────────────┐                                                   │
│  │  --adapter_enable │ ← 必须先启用                                      │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  Level 1: GSPO-Adapter 阶段1                                            │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  --gspo_adapter_enable                               │                │
│  │                                                      │                │
│  │  依赖：                                              │                │
│  │  ├─ --adapter_enable (必须)                         │                │
│  │  └─ --gspo_enable (自动启用，如果未设置)             │                │
│  └───────────────────────────┬─────────────────────────┘                │
│                              │                                           │
│                              ▼                                           │
│  Level 2: GSPO-Adapter 阶段2                                            │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  --gspo_adapter_scale_adaptation                     │                │
│  │                                                      │                │
│  │  依赖：                                              │                │
│  │  ├─ --gspo_adapter_enable (必须)                    │                │
│  │  └─ --adapter_enable (必须)                         │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 超参数调优建议

### 5.1 Adapter 维度

| 维度 | 参数量 | 适用场景 |
|------|--------|----------|
| 32 | ~2.7M | 轻量级，快速验证 |
| 64 | ~5.3M | 标准，推荐 |
| 128 | ~10.5M | 高容量，复杂任务 |

### 5.2 GSPO 参数

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `gspo_group_size` | 2-6 | 越大越准确，但越慢 |
| `gspo_warmup_epochs` | 3-10 | 太小可能不稳定，太大浪费 |
| `gspo_contrastive_weight` | 0.05-0.2 | 太大可能干扰主损失 |
| `gspo_quality_momentum` | 0.85-0.95 | 太小震荡，太大迟钝 |

### 5.3 GSPO-Adapter 参数

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `gspo_adapter_momentum` | 0.9-0.95 | 与 GSPO 动量类似 |

---

## 6. 常见问题

### Q1: 是否必须同时使用 GSPO 和 Adapter？

**A**: 不是。你可以：
- 只用 GSPO：`--gspo_enable`
- 只用 Adapter：`--adapter_enable`
- 两者结合但不开启 GSPO-Adapter：`--gspo_enable --adapter_enable`
- 完整方案：`--gspo_enable --adapter_enable --gspo_adapter_enable`

### Q2: GSPO-Adapter 会影响推理速度吗？

**A**: 几乎不会。质量追踪只在训练时进行，推理时 Adapter 使用固定的 scale。

### Q3: 如何判断 GSPO-Adapter 是否工作？

**A**: 检查训练日志中的 Adapter 质量历史：
```python
# 训练后检查
stats = model.named_modules()
for name, m in stats:
    if hasattr(m, 'get_gspo_stats'):
        print(name, m.get_gspo_stats())
```

### Q4: 阶段2的 Scale 自适应是否总是有益？

**A**: 不一定。在某些情况下，固定 scale 可能更稳定。建议：
- 先用阶段1验证框架
- 如果效果好，再尝试阶段2
- 比较两者的验证集性能

---

## 7. 配置示例文件

### 7.1 最小配置（阶段1）

```bash
#!/bin/bash
# train_phase1.sh

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --output_dir outputs/phase1
```

### 7.2 完整配置（阶段2）

```bash
#!/bin/bash
# train_phase2.sh

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --gspo_group_size 4 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --gspo_quality_momentum 0.9 \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --gspo_adapter_momentum 0.9 \
    --gspo_adapter_scale_adaptation \
    --output_dir outputs/phase2_full
```

---

*最后更新: 2024-12-20*


