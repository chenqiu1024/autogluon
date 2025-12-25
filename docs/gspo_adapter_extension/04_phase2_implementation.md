# 阶段2实现：Adapter Scale 自适应调节

## 1. 实现目标

**阶段2的核心思想**：根据质量历史动态调节 Adapter 的贡献度。

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段2：Scale 自适应调节                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  核心机制：                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  contribution_score = σ((quality_history - 0.5) × 4)    ││
│  │                                                          ││
│  │  effective_scale = learned_scale × contribution_score   ││
│  │                                                          ││
│  │  output = adapter_transform(x) × effective_scale        ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  效果：                                                       │
│  - quality_history 高 → contribution_score ↑ → 贡献更多     │
│  - quality_history 低 → contribution_score ↓ → 贡献减少     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 数学原理

### 2.1 贡献分数计算

$$
\text{contribution\_score} = \sigma\left(k \cdot (H_{adapter} - 0.5)\right)
$$

其中：
- $\sigma(\cdot)$ = sigmoid 函数
- $H_{adapter}$ = Adapter 质量历史 (范围 [0, 1])
- $k = 4$ = 放大系数
- $0.5$ = 中心点（中性质量）

### 2.2 有效缩放

$$
\text{scale}_{eff} = \text{scale}_{learned} \times \text{contribution\_score}
$$

$$
\text{output} = f_{adapter}(x) \times \text{scale}_{eff}
$$

### 2.3 行为分析

| quality_history | contribution_score | 含义 |
|-----------------|-------------------|------|
| 0.2 | σ(4×(-0.3)) = 0.23 | 低质量历史 → 大幅降低贡献 |
| 0.4 | σ(4×(-0.1)) = 0.40 | 略低 → 略微降低 |
| 0.5 | σ(4×0) = 0.50 | 中性 → 基准贡献 |
| 0.6 | σ(4×0.1) = 0.60 | 略高 → 略微增加 |
| 0.8 | σ(4×0.3) = 0.77 | 高质量历史 → 增加贡献 |

---

## 3. 代码实现

### 3.1 AdapterLayer 前向传播修改

```python
def forward(self, x):
    down = self.down_proj(x)
    act = self.activation(down)
    up = self.up_proj(act)
    
    # GSPO Phase 2: Adaptive scale based on quality history
    if self.gspo_enabled and self.gspo_scale_adaptation and self.training:
        effective_scale = self.scale * self.contribution_score
    else:
        effective_scale = self.scale
        
    return self.dropout(up) * effective_scale
```

### 3.2 贡献分数更新

```python
def update_quality_feedback(self, quality_score: torch.Tensor):
    if not self.gspo_enabled:
        return
    
    with torch.no_grad():
        quality_value = quality_score.mean().item()
        
        # 动量更新质量历史
        self.quality_history = (
            self.gspo_momentum * self.quality_history +
            (1 - self.gspo_momentum) * quality_value
        )
        
        # 计算贡献分数
        # 质量历史 0.5 → score 0.5 (中性)
        # 质量历史 > 0.5 → score > 0.5 (增加贡献)
        # 质量历史 < 0.5 → score < 0.5 (减少贡献)
        self.contribution_score = torch.sigmoid(
            (self.quality_history - 0.5) * 4.0
        )
        
        self.update_count += 1
```

---

## 4. 使用方法

### 4.1 命令行

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
    --gspo_adapter_amplification_factor 4.0 \
    --output_dir outputs/gspo_adapter_phase2
```

### 4.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_adapter_scale_adaptation` | flag | False | 启用质量自适应缩放 |
| `--gspo_adapter_amplification_factor` | float | 4.0 | 放大系数k，用于计算贡献分数 |

**注意**：`--gspo_adapter_scale_adaptation` 需要同时启用 `--gspo_adapter_enable`。

---

## 5. 工作原理

### 5.1 训练阶段行为

```
训练初期 (epoch 1-5, warmup)：
┌─────────────────────────────────────────────────────────────┐
│  quality_history ≈ 初始值 0.5                                │
│  contribution_score ≈ 0.5                                    │
│  effective_scale = learned_scale × 0.5                       │
│  → Adapter 贡献较小，让模型先学习主干特征                     │
└─────────────────────────────────────────────────────────────┘

训练中期 (epoch 5-20)：
┌─────────────────────────────────────────────────────────────┐
│  随着训练进行，quality 逐渐提升                               │
│  quality_history 开始累积                                    │
│                                                              │
│  如果 Adapter 贡献正向：                                      │
│    quality_history ↑ → contribution_score ↑ → 贡献增加       │
│                                                              │
│  如果 Adapter 贡献负向：                                      │
│    quality_history ↓ → contribution_score ↓ → 贡献减少       │
│    → 自动抑制不良的 Adapter 变换                              │
└─────────────────────────────────────────────────────────────┘

训练后期 (epoch 20-30)：
┌─────────────────────────────────────────────────────────────┐
│  quality_history 稳定在较高水平 (假设 0.75)                   │
│  contribution_score ≈ 0.77                                   │
│  effective_scale = learned_scale × 0.77                      │
│  → Adapter 稳定贡献，与 MLP 协同工作                          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 推理阶段行为

```python
if self.gspo_enabled and self.gspo_scale_adaptation and self.training:
    effective_scale = self.scale * self.contribution_score
else:
    effective_scale = self.scale  # 推理时使用固定 scale
```

**设计选择**：
- 训练时：使用动态 contribution_score
- 推理时：使用固定的 learned_scale

这样设计的原因：
1. 推理时不需要动态调整
2. 保持推理行为的一致性
3. contribution_score 的信息已经通过梯度传递到 learned_scale

---

## 6. 信号流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       阶段2 信号流（训练阶段）                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        前向传播                                         │ │
│  │                                                                         │ │
│  │  hidden_states                                                         │ │
│  │       │                                                                 │ │
│  │       ├──────────────────────┬────────────────────────────────────────┐│ │
│  │       ▼                      ▼                                        ││ │
│  │  ┌─────────┐          ┌─────────────────────────────────────────────┐ ││ │
│  │  │   MLP   │          │           Adapter (with GSPO)               │ ││ │
│  │  │ C→4C→C  │          │  ┌───────────────────────────────────────┐  │ ││ │
│  │  └────┬────┘          │  │  down_proj → GELU → up_proj           │  │ ││ │
│  │       │               │  │              ↓                         │  │ ││ │
│  │       │               │  │  ┌─────────────────────────────────┐  │  │ ││ │
│  │       │               │  │  │ effective_scale =               │  │  │ ││ │
│  │       │               │  │  │   learned_scale × contrib_score │  │  │ ││ │
│  │       │               │  │  └─────────────────────────────────┘  │  │ ││ │
│  │       │               │  │              ↓                         │  │ ││ │
│  │       │               │  │  output = transform × effective_scale │  │ ││ │
│  │       │               │  └───────────────────────────────────────┘  │ ││ │
│  │       │               └────────────────────────┬────────────────────┘ ││ │
│  │       │                                        │                       ││ │
│  │       ▼                                        ▼                       ││ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ ││ │
│  │  │  hidden_states = residual + mlp_out + adapter_out                │ ││ │
│  │  └──────────────────────────────────────────────────────────────────┘ ││ │
│  │                                                                        ││ │
│  └────────────────────────────────────────────────────────────────────────┘│ │
│                                    ▼                                        │
│                           最终预测 & 质量评估                                │
│                           quality = IoU(pred, gt)                           │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        反馈更新                                         │ │
│  │                                                                         │ │
│  │  quality_history = momentum × history + (1-momentum) × quality         │ │
│  │                           ↓                                             │ │
│  │  contribution_score = σ((quality_history - 0.5) × 4)                   │ │
│  │                           ↓                                             │ │
│  │                    [用于下一次前向传播]                                  │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 与阶段1的对比

| 特性 | 阶段1 | 阶段2 |
|------|-------|-------|
| 质量追踪 | ✅ 有 | ✅ 有 |
| 贡献分数计算 | ✅ 有（但不使用） | ✅ 有（使用） |
| 前向传播影响 | ❌ 无 | ✅ 动态调整 scale |
| 梯度影响 | 统一加权 | 统一加权 + scale 调节 |
| 适用场景 | 验证框架 | 自适应优化 |

---

## 8. 调参建议

### 8.1 放大系数 k

当前默认 k=4，可以通过命令行参数调整：

```bash
--gspo_adapter_amplification_factor 2.0  # 更平滑
--gspo_adapter_amplification_factor 4.0  # 默认
--gspo_adapter_amplification_factor 8.0  # 更激进
```

| k 值 | 效果 |
|------|------|
| 2 | 更平滑，contribution 变化小 |
| 4 | 默认，适中 |
| 8 | 更激进，contribution 变化大 |

### 8.2 动量 momentum

```bash
--gspo_adapter_momentum 0.95  # 更平滑，历史影响更大
--gspo_adapter_momentum 0.85  # 更敏感，近期影响更大
```

### 8.3 推荐配置

**保守配置**（稳定性优先）:
```bash
--gspo_adapter_momentum 0.95 \
--gspo_adapter_amplification_factor 2.0 \
--gspo_adapter_scale_adaptation
```

**激进配置**（自适应性优先）:
```bash
--gspo_adapter_momentum 0.85 \
--gspo_adapter_amplification_factor 8.0 \
--gspo_adapter_scale_adaptation
```

---

## 9. 预期效果

### 9.1 理想情况

- 训练初期：Adapter 贡献较小，避免干扰主干学习
- 训练中期：根据质量反馈自动调整贡献
- 训练后期：有效的 Adapter 获得更多权重，无效的被抑制

### 9.2 可能的问题

1. **贡献过低**：如果所有 Adapter 的 quality_history 都偏低，可能导致 Adapter 完全失效
   - 解决：调低放大系数 k，或增加 momentum

2. **震荡**：如果 momentum 太低，contribution_score 可能震荡
   - 解决：增加 momentum（如 0.95）

3. **收敛慢**：如果 contribution_score 变化太小，可能看不出效果
   - 解决：增加放大系数 k

---

*下一篇：[05_phase3_implementation.md](./05_phase3_implementation.md) - 阶段3实现细节（可选高级功能）*


