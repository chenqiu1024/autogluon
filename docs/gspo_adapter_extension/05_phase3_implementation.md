# 阶段3实现：Adapter MoE 架构（可选高级功能）

## 1. 实现目标

**阶段3的核心思想**：将单一 Adapter 扩展为多变体 MoE 架构，实现更精细的质量驱动选择。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    阶段3：Adapter MoE 架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  当前架构（阶段1-2）：                  目标架构（阶段3）：              │
│  ┌───────────────────┐                ┌───────────────────────────────┐ │
│  │  Single Adapter   │                │     Adapter MoE               │ │
│  │  ┌─────────────┐  │                │  ┌──────────────────────────┐ │ │
│  │  │ C→64→C      │  │      →        │  │ Gate + M Adapter Variants│ │ │
│  │  │ + GSPO      │  │                │  │ ┌─────┐ ┌─────┐ ┌─────┐ │ │ │
│  │  └─────────────┘  │                │  │ │ A1  │ │ A2  │ │ AM  │ │ │ │
│  └───────────────────┘                │  │ └──┬──┘ └──┬──┘ └──┬──┘ │ │ │
│                                        │  │    └──────┼──────┘     │ │ │
│                                        │  │           ▼            │ │ │
│                                        │  │    Weighted Sum        │ │ │
│                                        │  │    + GSPO Feedback     │ │ │
│                                        │  └──────────────────────────┘ │ │
│                                        └───────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 设计说明

### 2.1 为什么需要 Adapter MoE？

**阶段2的局限**：
- 单一 Adapter 只能做一种变换
- 质量反馈只能调节**贡献强度**，不能调节**变换类型**

**阶段3的优势**：
- 多个 Adapter 变体提供不同的变换选择
- 门控网络动态选择最佳变体
- GSPO 反馈优化专家选择策略

### 2.2 与 Conv-LoRA MoE 的对比

| 维度 | Conv-LoRA MoE | Adapter MoE |
|------|---------------|-------------|
| 位置 | Attention Q/K/V | MLP 之后 |
| 优化维度 | 空间 (H×W) | 通道 (C) |
| 专家类型 | 卷积（不同感受野） | 全连接（不同瓶颈） |
| 专家数量 | 8（默认） | 4（推荐） |
| 门控 | 空间特征驱动 | 全局特征驱动 |

---

## 3. 架构设计

### 3.1 Adapter MoE 结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AdapterMoE Module                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: x [B, H, W, C]                                                  │
│         │                                                                │
│         ├─────────────────────────────────────────────────────────────┐ │
│         │                                                             │ │
│         ▼                                                             ▼ │
│  ┌──────────────────────┐                    ┌─────────────────────────┐│
│  │    Gate Network      │                    │   Expert Adapters (M)   ││
│  │  ┌─────────────────┐ │                    │  ┌───────────────────┐  ││
│  │  │ Global Pool     │ │                    │  │ Expert 1 (dim=32) │  ││
│  │  │      ↓          │ │                    │  │ C→32→C            │  ││
│  │  │ Linear(C, M)    │ │                    │  └───────────────────┘  ││
│  │  │      ↓          │ │                    │  ┌───────────────────┐  ││
│  │  │ Softmax + Noise │ │                    │  │ Expert 2 (dim=64) │  ││
│  │  │      ↓          │ │                    │  │ C→64→C            │  ││
│  │  │ TopK Selection  │ │                    │  └───────────────────┘  ││
│  │  └─────────────────┘ │                    │  ┌───────────────────┐  ││
│  │         │            │                    │  │ Expert 3 (dim=128)│  ││
│  │  weights [B, M]      │                    │  │ C→128→C           │  ││
│  └─────────│────────────┘                    │  └───────────────────┘  ││
│            │                                 │  ┌───────────────────┐  ││
│            │                                 │  │ Expert 4 (dim=96) │  ││
│            │                                 │  │ + Skip Connection │  ││
│            │                                 │  └───────────────────┘  ││
│            │                                 └───────────┬─────────────┘│
│            │                                             │               │
│            └─────────────────────┬───────────────────────┘               │
│                                  │                                       │
│                                  ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │  Weighted Sum           │                          │
│                    │  output = Σ(w_i * E_i)  │                          │
│                    └─────────────────────────┘                          │
│                                  │                                       │
│                                  ▼                                       │
│  Output: [B, H, W, C]                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 专家变体设计

**推荐的4个专家变体**：

| 专家 | 瓶颈维度 | 特点 | 适用场景 |
|------|----------|------|----------|
| E1 | 32 | 轻量级 | 简单特征适应 |
| E2 | 64 | 标准 | 通用变换 |
| E3 | 128 | 高容量 | 复杂变换 |
| E4 | 64 + Skip | 残差增强 | 保持原始特征 |

### 3.3 GSPO 集成

```python
class AdapterMoE(nn.Module):
    def __init__(self, in_features, expert_dims=[32, 64, 128, 64], num_experts=4):
        # Gate network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, num_experts),
        )
        
        # Expert adapters
        self.experts = nn.ModuleList([
            AdapterLayer(in_features, dim) for dim in expert_dims
        ])
        
        # GSPO quality tracking (per expert)
        self.register_buffer("expert_quality_history", torch.ones(num_experts) * 0.5)
        self.register_buffer("expert_usage_count", torch.zeros(num_experts))
    
    def forward(self, x, output_moe_loss=False):
        B, H, W, C = x.shape
        
        # Gate: compute expert weights
        gate_input = x.mean(dim=(1, 2))  # [B, C]
        logits = self.gate(gate_input)   # [B, M]
        
        # Add quality bias (GSPO)
        if self.training:
            quality_bias = self._compute_quality_bias()
            logits = logits + quality_bias
        
        weights = F.softmax(logits, dim=-1)  # [B, M]
        
        # Expert forward
        expert_outputs = [exp(x) for exp in self.experts]  # List of [B, H, W, C]
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, H, W, C, M]
        
        # Weighted sum
        weights_expanded = weights.view(B, 1, 1, 1, -1)  # [B, 1, 1, 1, M]
        output = (expert_stack * weights_expanded).sum(dim=-1)  # [B, H, W, C]
        
        if output_moe_loss:
            moe_loss = self._compute_load_balance_loss(weights)
            return output, moe_loss
        
        return output
```

---

## 4. 实现状态

> ⚠️ **注意**：阶段3目前为**设计阶段**，代码尚未实现。

### 4.1 实现优先级

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | 阶段1 - 统一优势加权 | ✅ 已完成 |
| P0 | 阶段2 - Scale 自适应 | ✅ 已完成 |
| P1 | 阶段3 - Adapter MoE | 📝 设计完成 |

### 4.2 实现计划

如果阶段1-2在实验中证明有效，将在后续版本中实现阶段3：

1. **实现 AdapterMoE 类**
2. **添加配置参数**：
   - `--adapter_moe_enable`
   - `--adapter_moe_num_experts`
   - `--adapter_moe_expert_dims`
3. **集成 GSPO 反馈**
4. **实验验证**

---

## 5. 预期配置（未来）

```bash
# 阶段3 完整配置（未来版本）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --adapter_moe_enable \
    --adapter_moe_num_experts 4 \
    --output_dir outputs/gspo_adapter_phase3
```

---

## 6. 与完整 GSPO 框架的对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    完整 GSPO 混合架构（阶段3）                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SamVisionLayer:                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  Attention Path:                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Conv-LoRA MoE (8 experts)                                        │  │ │
│  │  │  - 8 种卷积核：不同感受野                                          │  │ │
│  │  │  - GSPO 质量反馈：空间选择优化                                      │  │ │
│  │  │  - 优化目标：空间维度 (H×W)                                        │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  MLP Path:                                                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  MLP (原始)     +     Adapter MoE (4 experts)                     │  │ │
│  │  │                        - 4 种变体：不同容量                        │  │ │
│  │  │                        - GSPO 质量反馈：变换选择优化                │  │ │
│  │  │                        - 优化目标：通道维度 (C)                    │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  统一 GSPO 框架：                                                            │
│  - 共享质量信号                                                              │
│  - 独立专家选择                                                              │
│  - 协同优化：空间 + 通道                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 潜在挑战

### 7.1 参数量增加

| 配置 | Adapter 参数量 |
|------|----------------|
| 阶段1-2（单一 Adapter） | ~5.3M (32层 × 165K) |
| 阶段3（4 专家 MoE） | ~21M (32层 × 4 × 165K) |

**缓解策略**：
- 使用更小的专家维度
- 减少专家数量
- 使用 TopK 稀疏选择

### 7.2 训练稳定性

**潜在问题**：
- 两个 MoE 系统（Conv-LoRA + Adapter）可能相互干扰
- 门控网络训练可能不稳定

**缓解策略**：
- 使用更大的 warmup
- 逐步启用（先 Conv-LoRA MoE，再 Adapter MoE）
- 使用负载均衡损失

### 7.3 推理效率

**问题**：MoE 增加推理时间

**缓解策略**：
- 推理时使用 Top1 而非 TopK
- 或者合并专家权重为单一 Adapter

---

## 8. 总结

阶段3是一个**可选的高级功能**，适用于：
- 已验证阶段1-2有效的场景
- 需要更精细特征适应的任务
- 有足够计算资源的项目

对于大多数应用，**阶段2**（Scale 自适应）应该已经足够。

---

*下一篇：[06_configuration_guide.md](./06_configuration_guide.md) - 配置参数完整说明*


