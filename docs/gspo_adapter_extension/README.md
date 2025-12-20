# GSPO 扩展到 Encoder Adapter 的设计与实现

**版本**: v1.0  
**创建日期**: 2024-12-20  
**作者**: AutoGluon Team

---

## 📋 目录

1. [项目概述](#项目概述)
2. [设计目标](#设计目标)
3. [实现阶段](#实现阶段)
4. [文档索引](#文档索引)

---

## 项目概述

本项目扩展 GSPO（Group Sequence Policy Optimization）强化学习框架，使其不仅能够优化 Conv-LoRA 的 MoE 专家选择，还能同时优化 Vision Encoder 中的 FC Adapter 模块。

### 当前状态 vs 目标状态

```
当前状态：                              目标状态：
┌────────────────────────┐              ┌────────────────────────┐
│  Conv-LoRA MoE         │              │  Conv-LoRA MoE         │
│  ├─ GSPO 优化 ✅        │              │  ├─ GSPO 优化 ✅        │
│  └─ 质量反馈 ✅         │              │  └─ 质量反馈 ✅         │
│                        │     →        │                        │
│  Encoder Adapter       │              │  Encoder Adapter       │
│  ├─ GSPO 优化 ❌        │              │  ├─ GSPO 优化 ✅        │
│  └─ 质量反馈 ❌         │              │  └─ 质量反馈 ✅         │
└────────────────────────┘              └────────────────────────┘
```

---

## 设计目标

1. **功能扩展**：GSPO 同时微调 Conv-LoRA MoE 和 Encoder Adapter
2. **向后兼容**：通过开关参数保持与现有实现的兼容性
3. **渐进式实现**：分三阶段逐步增加复杂度
4. **可验证性**：每阶段可独立测试和验证

---

## 实现阶段

### 阶段 1：统一优势加权（最小改动验证）

- 在现有 GSPO 训练循环中，将优势加权应用到整个模型
- 添加 Adapter 的质量历史追踪
- **开关参数**：`gspo_adapter_enabled`

### 阶段 2：Adapter Scale 自适应调节

- 根据质量反馈动态调整 Adapter 的 scale 参数
- 引入 Adapter 贡献分数机制
- **开关参数**：`gspo_adapter_scale_adaptation`

### 阶段 3：Adapter MoE 架构（可选高级功能）

- 将单一 Adapter 扩展为多变体 MoE 架构
- 引入 Adapter 门控网络和质量追踪
- **开关参数**：`adapter_moe_enabled`, `adapter_moe_num_experts`

---

## 文档索引

| 文档 | 描述 |
|------|------|
| [01_design_overview.md](./01_design_overview.md) | 整体设计概述与架构框图 |
| [02_gspo_math_formulation.md](./02_gspo_math_formulation.md) | GSPO 数学公式与强化学习原理 |
| [03_phase1_implementation.md](./03_phase1_implementation.md) | 阶段1实现细节 |
| [04_phase2_implementation.md](./04_phase2_implementation.md) | 阶段2实现细节 |
| [05_phase3_implementation.md](./05_phase3_implementation.md) | 阶段3实现细节 |
| [06_configuration_guide.md](./06_configuration_guide.md) | 配置参数说明 |
| [07_experiments.md](./07_experiments.md) | 实验设计与预期结果 |

---

## 快速开始

### 阶段1（推荐首先尝试）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --gspo_adapter_enable \
    --adapter_enable \
    --output_dir outputs/gspo_adapter_phase1
```

### 阶段2

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --adapter_enable \
    --output_dir outputs/gspo_adapter_phase2
```

---

*最后更新: 2024-12-20*


