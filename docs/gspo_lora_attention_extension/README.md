# GSPO 扩展到 Decoder LoRA on Attention 的设计与实现

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

本项目将 GSPO（Group Sequence Policy Optimization）强化学习框架进一步扩展，使其能够同时优化：

1. ✅ **Conv-LoRA MoE**（已完成）：Vision Encoder 中的空间适应
2. ✅ **Encoder Adapter**（已完成）：Vision Encoder 中的通道适应
3. 🆕 **Decoder LoRA on Attention**（本次新增）：Mask Decoder 中的注意力适应

### 当前状态 vs 目标状态

```
当前状态：                              目标状态：
┌────────────────────────────────┐      ┌────────────────────────────────┐
│  Vision Encoder                │      │  Vision Encoder                │
│  ├─ Conv-LoRA MoE: GSPO ✅     │      │  ├─ Conv-LoRA MoE: GSPO ✅     │
│  └─ Encoder Adapter: GSPO ✅   │      │  └─ Encoder Adapter: GSPO ✅   │
│                                │  →   │                                │
│  Mask Decoder                  │      │  Mask Decoder                  │
│  ├─ LoRA on Attention: 标准 ❌ │      │  ├─ LoRA on Attention: GSPO ✅ │
│  └─ (其他组件冻结)              │      │  └─ (其他组件冻结)              │
└────────────────────────────────┘      └────────────────────────────────┘
```

---

## 设计目标

1. **功能扩展**：GSPO 同时微调 Encoder（Conv-LoRA + Adapter）和 Decoder（LoRA on Attention）
2. **向后兼容**：通过开关参数保持与现有实现的兼容性
3. **统一框架**：使用相同的质量反馈机制
4. **端到端优化**：编码器和解码器协同进化

---

## 实现阶段

### 阶段 1：Scaling 自适应调节（推荐）

- 为 LoRALinear 添加质量历史追踪
- 根据质量反馈动态调整 LoRA 的 scaling 因子
- **开关参数**：`gspo_lora_attention_enabled`

### 阶段 2：分层质量追踪（可选）

- 分别追踪 Self-Attention、Cross-Attention 的质量贡献
- 对不同类型的注意力层应用差异化调节
- **开关参数**：`gspo_lora_layerwise_tracking`

---

## 文档索引

| 文档 | 描述 |
|------|------|
| [01_design_overview.md](./01_design_overview.md) | 整体设计概述与架构框图 |
| [02_gspo_integration.md](./02_gspo_integration.md) | GSPO 集成方案与数学公式 |
| [03_implementation.md](./03_implementation.md) | 实现细节与代码修改 |
| [04_configuration_guide.md](./04_configuration_guide.md) | 配置参数说明 |

---

## 快速开始

### 完整混合架构（GSPO 统一微调所有 PEFT 模块）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --decoder_attn_lora_enable \
    --gspo_lora_attention_enable \
    --output_dir outputs/gspo_full_hybrid
```

---

*最后更新: 2024-12-20*

