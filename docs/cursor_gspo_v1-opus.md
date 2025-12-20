# GSPO强化学习微调部分
_Exported on 12/20/2025 at 12:40:47 PST from Cursor (2.2.43)_

---

**User**

从@docs/cursor_gspo_adapter_convlora_followingups-251130.md 和@docs/cursor_gspo_adapter_convlora-251129.md 以及 examples/automm/Conv-LoRA下的代码了解之前的上下文。然后研究并回答：在当前的模型和训练方案设计中，被GSPO强化学习微调的是哪一部分？

---

**Cursor**

让我继续阅读更多相关代码来了解 GSPO 强化学习微调的具体部分：

让我查看 GSPO 与训练流程的集成代码：

根据我对代码和文档的深入分析，现在可以详细回答你的问题了。

---

## 📌 GSPO 强化学习微调的部分

在当前的模型和训练方案设计中，**GSPO（Group Sequence Policy Optimization）强化学习微调的是 Conv-LoRA 中的 MoE（Mixture-of-Experts）门控网络**，具体来说是以下组件：

### 🎯 被 GSPO 微调的核心模块

**1. `MoEGate` 中的门控参数**

```802:839:multimodal/src/autogluon/multimodal/models/adaptation_layers.py
class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        # ...
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
```

GSPO 通过以下方式影响 `w_gate` 和 `w_noise` 的学习：
- **质量偏置（Quality Bias）**：根据专家历史表现，为门控 logits 添加偏置
- **优势加权梯度**：高质量预测获得更大的损失权重，从而增强对应专家选择路径的梯度

**2. `ConvLoRALinear` 中的 MoE-Conv 专家**

```693:718:multimodal/src/autogluon/multimodal/models/adaptation_layers.py
if r > 0:
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
    
    # MoE-Conv experts with Noisy-TopK gating
    self.lora_moe_gating = MoEGate(
        M=conv_lora_expert_num, 
        d=self.r, 
        K=topk,
        gspo_enabled=gspo_enabled,
        gspo_group_size=gspo_group_size,
        gspo_quality_momentum=gspo_quality_momentum
    )
    self.lora_moe_experts = nn.ModuleList([
        nn.Sequential(nn.Conv2d(r, r, 3, 1, 1), nn.GELU())
        for _ in range(conv_lora_expert_num)
    ])
```

---

### 🔄 GSPO 的工作机制

根据 `gspo_trainer.py` 的实现：

```186:320:examples/automm/Conv-LoRA/gspo_trainer.py
def gspo_group_training_step(self, images, masks_gt, forward_fn, loss_fn):
    # 1. 生成 G 个预测（每次前向传播因 dropout/噪声不同，选择不同专家组合）
    for g in range(G):
        pred_masks, moe_loss, selected_experts = forward_fn(images)
        quality = self.compute_segmentation_quality(pred_masks, masks_gt)
        # ...
    
    # 2. 计算组级优势函数（GSPO 核心）
    advantages = quality_tensor - baseline
    
    # 3. 优势加权损失
    weight = torch.sigmoid(advantage * temperature)
    weighted_seg_loss = seg_loss * weight
    
    # 4. 对比损失（拉近高质量预测，推远低质量）
    contrastive_loss = self.compute_group_contrastive_loss(predictions, quality_scores)
```

---

### 📊 整体架构中 GSPO 的作用域

```text
SAM 模型整体结构：
┌─────────────────────────────────────────────────────────────────┐
│                     SAM Vision Encoder (ViT)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Attention Layer (q/k/v projections)                      │  │
│  │    └─ ConvLoRALinear ← 包含 MoE-Conv + MoEGate            │  │
│  │                        ↑                                   │  │
│  │                  ⭐ GSPO 微调的部分 ⭐                      │  │
│  │       - 门控网络 w_gate, w_noise (可学习参数)             │  │
│  │       - 各卷积专家 experts[i] (可学习参数)                │  │
│  │       - expert_quality_history (运行时反馈缓冲区)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MLP + Adapter (可选)                                      │  │
│  │    └─ Encoder-Adapter (标准监督学习，非 GSPO)             │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     SAM Mask Decoder                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Two-Way Transformer + MLP                                 │  │
│  │    └─ Decoder-Adapter (可选，标准监督学习，非 GSPO)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 📝 总结

| 模块 | 是否被 GSPO 微调 | 微调方式 |
|------|------------------|----------|
| **MoEGate (`w_gate`, `w_noise`)** | ✅ 是 | 质量偏置 + 优势加权梯度 |
| **MoE-Conv 专家 (`lora_moe_experts`)** | ✅ 是 | 优势加权梯度 |
| **LoRA 低秩矩阵 (`lora_A`, `lora_B`)** | ✅ 是 | 标准梯度（通过 GSPO 加权） |
| **Encoder-Adapter** | ❌ 否 | 标准监督学习 |
| **Decoder-Adapter** | ❌ 否 | 标准监督学习 |
| **SAM 主干参数** | ❌ 否 | 冻结 |

**核心洞察**：GSPO 本质上是用强化学习的思想（组级采样、优势函数、策略优化）来**增强 MoE 门控网络的专家选择策略**，使其能够：
1. 根据历史表现倾向于选择"高质量"专家
2. 通过对比损失学习更有区分度的专家组合
3. 在探索（尝试新专家）和利用（使用已知好专家）之间取得平衡

```python
class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        # ...
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
```

```python
if r > 0:
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
    
    # MoE-Conv experts with Noisy-TopK gating
    self.lora_moe_gating = MoEGate(
        M=conv_lora_expert_num, 
        d=self.r, 
        K=topk,
        gspo_enabled=gspo_enabled,
        gspo_group_size=gspo_group_size,
        gspo_quality_momentum=gspo_quality_momentum
    )
    self.lora_moe_experts = nn.ModuleList([
        nn.Sequential(nn.Conv2d(r, r, 3, 1, 1), nn.GELU())
        for _ in range(conv_lora_expert_num)
    ])
```

```python
def gspo_group_training_step(self, images, masks_gt, forward_fn, loss_fn):
    # 1. 生成 G 个预测（每次前向传播因 dropout/噪声不同，选择不同专家组合）
    for g in range(G):
        pred_masks, moe_loss, selected_experts = forward_fn(images)
        quality = self.compute_segmentation_quality(pred_masks, masks_gt)
        # ...
    
    # 2. 计算组级优势函数（GSPO 核心）
    advantages = quality_tensor - baseline
    
    # 3. 优势加权损失
    weight = torch.sigmoid(advantage * temperature)
    weighted_seg_loss = seg_loss * weight
    
    # 4. 对比损失（拉近高质量预测，推远低质量）
    contrastive_loss = self.compute_group_contrastive_loss(predictions, quality_scores)
```

```plaintext
SAM 模型整体结构：
┌─────────────────────────────────────────────────────────────────┐
│                     SAM Vision Encoder (ViT)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Attention Layer (q/k/v projections)                      │  │
│  │    └─ ConvLoRALinear ← 包含 MoE-Conv + MoEGate            │  │
│  │                        ↑                                   │  │
│  │                  ⭐ GSPO 微调的部分 ⭐                      │  │
│  │       - 门控网络 w_gate, w_noise (可学习参数)             │  │
│  │       - 各卷积专家 experts[i] (可学习参数)                │  │
│  │       - expert_quality_history (运行时反馈缓冲区)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MLP + Adapter (可选)                                      │  │
│  │    └─ Encoder-Adapter (标准监督学习，非 GSPO)             │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     SAM Mask Decoder                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Two-Way Transformer + MLP                                 │  │
│  │    └─ Decoder-Adapter (可选，标准监督学习，非 GSPO)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

```plaintext

---

### 🔄 GSPO 的工作机制

根据 `gspo_trainer.py` 的实现：

```186:320:examples/automm/Conv-LoRA/gspo_trainer.py
def gspo_group_training_step(self, images, masks_gt, forward_fn, loss_fn):
    # 1. 生成 G 个预测（每次前向传播因 dropout/噪声不同，选择不同专家组合）
    for g in range(G):
        pred_masks, moe_loss, selected_experts = forward_fn(images)
        quality = self.compute_segmentation_quality(pred_masks, masks_gt)
        # ...
    
    # 2. 计算组级优势函数（GSPO 核心）
    advantages = quality_tensor - baseline
    
    # 3. 优势加权损失
    weight = torch.sigmoid(advantage * temperature)
    weighted_seg_loss = seg_loss * weight
    
    # 4. 对比损失（拉近高质量预测，推远低质量）
    contrastive_loss = self.compute_group_contrastive_loss(predictions, quality_scores)
```

```plaintext
SAM 模型整体结构：
┌─────────────────────────────────────────────────────────────────┐
│                     SAM Vision Encoder (ViT)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Attention Layer (q/k/v projections)                      │  │
│  │    └─ ConvLoRALinear ← 包含 MoE-Conv + MoEGate            │  │
│  │                        ↑                                   │  │
│  │                  ⭐ GSPO 微调的部分 ⭐                      │  │
│  │       - 门控网络 w_gate, w_noise (可学习参数)             │  │
│  │       - 各卷积专家 experts[i] (可学习参数)                │  │
│  │       - expert_quality_history (运行时反馈缓冲区)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MLP + Adapter (可选)                                      │  │
│  │    └─ Encoder-Adapter (标准监督学习，非 GSPO)             │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     SAM Mask Decoder                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Two-Way Transformer + MLP                                 │  │
│  │    └─ Decoder-Adapter (可选，标准监督学习，非 GSPO)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

```python
class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        # ...
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
```

```python
if r > 0:
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
    
    # MoE-Conv experts with Noisy-TopK gating
    self.lora_moe_gating = MoEGate(
        M=conv_lora_expert_num, 
        d=self.r, 
        K=topk,
        gspo_enabled=gspo_enabled,
        gspo_group_size=gspo_group_size,
        gspo_quality_momentum=gspo_quality_momentum
    )
    self.lora_moe_experts = nn.ModuleList([
        nn.Sequential(nn.Conv2d(r, r, 3, 1, 1), nn.GELU())
        for _ in range(conv_lora_expert_num)
    ])
```

```python
def gspo_group_training_step(self, images, masks_gt, forward_fn, loss_fn):
    # 1. 生成 G 个预测（每次前向传播因 dropout/噪声不同，选择不同专家组合）
    for g in range(G):
        pred_masks, moe_loss, selected_experts = forward_fn(images)
        quality = self.compute_segmentation_quality(pred_masks, masks_gt)
        # ...
    
    # 2. 计算组级优势函数（GSPO 核心）
    advantages = quality_tensor - baseline
    
    # 3. 优势加权损失
    weight = torch.sigmoid(advantage * temperature)
    weighted_seg_loss = seg_loss * weight
    
    # 4. 对比损失（拉近高质量预测，推远低质量）
    contrastive_loss = self.compute_group_contrastive_loss(predictions, quality_scores)
```

```plaintext
SAM 模型整体结构：
┌─────────────────────────────────────────────────────────────────┐
│                     SAM Vision Encoder (ViT)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Attention Layer (q/k/v projections)                      │  │
│  │    └─ ConvLoRALinear ← 包含 MoE-Conv + MoEGate            │  │
│  │                        ↑                                   │  │
│  │                  ⭐ GSPO 微调的部分 ⭐                      │  │
│  │       - 门控网络 w_gate, w_noise (可学习参数)             │  │
│  │       - 各卷积专家 experts[i] (可学习参数)                │  │
│  │       - expert_quality_history (运行时反馈缓冲区)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MLP + Adapter (可选)                                      │  │
│  │    └─ Encoder-Adapter (标准监督学习，非 GSPO)             │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     SAM Mask Decoder                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Two-Way Transformer + MLP                                 │  │
│  │    └─ Decoder-Adapter (可选，标准监督学习，非 GSPO)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

**User**

本项目中加入了一类FC Adapter，它是加在哪里的？

---

**Cursor**

根据我之前阅读的文档和代码，本项目中的 **FC Adapter（全连接 Adapter）** 加在以下位置：

---

## 📍 FC Adapter 的位置

### 1️⃣ **Encoder-Adapter**（Vision Encoder 中）

**位置**：SAM Vision Encoder 的**每个 SamVisionLayer 的 MLP 之后**

**数量**：32 个（每层 1 个）

**插入方式**：与 MLP **并行**，然后三路残差相加

```text
SamVisionLayer (×32 层):
┌─────────────────────────────────────────┐
│  LayerNorm1                             │
│  Attention + Conv-LoRA                  │ ← 空间优化
│  Residual                               │
│  LayerNorm2                             │
│  ┌─────────────────────────────────┐    │
│  │  MLP (原始)                      │    │
│  │        +                         │    │
│  │  Encoder-Adapter ← 这里！        │    │ ← 通道优化
│  └─────────────────────────────────┘    │
│  hidden_states = residual + mlp_out + adapter_out  │
└─────────────────────────────────────────┘
```

**代码位置**：`modeling_sam_for_conv_lora.py` 第 984-988 行

```python
if self.adapter is not None:
    adapter_out = self.adapter(hidden_states)
    hidden_states = residual + mlp_out + adapter_out  # 三路相加
else:
    hidden_states = residual + mlp_out
```

---

### 2️⃣ **Decoder-Adapter**（Mask Decoder 中，可选）

**位置**：SAM Mask Decoder 的 **Two-Way Transformer Block 的 MLP 之后**

**数量**：2 个（Two-Way Transformer 有 2 个 block）

**插入方式**：同样与 MLP **并行**

```text
SamTwoWayAttentionBlock (queries 路径):
┌─────────────────────────────────────────┐
│  Self-Attn (+PE)                        │
│  Cross-Attn token→image                 │
│  LayerNorm2                             │
│  ┌─────────────────────────────────┐    │
│  │  MLP(queries)                    │    │
│  │        +                         │    │
│  │  Decoder-Adapter ← 这里！        │    │
│  └─────────────────────────────────┘    │
│  queries = queries + mlp_out + adapter_out │
│  LayerNorm3                             │
│  Cross-Attn image→token                 │
└─────────────────────────────────────────┘
```

---

## 🔧 Adapter 的结构

两种 Adapter 都使用相同的 **瓶颈架构（Bottleneck）**：

```python
class AdapterLayer(nn.Module):
    # 结构：Linear(C → r) → GELU → Linear(r → C)
    def __init__(self, in_features, adapter_dim):
        self.down_proj = nn.Linear(in_features, adapter_dim)  # 降维
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)    # 升维（零初始化）
```

**参数**：
- Encoder-Adapter：`C=1280, r=64`（约 165K 参数/层）
- Decoder-Adapter：`C=256, r=64`（约 33K 参数/层）

---

## 📊 总结表

| Adapter 类型 | 位置 | 数量 | 配置参数 |
|-------------|------|------|---------|
| **Encoder-Adapter** | Vision Encoder 每层 MLP 后 | 32 | `--adapter_enable`, `--adapter_dim` |
| **Decoder-Adapter** | Mask Decoder Two-Way Block MLP 后 | 2 | `--decoder_adapter_enable`, `--decoder_adapter_dim` |

---

## 🎯 关键点

1. **与 Conv-LoRA 分离**：FC Adapter 在 MLP 路径，Conv-LoRA 在 Attention 路径
2. **并行集成**：Adapter 与 MLP 并行，不是串行
3. **外部残差**：`output = residual + mlp_out + adapter_out`
4. **零初始化**：`up_proj` 权重零初始化，保证初始时 adapter 输出 ≈ 0

```plaintext
SamVisionLayer (×32 层):
┌─────────────────────────────────────────┐
│  LayerNorm1                             │
│  Attention + Conv-LoRA                  │ ← 空间优化
│  Residual                               │
│  LayerNorm2                             │
│  ┌─────────────────────────────────┐    │
│  │  MLP (原始)                      │    │
│  │        +                         │    │
│  │  Encoder-Adapter ← 这里！        │    │ ← 通道优化
│  └─────────────────────────────────┘    │
│  hidden_states = residual + mlp_out + adapter_out  │
└─────────────────────────────────────────┘
```

```python
if self.adapter is not None:
    adapter_out = self.adapter(hidden_states)
    hidden_states = residual + mlp_out + adapter_out  # 三路相加
else:
    hidden_states = residual + mlp_out
```

```plaintext
SamTwoWayAttentionBlock (queries 路径):
┌─────────────────────────────────────────┐
│  Self-Attn (+PE)                        │
│  Cross-Attn token→image                 │
│  LayerNorm2                             │
│  ┌─────────────────────────────────┐    │
│  │  MLP(queries)                    │    │
│  │        +                         │    │
│  │  Decoder-Adapter ← 这里！        │    │
│  └─────────────────────────────────┘    │
│  queries = queries + mlp_out + adapter_out │
│  LayerNorm3                             │
│  Cross-Attn image→token                 │
└─────────────────────────────────────────┘
```

```python
class AdapterLayer(nn.Module):
    # 结构：Linear(C → r) → GELU → Linear(r → C)
    def __init__(self, in_features, adapter_dim):
        self.down_proj = nn.Linear(in_features, adapter_dim)  # 降维
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)    # 升维（零初始化）
```

---

**User**

请研究如何使GSPO强化学习不仅仅用来微调 Conv-LoRA的MoE 还同时微调 FC Adapter？可能带来的好处是什么？

---

**Cursor**

这是一个非常有意思的研究问题！让我先分析当前 GSPO 的工作机制，然后探讨如何将其扩展到 FC Adapter。

根据代码分析，让我详细研究如何将 GSPO 扩展到 FC Adapter，以及可能带来的好处。

---

## 📊 当前状态分析

### GSPO 目前只作用于 Conv-LoRA MoE

```text
当前 GSPO 作用范围：
┌─────────────────────────────────────────────────────────┐
│  SamVisionLayer                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Attention Path                                       │ │
│  │   Conv-LoRA + MoEGate ← GSPO 优化 ✅                 │ │
│  │   - 质量反馈更新 expert_quality_history             │ │
│  │   - 优势加权梯度                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ MLP Path                                             │ │
│  │   FC Adapter ← 仅标准监督学习 ❌                      │ │
│  │   - 无质量反馈                                       │ │
│  │   - 无优势加权                                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 将 GSPO 扩展到 FC Adapter 的方案

### 方案 1：**Adapter Scale 的质量感知调节**

**核心思想**：利用 Adapter 已有的 `self.scale` 参数，根据质量反馈动态调整

```python
class AdapterLayerWithGSPO(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False, gspo_momentum=0.9):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        # GSPO 扩展
        self.gspo_enabled = gspo_enabled
        if gspo_enabled:
            self.register_buffer("quality_history", torch.tensor(0.5))  # 历史质量
            self.register_buffer("contribution_score", torch.tensor(1.0))  # 贡献分数
            self.gspo_momentum = gspo_momentum
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        
        # GSPO：根据历史质量调整有效缩放
        if self.gspo_enabled and self.training:
            effective_scale = self.scale * self.contribution_score
        else:
            effective_scale = self.scale
            
        return up * effective_scale
    
    def update_quality_feedback(self, quality_score):
        """GSPO 质量反馈"""
        if self.gspo_enabled:
            with torch.no_grad():
                # 动量更新质量历史
                self.quality_history = (
                    self.gspo_momentum * self.quality_history +
                    (1 - self.gspo_momentum) * quality_score
                )
                # 根据质量调整贡献分数
                self.contribution_score = torch.sigmoid(
                    (self.quality_history - 0.5) * 4  # 中心化并放大
                )
```

---

### 方案 2：**多 Adapter 变体的组级采样**（类似 MoE）

**核心思想**：为每个 Adapter 引入多个"变体"，通过 GSPO 选择最佳变体

```python
class AdapterMoE(nn.Module):
    """将 FC Adapter 扩展为 MoE 架构"""
    def __init__(self, in_features, adapter_dim, num_adapters=4, gspo_enabled=False):
        super().__init__()
        self.num_adapters = num_adapters
        self.gspo_enabled = gspo_enabled
        
        # 多个 Adapter 变体
        self.adapters = nn.ModuleList([
            AdapterLayer(in_features, adapter_dim)
            for _ in range(num_adapters)
        ])
        
        # 门控网络（类似 Conv-LoRA 的 MoEGate）
        self.gate = nn.Linear(in_features, num_adapters)
        
        # GSPO 质量追踪
        if gspo_enabled:
            self.register_buffer("adapter_quality_history", torch.zeros(num_adapters))
            self.register_buffer("adapter_usage_count", torch.zeros(num_adapters))
    
    def forward(self, x):
        # 计算门控权重
        gate_input = x.mean(dim=(1, 2)) if x.dim() == 4 else x.mean(dim=1)
        gate_logits = self.gate(gate_input)
        
        # GSPO：添加质量偏置
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            gate_logits = gate_logits + quality_bias
        
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 加权组合多个 Adapter 的输出
        output = 0
        for i, adapter in enumerate(self.adapters):
            adapter_out = adapter(x)
            output = output + gate_weights[..., i:i+1, None, None] * adapter_out
        
        return output
    
    def _compute_quality_bias(self):
        # 类似 MoEGate 的质量偏置计算
        quality_history = self.adapter_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        return quality_bias
    
    def update_quality_history(self, selected_adapters, quality_scores):
        """GSPO 反馈"""
        if self.gspo_enabled:
            # 类似 Conv-LoRA MoEGate 的更新逻辑
            ...
```

---

### 方案 3：**Adapter Dropout 的自适应调节**

**核心思想**：利用 Dropout 产生变体，通过 GSPO 学习最优 dropout 率

```python
class AdapterWithAdaptiveDropout(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        
        # GSPO：可学习的 dropout 率
        if gspo_enabled:
            self.dropout_rate = nn.Parameter(torch.tensor(0.1))
            self.register_buffer("quality_history", torch.tensor(0.5))
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        
        # GSPO：自适应 dropout
        if self.training and hasattr(self, 'dropout_rate'):
            # 根据质量历史调整 dropout
            effective_dropout = torch.clamp(
                self.dropout_rate * (1 - self.quality_history),
                0.0, 0.5
            )
            act = F.dropout(act, p=effective_dropout.item(), training=True)
        
        up = self.up_proj(act)
        return up
```

---

### 方案 4：**统一的 GSPO 优势加权**（最简单）

**核心思想**：在现有 GSPO 训练循环中，将优势加权应用到整个模型（包括 Adapter）

```python
# 修改 gspo_trainer.py 的 gspo_group_training_step
def gspo_group_training_step(self, ...):
    # ... 现有代码 ...
    
    # 3. GSPO 加权损失（现有）
    for g in range(G):
        advantage = advantages[g]
        weight = torch.sigmoid(advantage * self.advantage_temperature)
        
        # 加权分割损失 - 这已经会反向传播到 Adapter！
        weighted_seg_loss = (seg_loss * weight).mean()
        total_seg_loss += weighted_seg_loss
    
    # 🆕 4. Adapter 专属的质量反馈
    self._update_adapter_feedback(model, group_quality_scores)

def _update_adapter_feedback(self, model, quality_scores):
    """新增：更新所有 Adapter 的质量历史"""
    avg_quality = torch.stack(quality_scores).mean()
    
    for module in model.modules():
        if isinstance(module, AdapterLayer) and hasattr(module, 'quality_history'):
            module.update_quality_feedback(avg_quality)
```

---

## ✨ 可能带来的好处

### 1. **通道维度的自适应优化**

```text
当前：Adapter 对所有样本使用相同的特征变换强度
扩展后：根据样本质量动态调整 Adapter 贡献

好处：
- 高质量样本：Adapter 贡献更大 → 学习更多领域特征
- 低质量样本：Adapter 贡献减小 → 避免学习错误模式
```

### 2. **空间 + 通道的协同优化**

```text
当前：
  Conv-LoRA (空间) ← GSPO 优化
  Adapter (通道)  ← 独立优化

扩展后：
  Conv-LoRA (空间) ← GSPO 联合优化
  Adapter (通道)  ← GSPO 联合优化
  
好处：两个维度可以相互配合，找到最优组合
```

### 3. **端到端的质量驱动学习**

| 优化目标 | 当前 | 扩展后 |
|---------|------|--------|
| Conv-LoRA 专家选择 | 质量驱动 ✅ | 质量驱动 ✅ |
| Adapter 特征变换 | 损失驱动 ❌ | 质量驱动 ✅ |
| 整体分割质量 | 间接优化 | **直接优化** ✅ |

### 4. **更好的探索-利用平衡**

```text
Conv-LoRA MoE：8 个专家 → 探索空间大
FC Adapter：1 个模块 → 探索空间小

扩展方案 2（Adapter MoE）：
  - 4 个 Adapter 变体 → 增加探索空间
  - GSPO 选择最优变体 → 智能利用
```

### 5. **避免 Adapter 过拟合**

```text
问题：Adapter 容易过拟合到训练集的特定模式

GSPO 解决方案：
  - 质量反馈机制抑制低质量输出
  - 组级采样增加训练多样性
  - 自适应缩放避免过度贡献
```

---

## 📊 潜在收益估计

| 改进维度 | 预期收益 | 原因 |
|---------|---------|------|
| IoU/Dice 提升 | +1-3% | 双路径协同优化 |
| 训练稳定性 | 提升 | 质量反馈防止发散 |
| 泛化能力 | 提升 | 减少过拟合 |
| 边界精度 | 提升 | 空间+通道协同 |
| 参数效率 | 不变/略增 | 取决于具体方案 |
| 训练时间 | +10-30% | 额外的反馈计算 |

---

## 🎯 推荐实现路径

### **第一阶段：最小改动验证**（方案 4）
- 仅添加 Adapter 的质量历史追踪
- 利用现有 GSPO 的优势加权
- 验证是否有收益

### **第二阶段：Scale 自适应**（方案 1）
- 根据质量反馈调整 Adapter 的 scale 参数
- 较小的代码改动
- 较好的可解释性

### **第三阶段：Adapter MoE**（方案 2）
- 如果前两阶段有收益，考虑更复杂的设计
- 多 Adapter 变体 + 门控选择
- 与 Conv-LoRA MoE 形成完整的"双 MoE"架构

---

## 📝 总结

将 GSPO 扩展到 FC Adapter 是一个有价值的研究方向：

| 方面 | 评估 |
|------|------|
| **技术可行性** | ✅ 高（多种实现路径） |
| **预期收益** | ⭐⭐⭐ 中等到高 |
| **实现复杂度** | 方案 4 低 / 方案 2 中等 |
| **创新性** | ⭐⭐⭐⭐ 较高（双路径 GSPO 优化） |

如果你想实现这个扩展，建议从**方案 4（统一优势加权）**开始，这是最小改动且能快速验证效果的路径。如果需要我帮你实现具体代码，请切换到 Agent 模式。

```plaintext
当前 GSPO 作用范围：
┌─────────────────────────────────────────────────────────┐
│  SamVisionLayer                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Attention Path                                       │ │
│  │   Conv-LoRA + MoEGate ← GSPO 优化 ✅                 │ │
│  │   - 质量反馈更新 expert_quality_history             │ │
│  │   - 优势加权梯度                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ MLP Path                                             │ │
│  │   FC Adapter ← 仅标准监督学习 ❌                      │ │
│  │   - 无质量反馈                                       │ │
│  │   - 无优势加权                                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

```python
class AdapterLayerWithGSPO(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False, gspo_momentum=0.9):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        # GSPO 扩展
        self.gspo_enabled = gspo_enabled
        if gspo_enabled:
            self.register_buffer("quality_history", torch.tensor(0.5))  # 历史质量
            self.register_buffer("contribution_score", torch.tensor(1.0))  # 贡献分数
            self.gspo_momentum = gspo_momentum
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        
        # GSPO：根据历史质量调整有效缩放
        if self.gspo_enabled and self.training:
            effective_scale = self.scale * self.contribution_score
        else:
            effective_scale = self.scale
            
        return up * effective_scale
    
    def update_quality_feedback(self, quality_score):
        """GSPO 质量反馈"""
        if self.gspo_enabled:
            with torch.no_grad():
                # 动量更新质量历史
                self.quality_history = (
                    self.gspo_momentum * self.quality_history +
                    (1 - self.gspo_momentum) * quality_score
                )
                # 根据质量调整贡献分数
                self.contribution_score = torch.sigmoid(
                    (self.quality_history - 0.5) * 4  # 中心化并放大
                )
```

```python
class AdapterMoE(nn.Module):
    """将 FC Adapter 扩展为 MoE 架构"""
    def __init__(self, in_features, adapter_dim, num_adapters=4, gspo_enabled=False):
        super().__init__()
        self.num_adapters = num_adapters
        self.gspo_enabled = gspo_enabled
        
        # 多个 Adapter 变体
        self.adapters = nn.ModuleList([
            AdapterLayer(in_features, adapter_dim)
            for _ in range(num_adapters)
        ])
        
        # 门控网络（类似 Conv-LoRA 的 MoEGate）
        self.gate = nn.Linear(in_features, num_adapters)
        
        # GSPO 质量追踪
        if gspo_enabled:
            self.register_buffer("adapter_quality_history", torch.zeros(num_adapters))
            self.register_buffer("adapter_usage_count", torch.zeros(num_adapters))
    
    def forward(self, x):
        # 计算门控权重
        gate_input = x.mean(dim=(1, 2)) if x.dim() == 4 else x.mean(dim=1)
        gate_logits = self.gate(gate_input)
        
        # GSPO：添加质量偏置
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            gate_logits = gate_logits + quality_bias
        
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 加权组合多个 Adapter 的输出
        output = 0
        for i, adapter in enumerate(self.adapters):
            adapter_out = adapter(x)
            output = output + gate_weights[..., i:i+1, None, None] * adapter_out
        
        return output
    
    def _compute_quality_bias(self):
        # 类似 MoEGate 的质量偏置计算
        quality_history = self.adapter_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        return quality_bias
    
    def update_quality_history(self, selected_adapters, quality_scores):
        """GSPO 反馈"""
        if self.gspo_enabled:
            # 类似 Conv-LoRA MoEGate 的更新逻辑
            ...
```

```python
class AdapterWithAdaptiveDropout(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        
        # GSPO：可学习的 dropout 率
        if gspo_enabled:
            self.dropout_rate = nn.Parameter(torch.tensor(0.1))
            self.register_buffer("quality_history", torch.tensor(0.5))
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        
        # GSPO：自适应 dropout
        if self.training and hasattr(self, 'dropout_rate'):
            # 根据质量历史调整 dropout
            effective_dropout = torch.clamp(
                self.dropout_rate * (1 - self.quality_history),
                0.0, 0.5
            )
            act = F.dropout(act, p=effective_dropout.item(), training=True)
        
        up = self.up_proj(act)
        return up
```

```python
# 修改 gspo_trainer.py 的 gspo_group_training_step
def gspo_group_training_step(self, ...):
    # ... 现有代码 ...
    
    # 3. GSPO 加权损失（现有）
    for g in range(G):
        advantage = advantages[g]
        weight = torch.sigmoid(advantage * self.advantage_temperature)
        
        # 加权分割损失 - 这已经会反向传播到 Adapter！
        weighted_seg_loss = (seg_loss * weight).mean()
        total_seg_loss += weighted_seg_loss
    
    # 🆕 4. Adapter 专属的质量反馈
    self._update_adapter_feedback(model, group_quality_scores)

def _update_adapter_feedback(self, model, quality_scores):
    """新增：更新所有 Adapter 的质量历史"""
    avg_quality = torch.stack(quality_scores).mean()
    
    for module in model.modules():
        if isinstance(module, AdapterLayer) and hasattr(module, 'quality_history'):
            module.update_quality_feedback(avg_quality)
```

```plaintext
当前：Adapter 对所有样本使用相同的特征变换强度
扩展后：根据样本质量动态调整 Adapter 贡献

好处：
- 高质量样本：Adapter 贡献更大 → 学习更多领域特征
- 低质量样本：Adapter 贡献减小 → 避免学习错误模式
```

```plaintext
当前：
  Conv-LoRA (空间) ← GSPO 优化
  Adapter (通道)  ← 独立优化

扩展后：
  Conv-LoRA (空间) ← GSPO 联合优化
  Adapter (通道)  ← GSPO 联合优化
  
好处：两个维度可以相互配合，找到最优组合
```

```plaintext
Conv-LoRA MoE：8 个专家 → 探索空间大
FC Adapter：1 个模块 → 探索空间小

扩展方案 2（Adapter MoE）：
  - 4 个 Adapter 变体 → 增加探索空间
  - GSPO 选择最优变体 → 智能利用
```

```plaintext
问题：Adapter 容易过拟合到训练集的特定模式

GSPO 解决方案：
  - 质量反馈机制抑制低质量输出
  - 组级采样增加训练多样性
  - 自适应缩放避免过度贡献
```

```plaintext
当前 GSPO 作用范围：
┌─────────────────────────────────────────────────────────┐
│  SamVisionLayer                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Attention Path                                       │ │
│  │   Conv-LoRA + MoEGate ← GSPO 优化 ✅                 │ │
│  │   - 质量反馈更新 expert_quality_history             │ │
│  │   - 优势加权梯度                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ MLP Path                                             │ │
│  │   FC Adapter ← 仅标准监督学习 ❌                      │ │
│  │   - 无质量反馈                                       │ │
│  │   - 无优势加权                                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

```python
class AdapterLayerWithGSPO(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False, gspo_momentum=0.9):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        # GSPO 扩展
        self.gspo_enabled = gspo_enabled
        if gspo_enabled:
            self.register_buffer("quality_history", torch.tensor(0.5))  # 历史质量
            self.register_buffer("contribution_score", torch.tensor(1.0))  # 贡献分数
            self.gspo_momentum = gspo_momentum
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        
        # GSPO：根据历史质量调整有效缩放
        if self.gspo_enabled and self.training:
            effective_scale = self.scale * self.contribution_score
        else:
            effective_scale = self.scale
            
        return up * effective_scale
    
    def update_quality_feedback(self, quality_score):
        """GSPO 质量反馈"""
        if self.gspo_enabled:
            with torch.no_grad():
                # 动量更新质量历史
                self.quality_history = (
                    self.gspo_momentum * self.quality_history +
                    (1 - self.gspo_momentum) * quality_score
                )
                # 根据质量调整贡献分数
                self.contribution_score = torch.sigmoid(
                    (self.quality_history - 0.5) * 4  # 中心化并放大
                )
```

```python
class AdapterMoE(nn.Module):
    """将 FC Adapter 扩展为 MoE 架构"""
    def __init__(self, in_features, adapter_dim, num_adapters=4, gspo_enabled=False):
        super().__init__()
        self.num_adapters = num_adapters
        self.gspo_enabled = gspo_enabled
        
        # 多个 Adapter 变体
        self.adapters = nn.ModuleList([
            AdapterLayer(in_features, adapter_dim)
            for _ in range(num_adapters)
        ])
        
        # 门控网络（类似 Conv-LoRA 的 MoEGate）
        self.gate = nn.Linear(in_features, num_adapters)
        
        # GSPO 质量追踪
        if gspo_enabled:
            self.register_buffer("adapter_quality_history", torch.zeros(num_adapters))
            self.register_buffer("adapter_usage_count", torch.zeros(num_adapters))
    
    def forward(self, x):
        # 计算门控权重
        gate_input = x.mean(dim=(1, 2)) if x.dim() == 4 else x.mean(dim=1)
        gate_logits = self.gate(gate_input)
        
        # GSPO：添加质量偏置
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            gate_logits = gate_logits + quality_bias
        
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 加权组合多个 Adapter 的输出
        output = 0
        for i, adapter in enumerate(self.adapters):
            adapter_out = adapter(x)
            output = output + gate_weights[..., i:i+1, None, None] * adapter_out
        
        return output
    
    def _compute_quality_bias(self):
        # 类似 MoEGate 的质量偏置计算
        quality_history = self.adapter_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        return quality_bias
    
    def update_quality_history(self, selected_adapters, quality_scores):
        """GSPO 反馈"""
        if self.gspo_enabled:
            # 类似 Conv-LoRA MoEGate 的更新逻辑
            ...
```

```python
class AdapterWithAdaptiveDropout(nn.Module):
    def __init__(self, in_features, adapter_dim, gspo_enabled=False):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        
        # GSPO：可学习的 dropout 率
        if gspo_enabled:
            self.dropout_rate = nn.Parameter(torch.tensor(0.1))
            self.register_buffer("quality_history", torch.tensor(0.5))
    
    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        
        # GSPO：自适应 dropout
        if self.training and hasattr(self, 'dropout_rate'):
            # 根据质量历史调整 dropout
            effective_dropout = torch.clamp(
                self.dropout_rate * (1 - self.quality_history),
                0.0, 0.5
            )
            act = F.dropout(act, p=effective_dropout.item(), training=True)
        
        up = self.up_proj(act)
        return up
```

```python
# 修改 gspo_trainer.py 的 gspo_group_training_step
def gspo_group_training_step(self, ...):
    # ... 现有代码 ...
    
    # 3. GSPO 加权损失（现有）
    for g in range(G):
        advantage = advantages[g]
        weight = torch.sigmoid(advantage * self.advantage_temperature)
        
        # 加权分割损失 - 这已经会反向传播到 Adapter！
        weighted_seg_loss = (seg_loss * weight).mean()
        total_seg_loss += weighted_seg_loss
    
    # 🆕 4. Adapter 专属的质量反馈
    self._update_adapter_feedback(model, group_quality_scores)

def _update_adapter_feedback(self, model, quality_scores):
    """新增：更新所有 Adapter 的质量历史"""
    avg_quality = torch.stack(quality_scores).mean()
    
    for module in model.modules():
        if isinstance(module, AdapterLayer) and hasattr(module, 'quality_history'):
            module.update_quality_feedback(avg_quality)
```

```plaintext
当前：Adapter 对所有样本使用相同的特征变换强度
扩展后：根据样本质量动态调整 Adapter 贡献

好处：
- 高质量样本：Adapter 贡献更大 → 学习更多领域特征
- 低质量样本：Adapter 贡献减小 → 避免学习错误模式
```

```plaintext
当前：
  Conv-LoRA (空间) ← GSPO 优化
  Adapter (通道)  ← 独立优化

扩展后：
  Conv-LoRA (空间) ← GSPO 联合优化
  Adapter (通道)  ← GSPO 联合优化
  
好处：两个维度可以相互配合，找到最优组合
```

```plaintext
Conv-LoRA MoE：8 个专家 → 探索空间大
FC Adapter：1 个模块 → 探索空间小

扩展方案 2（Adapter MoE）：
  - 4 个 Adapter 变体 → 增加探索空间
  - GSPO 选择最优变体 → 智能利用
```

```plaintext
问题：Adapter 容易过拟合到训练集的特定模式

GSPO 解决方案：
  - 质量反馈机制抑制低质量输出
  - 组级采样增加训练多样性
  - 自适应缩放避免过度贡献
```

---

**User**

好的，我想要GSPO可以同时微调Conv-LoRA的MoE以及Vision Encoder 每层 MLP 后的Encoder Adapter（暂时不要包括decoder adapter)，请按照你说的上述路线设计并实现各阶段方案，并注意要通过加入可选开关参数使其兼容已有实现，即当不开启相应选项时，GSPO仍然只用于微调Conv-LoRA。为避免文档太多太乱，请在docs目录下创建子目录专门用于存放为此次设计而撰写的各种设计和说明文档，文件和目录名请用英文，内容用中文。设计文档应包含模型设计框图和必要的公式，清晰地说明强化学习是用在哪里、如何应用的（即在你的实际实现里，损失/奖励函数的形式是什么、其如何与GSPO的公式相对应），等等。

---

**Cursor**

我将按照三阶段路线设计并实现 GSPO 扩展到 Encoder Adapter 的方案。让我开始：

```markdown
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

```

```markdown
# 整体设计概述与架构框图

## 1. 系统架构图

### 1.1 扩展后的 GSPO 训练系统

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GSPO-Extended Training System                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐          ┌──────────────────┐                          │
│  │  Input Batch    │          │  Ground Truth    │                          │
│  │  Images [B,C,H,W│──────┐   │  Masks [B,H,W]   │─────────┐                │
│  └─────────────────┘      │   └──────────────────┘         │                │
│                           │                                 │                │
│                           ▼                                 ▼                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              Extended GSPOConvLoRATrainer                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Group Sampling (G predictions/image)                          │  │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  For g in 1..G:                                          │  │   │  │
│  │  │  │    - Forward through SAM Vision Encoder                  │  │   │  │
│  │  │  │    - Conv-LoRA MoE 专家选择 (空间维度)                    │  │   │  │
│  │  │  │    - Encoder Adapter 特征变换 (通道维度) ← 🆕             │  │   │  │
│  │  │  │    - Get prediction & quality score                      │  │   │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Advantage Function (GSPO 核心)                                │  │   │
│  │  │  advantage[g] = quality[g] - baseline                          │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Weighted Loss (优势加权损失)                                   │  │   │
│  │  │  loss = Σ sigmoid(advantage) * seg_loss                        │  │   │
│  │  │       + contrastive_loss                                       │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Quality Feedback (质量反馈) ← 扩展部分                         │  │   │
│  │  │  ┌─────────────────────────┐  ┌─────────────────────────────┐  │  │   │
│  │  │  │ MoEGate 反馈 (已有)     │  │ Adapter 反馈 (🆕 新增)       │  │  │   │
│  │  │  │ expert_quality_history  │  │ adapter_quality_history     │  │  │   │
│  │  │  │ expert_usage_count      │  │ adapter_contribution_score  │  │  │   │
│  │  │  └─────────────────────────┘  └─────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                           │                                                  │
│                           ▼                                                  │
│                    ┌──────────────┐                                         │
│                    │  Backprop &  │                                         │
│                    │  Optimize    │                                         │
│                    └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 单个 SamVisionLayer 中的 GSPO 作用点

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SamVisionLayer (×32 层)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: hidden_states [B, H, W, C=1280]                                     │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Attention Path (空间路径)                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  LayerNorm1                                                      │ │   │
│  │  │       │                                                          │ │   │
│  │  │       ▼                                                          │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐│ │   │
│  │  │  │  ConvLoRALinear (Q/K/V 投影)                                 ││ │   │
│  │  │  │  ┌───────────────────────────────────────────────────────┐  ││ │   │
│  │  │  │  │  MoEGate + Conv Experts                               │  ││ │   │
│  │  │  │  │  ┌─────────────────────────────────────────────────┐  │  ││ │   │
│  │  │  │  │  │  ⭐ GSPO 优化点 1：专家选择                       │  │  ││ │   │
│  │  │  │  │  │  - expert_quality_history 质量追踪               │  │  ││ │   │
│  │  │  │  │  │  - quality_bias 偏置调节                         │  │  ││ │   │
│  │  │  │  │  │  - 优势加权梯度更新                               │  │  ││ │   │
│  │  │  │  │  └─────────────────────────────────────────────────┘  │  ││ │   │
│  │  │  │  └───────────────────────────────────────────────────────┘  ││ │   │
│  │  │  └─────────────────────────────────────────────────────────────┘│ │   │
│  │  │       │                                                          │ │   │
│  │  │       ▼                                                          │ │   │
│  │  │  Multi-Head Attention                                            │ │   │
│  │  │       │                                                          │ │   │
│  │  │       ▼                                                          │ │   │
│  │  │  Residual: hidden_states = residual + attn_out                   │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MLP Path (通道路径)                                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  LayerNorm2                                                      │ │   │
│  │  │       │                                                          │ │   │
│  │  │       ├──────────────────┬──────────────────────────────────────┐│ │   │
│  │  │       ▼                  ▼                                      ││ │   │
│  │  │  ┌─────────┐      ┌─────────────────────────────────────────┐  ││ │   │
│  │  │  │   MLP   │      │  Encoder Adapter                        │  ││ │   │
│  │  │  │ C→4C→C  │      │  ┌───────────────────────────────────┐  │  ││ │   │
│  │  │  └────┬────┘      │  │  ⭐ GSPO 优化点 2 (🆕)              │  │  ││ │   │
│  │  │       │           │  │  - adapter_quality_history         │  │  ││ │   │
│  │  │       │           │  │  - contribution_score 贡献分数     │  │  ││ │   │
│  │  │       │           │  │  - scale 自适应调节                 │  │  ││ │   │
│  │  │       │           │  │  - 优势加权梯度更新                 │  │  ││ │   │
│  │  │       │           │  └───────────────────────────────────┘  │  ││ │   │
│  │  │       │           │  Linear(1280→64) → GELU → Linear(64→1280)│  ││ │   │
│  │  │       │           └────────────┬────────────────────────────┘  ││ │   │
│  │  │       │                        │                                ││ │   │
│  │  │       ▼                        ▼                                ││ │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐││ │   │
│  │  │  │  三路残差相加                                                │││ │   │
│  │  │  │  hidden_states = residual + mlp_out + adapter_out           │││ │   │
│  │  │  └─────────────────────────────────────────────────────────────┘││ │   │
│  │  └─────────────────────────────────────────────────────────────────┘│ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  Output: hidden_states [B, H, W, C=1280]                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流与信号流

### 2.1 前向传播数据流

```
Input Image [B, 3, 1024, 1024]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  SAM Vision Encoder                                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Patch Embedding → [B, 64, 64, 1280]                   │  │
│  │       │                                                 │  │
│  │       ▼                                                 │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │  SamVisionLayer ×32                              │   │  │
│  │  │    - Attention + Conv-LoRA (空间)                │   │  │
│  │  │    - MLP + Adapter (通道)                        │   │  │
│  │  │    ↓                                             │   │  │
│  │  │  [B, 64, 64, 1280]                               │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │       │                                                 │  │
│  │       ▼                                                 │  │
│  │  Neck → [B, 256, 64, 64]                               │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  SAM Mask Decoder                                             │
│  → Segmentation Mask [B, 1, 256, 256]                        │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 GSPO 信号流

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GSPO 信号流                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │ 前向传播 ×G │  (G = group_size, 默认 4)                              │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  质量评估                                                         │   │
│  │  quality[g] = IoU(pred[g], gt) + boundary_bonus + smooth_penalty│   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  优势函数计算                                                     │   │
│  │  baseline = mean(quality[1..G])                                  │   │
│  │  advantage[g] = quality[g] - baseline                            │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│                   ┌───────────────┴───────────────┐                    │
│                   ▼                               ▼                    │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│  │  Conv-LoRA MoE 反馈          │  │  Adapter 反馈 (🆕)            │   │
│  │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │   │
│  │  │ expert_quality_history │  │  │  │ adapter_quality_history│  │   │
│  │  │ += momentum * Δquality │  │  │  │ += momentum * Δquality │  │   │
│  │  └────────────────────────┘  │  │  └────────────────────────┘  │   │
│  │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │   │
│  │  │ quality_bias =         │  │  │  │ contribution_score =   │  │   │
│  │  │   f(quality_history)   │  │  │  │   f(quality_history)   │  │   │
│  │  └────────────────────────┘  │  │  └────────────────────────┘  │   │
│  └──────────────────────────────┘  └──────────────────────────────┘   │
│                   │                               │                    │
│                   └───────────────┬───────────────┘                    │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  优势加权损失                                                     │   │
│  │  weight[g] = sigmoid(advantage[g] * temperature)                 │   │
│  │  total_loss = Σ(weight[g] * seg_loss[g]) / G                     │   │
│  │             + contrastive_loss                                   │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  反向传播                                                         │   │
│  │  - 优势加权梯度同时更新 Conv-LoRA 和 Adapter                      │   │
│  │  - 高质量预测获得更强梯度                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         模块依赖图                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  run_semantic_segmentation.py                                           │
│         │                                                                │
│         │ 创建                                                           │
│         ▼                                                                │
│  MultiModalPredictor                                                    │
│         │                                                                │
│         │ 使用                                                           │
│         ▼                                                                │
│  SemanticSegmentationLitModule                                          │
│         │                                                                │
│         │ 包含                                                           │
│         ├──────────────────────────────────┐                            │
│         ▼                                  ▼                            │
│  SAMForSemanticSegmentation         GSPOConvLoRATrainer (扩展)          │
│         │                                  │                             │
│         │ 包含                             │ 调用                        │
│         ▼                                  ▼                             │
│  ┌──────────────────┐              ┌──────────────────────────┐         │
│  │ SamVisionLayer   │              │ gspo_group_training_step │         │
│  │   ├─ ConvLoRA    │◄─────────────│   ├─ 质量评估            │         │
│  │   │   └─ MoEGate │              │   ├─ 优势计算            │         │
│  │   └─ Adapter ←───│──────────────│   ├─ 加权损失            │         │
│  │       (扩展)     │              │   └─ 质量反馈 (扩展)     │         │
│  └──────────────────┘              └──────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 兼容性设计

### 4.1 开关参数层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      参数开关层次结构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 0: 基础功能                                                       │
│  ├─ --adapter_enable           # 启用 Encoder Adapter (已有)            │
│  └─ --gspo_enable              # 启用 GSPO 训练 (已有)                  │
│                                                                          │
│  Level 1: GSPO-Adapter 统一优化 (阶段1)                                  │
│  └─ --gspo_adapter_enable      # 启用 GSPO 对 Adapter 的优化 (🆕)       │
│                                                                          │
│  Level 2: Adapter Scale 自适应 (阶段2)                                   │
│  └─ --gspo_adapter_scale_adapt # 启用 Adapter scale 自适应调节 (🆕)     │
│                                                                          │
│  Level 3: Adapter MoE (阶段3, 可选)                                      │
│  ├─ --adapter_moe_enable       # 启用 Adapter MoE 架构 (🆕)             │
│  └─ --adapter_moe_num_experts  # Adapter 专家数量 (🆕)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 向后兼容性保证

```python
# 原有命令（不受影响）
python run_semantic_segmentation.py --gspo_enable --adapter_enable
# 结果：GSPO 仅优化 Conv-LoRA，Adapter 仅标准监督学习

# 新功能（显式启用）
python run_semantic_segmentation.py --gspo_enable --adapter_enable --gspo_adapter_enable
# 结果：GSPO 同时优化 Conv-LoRA 和 Adapter
```

---

*下一篇：[02_gspo_math_formulation.md](./02_gspo_math_formulation.md) - GSPO 数学公式与强化学习原理*

```

```markdown
# GSPO 数学公式与强化学习原理

## 1. GSPO 强化学习框架

### 1.1 基本概念映射

GSPO（Group Sequence Policy Optimization）将传统的监督学习问题转化为强化学习框架：

| 强化学习概念 | GSPO 中的对应 | 说明 |
|-------------|--------------|------|
| **状态 (State)** | 输入图像 x | 需要分割的医学图像 |
| **动作 (Action)** | 专家选择 / Adapter 变换 | MoE 选择哪些专家，Adapter 如何变换特征 |
| **策略 (Policy)** | 门控网络 / Adapter 参数 | 决定动作的参数化函数 |
| **奖励 (Reward)** | 分割质量 Q(ŷ, y) | IoU、Dice 等质量指标 |
| **轨迹 (Trajectory)** | 一次完整的前向传播 | 从输入到预测的完整路径 |

### 1.2 组级采样

不同于传统 RL 的单次采样，GSPO 采用组级采样策略：

$$
\mathcal{G} = \{(ŷ_1, a_1), (ŷ_2, a_2), ..., (ŷ_G, a_G)\}
$$

其中：
- $G$ = 组大小（默认 4）
- $ŷ_g$ = 第 g 个预测
- $a_g$ = 第 g 个动作（专家选择 + Adapter 变换）

---

## 2. 奖励函数设计

### 2.1 基础质量奖励

$$
R_{base}(ŷ, y) = \text{IoU}(ŷ, y) = \frac{|ŷ \cap y|}{|ŷ \cup y|}
$$

或使用 Dice：

$$
R_{base}(ŷ, y) = \text{Dice}(ŷ, y) = \frac{2|ŷ \cap y|}{|ŷ| + |y|}
$$

### 2.2 形状奖励（Shaped Reward）

为了提供更丰富的学习信号，我们引入形状奖励：

$$
R_{shaped}(ŷ, y) = R_{base} + w_b \cdot R_{boundary} - w_s \cdot L_{smooth} + w_t \cdot R_{thin}
$$

其中：

**边界奖励（Boundary Reward）**：
$$
R_{boundary} = 1 - \frac{1}{HW}\sum_{i,j}|\nabla ŷ_{i,j} - \nabla y_{i,j}|
$$

**平滑惩罚（Smoothness Penalty）**：
$$
L_{smooth} = \frac{1}{HW}\sum_{i,j}(\nabla^2 ŷ_{i,j})^2
$$

**细线奖励（Thin Structure Reward）**：
$$
R_{thin} = \text{Dice}(\text{skeleton}(ŷ), \text{skeleton}(y))
$$

**默认权重**：
- $w_b = 0.3$（边界权重）
- $w_s = 0.1$（平滑权重）
- $w_t = 0.05$（细线权重）

---

## 3. 优势函数

### 3.1 组内基线

GSPO 的核心思想是使用组内平均作为基线：

$$
b = \frac{1}{G}\sum_{g=1}^{G} R_{shaped}(ŷ_g, y)
$$

### 3.2 优势计算

$$
A_g = R_{shaped}(ŷ_g, y) - b
$$

**直观理解**：
- $A_g > 0$：预测 g 优于组内平均 → 增强其梯度
- $A_g < 0$：预测 g 劣于组内平均 → 抑制其梯度
- $A_g = 0$：预测 g 等于组内平均 → 标准梯度

---

## 4. 损失函数

### 4.1 优势加权分割损失

$$
\mathcal{L}_{seg} = \frac{1}{G}\sum_{g=1}^{G} \sigma(A_g \cdot \tau) \cdot L_{task}(ŷ_g, y)
$$

其中：
- $\sigma(\cdot)$ = sigmoid 函数
- $\tau$ = 温度系数（默认 5.0）
- $L_{task}$ = 任务损失（如 Structure Loss）

**温度的作用**：
- $\tau$ 大：权重分布更极化（高质量预测获得更多权重）
- $\tau$ 小：权重分布更均匀

### 4.2 对比损失

为了让模型学习区分高质量和低质量预测：

$$
\mathcal{L}_{contrast} = \text{MSE}(S_{feat}, S_{target})
$$

其中：
- $S_{feat}[i,j] = \cos(f_i, f_j)$：预测特征的相似度矩阵
- $S_{target}[i,j] = 1 - \frac{|Q_i - Q_j|}{\max(|Q_i - Q_j|)}$：质量差异决定的目标相似度

**直观理解**：质量相近的预测应有相似特征，质量差异大的预测应有不同特征。

### 4.3 总损失

$$
\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{moe} \cdot \mathcal{L}_{moe} + \lambda_{c} \cdot \mathcal{L}_{contrast}
$$

**默认超参数**：
- $\lambda_{moe} = 0.01$
- $\lambda_{c} = 0.1$

---

## 5. 质量反馈机制

### 5.1 Conv-LoRA MoE 反馈（已有）

对于每个专家 $e$，维护质量历史：

$$
H_e^{(t+1)} = \mu \cdot H_e^{(t)} + (1-\mu) \cdot \bar{Q}_e^{(t)}
$$

其中：
- $H_e^{(t)}$ = 专家 e 在时刻 t 的质量历史
- $\bar{Q}_e^{(t)}$ = 专家 e 在当前批次的平均质量
- $\mu = 0.9$（动量系数）

**质量偏置**：
$$
\text{bias}_e = \alpha \cdot \frac{H_e - \bar{H}}{\sigma_H + \epsilon} + \beta \cdot (1 - \frac{n_e}{\sum_i n_i})
$$

其中：
- 第一项：利用（高质量专家）
- 第二项：探索（使用少的专家）
- $\alpha = 0.8, \beta = 0.2$

### 5.2 Adapter 反馈（🆕 新增）

**阶段 1：质量历史追踪**

对于每层 Adapter，维护：

$$
H_{adapter}^{(t+1)} = \mu \cdot H_{adapter}^{(t)} + (1-\mu) \cdot \bar{Q}^{(t)}
$$

**阶段 2：贡献分数调节**

$$
\text{score} = \sigma\left(k \cdot (H_{adapter} - 0.5)\right)
$$

其中 $k = 4$ 是放大系数。

**Adapter 有效缩放**：

$$
\text{scale}_{eff} = \text{scale}_{learned} \cdot \text{score}
$$

$$
\text{output} = f_{adapter}(x) \cdot \text{scale}_{eff}
$$

**直观理解**：
- 质量历史高 → 贡献分数大 → Adapter 贡献更多
- 质量历史低 → 贡献分数小 → Adapter 贡献减少，避免学习错误模式

---

## 6. GSPO 与标准策略梯度的关系

### 6.1 标准策略梯度

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]
$$

### 6.2 GSPO 的简化形式

GSPO 可以看作策略梯度的一种简化和稳定化形式：

$$
\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{g=1}^{G} \sigma(A_g \cdot \tau) \cdot \nabla_\theta \mathcal{L}(ŷ_g, y)
$$

**关键简化**：
1. 使用优势函数代替原始奖励 → 减少方差
2. 使用 sigmoid 代替指数 → 防止梯度爆炸
3. 使用组内基线 → 自适应基准

---

## 7. 扩展到 Adapter 的理论基础

### 7.1 双策略优化视角

扩展后的系统可以看作同时优化两个策略：

$$
\pi_{total}(a|s) = \pi_{MoE}(a_{expert}|s) \cdot \pi_{Adapter}(a_{transform}|s)
$$

其中：
- $\pi_{MoE}$：专家选择策略（由 MoEGate 参数化）
- $\pi_{Adapter}$：特征变换策略（由 Adapter 参数化）

### 7.2 联合优化目标

$$
J(\theta_{MoE}, \theta_{Adapter}) = \mathbb{E}_{a \sim \pi_{total}}\left[R(s, a)\right]
$$

**梯度更新**：

$$
\nabla_{\theta_{MoE}} J = \mathbb{E}\left[A \cdot \nabla_{\theta_{MoE}} \log \pi_{MoE}\right]
$$

$$
\nabla_{\theta_{Adapter}} J = \mathbb{E}\left[A \cdot \nabla_{\theta_{Adapter}} \log \pi_{Adapter}\right]
$$

### 7.3 协同效应

```
空间优化 (Conv-LoRA MoE)          通道优化 (Adapter)
        │                               │
        │ 专家选择影响                   │ 特征变换影响
        │ 空间注意力模式                 │ 语义表示质量
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
                  分割预测质量
                        │
                        ▼
                   统一奖励信号
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
  更新 MoE 参数                   更新 Adapter 参数
```

---

## 8. 代码实现中的对应

### 8.1 奖励计算（gspo_trainer.py）

```python
# 基础质量
quality = self.compute_segmentation_quality(pred_masks, masks_gt)

# 形状奖励
shaped_quality = (
    quality
    + self.w_boundary * (1.0 - boundary_loss.detach())
    - self.w_smooth * smooth_loss.detach()
    + self.w_thin * thin_reward.detach()
)
```

### 8.2 优势函数（gspo_trainer.py）

```python
quality_tensor = torch.stack(group_quality_scores)  # [G, B]
baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
advantages = quality_tensor - baseline  # [G, B]
```

### 8.3 加权损失（gspo_trainer.py）

```python
weight = torch.sigmoid(advantage * self.advantage_temperature)
weighted_seg_loss = (seg_loss * weight).mean()
```

### 8.4 质量反馈（adaptation_layers.py, 扩展）

```python
# MoE 反馈（已有）
self.expert_quality_history[expert_id] = (
    self.gspo_quality_momentum * current_quality +
    (1 - self.gspo_quality_momentum) * quality
)

# Adapter 反馈（新增）
self.adapter_quality_history = (
    self.gspo_momentum * self.adapter_quality_history +
    (1 - self.gspo_momentum) * quality
)
self.contribution_score = torch.sigmoid(
    (self.adapter_quality_history - 0.5) * 4
)
```

---

*下一篇：[03_phase1_implementation.md](./03_phase1_implementation.md) - 阶段1实现细节*

```

现在让我修改 `AdapterLayer` 类，添加 GSPO 支持：

```worktree-python
# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def identity(x):
    return x


class AdapterLayer(nn.Module):
    """
    Standard Bottleneck Adapter Layer with optional GSPO enhancement.
    Structure: Linear(d, r) -> Activation -> Linear(r, d) -> Scale
    
    GSPO Enhancement (when gspo_enabled=True):
    - Tracks quality history for quality-aware adaptation
    - Supports contribution score for adaptive scaling (Phase 2)
    - Can be extended to MoE architecture (Phase 3)
    
    Parameters
    ----------
    in_features : int
        Input feature dimension (e.g., 1280 for SAM ViT-Huge)
    adapter_dim : int
        Bottleneck dimension (e.g., 64)
    scale : float
        Initial scale factor (default 1.0)
    dropout : float
        Dropout probability (default 0.0)
    gspo_enabled : bool
        Whether to enable GSPO quality tracking (default False)
    gspo_momentum : float
        Momentum for quality history updates (default 0.9)
    gspo_scale_adaptation : bool
        Whether to enable adaptive scale based on quality (Phase 2, default False)
    """
    def __init__(
        self, 
        in_features, 
        adapter_dim, 
        scale=1.0, 
        dropout=0.0,
        gspo_enabled=False,
        gspo_momentum=0.9,
        gspo_scale_adaptation=False,
    ):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # GSPO enhancement
        self.gspo_enabled = gspo_enabled
        self.gspo_momentum = gspo_momentum
        self.gspo_scale_adaptation = gspo_scale_adaptation
        
        if gspo_enabled:
            # Quality tracking buffers
            self.register_buffer("quality_history", torch.tensor(0.5))
            self.register_buffer("contribution_score", torch.tensor(1.0))
            self.register_buffer("update_count", torch.tensor(0))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

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
    
    def update_quality_feedback(self, quality_score: torch.Tensor):
        """
        GSPO: Update quality history based on actual performance.
        
        This provides the feedback loop for quality-aware adaptation.
        Called by GSPOConvLoRATrainer after each training step.
        
        Args:
            quality_score: Scalar tensor containing quality metric (e.g., IoU)
        """
        if not self.gspo_enabled:
            return
        
        with torch.no_grad():
            # Convert to scalar if needed
            if quality_score.numel() > 1:
                quality_score = quality_score.mean()
            quality_value = quality_score.item()
            
            # Momentum update of quality history
            self.quality_history = (
                self.gspo_momentum * self.quality_history +
                (1 - self.gspo_momentum) * quality_value
            )
            
            # Update contribution score (Phase 2)
            # Maps quality history [0, 1] to contribution score via sigmoid
            # quality_history = 0.5 -> score = 0.5 (neutral)
            # quality_history > 0.5 -> score > 0.5 (increase contribution)
            # quality_history < 0.5 -> score < 0.5 (decrease contribution)
            self.contribution_score = torch.sigmoid(
                (self.quality_history - 0.5) * 4.0  # Amplification factor
            )
            
            self.update_count += 1
    
    def get_gspo_stats(self):
        """Return GSPO statistics for logging."""
        if not self.gspo_enabled:
            return {}
        return {
            "quality_history": self.quality_history.item(),
            "contribution_score": self.contribution_score.item(),
            "update_count": self.update_count.item(),
        }



class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Paper alignment (Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model, ICLR 2024):
    - Sec. 3 (Method): Replace standard LoRA's linear residual with a spatial, convolutional expert branch (MoE-Conv).
    - Sec. 3.2 (MoE-Conv & Gating): Noisy-TopK gating over GAP features, plus load-balancing auxiliary loss.
    - Sec. 3.3 (Integration): Inject at attention projections (q/k/v) of the vision encoder.
    - Sec. 4 (Training): Return moe_loss for aggregation with task loss.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        gspo_enabled: bool = False,
        gspo_group_size: int = 3,
        gspo_quality_momentum: float = 0.9,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.gspo_enabled = gspo_enabled
        # Actual trainable parameters
        if r > 0:
            # Sec. 3: Low-rank LoRA factors around a frozen base linear
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
            # GSPO: Pass GSPO parameters to MoEGate
            topk = 1
            self.lora_moe_gating = MoEGate(
                M=conv_lora_expert_num, 
                d=self.r, 
                K=topk,
                gspo_enabled=gspo_enabled,
                gspo_group_size=gspo_group_size,
                gspo_quality_momentum=gspo_quality_momentum
            )
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        # Base frozen path + residual (LoRA principle)
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        moe_loss = 0
        selected_experts = None
        
        if self.r > 0:
            # 1) Project to rank-r LoRA space
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            # 2) Sec. 3.2: Compute Noisy-TopK gates over spatial features; also returns auxiliary moe_loss.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            
            # GSPO: Receive selected experts info
            if self.gspo_enabled:
                gates, moe_loss, selected_experts = self.lora_moe_gating(lora_res)
            else:
                gate_output = self.lora_moe_gating(lora_res)
                if len(gate_output) == 3:
                    gates, moe_loss, selected_experts = gate_output
                else:
                    gates, moe_loss = gate_output

            # Distribute data samples to experts.
            # 3) Route tokens to experts (SparseDispatcher)
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                # Per-expert resolution change to diversify receptive fields (paper Sec. 3.2)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            # 4) Combine expert outputs (weighted by gates) back to the batch layout
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            # 5) Project back to output dim with LoRA scale (final residual add)
            result += (lora_res @ self.lora_B.T) * self.scaling

        # Return 3 values if GSPO is enabled, 2 values otherwise for backward compatibility
        if self.gspo_enabled and selected_experts is not None:
            return result, moe_loss, selected_experts
        else:
            return result, moe_loss


class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
            gspo_enabled: whether to enable GSPO (Group Sequence Policy Optimization) enhancements.
            gspo_group_size: number of experts to select when GSPO is enabled (overrides K).
            gspo_quality_momentum: momentum coefficient for expert quality history updates.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.gspo_enabled = gspo_enabled
        self.gspo_group_size = gspo_group_size
        self.gspo_quality_momentum = gspo_quality_momentum
        
        # GSPO: Use group_size as K when enabled, otherwise use original K
        self.k = gspo_group_size if gspo_enabled else K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
        
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        """
        Paper alignment (Sec. 3.2):
        - GAP over spatial features -> linear logits (+ noise during training) -> TopK expert selection
        - Softmax over TopK logits yields gates; compute importance/load stats and CV^2-based load-balancing loss
        
        GSPO enhancement:
        - Quality-aware bias added to logits based on expert historical performance
        - Supports multi-expert selection (TopK > 1) for group-level optimization
        """
        batch_size = feats.shape[0]

        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        
        # GSPO: Add quality-aware bias to encourage high-performing experts
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            clean_logits = clean_logits + quality_bias.unsqueeze(0)
        
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss, top_k_indices

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def _compute_quality_bias(self):
        """
        GSPO: Compute quality-aware bias for expert selection.
        
        This method balances exploitation (favoring high-quality experts) and
        exploration (giving chances to less-used experts).
        
        Returns:
            bias: Tensor of shape [M] to be added to logits
        """
        if not hasattr(self, 'expert_quality_history'):
            return torch.zeros(self.M, device=self.w_gate.device)
        
        # Normalize quality history to compute exploitation bias
        quality_history = self.expert_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        
        # Compute exploration bonus for less-used experts
        usage_count = self.expert_usage_count
        if usage_count.sum() > 0:
            usage_normalized = usage_count / (usage_count.sum() + 1e-8)
            exploration_bonus = 1.0 - usage_normalized
        else:
            exploration_bonus = torch.ones_like(usage_count)
        
        # Combine: 80% exploitation + 20% exploration
        combined_bias = 0.8 * quality_bias + 0.2 * exploration_bonus
        
        return combined_bias
    
    def update_quality_history(self, selected_experts, quality_scores):
        """
        GSPO: Update expert quality history based on actual performance.
        
        This provides the feedback loop for quality-aware gating.
        
        Args:
            selected_experts: Tensor of shape [batch, k] containing selected expert indices
            quality_scores: Tensor of shape [batch] containing quality metrics (e.g., IoU, DICE)
        """
        if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
            return
        
        with torch.no_grad():
            for b in range(selected_experts.shape[0]):
                quality = quality_scores[b].item() if quality_scores[b].numel() == 1 else quality_scores[b].mean().item()
                for k in range(selected_experts.shape[1]):
                    expert_id = selected_experts[b, k].item()
                    # Momentum update
                    current_quality = self.expert_quality_history[expert_id]
                    self.expert_quality_history[expert_id] = (
                        self.gspo_quality_momentum * current_quality +
                        (1 - self.gspo_quality_momentum) * quality
                    )
                    self.expert_usage_count[expert_id] += 1


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

```

现在让我更新 `gspo_trainer.py` 来支持 Adapter 的质量反馈：

现在让我修改 `gspo_trainer.py` 来添加 Adapter 的质量反馈支持：

```worktree-python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA and Adapters

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support Encoder Adapters (Phase 1-3):
- Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
- Phase 2: Adapter scale adaptation (contribution score based on quality history)
- Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection AND adapter quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou'
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, C, H, W] or [B, H, W]
            metric: 'iou', 'dice', or 'both'
            
        Returns:
            quality_scores: Tensor of shape [B] containing quality scores
        """
        # Ensure masks are in the right format
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        
        # Binarize if needed
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        if metric == 'iou':
            return iou
        
        # DICE
        dice = (2 * intersection + 1e-6) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        
        if metric == 'dice':
            return dice
        
        # Both: average of IoU and DICE
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel probability maps for
        boundary/smoothness computation.
        """
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            # Use the maximum class probability as the foreground probability
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()


```

```worktree-python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA and Adapters

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support Encoder Adapters (Phase 1-3):
- Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
- Phase 2: Adapter scale adaptation (contribution score based on quality history)
- Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection AND adapter quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou'
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, C, H, W] or [B, H, W]
            metric: 'iou', 'dice', or 'both'
            
        Returns:
            quality_scores: Tensor of shape [B] containing quality scores
        """
        # Ensure masks are in the right format
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        
        # Binarize if needed
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        if metric == 'iou':
            return iou
        
        # DICE
        dice = (2 * intersection + 1e-6) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        
        if metric == 'dice':
            return dice
        
        # Both: average of IoU and DICE
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel probability maps for
        boundary/smoothness computation.
        """
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            # Use the maximum class probability as the foreground probability
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()


```

```worktree-python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA and Adapters

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support Encoder Adapters (Phase 1-3):
- Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
- Phase 2: Adapter scale adaptation (contribution score based on quality history)
- Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection AND adapter quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou'
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, C, H, W] or [B, H, W]
            metric: 'iou', 'dice', or 'both'
            
        Returns:
            quality_scores: Tensor of shape [B] containing quality scores
        """
        # Ensure masks are in the right format
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        
        # Binarize if needed
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        if metric == 'iou':
            return iou
        
        # DICE
        dice = (2 * intersection + 1e-6) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        
        if metric == 'dice':
            return dice
        
        # Both: average of IoU and DICE
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel probability maps for
        boundary/smoothness computation.
        """
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            # Use the maximum class probability as the foreground probability
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def update_adapter_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-Adapter Extension: Update adapter quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all AdapterLayer modules in the model.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing AdapterLayer modules
        """
        if not self.gspo_adapter_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Collect all AdapterLayer modules from the model
        adapters = []
        for name, module in model.named_modules():
            # Check if this is an AdapterLayer with GSPO enabled
            if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
                if module.gspo_enabled:
                    adapters.append((name, module))
        
        if not adapters:
            return
        
        # Update each adapter with quality feedback
        for name, adapter in adapters:
            adapter.update_quality_feedback(avg_quality)
        
        # Log adapter quality for analysis
        self.adapter_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_adapters': len(adapters),
        })
    
    def get_adapter_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all adapters in the model.
        
        Args:
            model: The model containing AdapterLayer modules
            
        Returns:
            Dictionary with adapter statistics
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_gspo_stats'):
                adapter_stats = module.get_gspo_stats()
                if adapter_stats:
                    stats[name] = adapter_stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
            'adapter_quality_log': self.adapter_quality_log,
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()
        self.adapter_quality_log.clear()


```

现在让我更新 `lit_semantic_seg.py` 中的 GSPO 训练步骤来调用 Adapter 反馈：

```worktree-python
import logging
from typing import Callable, Dict, Optional

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LOGITS, MOE_LOSS, SEMANTIC_MASK, WEIGHT
from ..models.utils import run_model
from .lit_module import LitModule
from .metrics.semantic_seg_metrics import Multiclass_IoU

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    
    GSPO enhancement:
    - Supports group-level training with quality-aware feedback
    - Integrates contrastive loss for better expert selection
    """
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
        # Training-time box prompt config (dict with keys: mode, p_no, p_gt, p_noisy, noise_frac, bbox_predictor)
        self.train_box_prompt_cfg = train_box_prompt_cfg or {
            "mode": "off",
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode
        self.train_bbox_predictor = train_bbox_predictor

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_labels = [mask_labels.to(per_output[LOGITS]) for mask_labels in kwargs["mask_labels"]]
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_mask_tokens, height, width
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_mask_tokens, num_classes
                    mask_labels=mask_labels,
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS],
                        target=label,
                    )
                    * weight
                )
            # MoE loss
            if MOE_LOSS in per_output:
                loss += per_output[MOE_LOSS]

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(metric, Multiclass_IoU):
            metric.update(kwargs["semantic_masks"], label)
        else:
            metric.update(logits.float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
            cfg = self.train_box_prompt_cfg or {}
            mode = cfg.get("mode", "off")
            box_mode = None
            if mode != "off":
                p_no = max(float(cfg.get("p_no", 0.0)), 0.0)
                p_gt = max(float(cfg.get("p_gt", 0.0)), 0.0)
                p_noisy = max(float(cfg.get("p_noisy", 0.0)), 0.0)
                noise_frac = float(cfg.get("noise_frac", 0.12))

                if mode == "mix":
                    total = p_no + p_gt + p_noisy
                    if total > 0:
                        p_no, p_gt, p_noisy = [x / total for x in (p_no, p_gt, p_noisy)]
                    r = torch.rand(1).item()
                    if r < p_no:
                        box_mode = "off"
                    elif r < p_no + p_gt:
                        box_mode = "gt"
                    else:
                        box_mode = "noisy"
                elif mode == "predict":
                    box_mode = "predict"
                else:
                    box_mode = mode

            def _compute_gt_boxes(mask: torch.Tensor) -> torch.Tensor:
                """mask: (B,H,W) or (B,1,H,W) int/long -> boxes (B,4) in pixel coords."""
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0, :, :]
                b, h, w = mask.shape
                boxes = torch.zeros((b, 4), device=mask.device, dtype=torch.float32)
                for i in range(b):
                    ys, xs = torch.nonzero(mask[i] > 0, as_tuple=True)
                    if xs.numel() == 0:
                        continue
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    boxes[i] = torch.tensor([x1, y1, x2, y2], device=mask.device, dtype=torch.float32)
                return boxes

            def _jitter_boxes(boxes: torch.Tensor, h: int, w: int, frac: float = 0.12) -> torch.Tensor:
                """Uniform jitter relative to box size."""
                jittered = boxes.clone()
                for i in range(jittered.shape[0]):
                    x1, y1, x2, y2 = jittered[i]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw = (x2 - x1).clamp(min=1.0)
                    bh = (y2 - y1).clamp(min=1.0)
                    dx = bw * frac
                    dy = bh * frac
                    rx1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    rx2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    x1n = torch.clamp(x1 + rx1, 0, w - 1)
                    y1n = torch.clamp(y1 + ry1, 0, h - 1)
                    x2n = torch.clamp(x2 + rx2, 0, w - 1)
                    y2n = torch.clamp(y2 + ry2, 0, h - 1)
                    jittered[i] = torch.stack(
                        [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
                    ).squeeze()
                return jittered

            def _predict_boxes_from_images(image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
                """Use BBoxPromptPredictor to predict boxes from image paths.
                
                Args:
                    image_paths: List of image file paths
                    target_h: Target image height (model input size)
                    target_w: Target image width (model input size)
                
                Returns:
                    boxes: (B, 4) tensor in pixel coords, scaled to target_h x target_w
                """
                import numpy as np
                from PIL import Image
                
                if self.train_bbox_predictor is None:
                    raise ValueError("train_bbox_predictor is required for mode='predict'")
                
                b = len(image_paths)
                boxes = torch.zeros((b, 4), device=label.device, dtype=torch.float32)
                
                for i, img_path in enumerate(image_paths):
                    if not img_path:
                        continue
                    try:
                        # predict() returns bbox in original image pixels: (x1, y1, x2, y2)
                        bbox_px = self.train_bbox_predictor.predict(img_path)
                        if bbox_px is None:
                            continue
                        
                        # Get original image size to scale bbox to target size
                        with Image.open(img_path) as img:
                            orig_w, orig_h = img.size
                        
                        # Scale bbox from original coords to target coords
                        scale_x = target_w / orig_w
                        scale_y = target_h / orig_h
                        x1 = bbox_px[0] * scale_x
                        y1 = bbox_px[1] * scale_y
                        x2 = bbox_px[2] * scale_x
                        y2 = bbox_px[3] * scale_y
                        
                        # Clamp to valid range
                        x1 = max(0, min(x1, target_w - 1))
                        y1 = max(0, min(y1, target_h - 1))
                        x2 = max(0, min(x2, target_w - 1))
                        y2 = max(0, min(y2, target_h - 1))
                        
                        # Ensure x1 < x2 and y1 < y2
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        boxes[i] = torch.tensor([x1, y1, x2, y2], device=label.device, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                        continue
                
                return boxes

            if box_mode in ["gt", "noisy"]:
                gt_boxes = _compute_gt_boxes(label)
                h, w = label.shape[-2], label.shape[-1]
                if box_mode == "noisy":
                    gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
                if not (gt_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
            elif box_mode == "predict":
                # Get image paths from batch
                image_path_key = self.model.image_path_key
                image_paths = batch.get(image_path_key, [])
                if image_paths:
                    h, w = label.shape[-2], label.shape[-1]
                    pred_boxes = _predict_boxes_from_images(image_paths, target_h=h, target_w=w)
                    if not (pred_boxes.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)
        # prepare_targets
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        if isinstance(self.loss_func, Mask2FormerLoss):
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
                semantic_masks=output[self.model.prefix][SEMANTIC_MASK],
            )
        else:
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
            )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
    
    def training_step(self, batch, batch_idx):
        """
        Per training step with GSPO enhancement.
        
        If GSPO is enabled and warmed up, use group-level optimization.
        Otherwise, fall back to standard training.
        
        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data
        batch_idx
            Index of mini-batch
            
        Returns
        -------
        Average loss of the mini-batch data
        """
        # Check if GSPO should be used
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        
        if use_gspo:
            # GSPO-enhanced training
            loss, metrics, selected_experts = self._gspo_training_step(batch)
            
            # Log GSPO-specific metrics
            for key, value in metrics.items():
                self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        else:
            # Standard training (same as parent class)
            output, loss = self._shared_step(batch)
            selected_experts = None
        
        # Handle manual optimization if needed
        if not self.automatic_optimization:
            if self.hparams.use_aug_optim:
                optimizer, aug_optimizer = self.optimizers()
            else:
                optimizer = self.optimizers()
                aug_optimizer = None

            lr_scheduler = self.lr_schedulers()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if aug_optimizer is not None:
                    aug_optimizer.step()
                    aug_optimizer.zero_grad()
        
        self.log("train_loss", loss)
        return loss
    
    def _gspo_training_step(self, batch):
        """
        GSPO group-level training step.
        
        This method generates multiple predictions per image and uses
        group-level advantage functions to weight the losses.
        
        Extended to support Encoder Adapter quality feedback when gspo_adapter_enabled=True.
        """
        images = batch[self.model.image_key] if hasattr(self.model, 'image_key') else batch['image']
        labels = batch[self.model.label_key]
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            logits = output[self.model.prefix][LOGITS]
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if hasattr(output[self.model.prefix], 'selected_experts'):
                selected_experts = output[self.model.prefix]['selected_experts']
            
            return logits, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                # Handle Mask2Former case
                return self.loss_func(predictions, targets)
            else:
                return self.loss_func(input=predictions, target=targets)
        
        # Run GSPO group training
        loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
            images=images,
            masks_gt=labels,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
        )
        
        # Update expert quality feedback (Conv-LoRA MoE)
        if selected_experts_groups:
            quality_scores = [
                self.gspo_trainer.compute_segmentation_quality(
                    pred, labels, self.gspo_trainer.quality_metric
                )
                for pred in [forward_fn(images)[0] for _ in range(self.gspo_trainer.group_size)]
            ]
            self.gspo_trainer.update_expert_feedback(
                selected_experts_groups, quality_scores, self.model
            )
            
            # Update Adapter quality feedback (GSPO-Adapter extension)
            if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
                self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
        
        return loss, metrics, selected_experts_groups

```

现在让我更新配置文件和训练脚本来支持新的 GSPO-Adapter 参数：

```worktree-yaml
optim:
  optim_type: "adamw"
  lr: 1.0e-4
  weight_decay: 0.001
  lr_choice: "layerwise_decay"
  lr_decay: 0.9
  lr_schedule: "cosine_decay"
  max_epochs: 20
  max_steps: -1
  warmup_steps: 0.1
  end_lr: 0
  lr_mult: 1 # multiply lr for downstream heads
  patience: 10
  val_check_interval: 0.5 # Check validation score a fraction of epoch if float and every n steps if integer with n < total step in epoch.
  check_val_every_n_epoch: 1 # Check validation score every n epochs.
  skip_final_val: False # Flag to skip the last validation
  gradient_clip_val: 1
  gradient_clip_algorithm: "norm"
  track_grad_norm: -1 # Whether to check gradient norm. We can set it to 2 to check for gradient norm.
  log_every_n_steps: 10
  label_smoothing: 0
  top_k: 3
  top_k_average_method:
    "greedy_soup" # We support averaging method described in https://arxiv.org/pdf/2203.05482.pdf.
    # Currently support "uniform_soup", "greedy_soup", and "best".
  peft: null # Can be 'bit_fit' (only finetune bias), 'norm_fit' (finetune the normalization terms + bias terms), lora (LoRA Adaptations only), lora_bias (LoRA Adaptation + bit_fit), lora_norm (LoRA Adaptation + norm_fit), or null
  lora:
    module_filter: null # Specify which module (if any) to adapt(e.g. ".*EncDecAttention|.*DenseReluDense"). Default consider all modules in a model (i.e. empty filter).
    filter: # Specify which layer in a module to adapt. Default fine-tune only query and value attention weights, recommended in https://arxiv.org/abs/2106.09685
      - "query"
      - "value"
      - "^q$" # The filter can also be a regex. We will use re.match(filter, name) to determine if LoRA should be applied.
      - "^v$"
      - "^k$"
      - "^o$"
    r: 8
    alpha: 8
    conv_lora_expert_num: 8 # default setting for Conv-LoRA
    gspo_enabled: False # Enable GSPO (Group Sequence Policy Optimization) enhancements
    gspo_group_size: 3 # Number of experts to select in GSPO mode (TopK)
    gspo_quality_momentum: 0.9 # Momentum coefficient for expert quality history updates
    gspo_warmup_epochs: 5 # Number of epochs before enabling full GSPO training
    gspo_contrastive_weight: 0.1 # Weight for contrastive loss in GSPO training
  loss_func:
    "auto" # The replaced loss for regression. Can only support loss function in torch.nn.
    # example
    # "BCEWithLogitsLoss" or "nn.BCEWithLogitsloss"
  focal_loss:
    alpha: null
    gamma: 2.0
    reduction: "mean"
  mask2former_loss:
    loss_cross_entropy_weight: 10.0
    loss_mask_weight: 5.0
    loss_dice_weight: 5.0
  extra_trainable_params: []
  cross_modal_align: null
  cross_modal_align_weight: 0
  automatic_optimization: True
  lemda:
    turn_on: False
    arch_type: "mlp_vae"
    z_dim: 8
    num_layers: 6
    kld_weight: 0.1
    mse_weight: 0.1
    adv_weight: 1.0e-4
    consist_weight: 0.01
    consist_threshold: 0.5
    lr: 1.0e-4
    optim_type: "adamw"
    weight_decay: 1.0e-5
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
    # GSPO-Adapter extension parameters
    adapter_enabled: False  # Enable GSPO quality feedback for Encoder Adapters
    adapter_momentum: 0.9   # Momentum for adapter quality history updates
    adapter_scale_adaptation: False  # Enable adaptive scale based on quality (Phase 2)

```

```worktree-python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
        })

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```worktree-python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
        })

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

现在让我更新 SAM 模型文件来支持 GSPO-Adapter 参数：

```worktree-python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        adapter_gspo_enabled: bool = False,
        adapter_gspo_momentum: float = 0.9,
        adapter_gspo_scale_adaptation: bool = False,
        decoder_attention_lora_r: int = 0,
        decoder_attention_lora_alpha: int = 1,
        decoder_attention_lora_dropout: float = 0.0,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        decoder_attention_lora_r
            LoRA rank for decoder attention layers. If 0 (default), LoRA is disabled.
            Recommended value: 8 for parameter-efficient fine-tuning.
        decoder_attention_lora_alpha
            LoRA scaling factor for decoder attention. Default: 1.
        decoder_attention_lora_dropout
            Dropout probability for LoRA in decoder attention. Default: 0.0.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.decoder_attention_lora_r = decoder_attention_lora_r
        self.decoder_attention_lora_alpha = decoder_attention_lora_alpha
        self.decoder_attention_lora_dropout = decoder_attention_lora_dropout

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
            for layer in self.model.vision_encoder.layers:
                # Manually create and assign adapter since config didn't have it at init time
                layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Load configuration and inject decoder attention LoRA parameters
            try:
                config = SamConfig.from_pretrained(checkpoint_name, local_files_only=True)
            except:
                config = SamConfig.from_pretrained(checkpoint_name)
            
            # Propagate decoder adapter settings into mask decoder config (if present)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            # Inject LoRA parameters into mask decoder config
            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, config=config, local_files_only=True)
                logger.info(f"Loaded model from local cache: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face: {checkpoint_name}")
                self.model = SamModel.from_pretrained(checkpoint_name, config=config)
            
            if self.decoder_attention_lora_r > 0:
                logger.info(f"Decoder Attention LoRA enabled: r={self.decoder_attention_lora_r}, "
                          f"alpha={self.decoder_attention_lora_alpha}, dropout={self.decoder_attention_lora_dropout}")
        else:
            config = SamConfig(name_or_path=checkpoint_name)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            self.model = SamModel(config)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    @property
    def box_key(self):
        return f"{self.prefix}_box"

    @property
    def image_path_key(self):
        return f"{self.prefix}_image_path"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

```worktree-python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        adapter_gspo_enabled: bool = False,
        adapter_gspo_momentum: float = 0.9,
        adapter_gspo_scale_adaptation: bool = False,
        decoder_attention_lora_r: int = 0,
        decoder_attention_lora_alpha: int = 1,
        decoder_attention_lora_dropout: float = 0.0,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        decoder_attention_lora_r
            LoRA rank for decoder attention layers. If 0 (default), LoRA is disabled.
            Recommended value: 8 for parameter-efficient fine-tuning.
        decoder_attention_lora_alpha
            LoRA scaling factor for decoder attention. Default: 1.
        decoder_attention_lora_dropout
            Dropout probability for LoRA in decoder attention. Default: 0.0.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.decoder_attention_lora_r = decoder_attention_lora_r
        self.decoder_attention_lora_alpha = decoder_attention_lora_alpha
        self.decoder_attention_lora_dropout = decoder_attention_lora_dropout

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.adapter_gspo_enabled = adapter_gspo_enabled
        self.adapter_gspo_momentum = adapter_gspo_momentum
        self.adapter_gspo_scale_adaptation = adapter_gspo_scale_adaptation
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            gspo_info = ""
            if self.adapter_gspo_enabled:
                gspo_info = f", GSPO enabled (momentum={self.adapter_gspo_momentum}"
                if self.adapter_gspo_scale_adaptation:
                    gspo_info += ", scale_adaptation=True"
                gspo_info += ")"
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder{gspo_info}")
            for layer in self.model.vision_encoder.layers:
                # Create adapter with GSPO parameters if enabled
                layer.adapter = AdapterLayer(
                    in_features=self.config.hidden_size, 
                    adapter_dim=self.adapter_dim,
                    gspo_enabled=self.adapter_gspo_enabled,
                    gspo_momentum=self.adapter_gspo_momentum,
                    gspo_scale_adaptation=self.adapter_gspo_scale_adaptation,
                )


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Load configuration and inject decoder attention LoRA parameters
            try:
                config = SamConfig.from_pretrained(checkpoint_name, local_files_only=True)
            except:
                config = SamConfig.from_pretrained(checkpoint_name)
            
            # Propagate decoder adapter settings into mask decoder config (if present)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            # Inject LoRA parameters into mask decoder config
            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, config=config, local_files_only=True)
                logger.info(f"Loaded model from local cache: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face: {checkpoint_name}")
                self.model = SamModel.from_pretrained(checkpoint_name, config=config)
            
            if self.decoder_attention_lora_r > 0:
                logger.info(f"Decoder Attention LoRA enabled: r={self.decoder_attention_lora_r}, "
                          f"alpha={self.decoder_attention_lora_alpha}, dropout={self.decoder_attention_lora_dropout}")
        else:
            config = SamConfig(name_or_path=checkpoint_name)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            self.model = SamModel(config)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    @property
    def box_key(self):
        return f"{self.prefix}_box"

    @property
    def image_path_key(self):
        return f"{self.prefix}_image_path"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

```worktree-yaml
model:
  names:
  categorical_mlp:
    hidden_size: 64
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    data_types:
      - "categorical"

  numerical_mlp:
    hidden_size: 128
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    token_dim: 8
    embedding_arch:
    data_types:
      - "numerical"
    merge: "concat"

  hf_text:
    checkpoint_name: "google/electra-base-discriminator"
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    use_fast: True  # Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  ner_text:
    checkpoint_name: "bert-base-cased"
    max_text_len: 512
    gradient_checkpointing: False
    low_cpu_mem_usage: False
    data_types:
      - "text_ner"
    tokenizer_name: "hf_auto"
    insert_sep: False
    text_segment_num: 2
    stochastic_chunk: False
    special_tags: 
      - X # CLS, SEP, and non-first tokens of a word will be labelled as X
      - O # Outside of a named entity

  document_transformer:
    checkpoint_name: "microsoft/layoutlmv3-base" # document foundation models
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "document"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]

  t_few:
    checkpoint_name: "t5-small" #"bigscience/T0_3B"
    gradient_checkpointing: False
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    length_norm: 1.0 # Normalizes length to adjust for length bias in target template
    unlikely_loss: 1.0 # Adds loss term that lowers probability of incorrect outputs
    mc_loss: 1.0 # Adds multiple choice cross entropy loss
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    text_segment_num: 2
    insert_sep: True
    low_cpu_mem_usage: False
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnititude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:

  timm_image:
    checkpoint_name: "swin_base_patch4_window7_224"
    mix_choice: "all_logits"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: null
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1

  mmdet_image:
    checkpoint_name: "yolov3_mobilenetv2_8xb24-320-300e_coco"
    config_file: ""
    data_types:
      - "image"
    max_img_num_per_col: 1
    output_bbox_format: "xyxy"  # now support xyxy or xywh, for bbox format details see https://keras.io/api/keras_cv/bounding_box/formats/
    frozen_layers: null
    coco_root: null

  mmocr_text_detection:
    checkpoint_name: "TextSnake"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  mmocr_text_recognition:
    checkpoint_name: "ABINet"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  clip:
    checkpoint_name: "openai/clip-vit-base-patch32"
    data_types:
      - "image"
      - "text"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "clip"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "clip"
    max_text_len: 77  # The maximum possible length.
    insert_sep: False
    text_segment_num: 1
    stochastic_chunk: False
    text_aug_detect_length: 10                     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0                      # scale randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  fusion_mlp:
    aux_loss_weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    dropout: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_ner:
    weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    drop_rate: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_transformer:
    aux_loss_weight:
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    adapt_in_features: "max"
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    normalization: "layer_norm"
    ffn_activation: "geglu"
    head_activation: "relu"
    data_types:
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"

  ft_transformer:
    data_types:
      - "categorical"
      - "numerical"
    embedding_arch:
      - 'linear'
    token_dim: 192
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    ffn_activation: "geglu"
    head_activation: "relu"
    normalization: "layer_norm"
    merge: "concat"
    requires_all_dtypes: False
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"
    pooling_mode: "cls"
    checkpoint_name: null

  sam:
    checkpoint_name: "facebook/sam-vit-huge"
    image_norm: "imagenet"
    data_types:
      - "semantic_segmentation_img"
    train_transforms:
      - "random_horizontal_flip"
    val_transforms: []
    img_transforms:
      - "resize_to_square"
    gt_transforms:
      - "resize_gt_to_square"
    max_img_num_per_col: 1
    frozen_layers: ["mask_decoder.iou_prediction_head", "prompt_encoder"]
    num_mask_tokens: 1
    ignore_label: 255
    adapter_enabled: False
    adapter_dim: 64
    adapter_gspo_enabled: False  # Enable GSPO quality feedback for Adapters
    adapter_gspo_momentum: 0.9   # Momentum for adapter quality history
    adapter_gspo_scale_adaptation: False  # Enable adaptive scale (Phase 2)
    decoder_attention_lora_r: 0
    decoder_attention_lora_alpha: 1
    decoder_attention_lora_dropout: 0.0
    decoder_adapter_enabled: False
    decoder_adapter_dim: 64

  meta_transformer:
    data_types:
      - "image"
      - "text"
      - "categorical"
      - "numerical"
    checkpoint_path: null
    model_version: "base"
    requires_all_dtypes: False
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1  # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:
    merge: "concat"

```

```markdown
# 阶段1实现：统一优势加权

## 1. 实现目标

**阶段1的核心思想**：以最小改动验证 GSPO 扩展到 Adapter 的可行性。

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段1：统一优势加权                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  改动范围：                                                   │
│  ✅ AdapterLayer 添加质量追踪 buffer                         │
│  ✅ GSPOConvLoRATrainer 添加 update_adapter_feedback        │
│  ✅ 训练脚本添加 --gspo_adapter_enable 开关                  │
│  ✅ 配置文件添加相应参数                                      │
│                                                              │
│  不改动：                                                     │
│  ❌ 前向传播逻辑（Adapter 输出计算方式不变）                   │
│  ❌ 损失函数（仍使用统一的优势加权损失）                       │
│  ❌ 现有 Conv-LoRA MoE 反馈机制                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 代码修改清单

### 2.1 AdapterLayer 类修改

**文件**: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`

**新增参数**:

```python
def __init__(
    self, 
    in_features, 
    adapter_dim, 
    scale=1.0, 
    dropout=0.0,
    gspo_enabled=False,        # 🆕 启用 GSPO 质量追踪
    gspo_momentum=0.9,         # 🆕 质量历史动量
    gspo_scale_adaptation=False,  # 🆕 阶段2功能，暂时不使用
):
```

**新增 buffer**（当 `gspo_enabled=True`）:

```python
if gspo_enabled:
    self.register_buffer("quality_history", torch.tensor(0.5))
    self.register_buffer("contribution_score", torch.tensor(1.0))
    self.register_buffer("update_count", torch.tensor(0))
```

**新增方法**:

```python
def update_quality_feedback(self, quality_score: torch.Tensor):
    """
    GSPO: 根据实际分割质量更新质量历史。
    
    由 GSPOConvLoRATrainer 在每个训练步骤后调用。
    """
    if not self.gspo_enabled:
        return
    
    with torch.no_grad():
        quality_value = quality_score.mean().item()
        
        # 动量更新质量历史
        self.quality_history = (
            self.gspo_momentum * self.quality_history +
            (1 - self.gspo_momentum) * quality_value
        )
        
        # 计算贡献分数（阶段2使用）
        self.contribution_score = torch.sigmoid(
            (self.quality_history - 0.5) * 4.0
        )
        
        self.update_count += 1

def get_gspo_stats(self):
    """返回 GSPO 统计信息用于日志记录。"""
    if not self.gspo_enabled:
        return {}
    return {
        "quality_history": self.quality_history.item(),
        "contribution_score": self.contribution_score.item(),
        "update_count": self.update_count.item(),
    }
```

### 2.2 GSPOConvLoRATrainer 类修改

**文件**: `examples/automm/Conv-LoRA/gspo_trainer.py`

**新增初始化参数**:

```python
def __init__(
    self,
    predictor,
    group_size: int = 4,
    # ... 现有参数 ...
    gspo_adapter_enabled: bool = False,  # 🆕
    gspo_adapter_momentum: float = 0.9,   # 🆕
):
    # ...
    self.gspo_adapter_enabled = gspo_adapter_enabled
    self.gspo_adapter_momentum = gspo_adapter_momentum
    self.adapter_quality_log = []  # 🆕 追踪日志
```

**新增方法**:

```python
def update_adapter_feedback(
    self,
    quality_scores: List[torch.Tensor],
    model
):
    """
    GSPO-Adapter: 根据 GSPO 反馈更新 Adapter 质量历史。
    """
    if not self.gspo_adapter_enabled:
        return
    
    # 计算组内平均质量
    avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
    
    # 收集所有启用 GSPO 的 AdapterLayer
    adapters = []
    for name, module in model.named_modules():
        if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
            if module.gspo_enabled:
                adapters.append((name, module))
    
    # 更新每个 Adapter 的质量反馈
    for name, adapter in adapters:
        adapter.update_quality_feedback(avg_quality)
    
    # 记录日志
    self.adapter_quality_log.append({
        'epoch': self.current_epoch,
        'avg_quality': avg_quality.item(),
        'num_adapters': len(adapters),
    })
```

### 2.3 lit_semantic_seg.py 修改

**文件**: `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`

**在 `_gspo_training_step` 末尾添加**:

```python
# Update expert quality feedback (Conv-LoRA MoE)
if selected_experts_groups:
    # ... 现有代码 ...
    
    # 🆕 Update Adapter quality feedback (GSPO-Adapter extension)
    if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
        self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
```

### 2.4 配置文件修改

**文件**: `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`

```yaml
gspo:
  # ... 现有参数 ...
  adapter_enabled: False       # 🆕 启用 GSPO-Adapter
  adapter_momentum: 0.9        # 🆕 Adapter 质量历史动量
  adapter_scale_adaptation: False  # 🆕 阶段2功能
```

**文件**: `multimodal/src/autogluon/multimodal/configs/model/default.yaml`

```yaml
sam:
  # ... 现有参数 ...
  adapter_gspo_enabled: False       # 🆕
  adapter_gspo_momentum: 0.9        # 🆕
  adapter_gspo_scale_adaptation: False  # 🆕
```

### 2.5 训练脚本修改

**文件**: `examples/automm/Conv-LoRA/run_semantic_segmentation.py`

**新增命令行参数**:

```python
parser.add_argument("--gspo_adapter_enable", action="store_true", 
                    help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                    help="Momentum for adapter quality history updates")
parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                    help="Enable adaptive scale based on quality history (Phase 2)")
```

**配置传递**:

```python
if args.gspo_adapter_enable:
    if not args.gspo_enable:
        print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
        args.gspo_enable = True
    print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
    hyperparameters.update({
        "optim.gspo.adapter_enabled": True,
        "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
    })
```

---

## 3. 使用方法

### 3.1 基本用法（阶段1）

```bash
cd examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --output_dir outputs/gspo_adapter_phase1
```

### 3.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_adapter_enable` | flag | False | 启用 GSPO-Adapter 质量追踪 |
| `--gspo_adapter_momentum` | float | 0.9 | 质量历史更新动量 |

### 3.3 向后兼容性

**原有命令（不受影响）**:

```bash
# 仅 Conv-LoRA + GSPO，不启用 Adapter
python3 run_semantic_segmentation.py --task isic2017 --gspo_enable

# Conv-LoRA + GSPO + Adapter（Adapter 无 GSPO 反馈）
python3 run_semantic_segmentation.py --task isic2017 --gspo_enable --adapter_enable
```

这些命令的行为与修改前完全一致。

---

## 4. 验证方法

### 4.1 功能验证

```python
# 在训练完成后，检查 Adapter 质量历史是否被更新
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor.load("outputs/gspo_adapter_phase1")
model = predictor._learner.model

# 遍历所有 Adapter，检查 GSPO 统计
for name, module in model.named_modules():
    if hasattr(module, 'get_gspo_stats'):
        stats = module.get_gspo_stats()
        if stats:
            print(f"{name}: {stats}")
```

### 4.2 预期输出

```
model.vision_encoder.layers.0.adapter: {'quality_history': 0.72, 'contribution_score': 0.81, 'update_count': 15000}
model.vision_encoder.layers.1.adapter: {'quality_history': 0.69, 'contribution_score': 0.77, 'update_count': 15000}
...
model.vision_encoder.layers.31.adapter: {'quality_history': 0.75, 'contribution_score': 0.84, 'update_count': 15000}
```

### 4.3 性能验证

比较以下配置的测试集性能：

1. **基线**: `--gspo_enable --adapter_enable`（Adapter 无 GSPO）
2. **阶段1**: `--gspo_enable --adapter_enable --gspo_adapter_enable`

预期：阶段1的性能应该与基线相近或略有提升（因为质量追踪在阶段1不直接影响前向传播）。

---

## 5. 工作原理

### 5.1 信息流

```
┌─────────────────────────────────────────────────────────────────┐
│                     阶段1 信息流                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 前向传播（×G 次）                                            │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  Image → Vision Encoder → Mask Decoder → Prediction      │ │
│     │           ↓                                              │ │
│     │    Conv-LoRA (空间) + Adapter (通道)                      │ │
│     │           ↓                                              │ │
│     │    [正常计算，无变化]                                      │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. 质量评估                                                     │
│     quality[g] = IoU(pred[g], gt)                                │
│                                                                  │
│  3. 优势计算                                                     │
│     advantage[g] = quality[g] - mean(quality)                    │
│                                                                  │
│  4. 加权损失（同时影响 Conv-LoRA 和 Adapter）                     │
│     loss = Σ sigmoid(advantage) * seg_loss                       │
│                                                                  │
│  5. 反馈更新                                                     │
│     ┌─────────────────────────┐  ┌─────────────────────────┐    │
│     │ Conv-LoRA MoE 反馈      │  │ Adapter 反馈 (🆕)       │    │
│     │ expert_quality_history  │  │ quality_history         │    │
│     │ += momentum * Δq        │  │ += momentum * avg_q     │    │
│     └─────────────────────────┘  └─────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 关键理解

**阶段1 的"统一优势加权"意味着**：

1. **相同的权重**：Conv-LoRA 和 Adapter 接收相同的优势加权梯度
2. **共享质量信号**：两者都根据最终分割质量被更新
3. **质量追踪独立**：虽然梯度共享，但质量历史分别追踪

这种设计的好处是：
- 最小改动，风险低
- 验证 GSPO 框架可以扩展到 Adapter
- 为阶段2的差异化优化打下基础

---

*下一篇：[04_phase2_implementation.md](./04_phase2_implementation.md) - 阶段2实现细节*

```

```markdown
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
    --output_dir outputs/gspo_adapter_phase2
```

### 4.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_adapter_scale_adaptation` | flag | False | 启用质量自适应缩放 |

**注意**：此参数需要同时启用 `--gspo_adapter_enable`。

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

当前默认 k=4，可以调整：

```python
self.contribution_score = torch.sigmoid(
    (self.quality_history - 0.5) * k  # k 可调
)
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
--gspo_adapter_momentum 0.95 --gspo_adapter_scale_adaptation
```

**激进配置**（自适应性优先）:
```bash
--gspo_adapter_momentum 0.85 --gspo_adapter_scale_adaptation
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

```

```markdown
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

```

```markdown
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
    checkpoint_name: "facebook/sam-vit-huge"
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

### 3.3 Conv-LoRA + GSPO + Adapter（无 GSPO-Adapter）

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

### 3.4 完整混合架构（阶段1）

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

### 3.5 完整混合架构（阶段2）

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

```

现在让我验证向后兼容性，确保不开启 GSPO-Adapter 时，代码行为与之前相同：

让我列出已创建的所有文件，并给你一个总结：

我已完成 GSPO 扩展到 Encoder Adapter 的设计与实现。让我为你总结：

---

## ✅ 完成总结

### 1. 文档目录结构

已在 `docs/gspo_adapter_extension/` 目录下创建了完整的设计文档：

| 文件 | 内容 |
|------|------|
| `README.md` | 项目概述、目标、快速开始 |
| `01_design_overview.md` | 整体架构框图、数据流、模块依赖 |
| `02_gspo_math_formulation.md` | **GSPO 数学公式、强化学习原理、损失函数详解** |
| `03_phase1_implementation.md` | 阶段1实现细节和代码修改清单 |
| `04_phase2_implementation.md` | 阶段2 Scale 自适应调节实现 |
| `05_phase3_implementation.md` | 阶段3 Adapter MoE 设计（未实现，仅设计） |
| `06_configuration_guide.md` | 完整配置参数说明 |

### 2. 代码修改

**已修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `adaptation_layers.py` | `AdapterLayer` 添加 GSPO 质量追踪 |
| `gspo_trainer.py` | 添加 `update_adapter_feedback` 方法 |
| `lit_semantic_seg.py` | 训练步骤中调用 Adapter 反馈 |
| `sam.py` | 传递 GSPO 参数到 Adapter |
| `run_semantic_segmentation.py` | 添加新命令行参数 |
| `optim/default.yaml` | 添加 GSPO-Adapter 配置 |
| `model/default.yaml` | 添加 Adapter GSPO 配置 |

### 3. 新增开关参数

```bash
--gspo_adapter_enable           # 阶段1：启用 GSPO-Adapter 质量追踪
--gspo_adapter_momentum         # 质量历史动量 (默认 0.9)
--gspo_adapter_scale_adaptation # 阶段2：启用 Scale 自适应
```

### 4. 向后兼容性

不开启 `--gspo_adapter_enable` 时，行为与之前完全一致：
- GSPO 仍只用于微调 Conv-LoRA
- Adapter 仍使用标准监督学习

### 5. 使用示例

**阶段1（统一优势加权）**：
```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --output_dir outputs/gspo_adapter_phase1
```

**阶段2（Scale 自适应）**：
```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --output_dir outputs/gspo_adapter_phase2
```

### 6. 关键设计说明

强化学习在 GSPO 中的应用（详见 `02_gspo_math_formulation.md`）：

1. **奖励函数**：`R = IoU(pred, gt)` + 边界奖励 - 平滑惩罚
2. **优势函数**：`A[g] = quality[g] - baseline`（组内相对质量）
3. **加权损失**：`loss = Σ sigmoid(A) × seg_loss`
4. **质量反馈**：通过动量更新质量历史，用于调节专家选择/Adapter 贡献

```shellscript
--gspo_adapter_enable           # 阶段1：启用 GSPO-Adapter 质量追踪
--gspo_adapter_momentum         # 质量历史动量 (默认 0.9)
--gspo_adapter_scale_adaptation # 阶段2：启用 Scale 自适应
```

```shellscript
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --output_dir outputs/gspo_adapter_phase1
```

```shellscript
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --gspo_enable \
    --adapter_enable \
    --gspo_adapter_enable \
    --gspo_adapter_scale_adaptation \
    --output_dir outputs/gspo_adapter_phase2
```

