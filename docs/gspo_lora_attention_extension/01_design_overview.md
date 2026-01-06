# 整体设计概述与架构框图

## 1. SAM 模型中的 PEFT 模块分布

### 1.1 完整 GSPO 混合架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAM + GSPO 完整混合架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Vision Encoder (ViT-Huge, 冻结)                      │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  SamVisionLayer ×32                                               │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  Attention Path                                              │ │  │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────┐│ │  │ │
│  │  │  │  │  ⭐ Conv-LoRA MoE (8 卷积专家)                           ││ │  │ │
│  │  │  │  │  - GSPO 质量反馈：优化空间专家选择                        ││ │  │ │
│  │  │  │  │  - 优化维度：空间 (H×W)                                  ││ │  │ │
│  │  │  │  └─────────────────────────────────────────────────────────┘│ │  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  MLP Path                                                    │ │  │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────┐│ │  │ │
│  │  │  │  │  ⭐ Encoder Adapter (瓶颈结构)                           ││ │  │ │
│  │  │  │  │  - GSPO 质量反馈：优化 Scale 自适应                      ││ │  │ │
│  │  │  │  │  - 优化维度：通道 (C)                                    ││ │  │ │
│  │  │  │  └─────────────────────────────────────────────────────────┘│ │  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  Output: image_embeddings [B, 256, 64, 64]                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Prompt Encoder (冻结)                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Mask Decoder (部分冻结)                              │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  SamTwoWayTransformer                                             │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  SamTwoWayAttentionBlock ×2                                  │ │  │ │
│  │  │  │                                                              │ │  │ │
│  │  │  │  ┌───────────────────────────────────────────────────────┐  │ │  │ │
│  │  │  │  │  Self-Attention                                        │  │ │  │ │
│  │  │  │  │  ┌─────────────────────────────────────────────────┐  │  │ │  │ │
│  │  │  │  │  │  ⭐ LoRA on Q/K/V (🆕 GSPO 微调)                 │  │  │ │  │ │
│  │  │  │  │  │  - GSPO 质量反馈：优化 Scaling 自适应            │  │  │ │  │ │
│  │  │  │  │  │  - 优化维度：注意力投影                          │  │  │ │  │ │
│  │  │  │  │  └─────────────────────────────────────────────────┘  │  │ │  │ │
│  │  │  │  └───────────────────────────────────────────────────────┘  │ │  │ │
│  │  │  │                                                              │ │  │ │
│  │  │  │  ┌───────────────────────────────────────────────────────┐  │ │  │ │
│  │  │  │  │  Cross-Attention (Token → Image)                       │  │ │  │ │
│  │  │  │  │  ┌─────────────────────────────────────────────────┐  │  │ │  │ │
│  │  │  │  │  │  ⭐ LoRA on Q/K/V (🆕 GSPO 微调)                 │  │  │ │  │ │
│  │  │  │  │  └─────────────────────────────────────────────────┘  │  │ │  │ │
│  │  │  │  └───────────────────────────────────────────────────────┘  │ │  │ │
│  │  │  │                                                              │ │  │ │
│  │  │  │  ┌───────────────────────────────────────────────────────┐  │ │  │ │
│  │  │  │  │  Cross-Attention (Image → Token)                       │  │ │  │ │
│  │  │  │  │  ┌─────────────────────────────────────────────────┐  │  │ │  │ │
│  │  │  │  │  │  ⭐ LoRA on Q/K/V (🆕 GSPO 微调)                 │  │  │ │  │ │
│  │  │  │  │  └─────────────────────────────────────────────────┘  │  │ │  │ │
│  │  │  │  └───────────────────────────────────────────────────────┘  │ │  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │  │  │  Final Attention                                             │ │  │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────┐│ │  │ │
│  │  │  │  │  ⭐ LoRA on Q/K/V (🆕 GSPO 微调)                         ││ │  │ │
│  │  │  │  └─────────────────────────────────────────────────────────┘│ │  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  Output: segmentation_mask [B, 1, H, W]                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 PEFT 模块统计

| 位置 | 模块 | GSPO 优化 | 参数量 | 优化维度 |
|------|------|-----------|--------|----------|
| Vision Encoder | Conv-LoRA MoE | ✅ | ~8M | 空间 (H×W) |
| Vision Encoder | Encoder Adapter | ✅ | ~5M | 通道 (C) |
| Mask Decoder | LoRA on Attention | 🆕 | ~86K | 注意力投影 |
| **总计** | - | - | **~13M** | - |

---

## 2. Decoder LoRA on Attention 的 GSPO 扩展设计

### 2.1 LoRA 数学基础

标准 LoRA 公式：

$$
y = W_0 \cdot x + (B \cdot A) \cdot x \cdot \text{scaling}
$$

其中：
- $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$：冻结的预训练权重
- $A \in \mathbb{R}^{r \times d_{in}}$：下投影矩阵（可训练）
- $B \in \mathbb{R}^{d_{out} \times r}$：上投影矩阵（可训练）
- $\text{scaling} = \frac{\alpha}{r}$：固定缩放因子

### 2.2 GSPO 扩展后的公式

GSPO 引入**动态缩放**：

$$
y = W_0 \cdot x + (B \cdot A) \cdot x \cdot \text{scaling} \cdot \text{contribution\_score}
$$

其中：

$$
\text{contribution\_score} = \sigma\left(k \cdot (H_{lora} - 0.5)\right)
$$

- $H_{lora}$：LoRA 层的质量历史（通过动量更新）
- $\sigma$：sigmoid 函数
- $k = 4$：放大系数

### 2.3 质量反馈机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LoRA on Attention GSPO 反馈机制                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 前向传播                                                                 │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Query  ←── LoRALinear(Q) ──→ Q_proj                             │    │
│     │  Key    ←── LoRALinear(K) ──→ K_proj                             │    │
│     │  Value  ←── LoRALinear(V) ──→ V_proj                             │    │
│     │                                                                   │    │
│     │  y = W_0·x + (B·A)·x · scaling · contribution_score              │    │
│     └──────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  2. 分割预测 & 质量评估                                                       │
│     quality = IoU(pred, gt)                                                 │
│                                      │                                       │
│                                      ▼                                       │
│  3. GSPO 反馈更新                                                            │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  H_lora = μ · H_lora + (1-μ) · quality                           │    │
│     │  contribution_score = σ((H_lora - 0.5) × 4)                      │    │
│     └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│  实现位置：Trainer 聚合组质量并调用 `update_lora_attention_feedback`，详见
│  [examples/automm/Conv-LoRA/gspo_trainer.py](examples/automm/Conv-LoRA/gspo_trainer.py#L504),
│  LoRA 层内部的更新实现见
│  [multimodal/src/autogluon/multimodal/models/adaptation_layers.py](multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L464-L472)
│                                      │                                       │
│                                      ▼                                       │
│  4. 下一次前向传播使用更新后的 contribution_score                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 2.4 Conv‑LoRA MoE 的 GSPO 反馈机制（详解）

背景与目标：Conv‑LoRA 将 LoRA 的低秩残差替换为一个空间 MoE（不同专家具有不同感受野 / 上采样比），GSPO 在此处目标是：通过组级采样与质量反馈偏好长期表现较好的专家，同时保留对冷门专家的探索。

流程图（简要）：

```
Forward (single candidate)
        │
        ▼
    Project -> rank-r features (spatial)
        │
        ▼
    MoEGate: GAP(features) @ W_gate  (+ quality_bias) + noise  -> TopK indices
        │                          │
        │                          ▼
        │                    selected_experts (per-sample)
        ▼                          │
Dispatch -> experts process spatial patches
        │
        ▼
Combine expert outputs -> reconstruct LoRA residual -> add to W0·x
        │
        ▼
Prediction -> quality (IoU)
        │
        ▼
Update: only the experts that were selected this forward receive quality feedback
```

关键公式（门控与质量偏置）：

1) 门控 logits（带质量偏置与噪声）

$$
	ext{clean\_logits} = \text{GAP}(F) \cdot W_{gate}
$$
$$
	ext{logits} = \text{clean\_logits} + \text{quality\_bias} + \epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma^2(F))
$$
Top‑K 选择后：
$$
	ext{gates} = \operatorname{softmax}(\text{top\_k}(\text{logits}))
$$

2) 质量偏置（实现中用于平衡 exploitation/exploration）：

$$
	ext{quality\_bias} = 0.8 \cdot \frac{Q_{hist} - \mu_Q}{\sigma_Q + \epsilon} + 0.2 \cdot (1 - \text{usage\_norm})
$$

3) 专家质量历史更新（仅对 selected_experts）：

对于每个被选中的 expert i（样本 b）：

$$
H^{(t+1)}_{expert,i} = \mu_e \cdot H^{(t)}_{expert,i} + (1-\mu_e) \cdot q_{b}
$$

并更新使用计数： usage_count[i] += 1。

说明与实现要点：
- 采样差异来源包括：`MoEGate` 的 noisy‑TopK（训练时注入噪声）、`ConvLoRALinear` 中的 dropout 与 experts 的多尺度上采样策略，这些使同一输入在 G 次前向中被路由到不同专家，生成 G 个 joint 候选。
- 代码中对专家的 credit assignment 是逐候选更新的：只有被选中的专家通过 [MoEGate.update_quality_history](multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L1168-L1185) 收到该候选的质量反馈（对应 `ConvLoRALinear` 返回的 `selected_experts`，见 [adaptation_layers.py](multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L927-L966)）; trainer 的 G 候选生成点见 [gspo_trainer.gspo_group_training_step](examples/automm/Conv-LoRA/gspo_trainer.py#L252).
- 因为专家是按样本/空间位置选择的，expert-level 的 H 会更细粒度地反映哪些专家对哪些空间位置有效。

### 2.5 Encoder Adapter 的 GSPO 反馈机制（详解）

背景与目标：Encoder Adapter（标准瓶颈 Adapter）通过 GSPO 进行 scale 自适应，使表现好的适配器放大其贡献，表现差的适配器降低贡献，从而提升整体稳定性与性能。

流程图（简要）：

```
Forward (single candidate)
        │
        ▼
    x -> AdapterDown -> Act -> AdapterUp -> up_out
        │
        ▼
    Output = base + up_out * s_eff
        │
        ▼
Prediction -> quality (IoU)
        │
        ▼
Trainer aggregates quality (e.g. group/batch) -> adapter.update_quality_feedback(q)
        │
        ▼
H_adapter ← μ · H_adapter + (1-μ) · q
contribution_score = σ(k·(H_adapter − 0.5))
s_eff ← s_0 · contribution_score
```

关键公式：

1) Adapter 前向（含自适应 scale）

$$
y = x + \text{Adapter}_{up}(\text{Act}(\text{Adapter}_{down}(x))) \cdot s_{eff}
$$
其中：
$$
s_{eff} = s_0 \cdot \text{contribution\_score},\quad \text{contribution\_score} = \sigma\big(k\cdot(H_{adapter}-0.5)\big)
$$

2) 质量历史更新（模块级 EMA）：

$$
H_{adapter}^{(t+1)} = \mu_a \cdot H_{adapter}^{(t)} + (1-\mu_a) \cdot q
$$

实现要点与差异：
- Trainer 中的 `update_adapter_feedback` 在当前实现会对组内质量做聚合（例如按 batch/G 求均值）然后传入每个 Adapter 的 `update_quality_feedback`，因此 Adapter 的更新通常是模块级的整体反馈而非 per‑spatial 的细粒度反馈（实现见 [examples/automm/Conv-LoRA/gspo_trainer.py](examples/automm/Conv-LoRA/gspo_trainer.py#L440) 与 Adapter 的实现 [multimodal/src/autogluon/multimodal/models/adaptation_layers.py](multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L99-L111)）。
- 若需要更细粒度的 credit assignment，可以将 Adapter 的更新改为按候选或按样本分别传入质量，但会带来更高的噪声与方差，需要额外平滑或正则化策略。

## 3. 与其他 PEFT 模块的协同

### 3.1 信号流

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Vision Encoder                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Conv-LoRA MoE (空间专家选择)  ←── GSPO 反馈                   │  │
│  │  Encoder Adapter (通道变换)    ←── GSPO 反馈                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                               │                                      │
│                               ▼                                      │
│                      image_embeddings                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Mask Decoder                                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  LoRA on Attention (注意力适应)  ←── GSPO 反馈 (🆕)             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                               │                                      │
│                               ▼                                      │
│                      segmentation_mask                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    Quality Evaluation (IoU/Dice)
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    Conv-LoRA MoE        Encoder Adapter      LoRA on Attention
    反馈更新              反馈更新             反馈更新 (🆕)
```

### 3.2 统一优势加权

所有 PEFT 模块共享相同的优势信号：

$$
\text{advantage}_g = Q_g - \bar{Q}
$$

其中 $Q_g$ 是第 g 个预测的质量，$\bar{Q}$ 是组内平均质量。

**加权损失**：

$$
\mathcal{L} = \frac{1}{G} \sum_{g=1}^{G} \sigma(\text{advantage}_g \cdot \tau) \cdot \mathcal{L}_{seg}^{(g)}
$$

---

## 4. 开关参数设计

### 4.1 参数层次

```
Level 0: 基础功能
├── --decoder_attn_lora_enable     # 启用 Decoder LoRA
└── --gspo_enable                  # 启用 GSPO 训练

Level 1: GSPO-LoRA 扩展
└── --gspo_lora_attention_enable   # 启用 GSPO 对 LoRA 的优化 (🆕)

Level 2: 高级选项
├── --gspo_lora_momentum           # LoRA 质量历史动量 (🆕)
└── --gspo_lora_scale_adaptation   # 启用 Scaling 自适应 (🆕)
```

### 4.2 向后兼容性

```python
# 原有命令（不受影响）
python run_semantic_segmentation.py --decoder_attn_lora_enable
# 结果：LoRA 使用标准固定 scaling

# 新功能（显式启用）
python run_semantic_segmentation.py --decoder_attn_lora_enable --gspo_lora_attention_enable
# 结果：LoRA 使用 GSPO 动态 scaling
```

---

*下一篇：[02_gspo_integration.md](./02_gspo_integration.md) - GSPO 集成方案与数学公式*

