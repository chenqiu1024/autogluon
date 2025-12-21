## 目标

在现有 **Conv-LoRA + GSPO** 以及（可选）**Encoder Adapter GSPO** 的基础上，将 **LoRA on Attention**（本项目中指 *SAM Mask Decoder 的 SamAttention Q/K/V 投影上的 LoRA*）也纳入 GSPO 的强化学习式微调。

约束与原则：

- **默认向后兼容**：不开启新开关时，GSPO active step 仍只优化 Conv‑LoRA（并且会冻结 decoder-attn-LoRA，避免“被动纳入”）。
- **仅纳入 decoder attention LoRA**：不改变 Conv‑LoRA（encoder attention）与 decoder adapter 的设计边界。
- **不触碰任何 `.pdf` 文件**。

---

## LoRA on Attention 在哪里？

Decoder 侧的 `SamAttention` 对 query/key/value 的投影：

```text
Mask Decoder (Two-Way Transformer)
  SamTwoWayAttentionBlock
    - self_attn:      q_proj/k_proj/v_proj (+ LoRA)
    - cross_token2img q_proj/k_proj/v_proj (+ LoRA)
    - cross_img2token q_proj/k_proj/v_proj (+ LoRA)
  final_attn_token_to_image: q_proj/k_proj/v_proj (+ LoRA)
```

LoRA 形式：

\[
y = W_0 x + \Delta W x,\quad \Delta W = B A,\quad \text{scaling}=\alpha/r
\]

实现等价于：

\[
\Delta y = (xA^\top)B^\top \cdot \text{scaling}
\]

---

## GSPO（现有实现）的核心公式与对应

对每张图像做 \(G\) 次 rollouts（通过 dropout/噪声产生差异）：

- 预测：\(\hat{y}^{(g)} = f_\theta(x; \xi_g)\)
- 质量：\(R^{(g)} = \text{Quality}(\hat{y}^{(g)}, y)\)
- baseline：\(\bar{R} = \frac{1}{G}\sum_g R^{(g)}\)
- advantage：\(A^{(g)} = R^{(g)} - \bar{R}\)
- 权重：\(w^{(g)} = \sigma(\tau A^{(g)})\)

优化目标（主项）：

\[
\mathcal{L}
= \frac{1}{G}\sum_{g=1}^G w^{(g)} \cdot \mathcal{L}_{seg}(\hat{y}^{(g)}, y)
 + \lambda_{moe}\mathcal{L}_{moe}
 + \lambda_{ctr}\mathcal{L}_{ctr}
\]

> 这是一种 GSPO-inspired surrogate：用组内相对质量作为“策略优势”，通过加权分割损失来做 credit assignment。

---

## 阶段 0（基线，向后兼容）

当不启用本扩展时（默认）：

- 即使你开启了 decoder attention LoRA（`decoder_attention_lora_r > 0`），在 **GSPO active step** 中也会临时冻结这些 LoRA 参数：
  - `optim.gspo.train_decoder_attention_lora = False`（默认）

从语义上保证：**GSPO 仍然只用于 Conv‑LoRA**（以及你是否额外开启的 encoder adapter GSPO）。

---

## 阶段 1：对 decoder-attn-LoRA 注入探索噪声（最小改动）

### 设计

在 GSPO active 的 rollouts 中，对 LoRA 的 delta 分支加入零均值噪声：

\[
\Delta y = (xA^\top)B^\top \cdot \text{scaling} + \epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma_{\text{lora}}^2)
\]

仅在 GSPO active 且训练模式下生效，用于提高 rollouts 多样性。

### 好处

- 同一输入产生多样预测，使 advantage-weighting 更有效地对齐“更好”的 LoRA 扰动方向。
- 保持参数量与结构几乎不变，调参成本低。

---

## 阶段 2：Gated decoder-attn-LoRA（显式策略）

### 设计

只对 LoRA 的 delta 分支加 gate（不动 base path）：

\[
y = W_0x + g(x)\cdot \Delta y
\]

其中 gate 是 per-sample 的标量：

\[
g(x) = \sigma(w\cdot \text{pool}(x) + \eta),\quad \eta\sim\mathcal{N}(0,\sigma_{\text{gate}}^2)
\]

- `pool(x)`：对 `[B,P,T,C]` 或 `[B,T,C]` 的 token/point 维度做平均，得到 `[B,C]`
- \(\eta\) 仅在 GSPO active rollouts 时注入，用于探索

### 好处

- 将“LoRA 应该介入多大强度”显式变成一个可学习策略。
- GSPO 的组内比较可以更清晰地对 gate 做 credit assignment（哪些 gate 值带来更好分割质量）。

---

## 与现有组件的协同（直观框图）

```text
Encoder (Vision)                       Decoder (Mask)
┌───────────────────────────┐         ┌──────────────────────────────┐
│ Attention: Conv-LoRA MoE  │         │ SamAttention Q/K/V (+LoRA)    │
│   - GSPO feedback on gate │         │   - Stage1: delta noise       │
│ MLP: Encoder Adapter      │         │   - Stage2: gated delta (policy)│
│   - (可选) GSPO gating     │         └──────────────────────────────┘
└───────────────────────────┘

GSPO group rollouts:
  同一图像做 G 次 forward (dropout/噪声/门控采样不同) → 质量对比 → advantage-weighted loss → 反传更新
```

