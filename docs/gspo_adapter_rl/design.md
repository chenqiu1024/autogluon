## 目标

在现有 **Conv-LoRA + GSPO** 方案基础上，让 GSPO 在 **Vision Encoder 每层 MLP 后的 FC Adapter（Encoder Adapter）** 上也产生“强化学习式”的优化效果，并且通过开关保持向后兼容：

- **默认行为不变**：当不开启本扩展选项时，GSPO 仍按原逻辑用于 Conv‑LoRA（Adapter 在 GSPO step 中默认冻结，避免被 GSPO 的组级加权间接影响）。
- **开启扩展**：GSPO active 时同时训练 encoder adapters，并可选启用两阶段增强：
  - **阶段 1（最小改动）**：在 GSPO rollouts 中对 encoder adapter 输出注入探索噪声，增强组内多样性与 credit assignment。
  - **阶段 2（显式策略）**：将 encoder adapter 替换为 *gated adapter*，引入输入条件化 gate（policy-like），并在 GSPO rollouts 中对 gate logits 注入探索噪声。

> 本次明确 **不包含 decoder adapter**（mask decoder 里的 adapter_queries 不参与本扩展）。

---

## 总体结构与插入位置

Encoder 的每个 ViT block（`SamVisionLayer`）里，Adapter 位于 **MLP 路径**，与 MLP 并行，外部残差相加：

```text
SamVisionLayer (×N layers)

  residual
    │
    ├─ Attention(...)  ← Conv-LoRA 注入在 q/k/v 线性投影内部（MoE-Conv）
    │
    └─ MLP path:
        LN2(x)
          ├─ MLP(x)      -> mlp_out
          └─ Adapter(x)  -> adapter_out   (FC bottleneck, encoder adapter)
        y = residual + mlp_out + adapter_out
```

阶段 2 时，Adapter 变为 gated adapter：

```text
adapter_out = AdapterCore(x) * gate(x)
gate(x) = sigmoid( w · pool(x) + ε ),   ε ~ N(0, σ_gate^2)  (仅 GSPO rollouts 时注入)
```

---

## GSPO 的公式（与实现对应）

在每个训练 step，对每张图像生成 \(G\) 次 rollouts（利用 dropout/噪声产生差异），得到 \(G\) 组预测：

- 预测：\(\hat{y}^{(g)} = f_\theta(x; \xi_g)\)，其中 \(\xi_g\) 表示 rollout 噪声（dropout、adapter 噪声、gate 噪声等）
- 质量（奖励）：\(R^{(g)} = \text{Quality}(\hat{y}^{(g)}, y)\)
  - 实现中使用 IoU/Dice，并加入边界/平滑/细结构 shaping 得到 `shaped_quality`
- baseline：\(\bar{R} = \frac{1}{G}\sum_{g=1}^G R^{(g)}\)
- advantage：\(A^{(g)} = R^{(g)} - \bar{R}\)
- 权重（温度 \(\tau\)）：\(w^{(g)} = \sigma(\tau \cdot A^{(g)})\)

最终优化目标（实现中的主要项）：

\[
\mathcal{L}
= \frac{1}{G}\sum_{g=1}^G w^{(g)} \cdot \mathcal{L}_{seg}(\hat{y}^{(g)}, y)
 + \lambda_{moe}\mathcal{L}_{moe}
 + \lambda_{ctr}\mathcal{L}_{ctr}
\]

其中：
- \(\mathcal{L}_{seg}\)：分割损失（并叠加 smooth/boundary 约束）
- \(\mathcal{L}_{moe}\)：MoE load-balancing 辅助损失（Conv‑LoRA gate）
- \(\mathcal{L}_{ctr}\)：组内对比损失（鼓励高质量预测聚合）

> 这是一种“GSPO inspired”的可微 surrogate：通过 advantage 形成 **组内相对质量** 的权重，放大/缩小对应 rollout 的梯度贡献，达到类似策略优化的 credit assignment 效果。

---

## 阶段 0（基线，向后兼容）

当不启用本扩展（`optim.gspo.train_encoder_adapter=False`）时：

- Encoder Adapter 仍可存在（`model.sam.adapter_enabled=True`），但在 **GSPO active 的 step** 中会被临时冻结（`requires_grad=False`），确保“GSPO 只用于 Conv‑LoRA”这一语义成立。
- 在 GSPO warmup 阶段（GSPO 未激活）仍按普通监督训练逻辑运行。

---

## 阶段 1：Encoder Adapter 探索噪声（最小改动）

### 设计

在 GSPO active 的 rollouts 期间，对每层 encoder adapter 输出加入零均值噪声：

\[
\Delta h = \text{Adapter}(x) + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma_{adp}^2)
\]

噪声只在 GSPO active 且训练模式下生效，用于：
- 增强 rollouts 多样性（同一输入得到不同预测）
- 强化 advantage 权重对“更好预测路径”的梯度分配

### 预期收益

- 更强的组内探索：让 Adapter 不再在 \(G\) 次 rollouts 中“几乎相同”，从而使 GSPO 的优势加权更有效。
- 更稳的 credit assignment：帮助模型把“质量提升”更明确地归因到 Adapter 分支的改动。

---

## 阶段 2：Gated Encoder Adapter（显式策略）

### 设计

将每层 encoder adapter 替换为 gated adapter：

\[
\Delta h = \text{AdapterCore}(x)\cdot g(x),\quad g(x)=\sigma(\text{MLP}_g(\text{pool}(x)) + \epsilon)
\]

其中：
- \(\text{pool}(x)\) 为 token 的均值池化（对 `[B,H,W,C]` 做 H/W 平均）
- \(\epsilon \sim \mathcal{N}(0,\sigma_{gate}^2)\) 仅在 GSPO active rollouts 中注入

### 直观解释

- gate 是“每个样本、每层一个可学习开关/强度”，是一个轻量 policy。
- GSPO 的 rollouts 会对 gate 进行不同采样（通过 \(\epsilon\)），得到不同预测质量；advantage-weighted loss 会促使 gate 网络朝“更高质量的门控策略”收敛。

### 预期收益

- 自适应层级分配：让模型学会“哪些层的 adapter 应该更强/更弱地介入”。
- 更强的空间-通道协同：Conv‑LoRA（空间 MoE）与 gated adapter（通道强度策略）同时受质量信号驱动。

---

## 不包含 Decoder Adapter 的原因

Decoder adapter 直接作用于 mask decoder 的 queries 分支，属于解码阶段适配；本次扩展聚焦于 **encoder representation** 的 GSPO credit assignment，以避免同时改动两个阶段导致归因混乱与调参复杂。


