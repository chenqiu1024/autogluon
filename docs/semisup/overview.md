# 半/弱监督方法概览（针对 SAM + Conv‑LoRA + GSPO 框架）

本文档简洁说明三类可直接落地的方法：

- A. Teacher‑Student（EMA 教师）+ 一致性正则（半监督核心推荐）
- B. 伪标签自训练（Pseudo‑labeling）结合 box‑stability 过滤
- C. 弱监督（仅 box / 点 / 涂鸦 / 图像级）损失与流程

总体设计约束与现有能力（来自项目现状）：

- SAM 主干冻结，训练小参数：Conv‑LoRA (MoE) + Encoder/Decoder Adapter。
- GSPO 提供组级 G 候选机制与质量/advantage 信号（参见 `gspo_group_training_step`）。
- Code 已支持训练时注入 box prompt（`bbox_prompt_source` / `train_box_prompt_mode`）。

原则性集成点（不破坏现有 pipeline）：

- 数据层：在 dataset 构建/loader 层标注 `labeled_fraction`，决定样本走 supervised / semi / weak path。
- 前向层：使用 EMA 教师网络（参数 EMA）在弱增强下生成稳定伪标签；学生在强增强下训练并最小化一致性损失。
- Trainer：保留 GSPO 的 G 候选生成，用 proxy reward（teacher agreement / stability / entropy）替代无标签 IoU 以驱动 MoE（可选）。

方法细节：

A. Teacher‑Student（EMA 教师）+ 一致性正则

流程图：

```
Unlabeled image u
  ├─> weak augment (u_w), stable prompt (predict box / small jitter)
  │     └─> teacher (EMA) -> pseudo mask \hat{y}_T
  └─> strong augment (u_s), noisy prompt (box jitter / TTA)
        └─> student -> p_S (prob map)

Loss_cons = w(u) · D( p_S, stop_grad(\hat{y}_T) )
```

公式：

Teacher 参数 EMA 更新：
$$
\theta_T^{(t)} = m \cdot \theta_T^{(t-1)} + (1-m) \cdot \theta_S^{(t)}
$$

一致性损失（示例，Dice 或 CE）：
$$
\mathcal{L}_{cons}(u) = w(u) \cdot \mathrm{Dice}\big( p_S(u_s), \; \mathrm{Binarize}(p_T(u_w); \tau) \big)
$$
其中 $w(u)$ 是伪标签可信度权重（见下）。

可信度/权重 $w(u)$ 的常用设计：
- Teacher 置信度阈值：$\max p_T > \tau_{conf}$
- 稳定性检验：在多次 prompt jitter / TTA 下，mask IoU 的方差低于 $\sigma_{stab}$
- 结合两者：$w(u)=\mathbf{1}[conf>\tau] \cdot \sigma_{stab\_weight}$。

B. 伪标签自训练（伪标签 + box‑稳定性过滤）

流程：

1. 用 EMA 教师或当前模型在 `bbox_prompt_source=predict` 下对无标签图生成候选 mask(s)。
2. 对每个候选做稳定性测试：多次 jitter box / 少量图像增强，计算 mask‑mask IoU（或 mask 内面积与 box 的覆盖比）
3. 只保留满足：conf>τ_conf 且 IoU_stab>τ_stab 的伪标签进入训练集。

伪标签过滤度量示例：
$$
\text{mask\_box\_score} = \frac{Area(\hat{M} \cap B)}{Area(\hat{M})} \quad\text{and}\quad \frac{Area(\hat{M} \cap B)}{Area(B)}
$$
只接受满足两者均在区间 $[\alpha, \beta]$ 的候选（例如 $[0.6, 1.05]$）。

C. 弱监督（box / point / scribble）损失

常用弱监督项（可与有监督损失联合）：

1) Box‑outside penalty（鼓励 box 外为背景）：
$$
\mathcal{L}_{out} = \frac{1}{|\Omega\setminus B|} \sum_{x\notin B} p_S(x)
$$
2) Box‑inside soft positive（只对 box 内高置信区间施加正项）：
$$
\mathcal{L}_{in} = -\frac{1}{|B|} \sum_{x\in B} w_x \big[ y_x \log p_S(x) + (1-y_x)\log(1-p_S(x)) \big]
$$
其中 $w_x$ 可根据点标签或 teacher 置信度设定，未标注像素可被忽略。
3) Projection / area constraint：预测 mask 在 x/y 投影应该覆盖 box 的投影。

半弱监督下与 GSPO 的协同（拓展）

- 在无标签样本上，用 proxy reward 替代 IoU：
$$
r(u) = \mathrm{IoU}\big( p_S(u_s), \; p_T(u_w) \big)\quad\text{或}\quad r(u)=1 - H(p_T(u_w))
$$
然后把 $r(u)$ 用作 GSPO 的 quality 替代或与真实 IoU 混合（加权），以驱动 MoE 门控偏好稳定专家。
- 在实现上，修改 `gspo_trainer` 中的 `compute_segmentation_quality` 用代理指标对无标签样本打分，后续 `update_expert_feedback` / `update_lora_attention_feedback` 使用这些 scores。

工程要点与风险提醒

- 伪标签/一致性策略容易出现确认偏差（confirmation bias），务必使用保守过滤和逐步放开阈值（warmup）——例如前 5 个 epoch 仅用 labeled 数据训练，再逐步加入无标签的一致性损失或伪标签。
- 对于 GSPO：若把无标签 proxy reward 直接换入，建议先混合真实 IoU 的权重（如仅在部分 batch 或某些 epoch 使用 proxy），以避免策略朝错误 reward 偏移。

下游实现细节见 `implementation.md`。
