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

Prompt jitter (实现细节，简洁说明)：当使用 `bbox_prompt_source=predict` 或在弱监督路径上做稳定性测试时，代码会对预测得到的 box 进行“小抖动”（small jitter）：先根据 `weak_box_jitter_mode`（"box"/"image"/"pixel"）和 `weak_box_jitter_amount` 计算可变幅度 dx/dy，然后为 box 的四条边分别采样非负增量（uniform[0,dx] / uniform[0,dy]），通常以 outward‑only 模式扩展 box（可选允许偶尔向内收缩）；最后把 jitter 后的坐标截断到图像边界。该实现在 `run_semantic_segmentation.py` 的训练/弱监督配置中和文档 `cursor_semisup_gpt-251231.md` 的 `_jitter_boxes_outward` 中有对应实现。

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

**教师‑学生（通俗数据流与流程图）**

简短说明：教师（Teacher）是一个参数慢变体（通常用 EMA 维护）用于生成“稳定的伪标签”；学生（Student）是正在训练的模型，用这些伪标签对无标注样本进行一致性训练，同时在有标签样本上使用标准监督损失。教师不直接参与反向传播，教师参数通过学生参数的指数移动平均更新。

数据流（ASCII 流程图）：

Unlabeled image u
  ├─> weak augment + predict box (u_w, stable prompt)
  │     └─> Teacher (EMA params) --forward--> pseudo mask \hat{y}_T  (stop_grad)
  └─> strong augment + noisy prompt (u_s, noisy prompt)
        └─> Student (trainable params) --forward--> p_S (prob map)

Losses on unlabeled sample:
- Consistency loss: D( p_S(u_s), stop_grad(\hat{y}_T) ) weighted by $w(u)$

有标签样本流程（并行）：
Labelled image x
  └─> (augment) -> Student -> p_S
  └─> supervised mask loss: L_sup( p_S, y )

训练更新规则：
- 学生参数通过反向传播最小化 L_sup (有标签) + L_cons (无标签) 等组合损失。
- 教师参数在线更新：
  $$\theta_T \leftarrow m\theta_T + (1-m)\theta_S$$
  其中 $m$ 是 EMA 衰减（接近 1），保证教师比学生更稳定。

为什么能用部分无标注样本学习：
- 有标签样本提供监督锚点，约束模型不偏离真实分布。
- 教师为无标签样本提供“伪目标”（比直接用模型输出更稳定），学生学习在强扰动下与这些目标一致，从而从无标签样本中吸收结构化信息（如对象形状和不变性）。
- 稳定的伪标签（EMA + 弱增强 + 稳定性筛选）可减少噪声，逐步扩展有效训练集，达到用少量标注和大量未标注混合训练的目的。

工程细节提示：
- 教师的输入通常用弱增强/小抖动以保证目标稳定；学生用强增强制造困难视图以提升鲁棒性。
- 仅在满足置信度与稳定性阈值（$\tau_{conf}$ 与 $\tau_{stab}$）时才使用伪标签，可以显著减少确认偏差。

**符号表（便于阅读公式）**

- $u$：无标签图像（unlabeled image）。
- $u_w$：对 $u$ 做弱增强得到的视图（weak augment, teacher 输入）。
- $u_s$：对 $u$ 做强增强得到的视图（strong augment, student 输入）。
- $B$：box prompt（bounding box），可来自 GT、预测器或 jitter。
- $\hat{M}$ 或 $\hat{y}_T$：教师生成的伪标签（pseudo mask），常为二值化后的 mask；$\hat{y}_T=\mathrm{Binarize}(p_T;\tau)$。
- $p_T$：教师（Teacher）对图像/视图的概率图（probability map，通常为 sigmoid/logit 后的概率）。
- $p_S$：学生（Student）对图像/视图的概率图（student 输出的概率）。
- $D(\cdot,\cdot)$：一致性损失度量（例如 Dice loss、binary CE 等），用来比较概率图或二值掩码。
- $w(u)$：伪标签权重/可信度，用于给每个无标签样本的 consistency loss 加权（可由 $\tau_{conf}$、$\tau_{stab}$ 等组合计算得到）。
- $\tau_{conf}$：置信度阈值（confidence threshold），例如基于 $\max_x p_T(x)$ 或 box 内均值判定伪标签是否足够置信。 
- $\tau_{stab}$：稳定性阈值（stability threshold），用于判定多次 jitter/TTA 下的 mask 一致性是否足够高（例如 mean IoU or min IoU > $\tau_{stab}$）。
- $\mathrm{IoU}(A,B)$：交并比（Intersection over Union），用于衡量两个掩码的重合度。
- $\mathrm{Area}(S)$：区域 $S$ 的面积（像素数或相对面积），例如 $\mathrm{Area}(\hat{M}\cap B)$ 表示掩码与盒子的重叠像素数。
- $\theta_S,\ \theta_T$：学生与教师的参数向量（weights）。
- $m$：EMA 衰减（momentum），教师参数按 $\theta_T\leftarrow m\theta_T+(1-m)\theta_S$ 更新。
- $\mathcal{L}_{cons}$：一致性损失（consistency loss），通常写为 $\mathcal{L}_{cons}(u)=w(u)\cdot D\big(p_S(u_s),\;\mathrm{stop\_grad}(\hat{y}_T)\big)$。
- $\mathcal{L}_{sup}$：有监督的 mask loss（例如 Dice + BCE），在有标签样本上计算并反向传播。
- $G$：GSPO 的 group size（每组候选数量），用于生成并比较多候选预测。

如需我把这些符号直接内联到公式下方或添加一页符号速查表，请告诉我你偏好哪种格式。

示例代码：伪标签置信度 + 稳定性（最小可运行）

下面的最小示例展示了如何从教师输出计算置信度、如何基于 box 抖动计算稳定性 IoU_stab，以及如何基于两者决定是否接受伪标签；这段代码用法与仓库中现有函数（`_weak_box_losses`, `_jitter_boxes_outward`, `compute_segmentation_quality`）相对应。

```python
import torch
import torch.nn.functional as F

def compute_iou(mask_a, mask_b):
    a = (mask_a > 0.5).float()
    b = (mask_b > 0.5).float()
    inter = (a * b).sum()
    union = a.sum() + b.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

def pseudo_label_confidence_and_stability(teacher_logits, student_logits, boxes, jitter_fn, model_forward,
                                         tau_conf=0.9, tau_stab=0.75, n_jitter=4):
    # teacher_probs: p_T
    p_T = torch.sigmoid(teacher_logits)  # [H, W] or [1, H, W]

    # confidence metrics
    max_prob = p_T.max()                 # max over image
    # or box mean: mean over pixels inside prompt box
    # box = [x1,y1,x2,y2] (example for first box)
    x1,y1,x2,y2 = boxes[0]
    bx = p_T[int(y1):int(y2)+1, int(x1):int(x2)+1]
    box_mean = bx.mean()
    # entropy (low entropy -> high confidence)
    p = p_T.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    # pick a confidence statistic (example: box_mean)
    conf = box_mean.item()

    # baseline mask from teacher (binarized)
    base_mask = (p_T > 0.5).float()

    # compute stability via jittered boxes
    ious = []
    for j in range(n_jitter):
        jittered_boxes = jitter_fn(boxes)
        # forward the teacher (or student) with jittered prompt to get mask
        jitter_output = model_forward(jittered_boxes)
        jitter_mask = (torch.sigmoid(jitter_output) > 0.5).float()
        ious.append(compute_iou(base_mask, jitter_mask).item())

    iou_stab = float(sum(ious) / len(ious))

    accept = (conf > tau_conf) and (iou_stab > tau_stab)
    return {
        'conf': conf,
        'iou_stab': iou_stab,
        'accept_pseudo': accept,
    }
```

说明：上面 `jitter_fn` 可用仓库中的 `_jitter_boxes_outward`，`model_forward` 可是调用 EMA 教师的前向函数（返回 logits）。`accept_pseudo=True` 时把二值化的教师掩码作为伪标签或用于一致性损失。

一致性损失（如何计算）

一致性损失的目标是让学生在强增强视图下的输出与教师在弱增强视图下的伪标签一致。常见实现方式：

- 概率级别一致性（soft）: 直接对概率图做 L2 / KL / BCE：
  $$\mathcal{L}_{cons} = \frac{1}{|\Omega|}\sum_x \| p_S(x) - p_T(x) \|^2$$

- 二值化后的一致性（hard）: 先二值化教师的概率图为 $\hat{y}_T$，再用 Dice/BCE 等：
  $$\mathcal{L}_{cons} = \mathrm{Dice}\big( p_S, \; \hat{y}_T \big)\quad\text{或}\quad \mathrm{BCE}(p_S,\hat{y}_T).$$

在本仓库的示例（概念上）使用了二值化教师伪标签与 Dice/BCE 组合，并通过权重 $w(u)$（由 $\tau_{conf}$ 與 $\tau_{stab}$ 决定）对每个无标签样本加权：
$$
\mathcal{L}_{cons}(u) = w(u) \cdot D\big( p_S(u_s), \; \mathrm{stop\_grad}(\hat{y}_T) \big)
$$
其中 `stop_grad` 表示教师伪标签不会反向传播回教师（教师仅由 EMA 更新）。

实务建议：当伪标签可能有噪声时，优先使用概率级别的一致性或混合损失（soft + hard），并对 $w(u)$ 做保守设置或逐步放宽（warmup）。


说明：$\tau_{stab}$ 与 $\mathrm{IoU}_{stab}$

- 定义：$\tau_{stab}$ 是用于判定伪标签稳定性的阈值；当多次对 prompt 进行微抖动或做 TTA 时，如果候选 mask 之间的一致性度量（如平均 IoU 或最小 IoU）高于 $\tau_{stab}$，则认为该伪标签稳定可信。
- $\mathrm{IoU}_{stab}$：通常指在多次扰动下计算得到的 mask‑to‑mask IoU 的统计量（例如 mean IoU、min IoU 或 1−std），常用实现为先对同一图像做 N 次 box‑jitter/TTA，得到 masks {M_i}，然后取
  $$\mathrm{IoU}_{stab}=\frac{1}{N}\sum_{i=1}^N \mathrm{IoU}(M_0, M_i)$$
  或者取 $\min_i\mathrm{IoU}(M_0,M_i)$ 作为更保守的度量。

示例判定规则：接受伪标签当且仅当 $\max p_T>\tau_{conf}$ 且 $\mathrm{IoU}_{stab}>\tau_{stab}$。

B. 伪标签自训练（伪标签 + box‑稳定性过滤）

流程：

1. 用 EMA 教师或当前模型在 `bbox_prompt_source=predict` 下对无标签图生成候选 mask(s)。

  说明（通俗）：这里的 `bbox_prompt_source=predict` 就是“让模型自己先预测一个 bounding box，然后把这个 box 作为提示去生成 mask”。换句话说，流程是：模型先在无标签图上运行一次（或用 EMA 教师运行一次）得到一个或多个 box（作为 model‑predicted prompts），再把这些 box 提示传入 mask 解码器/adapter，得到候选 mask，这些 mask 就是后续稳定性检测或伪标签筛选的输入。
2. 对每个候选做稳定性测试：多次 jitter box / 少量图像增强，计算 mask‑mask IoU（或 mask 内面积与 box 的覆盖比）
3. 只保留满足：conf>τ_conf 且 IoU_stab>τ_stab 的伪标签进入训练集。

伪标签过滤度量示例：
$$
\text{mask\_box\_score} = \frac{Area(\hat{M} \cap B)}{Area(\hat{M})} \quad\text{and}\quad \frac{Area(\hat{M} \cap B)}{Area(B)}
$$
只接受满足两者均在区间 $[\alpha, \beta]$ 的候选（例如 $[0.6, 1.05]$）。

注意（重要担忧）：对于 SAM 而言，如果我们用 box 作为提示输入，模型通常会生成与 box 较好对齐的 mask——在很多情况下预测的 mask 几乎覆盖整个 box，因此 $\mathrm{Area}(\hat{M} \cap B)$ 与 $\mathrm{Area}(B)$ 会非常接近，导致上面定义的 `mask_box_score` 在判别“伪标签质量”时可能无效或失准。换句话说，仅靠掩码与框的覆盖比很容易出现虚假的高分，无法反映 mask 的形状准确性或细节质量。

短期建议（文档记录，暂不改代码）：保留该度量作为一个初筛条件，但务必结合其他判据（如 mask 内部置信度分布、teacher 在多次 prompt/jitter 下的稳定性、mask 边界复杂度或 mask 与 box 的投影一致性）来复核伪标签；在后续 `implementation.md` 中我们会列出具体替代或补充度量供实装参考。

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

实现说明（代码证据）：在本代码库中，box‑outside penalty 的实现位于 `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py` 中的 `_weak_box_losses` 函数。实际实现通过对 box 之外的像素使用 `binary_cross_entropy_with_logits` 对应目标 0（即把 outside 视为 0 的软标签）来计算 outside loss，并按 `weak_loss_outside_weight` 加权后加入总 loss；同时在 box 内部使用熵最小化和可选 TV 平滑。换言之，当前实现并不依赖真实前景/背景真值，而是把 outside 区域作为“应当接近 0 的软约束”来鼓励模型抑制盒外前景概率（实现见 `_weak_box_losses`）。

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
