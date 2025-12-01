# 实验Exp4a参数修改与噪声解释
_Exported on 11/30/2025 at 21:15:25 PST from Cursor (2.1.41)_

---

**User**

根据对话历史docs/cursor_rloo_conv_lora-251127.md，在实验Exp4a时，我们改了哪些参数？其中噪声指的是什么参数？

---

**Cursor**

我先去查看你提到的 `cursor_rloo_conv_lora-251127.md` 中与 Exp4a 相关的部分，看看当时记录了哪些超参改动以及“噪声”的具体含义，然后给你一个总结。

**1）Exp4a 改了哪些参数 / 设置？**

根据文档中“实验 4：混合训练”部分，Exp4a 相比之前的纯 RLOO 实验（尤其是 Exp3 大噪声版）主要有三点改动：

- **采样噪声相关参数**  
  - 代码里把候选 mask 采样时加入 logits 的噪声强度区间，从 **1.0–3.0 降到 0.1–0.5**（更保守的采样）。  
- **加入监督损失权重**  
  - 新增混合训练公式  
    \[
    \text{total\_loss} = \text{supervised\_weight} \cdot \text{supervised\_loss} + (1-\text{supervised\_weight}) \cdot \text{rloo\_loss}
    \]  
  - 在 Exp4a 中设置 **`supervised_weight = 0.4`**（40% 监督 + 60% RLOO）。  
- **KL / diversity 计算方式**  
  - diversity（KL 散度）改为**除以像素数**，得到更合理的监控指标（这属于实现细节上的“参数/归一化”调整）。

命令行上直接可见的主要差异是：`--supervised_weight 0.4`、`--beta 0.0`（不再额外加 KL 正则），以及背后实现层面采样噪声区间从 1.0–3.0 改到 0.1–0.5。

---

**2）“噪声”具体指的是哪个参数？**

文档里说的“噪声从 1.0–3.0 降到 0.1–0.5”，指的是在 RLOO 训练中，**对 `pred_logits` 加噪声时使用的噪声强度 `noise_scale`**，即类似下面这行里的系数：

```python
noisy_logits = scaled_logits + torch.randn_like(pred_logits) * noise_scale
```

所以，“噪声”本质上就是这个 `noise_scale`（噪声强度）的取值区间，从原来的 1.0–3.0 调整到了 0.1–0.5，用来控制对 logits 加高斯噪声的幅度，从“激进探索”变成“保守探索”。

```python
noisy_logits = scaled_logits + torch.randn_like(pred_logits) * noise_scale
```

---

**User**

从Exp4a到Exp5又改了哪些？

---

**Cursor**

我去文档里把 Exp5 的配置和说明定位出来，对比 Exp4a，总结一下具体改动点（包括命令行参数和实现级别的差异），再用简要列表给你梳理。

**先说结论：Exp5 不是单一一个 RLOO 配置，而是一组“收尾方案”（5A/5B/5C），整体从“优化 RLOO（Exp4a）”转向“主要做监督 + TTA，只保留一个极保守的 RLOO 尝试”。**

### 1️⃣ 从 Exp4a 到 Exp5 整体思路上的变化

- **Exp4a**：  
  - 目标：用 **混合训练（监督 + RLOO）** 挽救纯 RLOO 性能下降的问题。  
  - 关键：`supervised_weight = 0.4`，噪声变保守（0.1–0.5），`beta = 0.0`，依然认真做 RLOO。

- **Exp5（5A/5B/5C）**：  
  - 思路转向：**承认 RLOO 在这个任务上效果有限**，主线改为：
    - `5A`：在已有监督模型上做 **TTA 推理** 提升 IoU。  
    - `5B`：**重新训练监督模型**（更多 epochs + 强数据增强 + 解冻更多层）。  
    - `5C`：只做 **一次“极保守”的混合 RLOO**，如果还不行就彻底放弃 RLOO。

### 2️⃣ 具体参数改动（Exp4a 的 RLOO vs Exp5C 的 RLOO）

对比两次真正跑 RLOO 的命令行（Exp4a vs Exp5C）：

- **Exp4a：混合训练（推荐）**

```bash
# 关键参数（已在文档中）
--output_dir       outputs/rloo/isic2017/exp4a_hybrid
--num_generations  4
--beta             0.0
--reward_type      combo
--supervised_weight 0.4     # 40% 监督 + 60% RLOO
--learning_rate    1e-5
--epochs           3
--batch_size       2
```

- **Exp5C：最后一次极保守 RLOO 尝试**

```bash
--output_dir       outputs/rloo/isic2017/exp5c_last_try
--num_generations  4              # 不变
--beta             0.0            # 不变
--reward_type      combo          # 不变
--supervised_weight 0.8           # 从 0.4 提到 0.8，几乎变成以监督为主
--learning_rate    2e-6           # 从 1e-5 降到 2e-6，更小更稳
--epochs           1              # 从 3 降到 1，只试水一下
--batch_size       2              # 不变
```

**可以理解为：Exp5C 在 Exp4a 的基础上，把“RL 权重、学习率、训练轮数”都往“更保守、更像纯监督”方向收缩，只做一个低风险的小实验。**

### 3️⃣ 5A / 5B 相对于 Exp4a 的变化点

- **实验 5A：TTA**  
  - 不再改训练参数，而是在 **推理阶段** 增加：翻转、旋转、多次预测再平均。  
  - 目标：在不重新训练的前提下，用测试时增强，**白嫖 +0.5–2% IoU**。

- **实验 5B：重新训练监督模型**  
  - 关键超参改动（相较基线监督模型，而不是 Exp4a）：
    - `max_epochs`：从 3 提到 **15**。  
    - `data.augmentation`：设为 `"strong"`，引入更强的数据增强。  
    - `model.sam.freeze_image_encoder = False`：**解冻更多层**，让 backbone 也跟着学。  
    - 加上 `warmup_steps=100`、`patience=5` 做更稳的优化。  
  - 完全 **放弃 RLOO，只做监督训练**。

---

**一句话总结**：  
- 从 Exp4a 到 Exp5，方向从“继续优化 RLOO（保守噪声 + 0.4 混合比）”变成“主推 TTA + 强监督训练”，只在 Exp5C 里做一个 **supervised_weight=0.8、lr=2e-6、epochs=1** 的极保守 RLOO 收尾实验。

```shellscript
# 关键参数（已在文档中）
--output_dir       outputs/rloo/isic2017/exp4a_hybrid
--num_generations  4
--beta             0.0
--reward_type      combo
--supervised_weight 0.4     # 40% 监督 + 60% RLOO
--learning_rate    1e-5
--epochs           3
--batch_size       2
```

```shellscript
--output_dir       outputs/rloo/isic2017/exp5c_last_try
--num_generations  4              # 不变
--beta             0.0            # 不变
--reward_type      combo          # 不变
--supervised_weight 0.8           # 从 0.4 提到 0.8，几乎变成以监督为主
--learning_rate    2e-6           # 从 1e-5 降到 2e-6，更小更稳
--epochs           1              # 从 3 降到 1，只试水一下
--batch_size       2              # 不变
```

---

**User**

相比Exp2，Exp3又改了什么？为什么当时要这样改？Exp2的测试集结果是什么？

---

**Cursor**

我去文档里把 Exp2 和 Exp3 的配置及结果定位出来，对比参数改动，看看当时写下的动机说明和测试集指标，然后给你一个简要总结。

先把结论说在前面：

### 1️⃣ 相比 Exp2，Exp3 改了什么？

这里说的 Exp2 / Exp3 对应文档里的这两段命令：

- **Exp2：更大的探索空间**（`exp2_gen8_beta0.005`）  

```bash
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp2_gen8_beta0.005 \
  --num_generations 8 \
  --beta 0.005 \            # 更小的 KL 惩罚
  --reward_type combo \
  --learning_rate 2e-5 \    # 稍高的学习率
  --epochs 10 \
  --batch_size 2
```

- **Exp3：不同 reward 类型**（`exp3_dice_only`）  

```bash
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp3_dice_only \
  --num_generations 4 \
  --beta 0.01 \
  --reward_type dice \       # 只用 Dice
  --epochs 10 \
  --batch_size 2
```

**从 Exp2 到 Exp3 的主要改动：**

- **候选数量**：`num_generations` 从 **8 → 4**（探索样本数减半，更保守）。  
- **KL 系数**：`beta` 从 **0.005 → 0.01**（KL 约束稍微变强一点）。  
- **reward 类型**：从 **`combo`（IoU+Dice 组合）→ `dice`（只用 Dice）**，这是最核心的变化。  
- **学习率**：Exp2 显式设为 **`2e-5`**，Exp3 没写 `--learning_rate`，按脚本默认是 **1e-5**，也就是从更激进的学习率收回到较保守的学习率。  

### 2️⃣ 当时为什么这么改？

从文档的命名和注释可以看出当时的思路：

- **Exp2「更大的探索空间」的目的**  
  - 提高 `num_generations` 到 8：希望通过 **更多候选**，让 RLOO 的 baseline 估计更稳定，同时增加探索。  
  - 把 `beta` 降到 0.005：**减弱 KL 惩罚**，允许策略在参考策略附近有更大偏移，以便更大胆地试探更好的 mask。  
  - 提高学习率到 2e-5：在探索更大空间的同时，加快收敛（更“激进”的优化）。  

- **Exp3「不同 reward 类型」的目的**  
  - 看到单纯调 KL、候选数、学习率这类“优化超参”可能不够，就想直接验证：  
    - **reward 设计本身是不是问题？**  
  - Dice 更关注重叠区域，对分割任务，Dice 往往比单纯 IoU 更敏感，所以：  
    - Exp3 把 `reward_type` 换成 `dice`，其它设定（epochs=10、batch_size=2 等）大致维持在同一量级，  
    - 意图是：**在相对保守的采样/学习率下，单独考察 reward 形式对 RLOO 的影响**。  

所以可以简单理解为：

- **Exp2**：侧重于“**加大探索力度**”（更多候选、更小 KL、更大 lr）。  
- **Exp3**：侧重于“**换 reward 形状试试看**”（Dice-only），同时把探索/学习率拉回相对保守一点的设置。

### 3️⃣ Exp2 的测试集结果是多少？

在当前这份 `cursor_rloo_conv_lora-251127.md` 里，**结果表只列了：**

- 基线 (3 epochs)：测试 IoU = **0.7707**  
- Exp1 (10 epochs, beta=0.01, noise=0.5)：测试 IoU = **0.7624**  
- Exp3（fixed KL，大噪声 1.0–3.0 的那个版本）：测试 IoU = **0.6568**

也就是说：

- 文档里 **没有出现 Exp2（`exp2_gen8_beta0.005`）的测试集 IoU / Dice 数值**，  
- 只给了如何运行 Exp2 的命令，没有后续 “测试 IoU = …” 的记录。  

所以基于当前 md 内容，我**查不到 Exp2 的测试集结果**；要知道确切数值，需要去对应输出目录（例如 `outputs/rloo/isic2017/exp2_gen8_beta0.005`）里看 `results_*.json` 或你当时保存的对比报告。

```shellscript
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp2_gen8_beta0.005 \
  --num_generations 8 \
  --beta 0.005 \            # 更小的 KL 惩罚
  --reward_type combo \
  --learning_rate 2e-5 \    # 稍高的学习率
  --epochs 10 \
  --batch_size 2
```

```shellscript
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp3_dice_only \
  --num_generations 4 \
  --beta 0.01 \
  --reward_type dice \       # 只用 Dice
  --epochs 10 \
  --batch_size 2
```

---

**User**

那么Exp1的结果是?

---

**Cursor**

**Exp1 的测试集结果是：**

- **测试 IoU：0.7624**  
- **测试 Dice：0.8466**  

（对比基线 3 epochs：IoU = 0.7707，Dice = 0.8521，Exp1 是略有下降。）

---

**User**

相对于Exp1，Exp3改了什么？

---

**Cursor**

**相对于 Exp1，Exp3（fixed KL）主要改动是：**

- **Epoch 数**：从 **10 → 5**。  
- **KL 系数 beta**：从 **0.01 → 0.0**，也就是**去掉 KL 正则**（只保留 RLOO loss，本身仍会记录 diversity）。  
- **采样噪声策略**：  
  - Exp1：采样时使用 **固定噪声 `noise = 0.5`**。  
  - Exp3：改为更激进的策略：  
    - 温度从 **0.5 → 2.0** 递增：`temperature = 0.5 + 1.5 * (g / num_generations)`  
    - 噪声强度从 **1.0 → 3.0** 递增：`noise_scale = 1.0 + 2.0 * (g / num_generations)`  
  → 本质上就是：**Epoch 减半，KL 取消，采样噪声从 0.5 提升到 1.0–3.0 的大噪声探索**。

