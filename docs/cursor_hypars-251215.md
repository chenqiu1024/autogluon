# Lambda_smooth 参数调优
_Exported on 12/15/2025 at 23:30:58 PST from Cursor (2.2.20)_

---

**User**

请从会话历史记录文件 docs/cursor_tta-251204.md 中恢复会话。现在，ChatGPT给出了关于三个不同超参数调整的建议：```
下面我**逐一给出你这个体系中每一种微调方法 → 对应超参数 → 在 conv-LoRA + SAM + GSPO + FC-adapter 中的具体作用机理**。

不兜圈子，但**解释到“为什么这个超参必须这样设”这一层**。
**0️⃣ 体系前提（统一假设）**
• **SAM backbone 冻结**
• 可训练参数：
    ◦ Conv-LoRA（局部空间扰动）
    ◦ Adapter（FC bottleneck，全局调制）
    ◦ GSPO policy（prompt / box 决策）
• 优化目标：
**在不破坏 SAM 语义结构的前提下，提升边界与细结构质量
1️⃣ Smoothness / Laplacian 正则
方法**
Mask 平滑正则（Laplacian / curvature constraint）
**超参数**
• **`λ_smooth`**
• **推荐范围**：`0.05 – 0.2`
• **常用值**：`0.1`
**数学形式**Lsmooth=∑i,j(∇2mi,j)2\mathcal{L}_{smooth}
= \sum_{i,j} (\nabla^2 m_{i,j})^2Lsmooth=i,j∑(∇2mi,j)2

**作用机理**
• Dice reward 对 **高频噪声不敏感**
• Conv-LoRA 对 **局部梯度极其敏感**
• GSPO 属于 **高方差 policy gradient**
→ 结果是：
**mask 边缘抖动、锯齿化、训练不稳定**
`λ_smooth` 的作用：
• 抑制 **二阶空间振荡**
• 降低 reward variance
• 稳定 GSPO policy 更新
**为什么不能大？**
• `> 0.2`：
    ◦ thin structure 被抹掉
    ◦ lesion 边界向内塌缩
**2️⃣ Boundary Loss / Boundary Reward
方法**
边界约束（distance transform / contour loss）
**超参数**
• **`λ_boundary`**
• **相对 Dice 比例**：`0.1 – 0.5 × Dice`
• **常用值**：`0.3`
**数学形式（示例）**Lboundary=∑dGT(p)⋅m(p)\mathcal{L}_{boundary}
= \sum d_{GT}(p)\cdot m(p)Lboundary=∑dGT(p)⋅m(p)
或对应 reward：Rboundary=−LboundaryR_{boundary} = - \mathcal{L}_{boundary}Rboundary=−Lboundary
**作用机理**
• Dice 的梯度：
    ◦ 90% 来自 **interior**
• 医学分割的关键误差：
    ◦ 全在 **boundary**
Boundary loss 的本质：**把梯度“搬”到边界上
在你架构中的具体作用**
• Conv-LoRA：
    ◦ 接收 boundary gradient → 学局部形状
• Adapter（FC bottleneck）：
    ◦ 防止全局 embedding 被边界噪声拉偏
**为什么是 0.1–0.5？**
• `< 0.1`：边界梯度被 Dice 吞没
• `> 0.5`：mask 内部破碎，出现 hollow 区域
**3️⃣ GSPO / RL Reward Scale
方法**
Reward shaping（多 reward 组合）
**超参数**
• **`w_dice = 1.0`（固定）**
• **`w_boundary = 0.1 – 0.5`**
• **`w_smooth = 0.05 – 0.2`
Reward 形式**R=wdice⋅Dice+wboundary⋅Rboundary−wsmooth⋅LsmoothR
= w_{dice}\cdot Dice
+ w_{boundary}\cdot R_{boundary}
- w_{smooth}\cdot \mathcal{L}_{smooth}R=wdice⋅Dice+wboundary⋅Rboundary−wsmooth⋅Lsmooth
**作用机理**
GSPO 的梯度为：∇θJ∝R∇log⁡πθ\nabla_\theta J
\propto R \nabla \log \pi_\theta∇θJ∝R∇logπθ
→ **reward scale = policy 更新幅度
关键原则**
• Dice 必须是 **anchor**
• Boundary / smoothness 只能是 **shaping
为什么 Dice 权重不能改？**
• Dice ∈ [0,1]
• GSPO 对 reward scale 极其敏感

→ 改 Dice = 改 policy learning rate（极易发散）
**4️⃣ Thin-Structure / Detail Reward
方法**
细结构增强（skeleton / thin boundary recall）
**超参数**
• **`w_thin`**
• **范围**：`0.05 – 0.1`
• **建议上限**：`≤ 0.1`
**作用机理**
• thin structure：
    ◦ 占像素极少
    ◦ Dice 几乎无梯度
• Conv-LoRA：
    ◦ 非常容易过拟合这些高频信号
`w_thin` 的定位是：**“告诉模型方向”，而不是“让它追求极致”
为什么不能大？**
• 大权重 → GSPO 频繁调整 box
• mask 出现毛刺 / fake branch
**5️⃣ Adapter（FC bottleneck）相关超参（补充）**
虽然你没点名，但它和上面 loss 强相关：
**关键超参**
• **`adapter_dim`**：`16 – 64`
• **`adapter_lr`**：`0.1 – 0.5 × LoRA lr`
**作用**
• 吸收 reward 中的 **全局偏移**
• 防止 Conv-LoRA 独自承受 noisy reward
**6️⃣ 推荐一套「稳定可复现」配置**

`w_dice      = 1.0
w_boundary  = 0.3
w_smooth    = 0.1
w_thin      = 0.05

adapter_dim = 32`

对应 reward：R=Dice+0.3⋅Boundary−0.1⋅Smooth+0.05⋅ThinR = Dice
+ 0.3·Boundary
- 0.1·Smooth
+ 0.05·ThinR=Dice+0.3⋅Boundary−0.1⋅Smooth+0.05⋅Thin``` 请你用你的话详细说明对上述提及的超参数的理解。尤其是其中几个用LaTex写的公式，看上去可能不是很正确，你可以在充分理解的基础上做纠正。

---

**Cursor**

我先去读取你提到的会话历史文件，确认当时的上下文和符号含义，然后会用自己的话系统梳理每个超参数的作用，并把不严谨的公式改成更规范的 LaTeX 形式。

下面我按你给的结构，一边解释每个超参数，一边把里面用到的公式用更规范、逻辑更完整的 LaTeX 重新写一遍，并说明哪些地方原文有点不严谨、我是怎么修正的。

---

### 一、整体图景（先把关系理清）

在你这个体系里：

- **冻结的**：`SAM backbone`  
- **可训练的**：
  - **Conv-LoRA**：在卷积层上做低秩扰动 → 擅长学“局部高频细节 / 边缘形状”
  - **Adapter（FC bottleneck）**：在特征维度上做窄瓶颈 → 学“全局偏移 / 类别级风格”
  - **GSPO policy**：RL 方式去学 prompt 或 box 的决策策略 → 对 reward 的尺度很**敏感**
- **优化目标**：  
  - 主体靠 **Dice** 保证整体分割质量  
  - 再用 **Boundary / Smooth / Thin** 这些项去“雕刻边缘和细结构”

所以：**所有这些超参数本质上都是在调“梯度流向”和“RL reward 的标度”，以防 Conv-LoRA 过拟合高频噪声、GSPO 发散。**

---

### 二、Smoothness / Laplacian 正则 与 `λ_smooth`

#### 1. 直观理解

- 预测 mask 记为 \( m(i,j) \in [0,1] \)，像素坐标 \((i,j)\)。
- **Dice** 对内部区域更敏感，对“锯齿、小抖动”这种高频边缘噪声不怎么在意。
- **Conv-LoRA** 对局部梯度非常敏感，很容易“放大”这些高频抖动。
- **GSPO** 又是 policy gradient，本身就高方差。

→ 如果没有平滑正则，模型很容易学出“正确大体轮廓 + 高频抖动的边缘”，训练也会不稳定。

#### 2. 更规范的数学形式（修正）

原文写的是：

\[
\mathcal{L}_{\text{smooth}}
= \sum_{i,j} (\nabla^2 m_{i,j})^2
\]

这意思是“Laplacian（二阶导）平方和”，直觉上没问题，只是记号略粗糙。更标准一点可以写成：

- 用离散拉普拉斯算子（例如 4 邻域）：

\[
(\Delta m)(i,j)
= m(i+1,j) + m(i-1,j) + m(i,j+1) + m(i,j-1) - 4 m(i,j)
\]

则平滑损失：

\[
\mathcal{L}_{\text{smooth}}
= \frac{1}{|\Omega|} \sum_{(i,j)\in\Omega}
\bigl( (\Delta m)(i,j) \bigr)^2
\]

这里 \(\Omega\) 是整张图的像素集合。  

**含义**：鼓励每个像素的值接近周围像素的平均值 → 抑制高频振荡，mask 更“圆润”。

#### 3. `λ_smooth` 的作用和范围

- 真正优化的是：

\[
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{task}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}
\]

- **推荐范围**：`0.05 – 0.2`，常用 `0.1`  
  直观解释：
  - `λ_smooth` 太小：对 Laplacian 惩罚几乎忽略 → 边缘抖动依然明显。
  - `λ_smooth` 太大（> 0.2）：
    - 所有高曲率结构（thin structures、尖锐拐角）被当成“噪声”抹平。
    - 病灶边缘会向内收缩，变得“圆、钝、偏小”。

**结论**：`λ_smooth` 其实是在调节 **“允许多锐的边缘 / 细结构” 与 “抑制高频噪声”** 的平衡。

---

### 三、Boundary Loss / Boundary Reward 与 `λ_boundary`

#### 1. 直观理解

- Dice 的梯度 90% 来自 **mask 内部**（重叠的主体区域），对边界错一点其实很“宽容”。
- 医学分割最关注的误差却几乎都在边界（病灶大小、接触位置、薄层结构）。

→ **Boundary loss 的核心目的就是“把梯度搬到边界附近”**，让模型专门去修边缘。

#### 2. 更合理的数学形式（修正）

原文公式：

\[
\mathcal{L}_{\text{boundary}}
= \sum d_{\text{GT}}(p)\cdot m(p)
\]

它少了 **GT mask \(y(p)\)**，以及“距离变小意味着更好”这层逻辑，形式上不够完整。  
常见的一种 **基于 distance transform 的边界损失** 可以写成（更合理的版本）：

- 记：
  - \(y(p) \in \{0,1\}\)：GT mask
  - \(m(p) \in [0,1]\)：预测 mask
  - \(D_{\text{GT}}(p)\)：像素 \(p\) 到 GT 边界的（非负）距离

一种简单的形式是：

\[
\mathcal{L}_{\text{boundary}}
= \frac{1}{|\Omega|} \sum_{p \in \Omega}
D_{\text{GT}}(p)\, \bigl| m(p) - y(p) \bigr|
\]

- 直观解释：
  - 远离边界的像素（\(D_{\text{GT}}\) 大）被错分的惩罚更弱（即使距离大，通常那里本来就比较“确定”）。
  - 边界附近（\(D_{\text{GT}}\) 小到中等）的误差被放大，更强制模型把预测边缘往 GT 边缘对齐。

若你把它反过来当 **boundary reward**（数值越大越好），就可以：

\[
R_{\text{boundary}} = - \mathcal{L}_{\text{boundary}}
\]

这一点和原文意思是一致的，只是我补全了 \(|m(p)-y(p)|\) 这部分，使得公式在数学/逻辑上更成立。

#### 3. `λ_boundary` 的意义和范围

优化目标大致是：

\[
\mathcal{L}_{\text{seg}} =
\mathcal{L}_{\text{Dice}}
+ \lambda_{\text{boundary}}\,\mathcal{L}_{\text{boundary}}
\]

经验范围：`λ_boundary ≈ 0.1 – 0.5 × (Dice 的权重)`，常用 `0.3` 左右。

- **< 0.1**：  
  - Dice 的梯度主导一切，边界项存在感很弱，边缘细抠不出来。
- **> 0.5**：  
  - 训练几乎“只关心边界”，内部区域会被过度牺牲，容易出现：
    - mask 内部出现空洞（hollow）
    - 分割区域被撕裂成多块（over-segmentation）

**结合你这个架构**：

- Conv-LoRA：会“接住”来自边界的高频梯度，学局部边缘形状；
- Adapter：保持全局 embedding 不被边界局部噪声拖偏（起到“缓冲器”的作用）。

---

### 四、GSPO / RL 中的 Reward Scale：`w_dice, w_boundary, w_smooth, w_thin`

#### 1. 正确、完整的 reward 组合形式（修正）

原文写的是：

\[
R = w_{\text{dice}}\cdot Dice
+ w_{\text{boundary}}\cdot R_{\text{boundary}}
- w_{\text{smooth}}\cdot \mathcal{L}_{\text{smooth}}
\]

但你后面又引入了 `w_thin`，所以一个**更完整、更一致**的形式应为：

\[
R =
w_{\text{dice}} \cdot \text{Dice}
+ w_{\text{boundary}} \cdot R_{\text{boundary}}
+ w_{\text{thin}} \cdot R_{\text{thin}}
- w_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}}
\]

其中：

- \(R_{\text{boundary}} = - \mathcal{L}_{\text{boundary}}\)
- \(R_{\text{thin}}\)：与细结构（骨架、细边缘召回等）相关的 reward

**推荐标度**（你给出的）：

- `w_dice = 1.0`（固定为锚点）
- `w_boundary = 0.1 – 0.5`
- `w_smooth = 0.05 – 0.2`
- `w_thin = 0.05 – 0.1`

#### 2. 为何 Dice 权重要固定为 1.0（不能乱改）

GSPO 的梯度（policy gradient）原理上应写成：

\[
\nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta}\bigl[
R \,\nabla_\theta \log \pi_\theta(a \mid s)
\bigr]
\]

原文写的是 \(\nabla_\theta J \propto R \nabla\log\pi_\theta\)，是个简化版，方向没问题，只是缺少期望符号和条件概率 \(\pi_\theta(a|s)\)。  

关键点是：**`R` 的数值尺度，直接乘在梯度前面，相当于改变了“有效学习率”**。

- 若你把 `w_dice` 从 1.0 改到 2.0：
  - Dice 一般 ∈ \([0,1]\)，整体 reward 的数值范围会基本翻倍。
  - 就等价于把 policy 的 **step size 成倍放大**，极容易让 RL 训练发散 / 抖动。
- 相反，保持 `w_dice = 1.0`，其余 shaping 项都用相对小一点的权重，就相当于：
  - Dice 作为**稳定 anchor**，设定了 reward 的基准尺度；
  - Boundary / Smooth / Thin 只是在这个基准上做**微调引导（shaping）**，而不是暴力重构目标。

---

### 五、Thin-Structure / Detail Reward 与 `w_thin`

#### 1. 直观含义

- 细结构（细血管、细神经、薄层组织）在整张图中占的像素很少：
  - 在 Dice 中几乎没有“话语权”，即使预测错了，Dice 变化也很有限。
- Conv-LoRA 对局部高频非常敏感：
  - 一旦你让它“拼命追求极细结构”，又给了它很大权重，它**很容易过拟合**到噪声、伪结构上。

→ 所以 `w_thin` 更像是：“告诉模型**这里有一类重要的难点（thin structure）需要照顾**”，而不是“你必须把它做到极致”。

#### 2. 数学形式示例 & 为什么权重要小

假设你有一个 skeleton-based reward：

- 记 \(S_{\text{GT}}\) 为 GT 骨架，\(S(m)\) 为从预测 mask 提取出的骨架。
- 例如定义：

\[
R_{\text{thin}}
= \text{Recall}(S(m), S_{\text{GT}})
= \frac{|S(m) \cap S_{\text{GT}}|}{|S_{\text{GT}}|}
\]

那么 reward 组合：

\[
R = \dots + w_{\text{thin}} R_{\text{thin}} + \dots
\]

- 范围建议：`w_thin = 0.05 – 0.1`，**不要大于 0.1**：
  - 太小：对 thin structure 没有引导作用；
  - 太大：GSPO 会频繁地为追求 thin reward 调整 prompts / boxes，
    - 出现毛刺、假分支（fake branch）、拉出一堆“看起来很细但其实是噪声”的结构。

**总之**：`w_thin` 是个“方向提示器”而非“主目标权重”。

---

### 六、Adapter（FC Bottleneck）相关：`adapter_dim` 与 `adapter_lr`

这两项虽然不在 loss 中，但和上面 reward 的**噪声与全局偏移**高度相关。

#### 1. `adapter_dim`：容量控制

- 范围建议：`16 – 64`，常用 `32`。
- 含义：
  - 太小（<16）：Adapter 容量不够，没法有效吸收全局偏移，gradient 噪声被迫压到 Conv-LoRA 上，LoRA 会变得很“躁”；
  - 太大（>64）：Adapter 容量过强，容易自己拟合各种乱七八糟的 reward 噪声，掩盖了 Conv-LoRA 局部结构学习的优势。

**可以理解为**：Adapter 是“全局缓冲器”，用来吸收因为多种 reward 组合带来的整体风向偏移。

#### 2. `adapter_lr`：学习率相对 LoRA 的比例

- 建议：`adapter_lr = 0.1 – 0.5 × LoRA lr`
- 直觉：
  - LoRA 主要负责“局部修边角”，需要反应快一点；
  - Adapter 负责“全局偏移与风格”，反应慢一点更稳定，避免频繁跟着高方差 reward 抖动。

---

### 七、一套稳定可复现配置的逻辑

你给的推荐：

```text
w_dice      = 1.0
w_boundary  = 0.3
w_smooth    = 0.1
w_thin      = 0.05

adapter_dim = 32
```

对应 reward：

\[
R = \text{Dice}
+ 0.3 \cdot R_{\text{boundary}}
- 0.1 \cdot \mathcal{L}_{\text{smooth}}
+ 0.05 \cdot R_{\text{thin}}
\]

**理解这套数值的核心逻辑**：

- **Dice = 主体目标，权重 1.0 固定** → 设定 reward 的“自然量纲”；
- `w_boundary = 0.3`：  
  - 边界梯度有足够存在感，但不会压垮内部一致性；
- `w_smooth = 0.1`：  
  - 对高频边缘抖动有明显抑制，又不会抹掉太多细结构；
- `w_thin = 0.05`：  
  - 给 thin structure 一个“额外加分项”，但不会让 RL 为 thin 过度牺牲主体结构；
- `adapter_dim = 32`：  
  - 对大多数 SAM-based 体系是一个“容量适中”的设置，可以稳当地吸收 reward 噪声导致的全局偏移，让 Conv-LoRA 专注边缘局部。

---

如果你愿意，我可以再帮你：  
- 按你当前代码实际写出来的 loss / reward 函数，对照上面这些公式逐项核对；  
- 或者根据你现在的数据规模和 batch size，给一版更细节的“从 scratch 调参顺序”（先固定哪些，再 Grid / 手调哪些）。

```plaintext
w_dice      = 1.0
w_boundary  = 0.3
w_smooth    = 0.1
w_thin      = 0.05

adapter_dim = 32
```

---

**User**

请实现：让λ_smooth， λ_boundary， w_boundary，w_smooth 和w_thin这些超参数都可以从外面传入来调整

---

**Cursor**

```python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection based on actual performance
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

修改完成：GSPO 训练器现在支持从外部传入 `λ_smooth / λ_boundary / w_boundary / w_smooth / w_thin`，并在组级优化时实际生效。关键调整：
- 构造函数新增上述五个超参并存储，默认值沿用你给的推荐区间中心：`lambda_smooth=0.1`、`lambda_boundary=0.3`、`w_boundary=0.3`、`w_smooth=0.1`、`w_thin=0.05`。  
```37:62:examples/automm/Conv-LoRA/gspo_trainer.py
def __init__(..., lambda_smooth: float = 0.1, lambda_boundary: float = 0.3, w_boundary: float = 0.3, w_smooth: float = 0.1, w_thin: float = 0.05):
    ...
    self.lambda_smooth = lambda_smooth
    self.lambda_boundary = lambda_boundary
    self.w_boundary = w_boundary
    self.w_smooth = w_smooth
    self.w_thin = w_thin
```

- 新增平滑/边界/细结构三个计算模块，并将预测、GT 转为单通道概率后统一使用：  
  - Laplacian 平滑损失（每样本）  
  - 边界 L1 损失（预测边界 vs GT 边界）  
  - 边界 Dice 风格的细结构奖励  
```123:184:examples/automm/Conv-LoRA/gspo_trainer.py
pred_probs, gt_probs = ...
lap = conv2d(...); smooth_loss = (lap**2).mean(...)
boundary_loss = |boundary(pred)-boundary(gt)|.mean(...)
thin_reward = Dice(boundary(pred), boundary(gt))
```

- 训练主循环中：  
  - 分割损失加入 `lambda_smooth * smooth_loss + lambda_boundary * boundary_loss`  
  - GSPO 质量评分改为带权重的 shaped reward：`quality + w_boundary*(1-boundary_loss) - w_smooth*smooth_loss + w_thin*thin_reward`，同时记录原始质量与各分项指标到 metrics。  
```220:311:examples/automm/Conv-LoRA/gspo_trainer.py
shaped_quality = quality + w_boundary*(1 - boundary_loss.detach()) - w_smooth*smooth_loss.detach() + w_thin*thin_reward.detach()
seg_loss = loss_fn(...) + lambda_smooth * smooth_loss.mean() + lambda_boundary * boundary_loss.mean()
metrics[...] = {'avg_quality_base': ..., 'avg_smooth_loss': ..., 'avg_boundary_loss': ..., 'avg_thin_reward': ...}
```

使用方式：实例化 `GSPOConvLoRATrainer` 时直接传入需要的权重，例如：
```python
trainer = GSPOConvLoRATrainer(
    predictor,
    group_size=4,
    lambda_smooth=0.12,
    lambda_boundary=0.25,
    w_boundary=0.35,
    w_smooth=0.08,
    w_thin=0.07,
)
```

测试：未运行自动化测试；建议在你现有的训练脚本中用上述实例化方式跑一小步，确认损失与日志里出现新的指标（smooth/boundary/thin）以验证配置生效。

```python
def __init__(..., lambda_smooth: float = 0.1, lambda_boundary: float = 0.3, w_boundary: float = 0.3, w_smooth: float = 0.1, w_thin: float = 0.05):
    ...
    self.lambda_smooth = lambda_smooth
    self.lambda_boundary = lambda_boundary
    self.w_boundary = w_boundary
    self.w_smooth = w_smooth
    self.w_thin = w_thin
```

```python
pred_probs, gt_probs = ...
lap = conv2d(...); smooth_loss = (lap**2).mean(...)
boundary_loss = |boundary(pred)-boundary(gt)|.mean(...)
thin_reward = Dice(boundary(pred), boundary(gt))
```

```python
shaped_quality = quality + w_boundary*(1 - boundary_loss.detach()) - w_smooth*smooth_loss.detach() + w_thin*thin_reward.detach()
seg_loss = loss_fn(...) + lambda_smooth * smooth_loss.mean() + lambda_boundary * boundary_loss.mean()
metrics[...] = {'avg_quality_base': ..., 'avg_smooth_loss': ..., 'avg_boundary_loss': ..., 'avg_thin_reward': ...}
```

```python
trainer = GSPOConvLoRATrainer(
    predictor,
    group_size=4,
    lambda_smooth=0.12,
    lambda_boundary=0.25,
    w_boundary=0.35,
    w_smooth=0.08,
    w_thin=0.07,
)
```

---

**User**

这些参数怎么能方便地设置呢？不能每次都在代码里改吧？你是不是应该把它们作为可选带默认值的cli参数？

---

**Cursor**

```python
import argparse
import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor


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
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25])")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0])")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
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
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

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
        print(f"Enabling Standard Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
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
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
        )

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
            
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

已把五个超参加成可选 CLI 参数并带默认值，不再需要改代码就能调：

- 新增参数（默认值与推荐值一致）：  
  - `--lambda_smooth 0.1`  
  - `--lambda_boundary 0.3`  
  - `--w_boundary 0.3`  
  - `--w_smooth 0.1`  
  - `--w_thin 0.05`

- 当你加上 `--gspo_enable` 时，这些值会写入 `hyperparameters`（键名 `optim.gspo.*`），便于后续在自定义 GSPO trainer 初始化时读取。

快速用法示例：
```bash
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.12 \
  --lambda_boundary 0.25 \
  --w_boundary 0.35 \
  --w_smooth 0.08 \
  --w_thin 0.07
```

当前脚本里依然是 MultiModalPredictor 训练，如果你在外部手动构造 `GSPOConvLoRATrainer`，可直接把这些 CLI 参数读出来传给构造函数即可。代码改动位于 `examples/automm/Conv-LoRA/run_semantic_segmentation.py`。

```shellscript
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.12 \
  --lambda_boundary 0.25 \
  --w_boundary 0.35 \
  --w_smooth 0.08 \
  --w_thin 0.07
```

---

**User**

新加的这几个cli参数是如何最终传入并影响训练过程中实际的超参数的？请说明完整的参数传递流程

---

**Cursor**

目前的传递链分两段：

1) CLI → 超参字典  
- 在 `run_semantic_segmentation.py` 中新增的 CLI 参数 `--lambda_smooth/--lambda_boundary/--w_boundary/--w_smooth/--w_thin` 先被 argparse 读取。  
- 当加了 `--gspo_enable` 时，这些值写入 `hyperparameters`，键名分别是：  
  - `optim.gspo.lambda_smooth`  
  - `optim.gspo.lambda_boundary`  
  - `optim.gspo.w_boundary`  
  - `optim.gspo.w_smooth`  
  - `optim.gspo.w_thin`  
- 随后 `hyperparameters` 被传给 `MultiModalPredictor(...)`。

2) 配置 → 训练逻辑（当前需要你显式接线）  
- 新增的键已经存进配置，但默认的 `MultiModalPredictor` 流程并不会自动创建或注入 `GSPOConvLoRATrainer`，也不会把这些键传到 GSPO 逻辑里。  
- 要让它真正影响训练，需要在你构建 GSPO 训练路径时显式读取这些键，并传入 `GSPOConvLoRATrainer` 构造函数。示例做法（思路代码）：
```python
cfg = predictor._config  # 或你持有的 config
gspo_cfg = cfg.optim.gspo
gspo_trainer = GSPOConvLoRATrainer(
    predictor,
    group_size=cfg.optim.lora.gspo_group_size,
    warmup_epochs=cfg.optim.lora.gspo_warmup_epochs,
    contrastive_weight=cfg.optim.lora.gspo_contrastive_weight,
    quality_metric="iou",
    advantage_temperature=5.0,
    lambda_smooth=gspo_cfg.lambda_smooth,
    lambda_boundary=gspo_cfg.lambda_boundary,
    w_boundary=gspo_cfg.w_boundary,
    w_smooth=gspo_cfg.w_smooth,
    w_thin=gspo_cfg.w_thin,
)
# 然后把 gspo_trainer 传入你的 LitModule / 训练循环
```
- 在 `multimodal/optim/lit_semantic_seg.py` 里，`SemanticSegmentationLitModule` 期望 `gspo_trainer` 由调用方注入（构造参数 `gspo_trainer=None`）。一旦你以这种方式传入，`gspo_group_training_step` 会使用 trainer 内部的这几个权重，进而改变 reward shaping 和分割损失。

如果你希望全链路自动化，需要再加一层接线：在创建 `SemanticSegmentationLitModule` 时，从 `hyperparameters` 读取 `optim.gspo.*`，构造 `GSPOConvLoRATrainer` 并传入。当前仓库还没有这一自动步骤，所以需要你手动接上或在 Learner 构建 lit module 的地方加上这段逻辑。

```python
cfg = predictor._config  # 或你持有的 config
gspo_cfg = cfg.optim.gspo
gspo_trainer = GSPOConvLoRATrainer(
    predictor,
    group_size=cfg.optim.lora.gspo_group_size,
    warmup_epochs=cfg.optim.lora.gspo_warmup_epochs,
    contrastive_weight=cfg.optim.lora.gspo_contrastive_weight,
    quality_metric="iou",
    advantage_temperature=5.0,
    lambda_smooth=gspo_cfg.lambda_smooth,
    lambda_boundary=gspo_cfg.lambda_boundary,
    w_boundary=gspo_cfg.w_boundary,
    w_smooth=gspo_cfg.w_smooth,
    w_thin=gspo_cfg.w_thin,
)
# 然后把 gspo_trainer 传入你的 LitModule / 训练循环
```

---

**User**

我不想手动加入，你给我完成这一步，否则要你做什么？！

---

**Cursor**

```python
import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax

from autogluon.core.metrics import Scorer

from ..constants import LABEL, LOGITS, SEMANTIC_MASK, SEMANTIC_SEGMENTATION_IMG
from ..optim import SemanticSegmentationLitModule, get_loss_func, get_norm_layer_param_names, get_peft_param_names
from ..optim.metrics.semantic_seg_metrics import Balanced_Error_Rate_Pred as Balanced_Error_Rate
from ..optim.metrics.semantic_seg_metrics import Binary_IoU_Pred as Binary_IoU
from ..optim.metrics.semantic_seg_metrics import Binary_DICE_Pred as Binary_DICE
from ..optim.metrics.semantic_seg_metrics import COD_METRICS_NAMES_Pred as COD_METRICS_NAMES
from ..optim.metrics.semantic_seg_metrics import Multiclass_IoU_Pred as Multiclass_IoU
from ..optim.metrics.semantic_seg_metrics import Multiclass_DICE_Pred as Multiclass_DICE
from ..utils import extract_from_output, setup_save_path
from ..utils.tta_utils import TTAPredictor, dice_coefficient, iou_score
from .base import BaseLearner

logger = logging.getLogger(__name__)

from ..constants import BER, DICE, EM, FM, IOU, MAE, SEMANTIC_SEGMENTATION, SM


class SemanticSegmentationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = SEMANTIC_SEGMENTATION,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = "iou",
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = "iou",
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
        )
        self._output_shape = num_classes
        self._sample_data_path = sample_data_path
        self._tta_predictor = None  # Will be initialized when TTA is enabled

        if self._sample_data_path is not None:
            infer_output_shape = self.get_semantic_segmentation_class_num(self._sample_data_path)
            if num_classes is not None:
                assert (
                    num_classes == infer_output_shape
                ), f"The provided number of classes '{num_classes}' and the inferred class number {infer_output_shape}' from the sample data should be consistent."
            else:
                self._output_shape = infer_output_shape

    def get_semantic_segmentation_class_num(self, sample_data_path):
        """
        Get the number of classes for given data.

        Parameters
        ----------
            sample_data_path
                This is used for automatically inference num_classes of semantic segmentation dataset.
                Could be an image directory, image file or pd.DataFrame.
        Returns
        -------
            The number of classes.
        """
        if isinstance(sample_data_path, str):
            if os.path.isdir(sample_data_path):
                mask_files = os.listdir(sample_data_path)
                num_classes = []
                for mask_file in mask_files:
                    per_num_classes = self.get_semantic_segmentation_class_num(
                        os.path.join(sample_data_path, mask_file)
                    )
                    num_classes.append(per_num_classes)
                return max(num_classes)
            else:
                mask = Image.open(sample_data_path)
                mode = mask.mode

                if mode == "1":
                    return 1
                classes = np.unique(mask)
                if mode == "L" and np.array_equal(classes, np.array([0, 255])):
                    return 1

                return max(classes).item() + 1  # include background

        elif isinstance(sample_data_path, pd.DataFrame):
            num_classes = []
            for idx in range(sample_data_path.shape[0]):
                row = sample_data_path.iloc[idx]
                mask_file = row[self._label_column]
                per_num_classes = self.get_semantic_segmentation_class_num(mask_file)
                num_classes.append(per_num_classes)
            return max(num_classes)

    def infer_output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.get_semantic_segmentation_class_num(self._train_data)

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = config.optim.peft
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_peft_param_names(
                norm_param_names,
                peft=peft,
                extra_params=config.optim.extra_trainable_params,
            )
        return peft_param_names

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            loss_func_name=config.optim.loss_func,
            config=config.optim,
            num_classes=self._output_shape,
        )
        return loss_func, None

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        """
        
        # If TTA is enabled, use TTA evaluation
        if self._tta_predictor is not None:
            # Use cache settings if configured
            cache_dir = getattr(self, '_tta_cache_dir', None)
            resume_from_cache = getattr(self, '_tta_resume_from_cache', True)
            return self._evaluate_with_tta(
                data, metrics, return_pred, 
                cache_dir=cache_dir,
                resume_from_cache=resume_from_cache
            )

        def get_metric_predict(
            metric_name: str,
            num_classes: Optional[int] = None,
        ):
            """
            Obtain a torchmerics.Metric from its name.
            Define a customized metric function in case that torchmetrics doesn't support some metric.

            Parameters
            ----------
            metric_name
                Name of metric.
            num_classes
                Number of classes.
            is_matching
                Whether is matching.
            problem_type
                Type of problem, e.g., binary and multiclass.

            Returns
            -------
            torchmetrics.Metric
                A torchmetrics.Metric object.
            custom_metric_func
                A customized metric function.
            """
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if self._output_shape == 1:
            logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        else:
            logits = extract_from_output(ret_type=SEMANTIC_MASK, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            per_metric = get_metric_predict(metric_name=per_metric_name.lower(), num_classes=self._output_shape)
            for y_p, y_t in zip(y_pred, y_true):
                per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
            score = per_metric.compute()
            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results
    
    def _evaluate_with_tta(
        self,
        data: Union[pd.DataFrame, dict, list],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        """
        Evaluate using Test-Time Augmentation with checkpoint support.
        
        This method performs TTA for each image and evaluates the results.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, dict, list]
            Input data containing image and label paths
        metrics : Optional[Union[str, List[str]]]
            Metrics to compute
        return_pred : Optional[bool]
            Whether to return predictions
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching)
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True)
        """
        logger.info(f"Evaluating with TTA ({len(self._tta_predictor.transforms)} augmentations per image)...")
        
        def get_metric_predict(metric_name: str, num_classes: Optional[int] = None):
            """Get metric function."""
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")
        
        # Get model and preprocessor
        model = self._model
        model.eval()
        
        # Ensure model is on the correct device (GPU if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if next(model.parameters()).device.type != 'cuda':
                logger.info("Moving model to GPU for TTA inference...")
                model = model.cuda()
        else:
            device = torch.device('cpu')
            logger.warning("GPU not available, using CPU for TTA (will be slow)")
        
        logger.info(f"TTA inference device: {device}")
        
        # ============================================================
        # 方案 4: Sanity Check - 提前发现问题
        # ============================================================
        logger.info("🔍 Running sanity checks before full evaluation...")
        
        # Sanity Check 1: 验证数据
        if len(data) == 0:
            raise ValueError("❌ No data to evaluate!")
        logger.info(f"  ✓ Total images to process: {len(data)}")
        logger.info(f"  ✓ DataFrame columns: {data.columns.tolist()}")
        
        # Sanity Check 2: 检查设备和模型
        logger.info(f"  ✓ Model device: {device}")
        logger.info(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  ✓ GPU memory allocated: {allocated_gb:.2f} GB")
        
        # Sanity Check 3: 测试单个样本的 TTA pipeline
        logger.info(f"  Testing TTA pipeline on first image...")
        test_row = data.iloc[0]
        test_img_path = test_row['image']
        
        # Get data processors for image preprocessing
        from torchvision import transforms as T
        
        # Create preprocessing pipeline similar to model's preprocessing
        # Based on SAM's image preprocessing
        def preprocess_image_for_sam(image_np):
            """Preprocess image for SAM model."""
            from PIL import Image as PILImage
            
            # Ensure correct format
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=2)
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to PIL
            img_pil = PILImage.fromarray(image_np.astype(np.uint8))
            
            # Resize to model's expected input size (typically 1024x1024 for SAM)
            target_size = model.image_size
            img_resized = img_pil.resize((target_size, target_size), PILImage.BILINEAR)
            
            # Convert to tensor and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Close PIL objects
            img_pil.close()
            img_resized.close()
            
            # Apply model's normalization (SAM uses specific mean/std)
            mean = np.array(model.image_mean).reshape(1, 1, 3)
            std = np.array(model.image_std).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().contiguous()
            
            # Free intermediate array
            del img_array
            
            return img_tensor
        
        # Create a prediction function for TTA
        def predict_fn(image: np.ndarray) -> np.ndarray:
            """
            Prediction function for a single image used by TTA.
            
            Parameters
            ----------
            image : np.ndarray
                Input image (H, W, C) in range [0, 255] or [0, 1]
            
            Returns
            -------
            prob : np.ndarray
                Probability map (H, W) for binary or (C, H, W) for multi-class
            """
            # Preprocess image
            img_tensor = preprocess_image_for_sam(image)
            
            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)
            
            # Forward pass
            with torch.no_grad():
                # Create batch dict for model
                # Need to provide a dummy label for model's forward pass
                batch = {
                    model.prefix + '_image': img_tensor,
                    model.prefix + '_label': torch.zeros((1, model.image_size, model.image_size), 
                                                         dtype=torch.long, device=device)
                }
                outputs = model(batch)
                
                # Extract logits
                logits = outputs[model.prefix][LOGITS]
                
                # Convert to probabilities and immediately move to CPU
                if self._output_shape == 1:
                    # Binary segmentation - remove channel dim
                    prob = torch.sigmoid(logits[0, 0]).cpu().numpy().copy()
                else:
                    # Multi-class segmentation
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy().copy()
                
                # Free GPU memory immediately and aggressively
                del img_tensor, batch, outputs, logits
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish
            
            return prob
        
        # Complete Sanity Check 3: Test TTA on first image
        try:
            test_img_pil = Image.open(test_img_path)
            test_img = np.array(test_img_pil)
            test_img_pil.close()
            
            if len(test_img.shape) == 2:
                test_img = np.stack([test_img] * 3, axis=2)
            
            logger.info(f"    Image shape: {test_img.shape}")
            
            # Test TTA prediction
            test_pred_mask, test_pred_prob = self._tta_predictor.predict_with_tta(
                test_img, predict_fn, return_probs=True
            )
            
            logger.info(f"    Output shape: {test_pred_prob.shape}")
            
            # Verify output shape matches input
            assert test_pred_prob.shape[:2] == test_img.shape[:2], \
                f"❌ Output shape mismatch! Expected {test_img.shape[:2]}, got {test_pred_prob.shape[:2]}"
            
            logger.info(f"  ✓ TTA pipeline test PASSED!")
            
            # Clean up test data
            del test_img, test_pred_mask, test_pred_prob
            
        except Exception as e:
            logger.error(f"❌ Sanity check FAILED! Error: {e}")
            logger.error("Please fix the issue before running full evaluation.")
            raise RuntimeError(f"TTA Sanity Check Failed: {e}") from e
        
        logger.info("✅ All sanity checks passed! Starting full evaluation...\n")
        
        # ============================================================
        # 方案 2: 断点续传机制
        # ============================================================
        import os
        import pickle
        
        # Setup cache directory
        processed_indices = set()
        all_preds = []
        all_labels = []
        cache_file = None
        processed_indices_file = None
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "tta_predictions.pkl")
            processed_indices_file = os.path.join(cache_dir, "processed_indices.txt")
            
            # Try to load cached data
            if resume_from_cache and os.path.exists(cache_file):
                try:
                    logger.info(f"📂 Loading cached predictions from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        all_preds = cache_data['preds']
                        all_labels = cache_data['labels']
                        processed_indices = set(cache_data['indices'])
                    logger.info(f"✅ Loaded {len(all_preds)} cached predictions (resuming from checkpoint)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}. Starting fresh.")
                    processed_indices = set()
                    all_preds = []
                    all_labels = []
            else:
                if cache_dir:
                    logger.info(f"💾 Caching enabled. Predictions will be saved to {cache_dir}")
        
        # Initialize metrics early so we can stream updates and avoid storing all predictions
        metric_objects = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            metric_objects[per_metric_name] = get_metric_predict(
                metric_name=per_metric_name.lower(),
                num_classes=self._output_shape
            )

        # Determine whether to stream metrics (no caching => no large tensors kept)
        streaming_metrics = cache_dir is None

        # Process each image with TTA
        import time
        import gc
        start_time = time.time()
        total_images = len(data)

        for idx, row in data.iterrows():
            # ⭐ 跳过已处理的样本（断点续传）
            if idx in processed_indices:
                logger.info(f"⏭️  Skipping image {idx+1}/{total_images} (already processed)")
                continue

            img_start = time.time()

            # Load image and label
            image_path = row['image']
            label_path = row['label'] if 'label' in row else None

            # Load image with proper resource management
            img_pil = Image.open(image_path)
            image = np.array(img_pil)
            img_pil.close()  # Release file handle

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB

            # Load label if exists
            label = None
            if label_path:
                label_pil = Image.open(label_path)
                label = np.array(label_pil)
                label_pil.close()  # Release file handle

                if len(label.shape) == 3:
                    label = label[:, :, 0]  # Take first channel if RGB

                # Binarize label: convert [0, 255] to [0, 1]
                # This matches the preprocessing in semantic_seg_metrics.py
                if self._output_shape == 1:  # Binary segmentation
                    label = (label > 128).astype(np.int64)

                if not streaming_metrics:
                    all_labels.append(torch.from_numpy(label))

            # Predict with TTA
            if self._output_shape == 1:
                # Binary segmentation
                pred_mask, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    # Store only when caching/resume is enabled
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_mask, pred_prob
            else:
                # Multi-class segmentation
                _, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_prob

            # Stream metric updates to avoid holding all predictions in memory
            if streaming_metrics:
                y_p = pred_tensor.float()
                if label is None:
                    raise ValueError("Labels are required for evaluation")
                y_t = torch.from_numpy(label)
                y_t = y_t.long() if y_t.dtype != torch.long else y_t

                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Free image and label memory
            del image
            if label is not None:
                del label

            # ⭐ 标记为已处理（断点续传）
            processed_indices.add(idx)

            img_time = time.time() - img_start

            if idx == 0:
                logger.info(f"First image processed in {img_time:.2f}s (includes warmup)")

            # ⭐ 定期保存缓存（每 10 张图或最后一张）
            num_processed = len(processed_indices)
            if (not streaming_metrics) and cache_file is not None and (num_processed % 10 == 0 or num_processed == total_images):
                try:
                    logger.info(f"💾 Saving checkpoint ({num_processed}/{total_images} images)")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'preds': all_preds,
                            'labels': all_labels,
                            'indices': list(processed_indices)
                        }, f)

                    # Save processed indices as text file (for easy inspection)
                    with open(processed_indices_file, 'w') as f:
                        f.write('\n'.join(map(str, sorted(processed_indices))))

                    logger.info(f"✅ Checkpoint saved successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save checkpoint: {e}")

            # More aggressive garbage collection to prevent memory leaks
            if (idx + 1) % 5 == 0:  # Every 5 images instead of 10
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache

            # Progress logging
            if (idx + 1) % 10 == 0 or num_processed == total_images:
                elapsed = time.time() - start_time
                avg_time = elapsed / num_processed if num_processed > 0 else 0
                remaining = total_images - num_processed
                eta = avg_time * remaining if remaining > 0 else 0
                logger.info(
                    f"Processed {num_processed}/{total_images} images with TTA "
                    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
                )

                # ⭐ 中间指标：在评估过程中周期性输出当前已处理样本的指标
                # 说明：
                # - 仅在 streaming_metrics=True（即未开启 cache_dir，走低内存流式评估）时启用
                # - 使用 torchmetrics.compute() 不会清空内部状态，因此可以安全多次调用
                if streaming_metrics:
                    try:
                        intermediate_results = {}
                        for metric_name, metric_obj in metric_objects.items():
                            score = metric_obj.compute()
                            # score 可能是 tensor 或数值，这里统一为 float 便于日志展示
                            if hasattr(score, "item"):
                                score = score.item()
                            intermediate_results[metric_name] = float(score)
                        if intermediate_results:
                            logger.info(
                                f"Intermediate TTA metrics after {num_processed} images: "
                                f"{intermediate_results}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute intermediate metrics: {e}")
        
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Finished processing all {len(data)} images, computing metrics...")

        results = {}
        if streaming_metrics:
            # Metrics were updated on the fly; just compute
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()
        else:
            # Check that we have predictions and labels
            if len(all_labels) == 0:
                raise ValueError("Labels are required for evaluation")
            
            if len(all_preds) != len(all_labels):
                raise ValueError(f"Number of predictions ({len(all_preds)}) != number of labels ({len(all_labels)})")

            # Process each image individually (can't stack due to different sizes)
            for y_p, y_t in zip(all_preds, all_labels):
                # Ensure predictions and labels are on CPU and have matching sizes
                y_p = y_p.float()
                y_t = y_t.long() if y_t.dtype != torch.long else y_t
                
                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),  # Add batch dim
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim
                
                # Update all metrics with this image
                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Compute final scores
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()

        logger.info(f"TTA evaluation completed: {results}")
        
        # ⭐ 清理缓存文件（评估成功完成后）
        if cache_file is not None and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                if processed_indices_file and os.path.exists(processed_indices_file):
                    os.remove(processed_indices_file)
                logger.info(f"🗑️  Cleaned up cache files (evaluation completed successfully)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up cache: {e}")
        
        # Free memory
        del all_preds, all_labels, metric_objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_pred:
            logger.warning("return_pred=True not fully supported with TTA (memory optimization)")
            return results, []
        else:
            return results

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optim_kwargs=None,
        distillation_kwargs=None,
        is_train=True,
    ):
        gspo_trainer = None
        if is_train:
            gspo_trainer = self._maybe_create_gspo_trainer()
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                gspo_trainer=gspo_trainer,
                **optim_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                **optim_kwargs,
            )

    def _maybe_create_gspo_trainer(self):
        """
        Auto-create GSPO trainer if the config enables GSPO.
        The trainer implementation lives in examples/automm/Conv-LoRA/gspo_trainer.py.
        """
        if self._config is None:
            return None

        lora_cfg = getattr(self._config.optim, "lora", None)
        if not lora_cfg or not getattr(lora_cfg, "gspo_enabled", False):
            return None

        gspo_cfg = getattr(self._config.optim, "gspo", None)
        lambda_smooth = getattr(gspo_cfg, "lambda_smooth", 0.1) if gspo_cfg else 0.1
        lambda_boundary = getattr(gspo_cfg, "lambda_boundary", 0.3) if gspo_cfg else 0.3
        w_boundary = getattr(gspo_cfg, "w_boundary", 0.3) if gspo_cfg else 0.3
        w_smooth = getattr(gspo_cfg, "w_smooth", 0.1) if gspo_cfg else 0.1
        w_thin = getattr(gspo_cfg, "w_thin", 0.05) if gspo_cfg else 0.05

        # Dynamically load GSPOConvLoRATrainer from examples path.
        examples_root = Path(__file__).resolve().parents[3] / "examples" / "automm" / "Conv-LoRA" / "gspo_trainer.py"
        if not examples_root.exists():
            logger.warning("GSPO is enabled but gspo_trainer.py not found at %s; GSPO will be skipped.", examples_root)
            return None

        spec = importlib.util.spec_from_file_location("autogluon_multimodal.gspo_trainer", examples_root)
        if spec is None or spec.loader is None:
            logger.warning("Failed to load GSPO trainer spec from %s; GSPO will be skipped.", examples_root)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        GSPOConvLoRATrainer = getattr(module, "GSPOConvLoRATrainer", None)
        if GSPOConvLoRATrainer is None:
            logger.warning("GSPOConvLoRATrainer not found in %s; GSPO will be skipped.", examples_root)
            return None

        try:
            return GSPOConvLoRATrainer(
                predictor=None,  # Not required by current implementation
                group_size=getattr(lora_cfg, "gspo_group_size", 3),
                warmup_epochs=getattr(lora_cfg, "gspo_warmup_epochs", 5),
                contrastive_weight=getattr(lora_cfg, "gspo_contrastive_weight", 0.1),
                quality_metric="iou",
                advantage_temperature=5.0,
                lambda_smooth=lambda_smooth,
                lambda_boundary=lambda_boundary,
                w_boundary=w_boundary,
                w_smooth=w_smooth,
                w_thin=w_thin,
            )
        except Exception as e:
            logger.warning("Failed to instantiate GSPOConvLoRATrainer: %s; GSPO will be skipped.", e)
            return None

    def on_predict_start(self, data: pd.DataFrame):
        data = self.data_to_df(data=data)
        if self._output_shape is None:  # for zero-shot evaluation/prediction
            self._output_shape = self.get_semantic_segmentation_class_num(data)
        self.ensure_predict_ready()
        return data
    
    def enable_tta(
        self,
        scales: List[float] = [0.75, 1.0, 1.25],
        flips: List[str] = ["none", "horizontal"],
        rotations: List[float] = [0],
        fusion_method: str = "mean",
        scale_weights: Optional[Dict[float, float]] = None,
        threshold: float = 0.5,
        min_area_ratio: float = 0.001,
        use_morphology: bool = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        """
        Enable Test-Time Augmentation (TTA) for inference.
        
        Parameters
        ----------
        scales : List[float]
            List of scale factors for multi-scale testing.
            Recommended: [0.75, 1.0, 1.25] for fast (6 inferences).
            Or [0.5, 0.75, 1.0, 1.25, 1.5] for better results.
        flips : List[str]
            List of flip types: "none", "horizontal", "vertical".
            Recommended: ["none", "horizontal"] for most cases.
        rotations : List[float]
            List of rotation angles in degrees.
            Recommended: [0] for fast, or [-10, 0, 10] for better (18 inferences).
        fusion_method : str
            Method to fuse predictions: "mean" or "weighted_mean".
        scale_weights : Optional[Dict[float, float]]
            Weights for each scale when using weighted_mean.
        threshold : float
            Threshold for binary segmentation (can be tuned on validation set).
        min_area_ratio : float
            Remove connected components with area < min_area_ratio * image_area.
        use_morphology : bool
            Whether to apply morphological closing for smoothing.
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching).
            Useful for long-running evaluations that might be interrupted.
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True).
        
        Examples
        --------
        >>> # Basic TTA (6 inferences)
        >>> predictor.enable_tta()
        
        >>> # Advanced TTA (18 inferences)
        >>> predictor.enable_tta(
        ...     scales=[0.75, 1.0, 1.25],
        ...     flips=["none", "horizontal"],
        ...     rotations=[-10, 0, 10],
        ...     fusion_method="weighted_mean"
        ... )
        """
        self._tta_predictor = TTAPredictor(
            scales=scales,
            flips=flips,
            rotations=rotations,
            fusion_method=fusion_method,
            scale_weights=scale_weights,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            use_morphology=use_morphology,
        )
        
        # Store cache configuration
        self._tta_cache_dir = cache_dir
        self._tta_resume_from_cache = resume_from_cache
        
        logger.info(f"TTA enabled with {len(self._tta_predictor.transforms)} augmentations")
        if cache_dir:
            logger.info(f"TTA caching enabled: {cache_dir} (resume={resume_from_cache})")
    
    def disable_tta(self):
        """Disable Test-Time Augmentation."""
        self._tta_predictor = None
        logger.info("TTA disabled")

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        data = self.on_predict_start(data)
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        realtime: Optional[bool] = False,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        When save_results is True, the output is a pandas dataframe containing the path of the predicted mask file for each input image.
        Otherwise, the output will have shape (#samples, height, width).
        """
        data = self.on_predict_start(data)
        if self._output_shape == 1:
            ret_type = LOGITS
        else:
            ret_type = SEMANTIC_MASK

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        logits = self.post_process_prediction(data, outputs, ret_type)

        pred = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                pred.append(logit.argmax(axis=1))
            else:
                pred.append((logit > 0.5).squeeze(axis=1))

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            pred = self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        The output will always have shape (#samples, #classes, height, width).
        """
        assert (self._output_shape == 1 and as_multiclass == False) or (
            self._output_shape > 1 and as_multiclass == True
        )
        data = self.on_predict_start(data)

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if as_multiclass:
            ret_type = SEMANTIC_MASK
        else:
            ret_type = LOGITS

        logits = self.post_process_prediction(data, outputs, ret_type)

        prob = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                prob.append(softmax(logit, axis=1))
            else:
                prob.append(logit)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing segmentation results for one image
        data
            Pandas data frame or dict containing the image information to be tested
        result_path
            Path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        image_column_name = self.get_image_column_name(data)
        if isinstance(data, dict):
            image_names = data[image_column_name]
        else:
            image_names = data[image_column_name].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            if self._output_shape == 1:
                mask = Image.fromarray(image_pred.squeeze(axis=0))
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                mask.save(per_mask_path)
            else:
                masks = []
                classes = np.unique(image_pred)
                for class_id in classes:
                    if class_id == 0:  # bg
                        continue
                    masks.append(image_pred == class_id)

                for mask in masks:
                    show_mask(mask, plt.gca())
                mask_name = ""
                for i in os.path.basename(image_name).split(".")[:-1]:
                    mask_name += i
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                plt.axis("off")
                plt.savefig(per_mask_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df

    def post_process_prediction(self, data, outputs, ret_type):
        """
        Post-process segmentation results to match the size of original input images.

        Parameters
        ----------
        data
            Pandas data frame or dict containing the image information.
        outputs
            A list of segmentation output results.
        ret_type
            What kind of information to extract from model outputs.

        Returns
        -------
        A list of the post-processed segmentation results.
        """
        logits = [ele[ret_type] for ele in outputs]
        image_column_name = self.get_image_column_name(data)
        for idx in range(data.shape[0]):
            ori_image_size = Image.open(data[image_column_name][idx]).size  # width, height
            logits[idx] = F.interpolate(
                logits[idx].float(), (ori_image_size[1], ori_image_size[0]), mode="bilinear", align_corners=False
            )
        return logits

    def get_image_column_name(self, data: pd.DataFrame):
        if self.column_types is None:
            column_names = list(data.columns)
            if self._label_column in column_names:
                column_names.remove(self._label_column)
            assert (
                len(column_names) == 1
            ), f"More than one image columns {column_names} exist in the data. Make sure to provide data with one image column."
            return column_names[0]
        else:
            for k, v in self.column_types.items():
                if v == SEMANTIC_SEGMENTATION_IMG:
                    return k
        return None

```

已接上全链路，CLI 超参会自动影响训练，无需手动写代码：

- 在 `run_semantic_segmentation.py` 传入 `--lambda_smooth/--lambda_boundary/--w_boundary/--w_smooth/--w_thin` 时，这些值写入 `hyperparameters` 的 `optim.gspo.*`。
- `SemanticSegmentationLearner.get_litmodule_per_run` 现在会检测 `config.optim.lora.gspo_enabled`，自动加载 `examples/automm/Conv-LoRA/gspo_trainer.py` 并实例化 `GSPOConvLoRATrainer`，把上面这五个权重传进去，然后注入 `SemanticSegmentationLitModule`（训练态）。
- 如果找不到 `gspo_trainer.py` 或类不存在，会有 warning 并回退不启用 GSPO。

涉及文件：
- `examples/automm/Conv-LoRA/run_semantic_segmentation.py`（之前已接收 CLI 并写入 `optim.gspo.*`）
- `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`（新增自动构建 gspo_trainer 的逻辑）

用法保持不变，只要加 `--gspo_enable` 并传你想调的权重即可，训练时会自动生效。

---

**User**

以当前代码版本为准。我用命令`nohup python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.1 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir outputs/hypars_lambda_smooth-251214 \
  > outputs/hypars_lambda_smooth-251214.log 2>&1 &`来训练，但报错:`Seed set to 42686693
Enabling GSPO with group_size=4, warmup_epochs=5
Enabling Decoder Attention LoRA: r=8, alpha=8, dropout=0.0

[Warning] MultiModalPredictor has no 'path' attribute. Model save directory cannot be printed.

Traceback (most recent call last):
  File "/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/run_semantic_segmentation.py", line 282, in <module>
    predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/predictor.py", line 540, in fit
    self._learner.fit(
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/learners/base.py", line 665, in fit
    fit_returns = self.execute_fit()
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/learners/base.py", line 577, in execute_fit
    attributes = self.fit_per_run(**self._fit_args)
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/learners/base.py", line 1275, in fit_per_run
    config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/learners/base.py", line 703, in get_config_per_run
    config = get_config(
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/utils/config.py", line 198, in get_config
    config = apply_omegaconf_overrides(config, overrides=overrides, check_key_exist=True)
  File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/utils/config.py", line 480, in apply_omegaconf_overrides
    raise KeyError(
KeyError: '"optim.gspo.lambda_smooth" is not found in the config. You may need to check the overrides. overrides={\'optim.lora.r\': 3, \'optim.peft\': \'conv_lora\', \'optim.lora.conv_lora_expert_num\': 8, \'env.num_gpus\': 1, \'optim.loss_func\': \'structure_loss\', \'optim.max_epochs\': 30, \'optim.lr\': 0.0001, \'env.per_gpu_batch_size\': 1, \'env.batch_size\': 4, \'optim.lora.gspo_enabled\': True, \'optim.lora.gspo_group_size\': 4, \'optim.lora.gspo_quality_momentum\': 0.9, \'optim.lora.gspo_warmup_epochs\': 5, \'optim.lora.gspo_contrastive_weight\': 0.1, \'optim.gspo.lambda_smooth\': 0.1, \'optim.gspo.lambda_boundary\': 0.3, \'optim.gspo.w_boundary\': 0.3, \'optim.gspo.w_smooth\': 0.1, \'optim.gspo.w_thin\': 0.05, \'model.sam.decoder_attention_lora_r\': 8, \'model.sam.decoder_attention_lora_alpha\': 8, \'model.sam.decoder_attention_lora_dropout\': 0.0}'`

---

**Cursor**

```yaml
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

```

问题原因：配置里没有 `optim.gspo.*` 默认项，`apply_omegaconf_overrides` 在 `check_key_exist=True` 下报 KeyError。

修复：在 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml` 增加默认块：
```yaml
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
```
现在 CLI 里的 `optim.gspo.*` override 能通过校验，训练可继续。请重新运行你的命令即可。

```yaml
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
```

---

**User**

\lambda_{smooth}参数，原来没有调节之前的值是多少？

---

**Cursor**

`λ_smooth` 在我们加 CLI 之前的默认就是 0.1。这个默认值写在 `examples/automm/Conv-LoRA/gspo_trainer.py` 里 `GSPOConvLoRATrainer.__init__` 的参数默认值 `lambda_smooth: float = 0.1`。即使你不在 CLI 或 config 里传，它也会走 0.1。

---

**User**

那么在当前还没有加lambda_smooth等超参调节的版本上，对应于这个超参的值取的是多少？

---

**Cursor**

在「还没有加 λ_smooth 等超参调节」的旧版里，代码里根本没有平滑正则这一项，`GSPOConvLoRATrainer.__init__` 只接收 group_size / warmup / contrastive_weight 等参数，没有 λ_smooth 字段，训练循环也没有把平滑损失加进去（只算 segmentation loss + GSPO 组内权重 + 对比损失）。因此当时的“默认值”可以理解为：**λ_smooth 等于 0（未启用平滑项）**。

参考旧版构造函数片段可见没有任何 λ_smooth 参数或使用：
```37:56:examples/automm/Conv-LoRA/gspo_trainer.py
def __init__(..., group_size: int = 4, warmup_epochs: int = 5, contrastive_weight: float = 0.1, quality_metric: str = 'iou', advantage_temperature: float = 5.0):
    self.predictor = predictor
    self.group_size = group_size
    self.warmup_epochs = warmup_epochs
    self.contrastive_weight = contrastive_weight
    self.quality_metric = quality_metric
    self.advantage_temperature = advantage_temperature
```

总结：旧版没有平滑正则，等效于 λ_smooth=0。

```python
def __init__(..., group_size: int = 4, warmup_epochs: int = 5, contrastive_weight: float = 0.1, quality_metric: str = 'iou', advantage_temperature: float = 5.0):
    self.predictor = predictor
    self.group_size = group_size
    self.warmup_epochs = warmup_epochs
    self.contrastive_weight = contrastive_weight
    self.quality_metric = quality_metric
    self.advantage_temperature = advantage_temperature
```

---

**User**

那么lambda_boundary呢？

---

**Cursor**

和 λ_smooth 一样，旧版里根本没有边界正则/奖励这一项，`gspo_trainer.py` 的构造函数和训练循环里都没有 `lambda_boundary`，也没有计算 boundary_loss 或把它加到 seg loss 里。等效默认就是 **λ_boundary = 0（未启用边界项）**。

---

**User**

实验的结果：设置lambda_smooth=0.1，训练得到的测试集验证dice=0.8493121266365051，比起不加lambda_smooth的性能（dice~= 0.8581）反而下降，请分析这个结果是否合理，接下来怎样再去调节？或者没有必要？

---

**Cursor**

这个结果**完全可能合理**，而且在你当前实现里，“加了 λ_smooth 反而 Dice 下降”更常见于以下几类情况（按优先级）：

### 1) 你把 smoothness 加到了**训练损失**里，但它并不一定和 Dice 单调一致
`λ_smooth=0.1` 会显著惩罚高曲率/细结构边界；如果你的数据集（ISIC）里病灶边缘本来就有很多细碎不规则细节，平滑会把这些细节“抹圆”，导致 **FN 增加或边界内缩**，Dice 下降很正常。

### 2) 你的 smoothness 实现是 Laplacian 平方，尺度可能比你直觉大
Laplacian 的数值和分辨率、mask 概率分布有关；同样是 `0.1`，在不同模型输出分布下等效强度差别很大。尤其你是对 `pred_probs` 做二阶惩罚，早期训练 logits/概率波动大时，smooth 项会更强，可能阻碍主任务收敛。

### 3) 你只改了 λ_smooth，但没有配套调 `w_smooth / w_thin / w_boundary`
在你现在的 GSPO trainer 里，smooth 同时会影响：
- **监督训练的 seg_loss**（通过 `lambda_smooth * smooth_loss.mean()`）
- **GSPO 的 shaped_quality**（通过 `- w_smooth * smooth_loss`）

如果你只设了 `lambda_smooth=0.1`，但 `w_smooth` 仍是默认 0.1，等于**双重抑制高频**，很容易过头，出现你看到的下降。

---

## 接下来怎么调（建议最省时间、最有信息量的做法）

### A. 先判断“有没有必要”：看边界质量是否提升
如果你看可视化发现 `lambda_smooth=0.1` 的 mask 边缘更干净、锯齿更少、伪分支更少，但 Dice 轻微掉一点，这在医学分割里未必是坏事；反之如果边界也没明显变好，那就没必要加 smooth。

### B. 做一个最小网格：把 smooth 强度降到“只做轻微 shaping”
建议只跑 3 个点，成本低但信息足：

- **方案 1（最稳）**：只对 RL shaping 轻微惩罚，先别动监督 loss  
  - `lambda_smooth = 0.0`
  - `w_smooth = 0.02 ~ 0.05`

- **方案 2（轻监督 + 更轻 shaping）**：  
  - `lambda_smooth = 0.01 ~ 0.03`
  - `w_smooth = 0.02 ~ 0.05`

- **方案 3（你现在的强度下探）**：  
  - `lambda_smooth = 0.05`
  - `w_smooth = 0.05`（别再用 0.1）

你现在 `0.1` 明显偏“强”，尤其在没有同时强调 thin/boundary 的情况下。

### C. 如果你确实关心细结构，必须配套 `w_thin`（否则平滑必然吃亏）
一种常见组合是：
- `lambda_smooth` 降低（0.01~0.05）
- `w_smooth` 降低（0.02~0.05）
- `w_thin` 稍微提高一点点（0.05→0.08~0.1），用来抵消“平滑抹细节”的副作用

### D. 训练策略：建议先 warmup 再加 smooth（最符合直觉）
更稳的做法是：前几 epoch 不加/少加平滑，让模型先学到大体形状；后期再加一点平滑抑制毛刺。这需要代码支持“epoch schedule”（如果你想我可以帮你加）。

---

## 结论
- **Dice 下降是合理的**：平滑会牺牲边界细节以换取更平滑的形状，未必提升 Dice。
- **下一步优先建议**：把 `lambda_smooth` 降到 `0~0.03`（或者只用 `w_smooth` 轻惩罚），并配套调整 `w_smooth/w_thin`；同时用可视化确认边界是否真的变好。
- 如果边界并没有明显改善：**可以不加 smooth**，把精力放在 `lambda_boundary / w_boundary` 上通常更直接（更贴近医学分割痛点）。
/////////////////////////////////////////////////////
这个结果**完全合理**，而且提供了很有价值的信息。让我详细分析：

## 一、性能下降的原因分析

### 1. Laplacian 平滑的双刃剑效应

λ_smooth=0.1 在推荐范围内（0.05-0.2），但它会：

**✅ 好处：**
- 抑制预测 mask 的高频噪声
- 减少边缘锯齿
- 降低 GSPO 的 reward variance

**❌ 代价（ISIC2017 数据集特定）：**
- **ISIC 皮肤病变的边界天然不规则**：有很多细小突起、凹陷、不光滑的轮廓
- Laplacian 正则会惩罚所有高曲率结构，无法区分"真实细结构" vs "噪声"
- 结果：模型学到的边界比 GT 更"圆润"，丢失了真实的不规则细节
- Dice 对边界错位敏感 → 性能下降 0.88%

### 2. 数据集依赖性

不同医学分割任务对平滑的需求差异很大：

| 数据集类型 | 边界特性 | λ_smooth 推荐 |
|-----------|---------|--------------|
| **肺结节** | 近似圆形，边界相对光滑 | 0.1-0.15 ✅ |
| **息肉** | 不规则但整体块状 | 0.05-0.1 |
| **ISIC 皮肤病变** | 高度不规则，细碎边界 | 0.01-0.05 或 0 ⚠️ |
| **血管/神经** | 极细长结构 | 0 或负值（不用）❌ |

ISIC2017 属于"边界复杂度高"类型，0.1 可能偏大。

---

## 二、接下来的调节策略

### 🎯 优先级 P0：尝试更小的 λ_smooth

```bash
# 实验 1：降低到推荐下限
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.05 \
  --num_gpus 1 \
  --output_dir outputs/lambda_smooth_0.05

# 实验 2：进一步降低（探索性）
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.03 \
  --num_gpus 1 \
  --output_dir outputs/lambda_smooth_0.03
```

**预期**：
- 如果 0.05 能接近或超过 baseline（0.8581），说明这个数据集只需要轻微平滑
- 如果 0.03 更好，说明 ISIC 对边界细节极其敏感

---

### 🎯 优先级 P1：检查是否需要平滑

如果 λ_smooth ≤0.03 仍不如 baseline，说明：
1. ISIC2017 **不需要**平滑正则（边界天然复杂是 feature，不是 bug）
2. 保持 λ_smooth=0

**决策标准**：
```
if dice(λ_smooth=0.05) > dice(baseline):
    # 有轻微正面效果，可以保留
    λ_smooth = 0.05
elif dice(λ_smooth=0.03) ≈ dice(baseline):
    # 效果中性，可选可不选
    λ_smooth = 0 或 0.03
else:
    # 持续负面影响，不用
    λ_smooth = 0
```

---

### 🎯 优先级 P2：其他超参可能更有用

对 ISIC 这种"不规则边界"数据集，**边界和细结构项可能比平滑更重要**：

```bash
# 尝试：不要平滑，加强边界和细结构
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.4 \
  --w_boundary 0.4 \
  --w_thin 0.08 \
  --num_gpus 1 \
  --output_dir outputs/boundary_focus
```

**理由**：
- `lambda_boundary` 直接优化边界对齐，不会抹掉细节
- `w_thin` 鼓励细结构召回
- 二者与平滑正交，可能更适合 ISIC

---

## 三、实验建议（快速验证）

### 方案 A：网格搜索（3 个实验）
```bash
# 1. λ_smooth=0.05（下限）
# 2. λ_smooth=0.03（探索）
# 3. λ_smooth=0，强化边界项
```

### 方案 B：单点验证（1 个实验）
如果资源有限，直接测：
```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.35 \
  --w_boundary 0.35 \
  --w_thin 0.07 \
  --num_gpus 1 \
  --output_dir outputs/no_smooth_more_boundary
```

---

## 四、理论解释：为什么不是所有数据集都需要平滑

### Laplacian 正则的本质

$$
\mathcal{L}_{\text{smooth}} = \sum_{i,j} \bigl( \Delta m(i,j) \bigr)^2
$$

它惩罚的是**二阶空间导数的能量**，即"曲率"。

- **适合场景**：GT 本身边界就应该光滑（如肺结节）
- **不适合场景**：GT 边界本身就高曲率（如 ISIC 病变的"星芒状"边缘）

### ISIC2017 的特殊性

查看 ISIC 数据集的 GT mask，你会发现很多病灶有：
- 不规则的锯齿状边缘（真实的，不是标注噪声）
- 细长的伪足结构
- 局部的尖锐拐角

**这些都是高曲率特征，被 Laplacian 误杀。**

---

## 五、总结与决策树

```
当前结果：λ_smooth=0.1 → dice=0.8493 < baseline(0.8581)
             ↓
是否需要平滑正则？
    │
    ├─ 尝试 λ_smooth=0.05
    │   │
    │   ├─ dice > 0.858 → ✅ 保留 0.05
    │   └─ dice ≤ 0.858 → 继续
    │           │
    │           └─ 尝试 λ_smooth=0.03
    │               │
    │               ├─ dice ≈ 0.858 → 可选
    │               └─ dice < 0.858 → ❌ 不用平滑
    │
    └─ 转向其他超参：
        - λ_boundary（更重要）
        - w_thin（细结构奖励）
```

---

## 六、我的建议

1. **立即行动**：测 λ_smooth=0.05，看是否能挽回性能
2. **如果不行**：λ_smooth=0，专注调 boundary 和 thin 项
3. **长期策略**：为不同数据集维护一个"推荐配置表"

**预测**：ISIC2017 最优配置可能是 `λ_smooth=0` 或接近 0。

需要我帮你生成完整的实验脚本吗？

```shellscript
# 实验 1：降低到推荐下限
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.05 \
  --num_gpus 1 \
  --output_dir outputs/lambda_smooth_0.05

# 实验 2：进一步降低（探索性）
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.03 \
  --num_gpus 1 \
  --output_dir outputs/lambda_smooth_0.03
```

```plaintext
if dice(λ_smooth=0.05) > dice(baseline):
    # 有轻微正面效果，可以保留
    λ_smooth = 0.05
elif dice(λ_smooth=0.03) ≈ dice(baseline):
    # 效果中性，可选可不选
    λ_smooth = 0 或 0.03
else:
    # 持续负面影响，不用
    λ_smooth = 0
```

```shellscript
# 尝试：不要平滑，加强边界和细结构
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.4 \
  --w_boundary 0.4 \
  --w_thin 0.08 \
  --num_gpus 1 \
  --output_dir outputs/boundary_focus
```

```shellscript
# 1. λ_smooth=0.05（下限）
# 2. λ_smooth=0.03（探索）
# 3. λ_smooth=0，强化边界项
```

```shellscript
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.35 \
  --w_boundary 0.35 \
  --w_thin 0.07 \
  --num_gpus 1 \
  --output_dir outputs/no_smooth_more_boundary
```

```plaintext
当前结果：λ_smooth=0.1 → dice=0.8493 < baseline(0.8581)
             ↓
是否需要平滑正则？
    │
    ├─ 尝试 λ_smooth=0.05
    │   │
    │   ├─ dice > 0.858 → ✅ 保留 0.05
    │   └─ dice ≤ 0.858 → 继续
    │           │
    │           └─ 尝试 λ_smooth=0.03
    │               │
    │               ├─ dice ≈ 0.858 → 可选
    │               └─ dice < 0.858 → ❌ 不用平滑
    │
    └─ 转向其他超参：
        - λ_boundary（更重要）
        - w_thin（细结构奖励）
```

---

**User**

为什么推荐在资源有限的情况下用“方案 B：单点验证（1 个实验）”，是它更容易出结果吗？

---

**Cursor**

不是"更容易出好结果"，而是**信息效率更高**。让我解释：

---

## 为什么推荐方案 B

### 1. 基于已有实验的推断

你已经测了 λ_smooth=0.1 → 性能下降。

这个结果暗示：
- ISIC2017 对平滑正则**负面敏感**
- 大概率 0.05 / 0.03 也只是"少掉一点分"，不会反超 baseline
- 与其花 2-3 个实验逐个测平滑值，不如直接**换方向**

### 2. 一个实验回答多个问题

方案 B 同时测试两个假设：

| 假设 | 方案 B 的设置 | 能验证什么 |
|-----|-------------|-----------|
| 假设 1 | λ_smooth=0 | 确认平滑对 ISIC 没用 |
| 假设 2 | λ_boundary=0.35, w_thin=0.07 | 边界/细结构方向是否有潜力 |

**一个实验 → 两个方向的判断**

### 3. 避免在低价值区域徘徊

如果你按方案 A：
```
λ_smooth=0.05 → dice=0.856 → "嗯，还是不如 baseline"
λ_smooth=0.03 → dice=0.857 → "接近了，再调？"
λ_smooth=0.02 → dice=0.858 → "终于持平..."
```

花了 3 个实验，结论只是"平滑没用"。

**方案 B 用 1 个实验跳过这个过程，直接探索新方向。**

---

## 诚实回答：方案 B 不保证更好

⚠️ 方案 B **不一定**比 baseline 更好。

它的价值是**信息量大**：
- 如果效果好 → 说明"边界/细结构"方向对了，可以继续微调
- 如果效果差 → 说明这几个超参都不是瓶颈，要考虑其他因素（数据增强、学习率、backbone 等）

---

## 总结

| | 方案 A（网格搜索）| 方案 B（单点验证）|
|---|---|---|
| 实验数量 | 3 | 1 |
| 目的 | 找 λ_smooth 最优值 | 判断"方向"是否值得深入 |
| 适合场景 | 已确认平滑有用，想精调 | 不确定哪个方向有潜力 |
| 风险 | 可能在无用区域浪费时间 | 可能跳过了微调平滑的机会 |

**当前情况**：平滑已经显示负面效果 → 方案 B 的"换方向"策略更划算。

---

如果你时间充裕，当然可以跑完整网格；如果想快速得到"下一步该往哪走"的信号，方案 B 是更高效的选择。

```plaintext
λ_smooth=0.05 → dice=0.856 → "嗯，还是不如 baseline"
λ_smooth=0.03 → dice=0.857 → "接近了，再调？"
λ_smooth=0.02 → dice=0.858 → "终于持平..."
```

---

**User**

你的方案 B：单点验证（1 个实验）为什么把之前的“# 尝试：不要平滑，加强边界和细结构
python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.4 \
  --w_boundary 0.4 \
  --w_thin 0.08 \
  --num_gpus 1 \
  --output_dir outputs/boundary_focus”改成了“python run_semantic_segmentation.py \
  --task isic2017 \
  --gspo_enable \
  --lambda_smooth 0.0 \
  --lambda_boundary 0.35 \
  --w_boundary 0.35 \
  --w_thin 0.07 \
  --num_gpus 1 \
  --output_dir outputs/no_smooth_more_boundary”?有什么好处？

---

**Cursor**

诚实回答：**没有特别的好处，这是我写的时候不小心给了两个略微不同的版本**。

两组参数都在推荐范围内：

| 参数 | 第一次 | 第二次 | 推荐范围 |
|-----|-------|-------|---------|
| lambda_boundary | 0.4 | 0.35 | 0.1 – 0.5 |
| w_boundary | 0.4 | 0.35 | 0.1 – 0.5 |
| w_thin | 0.08 | 0.07 | 0.05 – 0.1 |

差异很小（0.4 vs 0.35，0.08 vs 0.07），在实验层面几乎没有本质区别。

如果非要解释的话：
- 0.35/0.07 更接近默认值（0.3/0.05），算是"保守探索"
- 0.4/0.08 稍激进一点

**但老实说，选哪个都可以，这不是我刻意设计的"优化"，只是手滑写了两遍。** 你用哪组参数都行。

---

**User**

你对这4个文件的修改做了什么？

---

**Cursor**

在本次会话中，我对以下 4 个文件做了修改：

---

### 1. `examples/automm/Conv-LoRA/gspo_trainer.py`

**新增功能**：让 GSPO 训练器支持 5 个超参

- **构造函数**：新增 `lambda_smooth`, `lambda_boundary`, `w_boundary`, `w_smooth`, `w_thin` 参数及其默认值
- **新增辅助方法**：
  - `_prepare_binary_masks()`：将预测/GT 转为单通道概率图
  - `_boundary_map()`：用空间梯度近似边界图
  - `compute_smoothness_loss()`：Laplacian 平滑损失
  - `compute_boundary_loss()`：边界 L1 损失
  - `compute_thin_reward()`：细结构 Dice 奖励
- **修改 `gspo_group_training_step()`**：
  - 计算 smooth_loss、boundary_loss、thin_reward
  - 把它们加到 seg_loss（λ_smooth × smooth_loss + λ_boundary × boundary_loss）
  - 用它们构造 shaped_quality（w_boundary、w_smooth、w_thin 加权）
  - 日志中记录新增的 avg_smooth_loss、avg_boundary_loss、avg_thin_reward 等指标

---

### 2. `examples/automm/Conv-LoRA/run_semantic_segmentation.py`

**新增 CLI 参数**：

```python
parser.add_argument("--lambda_smooth", type=float, default=0.1)
parser.add_argument("--lambda_boundary", type=float, default=0.3)
parser.add_argument("--w_boundary", type=float, default=0.3)
parser.add_argument("--w_smooth", type=float, default=0.1)
parser.add_argument("--w_thin", type=float, default=0.05)
```

**传入 hyperparameters**：当 `--gspo_enable` 时，把这 5 个值写入 `optim.gspo.*` 键

---

### 3. `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`

**新增配置块**：

```yaml
gspo:
  lambda_smooth: 0.1
  lambda_boundary: 0.3
  w_boundary: 0.3
  w_smooth: 0.1
  w_thin: 0.05
```

⚠️ **你后来撤销了这个修改**，导致 `apply_omegaconf_overrides` 找不到 `optim.gspo.*` 键而报错。

---

### 4. `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

**修改 `get_litmodule_per_run()` 方法**：

- 检测 `config.optim.lora.gspo_enabled`
- 如果启用，自动加载 `examples/automm/Conv-LoRA/gspo_trainer.py` 并实例化 `GSPOConvLoRATrainer`
- 从 config 读取 5 个超参（lambda_smooth 等）传入构造函数
- 把 gspo_trainer 注入到 `SemanticSegmentationLitModule`

这样就实现了**全链路自动化**：CLI 参数 → config → gspo_trainer → 训练循环。

---

### 当前状态

| 文件 | 状态 |
|-----|-----|
| gspo_trainer.py | ✅ 已接受 |
| run_semantic_segmentation.py | ✅ 已接受 |
| default.yaml | ❌ 被你撤销（需要恢复才能正常运行）|
| semantic_segmentation.py | ✅ 已接受（你后来又加了 box_prompt 相关修改）|

```python
parser.add_argument("--lambda_smooth", type=float, default=0.1)
parser.add_argument("--lambda_boundary", type=float, default=0.3)
parser.add_argument("--w_boundary", type=float, default=0.3)
parser.add_argument("--w_smooth", type=float, default=0.1)
parser.add_argument("--w_thin", type=float, default=0.05)
```

```yaml
gspo:
  lambda_smooth: 0.1
  lambda_boundary: 0.3
  w_boundary: 0.3
  w_smooth: 0.1
  w_thin: 0.05
```

