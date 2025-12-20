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


