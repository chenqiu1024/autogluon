# 质量感知的半监督分割方案设计文档

## 1. 方案概述

### 1.1 核心创新点

本方案针对医学图像分割任务（ISIC2017），在**半监督设定**（10% full mask + 90% noisy box）下，提出了一套**质量感知的 Teacher-Student 自训练框架**，核心创新在于：

1. **多次采样一致性质量评估**：利用 Teacher 模型对未标注/弱标注样本进行 K 次前向传播（不同 dropout/噪声状态），通过预测间的互 IoU 和置信度估计伪标签质量。

2. **GSPO 策略优化与质量联动**：将质量分数作为 GSPO（Group Sequence Policy Optimization）的 advantage proxy，让 MoE 门控网络学习"何时相信伪标签、如何选择专家路径"，实现质量感知的参数更新。

3. **Box Prompt 抖动一致性**：对弱监督 box 进行随机扰动，约束模型在不同 prompt 下输出一致，提升推理时 box 增强的鲁棒性。

### 1.2 适用场景

- **数据形态**：医学 2D 图像分割（ISIC2017 皮肤病变分割）
- **标注设定**：
  - 10% 训练样本：完整 mask 标注（强监督）
  - 90% 训练样本：仅 bounding box（由 GT mask 生成 + 添加可控噪声模拟标注误差）
- **目标任务**：在标注受限情况下，最大化利用弱标注数据提升 DICE/IoU 指标

---

## 2. 核心公式

### 2.1 EMA Teacher 更新

$$
\theta_{\text{teacher}}^{(t)} \leftarrow m \cdot \theta_{\text{teacher}}^{(t-1)} + (1 - m) \cdot \theta_{\text{student}}^{(t)}
$$

其中 $m = 0.999$（动量系数），每个训练步更新一次。

### 2.2 质量评估

#### 一致性质量
$$
q_{\text{cons}} = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} \text{IoU}(p_i, p_j)
$$

#### 置信度质量
$$
q_{\text{conf}} = \frac{1}{K} \sum_{k=1}^K \left( \frac{1}{HW} \sum_{h,w} \max(p_k^{hw}, 1 - p_k^{hw}) \right)
$$

#### 融合质量
$$
q = \alpha \cdot q_{\text{cons}} + (1 - \alpha) \cdot q_{\text{conf}}, \quad \alpha = 0.7
$$

### 2.3 质量加权函数

$$
w(q) = \sigma(\beta (q - q_0)), \quad \beta = 10, \quad q_0 = 0.5
$$

### 2.4 总损失函数

$$
\mathcal{L} = \mathcal{L}_s + \lambda_u(t) \cdot \mathcal{L}_u + \lambda_c \cdot \mathcal{L}_{\text{cons}} + \mathcal{L}_{\text{GSPO}}
$$

其中：
- $\mathcal{L}_s$：监督损失（Dice + BCE）
- $\mathcal{L}_u = \sum_i \mathbb{1}(q_i > q_{\text{min}}) \cdot w(q_i) \cdot \mathcal{L}_{\text{seg}}$：质量加权伪监督
- $\mathcal{L}_{\text{cons}} = \text{KL}(p_1 \| p_2)$：Box jitter 一致性
- $\lambda_u(t) = \min(1.0, t / 5)$：线性 warmup

### 2.5 GSPO 扩展

$$
q_{\text{quality}} = 
\begin{cases}
\text{IoU}(\text{pred}, \text{gt}) & \text{if labeled} \\
q_{\text{cons}}(K\text{ samples}) & \text{if weak}
\end{cases}
$$

$$
A = q_{\text{quality}} - \frac{1}{G} \sum_{g=1}^G q_g
$$

---

## 3. 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| EMA momentum $m$ | 0.999 | Teacher 更新动量 |
| 采样次数 $K$ | 5 | 质量评估的前向次数 |
| 质量阈值 $q_{\text{min}}$ | 0.6 | 伪标签过滤下限 |
| 一致性权重 $\alpha$ | 0.7 | 质量融合中一致性占比 |
| 加权斜率 $\beta$ | 10 | sigmoid 加权函数斜率 |
| Box jitter $\sigma$ | 0.08 | 相对 box 宽高的噪声标准差 |
| Box noise (数据) | 0.10 | 训练数据中 box 噪声强度 |
| $\lambda_c$ | 0.1 | 一致性损失权重 |
| GSPO group size | 4 | 组级采样数量 |
| GSPO warmup epochs | 5 | GSPO 启用延迟 |

---

## 4. 创新点总结

1. **方法论**：首次将策略优化与半监督自训练结合，提出质量感知的门控专家选择
2. **技术**：多次采样一致性作为无标注质量 proxy
3. **应用**：在医学弱监督场景下显著降低标注成本（10% vs 100%）
