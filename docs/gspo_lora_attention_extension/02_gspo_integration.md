# GSPO 集成方案与数学公式

## 1. 强化学习在 LoRA on Attention 中的应用

### 1.1 问题建模

将 LoRA 的微调视为策略优化问题：

| RL 概念 | LoRA 对应 | 说明 |
|---------|-----------|------|
| 状态 (State) | 输入特征 x | Query/Key/Value 的输入 |
| 动作 (Action) | LoRA 变换强度 | contribution_score 控制 |
| 策略 (Policy) | scaling × contribution_score | 决定 LoRA 的有效贡献 |
| 奖励 (Reward) | 分割质量 Q | IoU、Dice 等指标 |

### 1.2 与 Conv-LoRA MoE 的区别

| 维度 | Conv-LoRA MoE | LoRA on Attention |
|------|---------------|-------------------|
| 位置 | Vision Encoder | Mask Decoder |
| 专家数 | 8 个卷积专家 | 无专家（单一 LoRA） |
| 选择机制 | 门控网络选择专家 | Scaling 自适应调节 |
| GSPO 作用 | 优化专家选择概率 | 优化贡献强度 |

### 1.3 设计选择理由

**为什么不为 LoRA 添加 MoE？**

1. **参数量考虑**：Decoder LoRA 仅 ~86K 参数，添加 MoE 会显著增加
2. **位置不同**：Decoder 已经是模型末端，不需要多尺度适应
3. **简洁性**：Scaling 自适应足以实现质量驱动优化

---

## 2. 数学公式详解

### 2.1 标准 LoRA 前向传播

$$
y = W_0 \cdot x + \underbrace{(B \cdot A) \cdot x \cdot \frac{\alpha}{r}}_{\text{LoRA 残差}}
$$

### 2.2 GSPO 增强的 LoRA

$$
y = W_0 \cdot x + (B \cdot A) \cdot x \cdot \frac{\alpha}{r} \cdot \underbrace{\text{contribution\_score}}_{\text{GSPO 动态调节}}
$$

### 2.3 贡献分数计算

贡献分数基于质量历史计算：

$$
\text{contribution\_score} = \sigma\left(k \cdot (H_{lora} - 0.5)\right)
$$

**参数说明**：
- $H_{lora} \in [0, 1]$：LoRA 层的质量历史
- $k = 4$：放大系数（控制敏感度）
- $0.5$：中心点（初始中性状态）

**行为分析**：

| $H_{lora}$ | contribution_score | 含义 |
|------------|-------------------|------|
| 0.3 | σ(-0.8) ≈ 0.31 | 质量差 → 减少 LoRA 贡献 |
| 0.5 | σ(0) = 0.50 | 中性 → 基准贡献 |
| 0.7 | σ(0.8) ≈ 0.69 | 质量好 → 增加 LoRA 贡献 |
| 0.9 | σ(1.6) ≈ 0.83 | 质量优秀 → 大幅增加贡献 |

### 2.4 质量历史更新

使用指数移动平均（EMA）更新：

$$
H_{lora}^{(t+1)} = \mu \cdot H_{lora}^{(t)} + (1 - \mu) \cdot Q^{(t)}
$$

其中：
- $\mu = 0.9$：动量系数（默认）
- $Q^{(t)}$：当前批次的平均分割质量

---

## 3. GSPO 训练流程

### 3.1 完整训练步骤

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GSPO 完整训练步骤（包含 LoRA）                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For each batch:                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Group Sampling (G=4 predictions per image)                         │ │
│  │     ┌────────────────────────────────────────────────────────────────┐ │ │
│  │     │  For g = 1..G:                                                  │ │ │
│  │     │    - 前向传播通过 Vision Encoder                                │ │ │
│  │     │      * Conv-LoRA MoE 应用（带 GSPO 质量偏置）                   │ │ │
│  │     │      * Encoder Adapter 应用（带 contribution_score）            │ │ │
│  │     │    - 前向传播通过 Mask Decoder                                  │ │ │
│  │     │      * LoRA on Attention 应用（带 contribution_score）🆕        │ │ │
│  │     │    - 获得预测 pred[g]                                           │ │ │
│  │     └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  2. Quality Evaluation                                                  │ │
│  │     quality[g] = IoU(pred[g], gt)                                      │ │
│  │                                                                         │ │
│  │  3. Advantage Computation                                               │ │
│  │     baseline = mean(quality[1..G])                                     │ │
│  │     advantage[g] = quality[g] - baseline                               │ │
│  │                                                                         │ │
│  │  4. Weighted Loss                                                       │ │
│  │     weight[g] = sigmoid(advantage[g] * temperature)                    │ │
│  │     loss = Σ(weight[g] * seg_loss[g]) / G + contrastive_loss          │ │
│  │                                                                         │ │
│  │  5. Quality Feedback Update                                             │ │
│  │     ┌────────────────────────────────────────────────────────────────┐ │ │
│  │     │  - Conv-LoRA MoE: update expert_quality_history                 │ │ │
│  │     │  - Encoder Adapter: update adapter_quality_history              │ │ │
│  │     │  - LoRA on Attention: update lora_quality_history 🆕            │ │ │
│  │     └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  6. Backpropagation                                                     │ │
│  │     优势加权梯度同时更新所有 PEFT 模块                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 损失函数

总损失包含三部分：

$$
\mathcal{L}_{total} = \mathcal{L}_{seg}^{weighted} + \lambda_{moe} \cdot \mathcal{L}_{moe} + \lambda_c \cdot \mathcal{L}_{contrastive}
$$

**加权分割损失**：

$$
\mathcal{L}_{seg}^{weighted} = \frac{1}{G} \sum_{g=1}^{G} \sigma(\text{advantage}_g \cdot \tau) \cdot \mathcal{L}_{seg}^{(g)}
$$

**MoE 负载均衡损失**：

$$
\mathcal{L}_{moe} = \text{load\_balance\_loss}(\text{expert\_selection})
$$

**对比损失**：

$$
\mathcal{L}_{contrastive} = \text{MSE}(S_{feat}, S_{quality})
$$

---

## 4. 代码实现对应

### 4.1 LoRALinear 类修改

```python
class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, ..., 
                 gspo_enabled=False, 
                 gspo_momentum=0.9,
                 gspo_scale_adaptation=False):
        # ... 原有初始化 ...
        
        # GSPO 扩展
        self.gspo_enabled = gspo_enabled
        self.gspo_momentum = gspo_momentum
        self.gspo_scale_adaptation = gspo_scale_adaptation
        
        if gspo_enabled:
            self.register_buffer("quality_history", torch.tensor(0.5))
            self.register_buffer("contribution_score", torch.tensor(1.0))
    
    def forward(self, x):
        result = F.linear(x, self.T(self.weight), self.bias)
        
        if self.r > 0:
            lora_delta = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            
            # GSPO: 动态调节 LoRA 贡献
            if self.gspo_enabled and self.gspo_scale_adaptation and self.training:
                lora_delta = lora_delta * self.contribution_score
            
            result = result + lora_delta
        
        return result
    
    def update_quality_feedback(self, quality_score):
        """GSPO: 更新质量历史"""
        if not self.gspo_enabled:
            return
        
        with torch.no_grad():
            quality_value = quality_score.mean().item()
            
            # EMA 更新
            self.quality_history = (
                self.gspo_momentum * self.quality_history +
                (1 - self.gspo_momentum) * quality_value
            )
            
            # 计算贡献分数
            self.contribution_score = torch.sigmoid(
                (self.quality_history - 0.5) * 4.0
            )
```

### 4.2 GSPOConvLoRATrainer 扩展

```python
def update_lora_feedback(self, quality_scores, model):
    """GSPO: 更新 LoRA 层的质量反馈"""
    if not self.gspo_lora_attention_enabled:
        return
    
    avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
    
    # 收集所有 LoRALinear 模块
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.gspo_enabled:
            lora_layers.append((name, module))
    
    # 更新每个 LoRA 层
    for name, lora in lora_layers:
        lora.update_quality_feedback(avg_quality)
```

---

## 5. 奖励函数设计

### 5.1 基础奖励

$$
R_{base} = \text{IoU}(\hat{y}, y)
$$

### 5.2 形状奖励（可选）

$$
R_{shaped} = R_{base} + w_b \cdot R_{boundary} - w_s \cdot L_{smooth}
$$

### 5.3 统一反馈

所有 PEFT 模块共享相同的奖励信号：

```python
avg_quality = mean([quality[g] for g in 1..G])

# 反馈到各模块
conv_lora_moe.update_quality_history(avg_quality)
encoder_adapter.update_quality_feedback(avg_quality)
lora_on_attention.update_quality_feedback(avg_quality)  # 🆕
```

---

*下一篇：[03_implementation.md](./03_implementation.md) - 实现细节与代码修改*

