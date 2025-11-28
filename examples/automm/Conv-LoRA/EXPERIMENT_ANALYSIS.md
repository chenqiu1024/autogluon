# RLOO 实验结果分析与改进建议

## 📊 实验结果对比

| 实验 | Epochs | Beta | 采样策略 | 测试 IoU | 测试 Dice | 训练 IoU 范围 |
|------|--------|------|----------|----------|-----------|--------------|
| **基线 (3 epochs)** | 3 | 0.05 | noise=0.5 | **0.7707** | **0.8521** | ~0.91-0.96 |
| **Exp1 (10 epochs)** | 10 | 0.01 | noise=0.5 | 0.7624 ↓ | 0.8466 ↓ | ~0.91-0.96 |
| **Exp3 (fixed KL)** | 5 | 0.0 | noise=1.0-3.0 | **0.6568 ↓↓** | **0.7678 ↓↓** | ~0.52-0.88 |

### 关键发现

❌ **Exp3 性能大幅下降**：
- IoU 下降 **11.4 个百分点**（0.7707 → 0.6568）
- Dice 下降 **8.4 个百分点**（0.8521 → 0.7678）

## 🔍 问题诊断

### 问题 1：Diversity 指标异常（3e+5）

**根本原因**：`bernoulli_kl` 函数返回的是像素级 KL 之**和**，而不是平均值。

**代码位置**：`rloo_utils.py` 第 76-79 行
```python
kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
while kl.ndim > 1:
    kl = kl.sum(dim=-1)  # ← 求和，不是平均！
return kl  # 对于 256x256 图像 = ~65536 个像素的总和
```

**影响**：
- diversity = 3e+5 不是真实的 KL 散度（应该是 ~4.5-5.0）
- 无法有效监控采样多样性

### 问题 2：采样噪声过大

**当前设置**：
```python
temperature = 0.5 + 1.5 * (g / num_generations)  # 0.5 → 2.0
noise_scale = 1.0 + 2.0 * (g / num_generations)  # 1.0 → 3.0
```

**后果**：
- 生成的候选质量极差
- 训练集 IoU 低至 **0.52-0.59**（部分样本）
- 模型学到的是**低质量策略**

**对比**：
- 基线（noise=0.5）：训练 IoU ~0.91-0.96，测试 IoU 0.77
- Exp3（noise=1.0-3.0）：训练 IoU ~0.52-0.88，测试 IoU 0.66 ↓

### 问题 3：过度探索 vs 利用平衡

**现象**：
- 太多探索（noise 太大）→ 候选质量差
- 模型优化错误的目标：学习生成低质量的 mask

**RLOO 的假设**：
- 需要多样化的**高质量**候选
- 候选之间应该有差异，但都应该是合理的分割

**当前情况**：
- 候选太随机，大部分是噪声
- 偏离原始预测太远，失去了监督学习的基础

## 🔧 根本问题

### 核心矛盾

**RLOO 适用场景**：
- 有一个合理的初始策略
- 通过小幅扰动探索改进
- 候选质量相近，细微差异决定 reward

**当前情况**：
- 采样噪声太大 → 候选质量极差
- 已经有很好的监督模型（IoU 0.77）
- RLOO 反而破坏了原有性能

### 为什么基线（noise=0.5）也没提升？

1. **KL 散度计算错误**：`ref_logits = pred_logits`，没有真正的参考
2. **采样多样性不足**：noise=0.5 可能还是太小
3. **RLOO 本身可能不适合这个任务**：
   - 分割是像素密集预测
   - 一次预测 65536 个像素
   - 微小噪声难以产生有意义的探索

## 💡 改进方案

### 方案 A：保守采样 + 混合训练（推荐）

**策略**：
1. 大幅降低采样噪声（0.1-0.5）
2. 添加监督损失（30-50%）
3. 使用较少的 epochs（3-5）

**优点**：
- 保持监督学习的稳定性
- RL 提供微调
- 不会破坏原有性能

**实现**：
```python
# 1. 降低噪声
temperature = 1.0  # 固定温度
noise_scale = 0.1 + 0.4 * (g / num_generations)  # 0.1 → 0.5

# 2. 混合损失
supervised_loss = self.loss_func(pred_logits, gt_mask)
rloo_loss_value = rloo_loss(log_probs, rewards)
total_loss = 0.4 * supervised_loss + 0.6 * rloo_loss_value
```

### 方案 B：放弃 RLOO，改用 PPO

**原因**：
- RLOO 的 leave-one-out baseline 可能不适合密集预测
- PPO 有更稳定的更新机制
- clip 机制防止过大的策略更新

**缺点**：
- 实现更复杂
- 需要更多工程工作

### 方案 C：回归监督学习 + 改进基础模型

**认清现实**：
- 当前监督模型（IoU 0.77）已经很好
- RLOO 没有带来提升，反而下降
- 应该改进数据、模型或训练方法

**建议**：
1. 增加数据增强（测试时增强 TTA）
2. 改进 loss 函数（focal loss, boundary loss）
3. 调整模型架构（更多 Conv-LoRA 层）
4. 使用更大的基础模型

### 方案 D：修复 Diversity + 极小噪声

**最后一试**：
1. 修复 diversity 计算（除以像素数）
2. 使用极小的噪声（0.05-0.2）
3. 只训练 1-2 个 epochs
4. 如果还是下降，就放弃 RLOO

## 📋 推荐的实验计划

### 优先级 1：混合训练（方案 A）

```bash
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4_hybrid \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 \
  --supervised_weight 0.4  # 需要添加此参数
```

**修改需求**：
1. 在 `lit_semantic_seg_rloo.py` 中降低采样噪声
2. 添加混合损失逻辑
3. 修复 diversity 计算

### 优先级 2：修复 Diversity + 微小噪声

只修复技术问题，看是否有帮助：

```bash
# 修改采样策略后
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4_tiny_noise \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --learning_rate 5e-6 \
  --epochs 2 \
  --batch_size 2
```

### 优先级 3：放弃 RLOO，改进基础模型

如果上述都失败：

```python
# 回到监督学习
predictor.fit(
    train_data="data/isic2017/train.csv",
    hyperparameters={
        "optimization.learning_rate": 1e-5,
        "optimization.max_epochs": 20,
        "data.augmentation": "strong",
        "loss": "focal_loss",  # 改进 loss
    }
)
```

## 🎯 成功标准

### 最低要求
- 测试 IoU ≥ 0.77（不比基线差）
- 训练稳定，不崩溃

### 理想目标
- 测试 IoU > 0.78（超过基线 1%）
- 训练集与测试集差距 < 10%

### 放弃标准
如果连续 3 个实验都无法达到最低要求，建议：
1. 放弃 RLOO
2. 专注于改进监督学习
3. 考虑其他方法（TTA、模型集成等）

## 🔍 深层思考

### 为什么 RLOO 在这个任务上困难？

1. **语义分割的特殊性**：
   - 65536 个像素的联合预测
   - 像素之间高度相关
   - 不是单一的离散决策

2. **监督信号已经很强**：
   - 每个像素都有 GT 标签
   - 监督学习已经很有效（IoU 0.77）
   - RL 很难超越密集监督

3. **采样空间过大**：
   - 2^65536 种可能的 mask
   - 随机采样很难找到好的候选
   - 需要结构化的探索

4. **Reward 稀疏性**：
   - IoU/Dice 是全局指标
   - 单个像素的改变影响很小
   - 梯度信号弱

### 结论

**RLOO 可能根本不适合密集预测任务**，特别是当监督学习已经表现很好时。

建议：
1. 先尝试方案 A（混合训练）
2. 如果失败，考虑放弃 RLOO
3. 转而改进监督学习或使用其他技术

---

**下一步行动**：实现方案 A（混合训练 + 保守采样）

