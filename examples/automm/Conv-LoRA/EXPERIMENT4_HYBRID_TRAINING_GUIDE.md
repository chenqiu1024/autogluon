# 实验 4：混合训练 (RLOO + 监督) 使用指南

## 🎯 目标

通过混合 RLOO 和监督学习，避免纯 RLOO 导致的性能下降问题。

## 📊 问题回顾

| 实验 | 策略 | 测试 IoU | 问题 |
|------|------|----------|------|
| 基线 | 监督 | 0.7707 | - |
| Exp1 | RLOO (beta=0.01) | 0.7624 | 略微下降 |
| Exp3 | RLOO (大噪声) | 0.6568 ↓↓ | 严重下降 |

**根本问题**：纯 RLOO 容易学到低质量策略

## ✅ 改进方案

### 三大改进

1. **保守采样**：噪声从 1.0-3.0 降低到 0.1-0.5
2. **混合训练**：添加 30-50% 监督损失
3. **修复 Diversity**：除以像素数，显示真实的 KL 散度

### 混合训练公式

```python
total_loss = supervised_weight * supervised_loss + (1 - supervised_weight) * rloo_loss
```

- `supervised_weight = 0.4`：40% 监督 + 60% RLOO
- 监督信号提供稳定性，RLOO 提供探索

## 🚀 运行实验

### 实验 4A：混合训练（推荐）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4a_hybrid \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --supervised_weight 0.4 \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 \
  > exp4a-hybrid-251127.log 2>&1 &

# 查看日志
tail -f exp4a-hybrid-251127.log
```

**参数解释**：
- `--supervised_weight 0.4`：**关键参数**，40% 监督 + 60% RLOO
- `--num_generations 4`：生成 4 个候选
- `--beta 0.0`：无 KL 惩罚
- `--epochs 3`：3 个 epochs（避免过拟合）

### 实验 4B：更保守的混合

如果 4A 效果不佳，尝试更高的监督权重：

```bash
nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4b_hybrid_conservative \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --supervised_weight 0.6 \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 \
  > exp4b-hybrid_conservative-251127.log 2>&1 &
```

**参数解释**：
- `--supervised_weight 0.6`：60% 监督 + 40% RLOO（更保守）

### 实验 4C：纯 RLOO with 小噪声（对比）

测试是否只是噪声太大导致的问题：

```bash
nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4c_pure_rloo_tiny_noise \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --supervised_weight 0.0 \
  --learning_rate 5e-6 \
  --epochs 2 \
  --batch_size 2 \
  > exp4c-pure_rloo_tiny_noise-251127.log 2>&1 &
```

**参数解释**：
- `--supervised_weight 0.0`：纯 RLOO（但使用小噪声）
- `--learning_rate 5e-6`：更小的学习率
- `--epochs 2`：只训练 2 个 epochs

## 📊 监控指标

训练时关注以下指标：

### 必看指标

| 指标 | 预期值 (Exp4A) | 说明 |
|------|----------------|------|
| `rloo_mean_diversity_step` | **0.001-0.01** | 真实的 KL 散度（归一化后） |
| `rloo_mean_reward_step` | 1.6-1.8 | 应该逐渐上升 |
| `rloo_mean_iou_step` | 0.75-0.85 | 训练集 IoU |
| `rloo_mean_dice_step` | 0.85-0.92 | 训练集 Dice |

### 混合训练特有指标

| 指标 | 说明 |
|------|------|
| `rloo_rloo_loss_step` | RLOO 损失分量 |
| `rloo_supervised_loss_step` | 监督损失分量 |

### 对比（预期改进）

| 训练类型 | Diversity | 训练 IoU | 测试 IoU（预期） |
|----------|-----------|----------|-----------------|
| **Exp3 (大噪声)** | ~3e+5 (错误) | 0.52-0.88 | 0.66 ↓↓ |
| **Exp4A (混合)** | 0.001-0.01 ✓ | 0.75-0.85 ✓ | **≥ 0.77** ✓ |

## ✅ 评估模型

训练完成后评估：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 找到最佳 checkpoint
# 假设是 rloo-epoch=XX-rloo_mean_reward=Y.YYYY.ckpt

python evaluate_rloo_lightning.py \
  --checkpoint_path outputs/rloo/isic2017/exp4a_hybrid/checkpoints/rloo-epoch=XX-rloo_mean_reward=Y.YYYY.ckpt \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --batch_size 4 \
  --output_file results_exp4a_hybrid.json
```

## 🎯 成功标准

### 最低要求（必须达到）
- ✅ 测试 IoU ≥ 0.77（不比基线差）
- ✅ Diversity 在合理范围（0.001-0.01）
- ✅ 训练稳定，不崩溃

### 理想目标
- 🎯 测试 IoU > 0.78（超过基线 1%）
- 🎯 训练-测试差距 < 10%

### 放弃标准
如果 Exp4A/B/C 都无法达到最低要求：
- 放弃 RLOO 方法
- 转而改进监督学习
- 考虑其他技术（TTA、模型集成等）

## 📝 代码改动

### 1. `lit_semantic_seg_rloo.py`

**采样策略（保守）**：
```python
# 旧代码（Exp3）：
noise_scale = 1.0 + 2.0 * (g / num_generations)  # 1.0 → 3.0

# 新代码（Exp4）：
noise_scale = 0.1 + 0.4 * (g / num_generations)  # 0.1 → 0.5
```

**Diversity 计算（修复）**：
```python
# 除以像素数得到真实的平均 KL
kl_sum = bernoulli_kl(noisy_logits, pred_logits).mean().item()
num_pixels = pred_logits.numel() / pred_logits.shape[0]
kl_div = kl_sum / num_pixels  # 归一化
```

**混合训练**：
```python
if self.supervised_weight > 0:
    supervised_loss = self.loss_func(pred_logits, label)
    loss = self.supervised_weight * supervised_loss + \
           (1 - self.supervised_weight) * rloo_loss
else:
    loss = rloo_loss
```

### 2. `run_semantic_segmentation_rloo_real.py`

**新参数**：
```python
parser.add_argument(
    "--supervised_weight",
    type=float,
    default=0.0,
    help="监督损失权重 (0-1)"
)
```

## 🔍 调试技巧

### 检查 Diversity 是否合理

```bash
# 在训练日志中搜索 diversity
grep "rloo_mean_diversity_step" exp4a-hybrid-251127.log | head -20
```

**预期**：0.001-0.01（不再是 3e+5）

### 检查混合训练是否生效

```bash
# 搜索混合训练特有的指标
grep "rloo_rloo_loss_step\|rloo_supervised_loss_step" exp4a-hybrid-251127.log | head -10
```

**预期**：能看到两个 loss 分量

### 检查训练稳定性

```bash
# 查看最后 50 行
tail -50 exp4a-hybrid-251127.log
```

**预期**：IoU 稳定在 0.75-0.85，没有 NaN 或崩溃

## 💡 理论分析

### 为什么混合训练可能有效？

1. **监督损失提供稳定基础**
   - 避免模型偏离太远
   - 保证基本分割质量

2. **RLOO 提供探索和微调**
   - 在监督基础上探索更好的策略
   - 通过 reward 信号优化特定指标（IoU/Dice）

3. **小噪声保证高质量候选**
   - 候选仍然接近监督学习的结果
   - 不会学到低质量策略

### 权重如何选择？

| Supervised Weight | 特点 | 适用场景 |
|-------------------|------|----------|
| **0.0** | 纯 RLOO | 如果对 RL 有信心 |
| **0.3-0.5** | 平衡 | 一般推荐，探索与稳定兼顾 |
| **0.6-0.8** | 保守 | 担心性能下降时 |
| **1.0** | 纯监督 | 退化为监督学习 |

**推荐**：先试 0.4，如果效果不好再增加到 0.6

## 📈 预期结果

### 乐观情况
- 测试 IoU: 0.78-0.79（+1-2%）
- 训练稳定，diversity 合理
- RLOO 确实有帮助

### 现实情况
- 测试 IoU: 0.77-0.78（持平或略有提升）
- 混合训练避免了性能下降
- RLOO 贡献有限但不有害

### 悲观情况
- 测试 IoU: 0.76-0.77（略有下降）
- 需要考虑放弃 RLOO
- 转而改进其他方面

---

## 🚀 快速开始

**立即运行 Exp4A**：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA && \
nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp4a_hybrid \
  --num_generations 4 \
  --beta 0.0 \
  --reward_type combo \
  --supervised_weight 0.4 \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 \
  > exp4a-hybrid-251127.log 2>&1 &

# 监控
tail -f exp4a-hybrid-251127.log
```

**祝实验顺利！** 🎉

