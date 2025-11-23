# RL Training修复清单

## 📋 修复总结

基于根本原因分析，对RL训练代码进行了以下关键修复：

---

## ✅ 已修复的问题

### 1. 降低init_bias (Priority 1 - 最关键) 

**问题**：
```python
# 之前
init_bias=2.0  # sigmoid(2.0) ≈ 0.88
# 导致所有层初始激活概率都在88%，陷入局部最优
```

**修复**：
```python
# 现在
init_bias=0.0  # sigmoid(0.0) = 0.50
# 从中性状态开始，给policy更多学习空间
```

**文件**：
- `train_rl_layer_selection.py` line 209
- 添加命令行参数 `--init_bias` (默认0.0)

**预期效果**：
- 概率分布会逐渐分化（部分层>0.7，部分层<0.3）
- Policy能够学会有区分度的层选择策略

---

### 2. 增加entropy_coef (Priority 2)

**问题**：
```python
# 之前
entropy_coef=0.01  # 探索太少
```

**修复**：
```python
# 现在  
entropy_coef=0.05  # 增加5倍，鼓励更多探索
```

**文件**：
- `train_rl_layer_selection.py` line 512 (argparse default)
- `train_rl_layer_selection.py` line 133 (function signature)

**预期效果**：
- Policy会尝试更多不同的层配置
- 减少过早收敛到局部最优的风险

---

### 3. 增强概率分布监控 (Priority 3)

**新增监控指标**：
```python
# 现在记录（每个episode）：
- prob_min, prob_max, prob_mean, prob_std
- high_prob_layers (>0.7的层数)
- low_prob_layers (<0.3的层数)
```

**文件**：
- `train_rl_layer_selection.py` line 289-310

**用途**：
- 判断policy是否在学习（概率应该逐渐分化）
- 提前发现"所有层相同概率"的问题

---

### 4. 监控Deterministic vs Stochastic (Priority 3)

**问题**：
- 训练用stochastic (Bernoulli采样)
- 评估用deterministic (>0.5阈值)
- 当所有概率>0.5时，两者差异巨大

**修复**：
```python
# 评估时同时记录两种模式
- num_layers_stochastic (训练时的行为)
- num_layers_deterministic (理想中的行为)
- 两者应该接近（差异<2层）
```

**文件**：
- `train_rl_layer_selection.py` line 320-348

**用途**：
- 及早发现"stochastic有效但deterministic失败"的问题
- 确保训练和评估的一致性

---

### 5. 改进进度输出 (Priority 4)

**之前**：
```
Episode 10 | Reward: 0.8341 | Layers: 27.2/32 | Loss: 0.0042
```

**现在**：
```
Episode 10 | Reward: 0.8341 | Layers: 27.2/32 | 
Prob: [0.234, 0.765] (std=0.187) | Loss: 0.0042
```

**新增信息**：
- `[min, max]`: 概率范围（应该逐渐扩大）
- `std`: 标准差（应该逐渐增大）

**用途**：
- 实时监控policy是否在学习区分不同层

---

## 🔧 修改的文件列表

| 文件 | 修改内容 |
|------|----------|
| `train_rl_layer_selection.py` | 核心修复：init_bias, entropy_coef, 监控增强 |
| `run_rl_training_fixed.sh` | 新脚本：使用修复后的参数 |
| `DIAGNOSIS_root_causes.md` | 诊断文档 |
| `FIXES_APPLIED.md` | 本文档 |

---

## 📊 如何验证修复有效

### 训练过程中观察

#### 1. 概率分布应该分化

**Good 👍**:
```
Episode 10:  Prob: [0.45, 0.55] (std=0.03)  ← 刚开始，还比较集中
Episode 100: Prob: [0.32, 0.68] (std=0.12)  ← 开始分化
Episode 300: Prob: [0.15, 0.85] (std=0.25)  ← 明显区分
```

**Bad 👎**:
```
Episode 10:  Prob: [0.48, 0.52] (std=0.01)
Episode 100: Prob: [0.47, 0.53] (std=0.02)  ← 没有变化
Episode 300: Prob: [0.46, 0.54] (std=0.02)  ← 失败
```

#### 2. High/Low prob layers应该增加

**Good 👍**:
```
Episode 10:  High: 0.0, Low: 0.0   ← 都在中间
Episode 100: High: 5.3, Low: 4.8   ← 开始有极端值
Episode 300: High: 12.5, Low: 10.2 ← 大约1/3层高，1/3层低
```

**Bad 👎**:
```
Episode 10:  High: 0.0, Low: 0.0
Episode 100: High: 0.0, Low: 0.0  ← 始终没有极端值
Episode 300: High: 0.0, Low: 0.0  ← 所有层都在0.3-0.7之间
```

#### 3. Stochastic和Deterministic应该接近

**Good 👍**:
```
Episode 25:  Stoch: 16.2, Det: 16.0  ← 非常接近
Episode 100: Stoch: 18.5, Det: 19.0  ← 差异<1层
```

**Bad 👎**:
```
Episode 25:  Stoch: 28.3, Det: 32.0  ← 差异4层
Episode 100: Stoch: 27.8, Det: 32.0  ← 始终差4层（deterministic全激活）
```

#### 4. Reward应该有改善趋势

**Good 👍**:
```
Episode 10:  Reward: 0.7523
Episode 100: Reward: 0.7845
Episode 300: Reward: 0.8012  ← 逐渐接近baseline
```

**Acceptable 🤔**:
```
Episode 10:  Reward: 0.7523
Episode 100: Reward: 0.7645
Episode 300: Reward: 0.7698  ← 有提升但不大（可能任务不需要选择性）
```

**Bad 👎**:
```
Episode 10:  Reward: 0.7523
Episode 100: Reward: 0.7489
Episode 300: Reward: 0.7512  ← 完全随机波动
```

---

## 🚀 使用修复后的训练脚本

### 快速测试（100 episodes，~45分钟）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

python train_rl_layer_selection.py \
    --task isic2017 \
    --ckpt_path AutogluonModels/ag-20251123_003958 \
    --output_dir rl_test_fixed \
    --num_episodes 100 \
    --batch_size 8 \
    --init_bias 0.0 \
    --entropy_coef 0.05 \
    --device cuda
```

**检查点**：
- 运行10 episodes后，查看`Prob`输出
- 如果`std`在增大，说明修复有效
- 如果仍然在0.48-0.52之间，说明仍有问题

### 完整训练（500 episodes，~4小时）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

chmod +x run_rl_training_fixed.sh
./run_rl_training_fixed.sh
```

### 监控训练（另一个终端）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

tensorboard --logdir rl_training_fixed_*/logs --port 6006
```

**关键图表**：
- `train/prob_std`: 应该逐渐上升
- `train/high_prob_layers` + `train/low_prob_layers`: 应该逐渐增加
- `eval/num_layers_deterministic` vs `eval/num_layers_stochastic`: 应该接近

---

## ⚠️ 已知限制（未修复的问题）

### 1. 使用Random Embeddings

**问题**：
```python
# train_rl_layer_selection.py line 255
patch_embeddings = torch.randn(batch_size, 64, 64, 1280).to(device)
```

**影响**：
- Policy看到的是随机噪声，不是真实图像特征
- 只能学到"平均策略"，不是真正的per-image dynamic selection

**为什么暂不修复**：
- 需要深入SAM内部实现，提取patch embeddings
- 实现复杂度高（需要hook或修改predictor）
- 先验证降低init_bias是否有效

**未来修复方案**：
```python
# 方案A：从SAM提取真实embeddings
def extract_patch_embeddings(predictor, batch):
    # Hook into SAM's vision encoder
    # Get output before transformer layers
    pass

# 方案B：使用图像统计特征作为proxy
def get_simple_features(images):
    # mean, std, histogram, texture features
    pass
```

### 2. 可能的Reward Signal不足

**假设**：
不同层配置对ISIC 2017任务的性能影响可能很小

**证据需求**：
- 需要做消融实验：逐层关闭，测量性能下降
- 或随机策略对比：16层 vs 24层 vs 32层

**如果假设成立**：
- Per-image dynamic selection可能不是最优方案
- 应该考虑固定层配置或layer grouping

---

## 🔄 回滚方案（如果修复无效）

如果训练500 episodes后仍然失败（概率不分化，性能无改善），考虑：

### 方案A：更激进的初始化
```bash
--init_bias -1.0  # sigmoid(-1.0) ≈ 0.27
# 从"大部分不激活"开始，强迫policy学会选择
```

### 方案B：固定层策略
```python
# 不是per-image，而是为整个ISIC 2017学习一个固定配置
# 简化问题，更容易学习
```

### 方案C：Layer Grouping
```python
# 将32层分为4组（每组8层）
# 只决策"激活哪些组"，而非逐层决策
# 决策空间从2^32降到2^4
```

---

## 📝 总结

### 修复优先级

| 优先级 | 修复内容 | 状态 | 预期影响 |
|--------|----------|------|----------|
| P1 | 降低init_bias | ✅ 完成 | 🔴 Critical |
| P2 | 增加entropy_coef | ✅ 完成 | 🟠 High |
| P3 | 增强监控 | ✅ 完成 | 🟡 Medium |
| P4 | 提取真实embeddings | ❌ 未完成 | 🟡 Medium |
| P5 | 改进reward设计 | ❌ 未完成 | 🟢 Low |

### 下一步

1. ✅ **立即运行**：使用`run_rl_training_fixed.sh`开始测试
2. ⏳ **监控训练**：观察概率分布是否分化
3. ⏳ **如果成功**：继续完整训练500 episodes
4. ⏳ **如果失败**：考虑方案A/B/C

### 成功标准

**Minimum (50 episodes内应该看到)**：
- Prob std > 0.10（从~0.01开始）
- High/low prob layers > 0（从0开始）
- Deterministic和stochastic差异<3层

**Full Success (500 episodes后)**：
- Prob std > 0.20
- 明确的层选择模式（部分层>0.7，部分<0.3）
- 性能≥baseline - 2%

---

**创建时间**: 2025-11-24  
**状态**: 修复完成，等待测试验证

