# 采样策略改进说明

## 问题诊断

之前的两次实验（beta=0.05, 3 epochs 和 beta=0.01, 10 epochs）都出现了：
- ❌ KL 散度始终为 0.000
- ❌ 模型没有真正的探索
- ❌ 测试集性能没有提升

**根本原因**：采样策略的噪声强度太小（0.5），导致生成的候选 mask 与原始预测几乎相同。

## 改进方案

### 修改位置
文件：`lit_semantic_seg_rloo.py`，第 108-122 行

### 改进策略

#### 1. **温度采样 (Temperature Sampling)**
```python
temperature = 0.5 + 1.5 * (g / num_generations)  # 0.5 → 2.0
scaled_logits = pred_logits / temperature
```
- 更高的温度使预测更"平滑"，增加不确定性
- 温度随 generation 索引递增

#### 2. **递增的噪声强度**
```python
noise_scale = 1.0 + 2.0 * (g / num_generations)  # 1.0 → 3.0
noisy_logits = scaled_logits + torch.randn_like(pred_logits) * noise_scale
```
- 之前：固定 noise_scale = 0.5
- 现在：1.0 到 3.0 的递增噪声
- 后面的候选与原始预测差异更大

#### 3. **随机丢弃 (Dropout-style)**
```python
if g % 2 == 1:
    dropout_mask = (torch.rand_like(noisy_logits) > 0.1).float()
    noisy_logits = noisy_logits * dropout_mask
```
- 对奇数索引的候选，随机丢弃 10% 的像素
- 进一步增加多样性

## 预期效果

✅ **KL 散度不再为 0**  
更大的噪声和温度采样会使候选 mask 显著偏离原始预测

✅ **真正的策略探索**  
模型能够探索不同的分割策略，不只是复制原始预测

✅ **更好的泛化**  
通过探索多样化的候选，模型学习到更鲁棒的策略

## 对比

| 策略 | 之前 | 现在 |
|------|------|------|
| **Temperature** | 1.0 (固定) | 0.5 → 2.0 (递增) |
| **Noise Scale** | 0.5 (固定) | 1.0 → 3.0 (递增) |
| **Dropout** | 无 | 10% (奇数候选) |
| **预期 KL** | ~0.000 | >0.001 |

## 实验建议

### 实验 2A：新采样 + 无 KL 惩罚
```bash
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp2a_new_sampling_no_kl \
  --num_generations 6 \
  --beta 0.0 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 5 \
  --batch_size 2
```

**为什么 beta=0.0？**
- 新的采样策略本身就引入了足够的多样性
- 移除 KL 惩罚让模型更自由地探索
- 如果效果好，可以后续尝试小的 beta（0.001-0.005）

### 实验 2B：新采样 + 更多候选
```bash
python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/exp2b_new_sampling_8gen \
  --num_generations 8 \
  --beta 0.0 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 5 \
  --batch_size 2
```

**为什么增加候选数？**
- 更多候选 → 更好的 baseline 估计
- 降低方差 → 更稳定的训练

## 监控指标

训练时关注：
- `rloo_mean_kl_step`: 应该 > 0.001（不再是 0）
- `rloo_mean_reward`: 应该逐渐上升
- `rloo_mean_iou/dice`: 训练集指标

评估时关注：
- 测试集 IoU：目标 > 0.7707（基线）
- 测试集 Dice：目标 > 0.8521（基线）
- 与训练集的差距：应该减小（表明泛化更好）

## 下一步

如果实验 2A/2B 仍然没有改进：
1. 考虑混合训练（RL + 监督）
2. 改进 reward 函数（添加边缘质量等）
3. 重新训练基线模型（改进泛化能力）

