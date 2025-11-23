# Bug Fix: Per-Image Layer Selection Evaluation

## 问题描述

### 原始实现的问题

训练脚本存在一个**严重的语义错误**，违背了"per-image动态层选择"的核心设计目标：

```python
# 错误的实现逻辑
for i in range(batch_size):
    mask_i = policy(image_i_features)  # ✓ 为图像i生成专属mask
    reward_i = env.step(mask_i)        # ✗ 但用mask_i评估整个150张图的val subset!
```

**问题**：
- Policy为**图像A**生成了专属的`mask_A`（基于图像A的特征）
- 但`env.step(mask_A)`内部却用`mask_A`评估**其他150张不同图像**的平均IoU
- 图像A的reward实际反映的是"`mask_A`对其他图像的效果"，而非"`mask_A`对图像A的效果"

**后果**：
- 训练信号错位：policy无法学习"为特定图像生成最优mask"
- 违背per-image语义：每张图像应该有自己的专属mask并只用于自己
- 性能提升困难：reward signal与policy决策不对应

## 正确的实现逻辑

### Per-Image Dynamic Selection的正确语义

```python
# 正确的实现逻辑
for i in range(batch_size):
    image_i_data = batch_data.iloc[[i]]     # 单张图像的数据
    mask_i = policy(image_i_features)       # ✓ 为图像i生成专属mask
    reward_i = evaluate(image_i_data, mask_i)  # ✓ 只用mask_i评估图像i自己!
```

**关键点**：
1. Policy根据图像特征生成mask
2. 该mask**只用于该图像**的前向传播
3. Reward**只反映**该mask对该图像的效果
4. 训练信号准确：`∇policy ∝ reward_i * ∇log P(mask_i | image_i)`

## 修改内容

### 1. ConvLoRAEnvironment (`conv_lora_env.py`)

#### 新增方法: `evaluate_single_image()`

```python
def evaluate_single_image(
    self,
    image_data: pd.DataFrame,  # 单张图像（1行）
    layer_mask: torch.Tensor,  # 该图像的专属mask [32]
) -> float:
    """
    Evaluate a single image with its specific layer mask.
    
    This is the correct method for per-image RL training.
    """
    # 确保只有1张图像
    assert len(image_data) == 1
    
    # 注入该图像的专属mask
    def forward_with_mask(*args, **kwargs):
        kwargs['layer_masks'] = layer_mask.unsqueeze(0)  # [1, 32]
        return original_forward(*args, **kwargs)
    
    model.forward = forward_with_mask
    
    # 只评估这一张图像
    metrics = self.predictor.evaluate(image_data, metrics=[self.reward_metric])
    return metrics[self.reward_metric]
```

#### 修改方法: `step()`

```python
def step(
    self,
    image_data: pd.DataFrame,  # 新增参数：图像数据
    layer_mask: torch.Tensor,  # 该图像的mask
) -> Dict[str, float]:
    """Execute one environment step: evaluate single image with its mask."""
    reward = self.evaluate_single_image(image_data, layer_mask)
    return {'reward': reward, 'num_active_layers': layer_mask.sum().item()}
```

#### 保留方法: `evaluate_policy()`

- 用于计算baseline reward（所有层激活，评估整个eval subset）
- **不再用于RL训练过程**

### 2. BatchedConvLoRAEnvironment

#### 修改方法: `evaluate_batch()`

```python
def evaluate_batch(
    self,
    batch_data: pd.DataFrame,      # 新增参数
    layer_masks_batch: torch.Tensor,
) -> torch.Tensor:
    """Evaluate each image with its corresponding mask."""
    for i in range(B):
        single_image_data = batch_data.iloc[[i]]
        reward_i = self.base_env.step(single_image_data, layer_masks_batch[i])
```

### 3. Training Script (`train_rl_layer_selection.py`)

#### 修改训练循环

```python
# 生成batch的masks
layer_masks, layer_probs, log_probs = policy(patch_embeddings)  # [16, 32]

# Per-image evaluation
rewards = []
for i in range(batch_size):
    # 取出单张图像的数据
    single_image_data = batch_data.iloc[[i]].reset_index(drop=True)
    # 用该图像的专属mask评估该图像
    result = env.step(single_image_data, layer_masks[i])
    rewards.append(result['reward'])
```

## 训练与推理的一致性

### 训练阶段（修正后）
```
image_1 → extract_features → policy → mask_1 → evaluate(image_1, mask_1) → reward_1
image_2 → extract_features → policy → mask_2 → evaluate(image_2, mask_2) → reward_2
...
```

### 推理阶段
```
new_image → extract_features → policy → mask_new → segment(new_image, mask_new)
```

✅ **完美对应**：训练时如何使用mask，推理时就如何使用！

## 为什么方案A优于方案B

### 方案A（已实现）：逐个评估
```python
for i in range(batch_size):
    reward_i = evaluate_single_image(image_i, mask_i)
```

**优点**：
- ✅ 在AutoGluon框架下技术可行
- ✅ 图像-mask对应关系清晰明确
- ✅ 完美符合per-image语义
- ✅ 训练-推理逻辑一致

**缺点**：
- ⚠️ 评估速度较慢（16次forward vs 1次）
- ⚠️ 单图像reward噪声较大

**缓解策略**：
- Baseline reduction（moving average）
- Batch averaging（16张图像平均reward）
- Entropy regularization
- 可调整`batch_size`和`eval_subset_size`

### 方案B（技术不可行）：批量评估
```python
rewards = evaluate_batch(images, masks)  # 理想但无法实现
```

**问题**：
- ❌ AutoGluon的`predictor.evaluate()`内部会重新shuffle和batch数据
- ❌ 无法追踪"当前forward的图像"对应"哪个mask"
- ❌ 高层API不暴露batch内索引控制

## 性能考虑

### 评估成本对比

- **Baseline（32层全激活）**：1次complete evaluation = ~31秒
- **方案A（per-image）**：16张图像 × 1次evaluation ≈ 0.2秒/张 × 16 ≈ 3.2秒/episode

### 优化策略
1. 减少`eval_subset_size`（200 → 100）
2. 减少`batch_size`（16 → 8）
3. 使用更高效的predictor内部方法（未来优化）

## 验证修正的正确性

### 测试checklist
- [ ] Policy为不同图像生成不同的mask
- [ ] 每个mask只用于对应图像的评估
- [ ] Reward能正确反映mask对该图像的效果
- [ ] 训练过程中policy能学习到有意义的层选择模式
- [ ] 最终性能优于baseline（全层激活）

## 总结

这次修正解决了一个**设计层面的根本性错误**：

| 维度 | 修正前 | 修正后 |
|------|--------|--------|
| **Policy输入** | 图像i的特征 | 图像i的特征 ✓ |
| **Policy输出** | mask_i（for 图像i） | mask_i（for 图像i）✓ |
| **Evaluation** | mask_i评估其他150张图 ❌ | mask_i只评估图像i ✓ |
| **Reward语义** | mask_i对其他图的效果 ❌ | mask_i对图像i的效果 ✓ |
| **训练信号** | 错位、噪声大 ❌ | 准确对应 ✓ |

**修正后的实现才真正符合"Per-Image Dynamic Layer Selection"的设计目标！**

---

**修改时间**: 2025-11-23  
**修改原因**: 纠正per-image评估逻辑，确保训练信号与policy决策对应  
**影响范围**: RL训练核心逻辑，预期显著提升训练效果

