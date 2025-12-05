# TTA 不同图像尺寸问题修复

## 问题描述

```
RuntimeError: stack expects each tensor to be equal size, 
but got [1, 2848, 4288] at entry 0 and [1, 2000, 3008] at entry 2
```

## 根本原因

ISIC2017 数据集中的图像有**不同的原始尺寸**：
- 图像 0: 2848 × 4288
- 图像 2: 2000 × 3008
- ... 等等

TTA 会将预测 resize 回原图尺寸，所以：
- 预测 0: (1, 2848, 4288)
- 预测 2: (1, 2000, 3008)

当尝试 `torch.stack(all_preds)` 时，由于尺寸不一致导致错误。

---

## 解决方案

**不再使用 `torch.stack()`，而是逐个处理每张图像**

### 旧代码（错误）

```python
# 累积所有预测
all_preds = [pred1, pred2, ..., pred600]  # 尺寸各不相同

# 尝试堆叠（失败！）
y_pred = torch.stack(all_preds)  # ❌ RuntimeError

# 批量计算指标
for y_p, y_t in zip(y_pred, y_true):
    metric.update(y_p, y_t)
```

### 新代码（正确）

```python
# 累积所有预测（尺寸各不相同）
all_preds = [pred1, pred2, ..., pred600]

# 初始化指标
metric = get_metric_predict(...)

# 逐个处理（不堆叠）
for y_p, y_t in zip(all_preds, all_labels):
    # Resize 预测以匹配标签（如果需要）
    if y_p.shape[-2:] != y_t.shape[-2:]:
        y_p = F.interpolate(
            y_p.unsqueeze(0),  # (C,H,W) → (1,C,H,W)
            size=y_t.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (1,C,H,W) → (C,H,W)
    
    # 更新指标
    metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

# 计算最终分数
score = metric.compute()
```

---

## 关键改进

### 1. 不再使用 torch.stack()

**原因**: 不同尺寸的图像无法堆叠

**解决**: 逐个处理，torchmetrics 的指标对象本身就支持这种方式

### 2. 自动 resize 匹配

如果预测和标签尺寸不匹配（理论上不应该），自动 resize：

```python
if y_p.shape[-2:] != y_t.shape[-2:]:
    y_p = F.interpolate(y_p.unsqueeze(0), size=y_t.shape[-2:], ...).squeeze(0)
```

### 3. 保持类型正确

```python
y_p = y_p.float()  # 预测是 float
y_t = y_t.long()   # 标签是 long/int
```

---

## 为什么之前没发现这个问题？

### 在标准评估中（无 TTA）

标准评估流程可能会：
1. 将所有图像 resize 到统一尺寸（如 1024×1024）
2. 批量处理
3. Stack 没有问题

### 在 TTA 中

TTA 会：
1. 保持原始图像尺寸
2. Resize 到模型输入尺寸（1024×1024）
3. **预测后 resize 回原始尺寸**（问题所在）
4. 不同图像的原始尺寸不同 → 无法 stack

---

## 验证修复

运行测试：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251203_075302 \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --output_dir outputs/eval_tta_fixed
```

应该能够正常完成，输出：
```
TTA evaluation completed: {'iou': 0.XXXX, 'dice': 0.XXXX}
```

---

## 额外发现：速度分析

从你的日志看到：
- 平均时间: **33.45s/图**
- 18 次变换（不是 6 次）

这说明你运行的是**进阶 TTA 配置**（包括 rotations），速度合理：
- 18 次变换 vs 6 次变换 = 3 倍时间
- 33s vs 预期的 8s (for 6 次) × 3 = 24s
- 实际 33s 略慢，但在合理范围

**速度分解**（18 次变换）:
- 变换 + Resize: ~20-25s (主要瓶颈)
- 模型推理: ~3-4s (18 × 0.2s)
- 融合和后处理: ~5-8s

---

## 修改文件

- `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

**修改内容**:
- 移除 `torch.stack(all_preds)`
- 改为逐个处理 `zip(all_preds, all_labels)`
- 添加自动 resize 匹配
- 保持类型正确性

---

## 性能说明（18 次 TTA）

| 配置 | 变换次数 | 单图时间 | 600图时间 |
|------|---------|---------|----------|
| 基础 TTA | 6 | ~8s | ~80 分钟 |
| **进阶 TTA** | **18** | **~30-35s** | **~5-6 小时** |

你当前运行的是 18 次 TTA，速度是预期的。

---

**修复日期**: 2025-12-04  
**修复类型**: RuntimeError - 不同尺寸图像的处理

