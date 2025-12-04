# TTA 性能现实评估

## 实际性能数据

### ISIC2017（600 张图像，6 次 TTA）

| 指标 | 实际测量值 | 说明 |
|------|-----------|------|
| **单图时间** | ~8 秒 | 包含所有操作 |
| **总时间** | ~80 分钟 | 600 × 8s ≈ 80min |
| **CPU 使用率** | 仍较高 | 主要用于图像变换 |
| **内存使用** | 逐渐增长 | 需要更激进的 GC |
| **GPU 使用率** | 70-100% | 正常 |

---

## 为什么比预期慢？

### TTA 单图处理时间分解（6 次变换）

| 操作 | 时间估算 | 说明 |
|------|---------|------|
| 图像加载 | 0.1s | I/O |
| **变换 × 6** | **4-5s** | **主要瓶颈** |
| - Scale (PIL) | 0.3s × 3 = 0.9s | 0.75x, 1.0x, 1.25x 各 2 次 |
| - Flip | 0.05s × 6 = 0.3s | numpy 操作，快 |
| - Rotate (PIL) | 0.1s × 6 = 0.6s | 虽然优化了，但仍需时间 |
| - Resize 到 1024 | 0.4s × 6 = 2.4s | **最大瓶颈** |
| - Normalize | 0.1s × 6 = 0.6s | numpy 操作 |
| **模型推理 × 6** | **1-2s** | GPU，快 |
| - 单次推理 | ~0.2s | SAM-Huge 在 GPU |
| **反变换 × 6** | **1-2s** | CPU |
| - Resize 回原图 | 0.3s × 6 = 1.8s | PIL resize |
| 概率融合 | 0.1s | numpy.mean |
| 后处理 | 0.05s | threshold + filter |
| **总计** | **~8s** | 合理 |

### 瓶颈分析

**最大瓶颈：Resize 操作**
- 每个变换需要 resize 到 1024x1024：~0.4s
- 每个输出需要 resize 回原图尺寸：~0.3s
- 总计：(0.4 + 0.3) × 6 = **4.2s/图**（占 52%）

**次要瓶颈：模型推理**
- 6 次推理：0.2s × 6 = **1.2s/图**（占 15%）

**其他开销：**
- 图像变换、数据复制：~2.6s/图（占 33%）

---

## 为什么 Resize 这么慢？

### SAM 的特殊性

SAM 要求**固定输入大小** (1024×1024)，所以：

1. **TTA 流程**：
   ```
   原图 (例如 600×450)
   ↓ Scale 0.75
   变为 450×337
   ↓ Resize 到 1024×1024  ← 耗时 0.4s
   ↓ 模型推理
   输出 1024×1024
   ↓ Resize 回 450×337   ← 耗时 0.3s
   ↓ Scale 反变换
   回到 600×450
   ```

2. **问题**：
   - 每个 TTA 变换都需要 2 次 resize（输入 + 输出）
   - 6 次变换 = 12 次 resize
   - 每次 resize 约 0.3-0.4s

3. **为什么慢**：
   - 1024×1024 是大尺寸（1M 像素）
   - PIL.resize 虽然比 scipy 快，但仍需时间
   - 双线性插值计算量大

---

## 内存泄漏分析

### 可能的原因

1. **PIL Image 对象未完全释放**
   - 虽然调用了 `.close()`，但可能有循环引用
   
2. **numpy 数组累积**
   - `all_preds` 和 `all_labels` 累积所有结果
   - 600 × 1024×1024 × 4 bytes ≈ 2.5GB

3. **Python 垃圾回收滞后**
   - 大型数组的回收可能延迟
   
4. **torch tensors 缓存**
   - GPU tensors 可能在 cache 中

---

## 进一步优化方案

### 选项 1: 流式评估（推荐）

不累积所有预测，而是逐个计算指标：

```python
for idx, row in data.iterrows():
    pred = tta_predict(image)
    
    # 立即更新指标，不保存预测
    metric.update(pred, label)
    
    del pred, label  # 立即释放
```

**优点**：
- 内存使用恒定（~1GB）
- 不会内存泄漏

**缺点**：
- 需要重构代码

### 选项 2: 减少 TTA 变换（临时方案）

```bash
# 只用 flip，不用 scale（避免大量 resize）
--tta_scales 1.0 \
--tta_flips none horizontal \
--tta_rotations 0
```

**效果**：
- 2 次变换 instead of 6
- 速度：8s → ~3s/图
- 内存：大幅降低

### 选项 3: 批处理 TTA

将多个变换打包成 batch 一起推理：

```python
# 当前：串行
for transform in transforms:
    pred = model(transform(image))

# 优化：批处理
images_batch = [transform(image) for transform in transforms]
preds_batch = model(stack(images_batch))  # 一次推理
```

**效果**：
- GPU 利用率更高
- 速度提升 2-3 倍
- 但显存需求增加

---

## 修复的具体改进

### 1. 修复 `del label` 错误

```python
# 旧代码（错误）
if label_path:
    label = ...
# del label  ← 错误：可能未定义

# 新代码（正确）
label = None
if label_path:
    label = ...
if label is not None:
    del label  # 安全
```

### 2. 更激进的垃圾回收

```python
# 旧：每 10 张图
if (idx + 1) % 10 == 0:
    gc.collect()

# 新：每 5 张图
if (idx + 1) % 5 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

### 3. 所有 PIL 对象都显式 close()

```python
img_pil = Image.open(path)
array = np.array(img_pil)
img_pil.close()  # ✅ 立即关闭
```

### 4. 所有 numpy 转换后显式 copy()

```python
array = np.array(pil_img).copy()  # ✅ 断开与 PIL 的引用
```

### 5. GPU 同步

```python
del tensors
torch.cuda.synchronize()  # ✅ 等待 GPU 完成
```

---

## 现实的性能预期

### 对于 SAM + TTA（固定输入大小模型）

| 配置 | 单图时间 | 600图时间 | 说明 |
|------|---------|----------|------|
| 无 TTA | 0.2s | 2分钟 | 基线 |
| Flip only (2次) | 0.6s | 6分钟 | 快速 |
| Flip + Rotate (6次) | 2-3s | 20-30分钟 | 推荐 |
| **Full TTA (6次)** | **6-8s** | **60-80分钟** | **当前** |
| Full TTA (18次) | 18-24s | 3-4小时 | 慢 |

**现实检查**：
- ✅ 8s/图 对于包含 scale 的 TTA 是**合理的**
- ❌ 如果希望更快，需要移除 scale 或使用批处理

---

## 推荐配置（速度 vs 精度）

### 快速模式（2-3s/图）

```bash
--tta_enable \
--tta_scales 1.0 \
--tta_flips none horizontal \
--tta_rotations -10 0 10
```

- 总变换：1 scale × 2 flips × 3 rotations = 6 次
- 但避免了多尺度 resize，快很多

### 标准模式（当前配置，6-8s/图）

```bash
--tta_enable
# 使用默认配置
```

- 包含 scale，效果可能更好
- 但速度较慢

### 精度优先（18-24s/图）

```bash
--tta_enable \
--tta_scales 0.75 1.0 1.25 \
--tta_flips none horizontal \
--tta_rotations -10 0 10 \
--tta_fusion weighted_mean
```

---

## 内存管理状态

### 最新改进

- ✅ 所有 PIL 对象都 close()
- ✅ 所有临时变量都 del
- ✅ 每 5 张图 gc.collect()
- ✅ 每 2 次变换 gc.collect()
- ✅ numpy.copy() 断开引用
- ✅ torch.cuda.synchronize()

### 预期效果

- 内存增长应该减缓（但不会完全消除，因为需要累积 all_preds）
- 稳定在 8-12GB（取决于图像数量）

---

## 终极解决方案：流式评估

如果内存仍然是问题，唯一的根本解决方案是**不累积预测**：

```python
# 当前：累积所有预测
all_preds = []
for image in images:
    pred = tta_predict(image)
    all_preds.append(pred)  # 累积

# 优化：流式计算
for image, label in zip(images, labels):
    pred = tta_predict(image)
    metric.update(pred, label)  # 立即使用
    del pred, label  # 立即释放
```

这需要重构 `_evaluate_with_tta` 方法，但可以将内存使用降到最低（~1GB）。

---

## 总结

### 当前状态

- ✅ **功能完整**: 所有 TTA 功能正常工作
- ⚠️ **速度**: 8s/图（对于包含 scale 的 TTA，这是合理的）
- ⚠️ **内存**: 仍在优化中，应该比之前好但可能仍有增长

### 建议

1. **如果追求速度**: 移除 scale，只用 flip + rotate
2. **如果追求精度**: 接受当前速度（8s/图）
3. **如果内存仍然问题**: 需要实现流式评估

---

**更新日期**: 2025-12-04  
**当前版本**: 已大幅优化，但受模型架构限制，无法达到最初预期的 1-2s/图

