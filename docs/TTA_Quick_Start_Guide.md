# TTA 快速验证和断点续传使用指南

本指南介绍如何使用 TTA 的三个新功能：
1. **快速验证模式** - 快速发现问题
2. **断点续传机制** - 不怕中断，节省时间
3. **Sanity Check** - 提前检测问题

---

## 🚀 方案 1: 快速验证模式

### 功能说明
在开发和调试阶段，不需要等待完整的评估（可能需要几小时），只需要处理前几张图片就能快速发现问题。

### 命令行参数

#### `--debug`
只处理前 5 张图片（快速调试）

```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --debug
```

**预期输出：**
```
🔍 DEBUG MODE: Processing only first 5 images (out of 600)
```

#### `--quick_test N`
只处理前 N 张图片（可自定义数量）

```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --quick_test 50
```

**预期输出：**
```
🔍 QUICK TEST MODE: Processing only first 50 images (out of 600)
```

### 使用场景
- ✅ 初次运行 TTA，验证配置是否正确
- ✅ 修改代码后，快速测试是否引入错误
- ✅ 尝试不同的 TTA 参数组合

---

## 💾 方案 2: 断点续传机制

### 功能说明
长时间运行的 TTA 评估（如 600 张图 × 18 次增强 = 10800 次推理）可能因为各种原因中断：
- 网络断开
- 服务器重启
- 内存不足导致崩溃
- 手动中断

使用断点续传，可以从中断处继续，不需要重新开始。

### 命令行参数

#### `--tta_cache_dir DIR`
指定缓存目录（自动创建）

```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_cache_dir outputs/tta_cache
```

#### `--tta_no_resume`
忽略缓存，从头开始（即使缓存存在）

```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_cache_dir outputs/tta_cache \
    --tta_no_resume
```

### 工作原理

#### 1. 自动保存进度
每处理 **10 张图片**就自动保存：
- `tta_predictions.pkl` - 预测结果（二进制文件）
- `processed_indices.txt` - 已处理图片索引列表（文本文件）

**示例日志：**
```
Processed 10/600 images with TTA (avg: 30.2s/img, ETA: 4.9h)
💾 Saving checkpoint (10/600 images)
✅ Checkpoint saved successfully

Processed 20/600 images with TTA (avg: 29.8s/img, ETA: 4.8h)
💾 Saving checkpoint (20/600 images)
✅ Checkpoint saved successfully
```

#### 2. 自动恢复进度
重新运行相同命令时，自动从缓存恢复：

**示例日志：**
```
📂 Loading cached predictions from outputs/tta_cache/tta_predictions.pkl
✅ Loaded 150 cached predictions (resuming from checkpoint)

⏭️  Skipping image 1/600 (already processed)
⏭️  Skipping image 2/600 (already processed)
...
⏭️  Skipping image 150/600 (already processed)

Processing image 151/600...  ← 从这里继续
```

#### 3. 自动清理缓存
评估成功完成后，自动删除缓存文件：

**示例日志：**
```
TTA evaluation completed: {'iou': 0.8234, 'dice': 0.9012}
🗑️  Cleaned up cache files (evaluation completed successfully)
```

### 查看缓存进度

```bash
# 查看已处理的图片数量
wc -l outputs/tta_cache/processed_indices.txt

# 查看最后处理的几张图片
tail outputs/tta_cache/processed_indices.txt

# 查看缓存文件大小
ls -lh outputs/tta_cache/
```

### 使用示例

#### 场景 1: 第一次运行（中途中断）

```bash
# 开始评估
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_cache_dir outputs/tta_cache

# 输出：
# Processed 150/600 images...
# [Ctrl+C] 手动中断或系统崩溃
```

#### 场景 2: 从断点继续

```bash
# 重新运行完全相同的命令
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_cache_dir outputs/tta_cache

# 输出：
# 📂 Loading cached predictions...
# ✅ Loaded 150 cached predictions
# ⏭️  Skipping 150 images (already processed)
# Processing image 151/600...  ← 继续处理
```

#### 场景 3: 强制重新开始

```bash
# 使用 --tta_no_resume 忽略缓存
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_cache_dir outputs/tta_cache \
    --tta_no_resume

# 或者手动删除缓存
rm -rf outputs/tta_cache
```

---

## 🔍 方案 4: Sanity Check（自动启用）

### 功能说明
在开始完整评估之前，自动进行快速检查，提前发现问题。

### 检查内容

#### 1. 数据验证
```
🔍 Running sanity checks before full evaluation...
  ✓ Total images to process: 600
  ✓ DataFrame columns: ['image', 'label']
```

#### 2. 设备检查
```
  ✓ Model device: cuda
  ✓ CUDA available: True
  ✓ GPU memory allocated: 2.34 GB
```

#### 3. TTA Pipeline 测试
```
  Testing TTA pipeline on first image...
    Image shape: (2848, 4288, 3)
    Output shape: (2848, 4288)
  ✓ TTA pipeline test PASSED!
```

#### 4. 成功提示
```
✅ All sanity checks passed! Starting full evaluation...
```

### 如果检查失败

**示例：**
```
❌ Sanity check FAILED! Error: Output shape mismatch! Expected (2848, 4288), got (1024, 1024)
Please fix the issue before running full evaluation.
RuntimeError: TTA Sanity Check Failed: ...
```

**处理方式：**
- 程序会立即终止（不会浪费时间）
- 查看错误信息
- 修复问题
- 重新运行

---

## 📊 推荐工作流程

### 第一次开发/调试

#### 步骤 1: 快速验证（5 张图，2 分钟）
```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --debug
```

**如果成功** → 进入步骤 2  
**如果失败** → 修复问题 → 重新运行步骤 1

#### 步骤 2: 中等规模测试（50 张图，25 分钟）
```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --quick_test 50
```

**如果成功** → 进入步骤 3  
**如果失败** → 修复问题 → 重新运行步骤 1

#### 步骤 3: 完整评估（带缓存，5 小时）
```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-xxx \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --tta_cache_dir outputs/tta_cache
```

**如果中断** → 重新运行相同命令，自动从断点继续

---

## 🎯 使用渐进式测试脚本

我们提供了自动化的渐进式测试脚本：

```bash
# 给脚本添加执行权限
chmod +x test_tta_progressive.sh

# 运行渐进式测试
./test_tta_progressive.sh
```

**脚本会自动执行：**
1. ✅ 阶段 1: 单样本测试（1 张图）
2. ✅ 阶段 2: 小批量测试（10 张图）
3. ✅ 阶段 3: 中等批量测试带缓存（50 张图）
4. ✅ 阶段 4: 验证断点续传（重新运行阶段 3）

**如果任何阶段失败，脚本会立即停止并显示错误。**

---

## 💡 最佳实践

### 1. 开发阶段
- ✅ 始终使用 `--debug` 或 `--quick_test` 快速验证
- ✅ 不要直接运行完整评估

### 2. 生产评估
- ✅ 始终使用 `--tta_cache_dir` 启用断点续传
- ✅ 定期检查缓存进度
- ✅ 如果中断，直接重新运行相同命令

### 3. 资源受限环境
- ✅ 使用 `--quick_test` 分批评估
- ✅ 例如：先 `--quick_test 100`，成功后再运行全部

### 4. 清理缓存
- ✅ 成功完成的评估会自动清理缓存
- ✅ 如果手动中断，可以保留缓存以便继续
- ✅ 如果确定不需要，手动删除：`rm -rf outputs/tta_cache`

---

## ⚠️ 注意事项

### 缓存文件大小
- 每张图片的预测约 10-50 MB（取决于分辨率）
- 600 张图片 ≈ 6-30 GB 缓存空间
- 确保有足够的磁盘空间

### 缓存一致性
如果更改了以下任何参数，**必须使用 `--tta_no_resume`** 或删除缓存：
- TTA 参数（scales, flips, rotations）
- 模型检查点
- 数据集

### 并发运行
- ❌ 不要同时运行多个使用相同 `--tta_cache_dir` 的进程
- ✅ 如果需要并发，使用不同的缓存目录

---

## 🎉 总结

| 功能 | 命令行参数 | 适用场景 |
|------|-----------|---------|
| **快速验证** | `--debug` 或 `--quick_test N` | 开发、调试、验证配置 |
| **断点续传** | `--tta_cache_dir DIR` | 长时间评估、不稳定环境 |
| **Sanity Check** | 自动启用 | 所有场景（提前发现问题） |

**推荐组合：**
```bash
# 开发阶段
python run_semantic_segmentation.py ... --tta_enable --debug

# 生产评估
python run_semantic_segmentation.py ... --tta_enable --tta_cache_dir outputs/cache
```

这样可以：
- ✅ 快速发现问题（--debug）
- ✅ 不怕中断（--tta_cache_dir）
- ✅ 提前检测（Sanity Check 自动运行）

---

## 📞 故障排除

### 问题 1: 缓存损坏
**症状：** 加载缓存时报错  
**解决：** 删除缓存重新开始
```bash
rm -rf outputs/tta_cache
```

### 问题 2: 磁盘空间不足
**症状：** "No space left on device"  
**解决：** 使用 `--quick_test` 分批处理，或清理缓存

### 问题 3: Sanity Check 失败
**症状：** 提前终止，显示错误信息  
**解决：** 根据错误信息修复问题，然后重新运行

---

**祝使用顺利！🚀**

