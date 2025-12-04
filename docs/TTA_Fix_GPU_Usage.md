# TTA GPU 使用问题修复

## 问题
运行 TTA 评估时：
- ✅ 日志显示正常
- ❌ GPU 使用率为 0%
- ❌ 显存使用为 0
- ❌ CPU 和内存使用率很高
- ❌ 程序"卡住"不动

## 原因
模型没有被移动到 GPU，所有计算都在 CPU 上进行，导致速度极慢（可能慢 50-100 倍）。

## 解决方案

### 1. 确保模型在 GPU 上

```python
# 检查 GPU 可用性
if torch.cuda.is_available():
    device = torch.device('cuda')
    if next(model.parameters()).device.type != 'cuda':
        logger.info("Moving model to GPU for TTA inference...")
        model = model.cuda()
else:
    device = torch.device('cpu')
    logger.warning("GPU not available, using CPU for TTA (will be slow)")

logger.info(f"TTA inference device: {device}")
```

### 2. 使用显式 device 而不是 model.device

```python
# 旧代码（可能不可靠）
img_tensor = img_tensor.unsqueeze(0).to(model.device)

# 新代码（显式指定）
img_tensor = img_tensor.unsqueeze(0).to(device)
```

### 3. 添加性能监控

添加了详细的计时信息：
- 第一张图像的处理时间（包括 warmup）
- 平均处理时间
- 预计完成时间（ETA）

---

## 预期行为

### 修复后的日志输出

```
TTA inference device: cuda
Evaluating with TTA (6 augmentations per image)...
First image processed in 2.35s (includes warmup)
Processed 10/600 images with TTA (avg: 1.2s/img, ETA: 11.8min)
Processed 20/600 images with TTA (avg: 1.1s/img, ETA: 10.6min)
...
```

### GPU 监控

运行后应该看到：
- ✅ GPU 使用率: 70-100%
- ✅ 显存使用: 取决于模型大小（SAM-Huge 约 2-4GB）
- ✅ CPU 使用率: 降低到 10-30%（仅用于数据加载）

---

## 性能对比

### CPU vs GPU（SAM-Huge，6次TTA）

| 设备 | 单图时间 | 600图总时间 | GPU使用 |
|------|---------|------------|---------|
| **CPU** | ~60-120s | ~10-20小时 | 0% |
| **GPU** | ~1-2s | ~10-20分钟 | 80-100% |

**速度提升**: **30-60倍**

---

## 故障排查

### 如果 GPU 仍然不被使用

1. **检查 PyTorch CUDA**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```
   应该输出 `True`

2. **检查模型是否在 GPU 上**
   ```bash
   python3 -c "
   from autogluon.multimodal import MultiModalPredictor
   predictor = MultiModalPredictor.load('AutogluonModels/ag-XXX')
   model = predictor._learner._model
   print('Model device:', next(model.parameters()).device)
   "
   ```
   应该输出 `cuda:0`

3. **检查 CUDA 版本兼容性**
   ```bash
   nvidia-smi
   python3 -c "import torch; print(torch.version.cuda)"
   ```

### 如果看到 "GPU not available" 警告

说明 PyTorch 无法访问 GPU，可能需要：
1. 重新安装 PyTorch with CUDA
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. 检查 CUDA 驱动是否正确安装

---

## 常见问题

### Q: 为什么第一张图像特别慢？

A: 第一次推理包括：
- CUDA 初始化
- 模型 warmup
- cuDNN 自动调优
- 内存分配

后续图像会快很多。

### Q: 为什么 CPU 使用率仍然较高？

A: 这是正常的，因为：
- 图像加载（I/O）
- TTA 变换（rotate, flip, scale）在 CPU 上
- 数据预处理（resize, normalize）部分在 CPU 上

但主要计算（模型推理）应该在 GPU 上。

### Q: 显存不足怎么办？

A: 可以：
1. 减少 batch size（目前是 1，已经最小）
2. 使用更小的模型（如 SAM-Base）
3. 减少 TTA 变换数量

---

## 修改文件

- `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

**修改内容**:
1. 添加 GPU 设备检查和模型移动
2. 使用显式 device 变量
3. 添加性能监控和 ETA 显示

---

**修复日期**: 2025-12-04  
**影响**: 性能提升 30-60 倍

