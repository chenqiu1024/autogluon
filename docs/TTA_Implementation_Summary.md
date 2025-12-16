# TTA 超参数控制功能实现总结

## 更新日期
2024-12-10

## 一、功能概述

根据 `docs/cursor_tta-251204.md` 中的讨论和截图建议，本次更新为 Test-Time Augmentation (TTA) 功能添加了更细粒度的超参数控制，支持对以下参数进行调节：

### 新增/增强的参数

1. **resize_method**: 缩放插值方法
   - 选项: `bilinear` (快速) / `bicubic` (高质量)
   - 影响所有 multi-scale 变换的图像缩放质量

2. **完整的 TTA 配置控制**:
   - `tta_scales`: 缩放倍率列表
   - `tta_flips`: 翻转类型列表
   - `tta_rotations`: 旋转角度列表
   - `tta_fusion`: 融合方法 (mean / weighted_mean)
   - `tta_threshold`: 二值化阈值
   - `tta_min_area`: 小连通域去除阈值
   - `tta_morphology`: 形态学后处理开关

---

## 二、修改文件清单

### 2.1 核心实现文件

| 文件 | 修改内容 |
|------|---------|
| `multimodal/src/autogluon/multimodal/utils/tta_utils.py` | 1. `ScaleTransform` 类新增 `resize_method` 参数<br>2. `TTAPredictor` 类新增 `resize_method` 参数<br>3. 支持 bilinear/bicubic 插值方法 |
| `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py` | `enable_tta()` 方法新增 `resize_method` 参数 |
| `multimodal/src/autogluon/multimodal/predictor.py` | `enable_tta()` 方法新增 `resize_method` 参数 |
| `examples/automm/Conv-LoRA/run_semantic_segmentation.py` | 新增 `--tta_resize_method` 命令行参数 |

### 2.2 文档与工具

| 文件 | 说明 |
|------|------|
| `docs/TTA_Hyperparameter_Tuning_Guide.md` | **核心文档**：完整的TTA调参指南，包含实验设计、命令行示例、结果分析 |
| `examples/automm/Conv-LoRA/experiments_tta_tuning.sh` | 批量实验执行脚本，支持分阶段运行 |
| `examples/automm/Conv-LoRA/analyze_tta_results.py` | 结果分析脚本，自动生成 Markdown 报告 |
| `docs/TTA_Implementation_Summary.md` | 本文档：实现总结 |

---

## 三、使用方法

### 3.1 基础用法

```bash
# 启用 TTA，使用 bilinear 插值
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_resize_method bilinear

# 使用 bicubic 插值（更高质量）
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_resize_method bicubic
```

### 3.2 完整配置示例

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bicubic \
  --tta_threshold 0.5 \
  --tta_min_area 0.001 \
  --tta_morphology \
  --tta_fusion mean \
  --output_dir outputs/tta_experiment
```

### 3.3 批量实验

```bash
# 快速验证（仅测试 3 个配置，每个 50 张图）
cd examples/automm/Conv-LoRA
bash experiments_tta_tuning.sh quick

# 运行阶段 1：Multi-scale 配置优化
bash experiments_tta_tuning.sh 1

# 运行阶段 2：TTA Augmentation 配置优化
bash experiments_tta_tuning.sh 2

# 运行阶段 3：后处理参数优化
bash experiments_tta_tuning.sh 3

# 运行所有阶段
bash experiments_tta_tuning.sh all
```

### 3.4 结果分析

```bash
# 分析实验结果，生成报告
python analyze_tta_results.py \
  --results_dir outputs \
  --output TTA_Results_Summary.md

# 查看报告
cat TTA_Results_Summary.md
```

---

## 四、超参数调节建议

### 4.1 TTA 方向超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tta_scales` | `[0.75, 1.0, 1.25]` 或 `[0.8, 1.0, 1.2]` | 缩放倍率，建议 2~4 个尺度 |
| `tta_flips` | `["none", "horizontal"]` | 水平翻转通常最有效 |
| `tta_rotations` | `[0]` 或 `[-10, 0, 10]` | 旋转会显著增加时间成本 |
| `tta_fusion` | `"mean"` 或 `"weighted_mean"` | mean 简单稳定，weighted_mean 可能略好 |

### 4.2 Multi-scale 方向超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tta_resize_method` | `"bilinear"` 或 `"bicubic"` | bilinear 快，bicubic 质量略好 |

### 4.3 后处理超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tta_threshold` | `0.4 ~ 0.6` | 二值化阈值，建议在验证集上调节 |
| `tta_min_area` | `0.0005 ~ 0.002` | 去除小连通域，避免噪声 |
| `tta_morphology` | `True` / `False` | 形态学闭运算，平滑边界 |

---

## 五、实验设计策略

### 5.1 逐步优化流程

1. **阶段 1**: 优化 Multi-scale 配置
   - 测试不同的 `tta_scales` 组合
   - 对比 `bilinear` vs `bicubic`
   - 选择最佳 scale 和 resize_method

2. **阶段 2**: 优化 TTA Augmentation
   - 测试不同的 `tta_flips` 组合
   - 测试是否添加 `tta_rotations`
   - 评估性能/时间权衡

3. **阶段 3**: 优化后处理参数
   - 在验证集上调节 `tta_threshold`
   - 测试 `tta_morphology` 的效果
   - 微调 `tta_min_area`

4. **阶段 4**: 最优组合验证
   - 组合前三阶段的最佳参数
   - 在完整测试集上验证

### 5.2 快速验证模式

对于时间有限的情况，推荐直接测试 3 个预设配置：

- **配置 A (轻量级)**: 6 次推理，速度快
- **配置 B (中等)**: 6 次推理 + bicubic + morphology
- **配置 C (完整)**: 27 次推理，最高精度

```bash
bash experiments_tta_tuning.sh quick
```

---

## 六、预期性能提升

根据医学图像分割的一般经验：

| 配置复杂度 | 推理次数 | 预期 IoU 提升 | 预期 Dice 提升 |
|-----------|---------|--------------|---------------|
| 轻量级 (配置 A) | 6 | +0.3~0.5% | +0.3~0.5% |
| 中等 (配置 B) | 6 | +0.5~0.8% | +0.5~0.8% |
| 完整 (配置 C) | 27 | +0.8~1.2% | +0.8~1.2% |

**注意事项**：
- TTA 对强模型效果更明显
- 不同数据集效果会有差异
- 推理时间会按推理次数线性增长

---

## 七、代码示例

### 7.1 Python API

```python
from autogluon.multimodal import MultiModalPredictor

# 加载模型
predictor = MultiModalPredictor.load("AutogluonModels/ag-20251208_054753")

# 启用 TTA
predictor.enable_tta(
    scales=[0.75, 1.0, 1.25],
    flips=["none", "horizontal"],
    rotations=[0],
    fusion_method="mean",
    resize_method="bicubic",  # 新增参数
    threshold=0.5,
    min_area_ratio=0.001,
    use_morphology=True,
)

# 评估
results = predictor.evaluate(test_df, metrics=["iou", "dice"])
print(results)
```

### 7.2 命令行

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bicubic \
  --tta_threshold 0.5 \
  --tta_min_area 0.001 \
  --tta_morphology \
  --tta_fusion mean
```

---

## 八、常见问题

### Q1: bilinear 和 bicubic 差异有多大？

**A**: 在大多数情况下，bicubic 相比 bilinear 可能带来 +0.1~0.3% 的性能提升，但速度会慢 10~20%。建议先用 bilinear 快速实验，确定其他超参后再尝试 bicubic。

### Q2: 旋转增强是否值得？

**A**: 对于皮肤病变分割（ISIC2017）等任务，旋转增强的收益通常有限（+0.1~0.2%），但会使推理时间增加 3 倍。建议优先调节 scales 和 flips。

### Q3: 如何选择最优的 threshold？

**A**: 可以在验证集上使用 `TTAPredictor.tune_threshold()` 方法自动搜索最优阈值，或在实验中尝试 0.4, 0.45, 0.5, 0.55 几个值。

### Q4: 实验结果保存在哪里？

**A**: 每个实验的结果保存在 `outputs/<实验名称>/metrics.txt` 文件中，包含 IoU 和 Dice 指标。

### Q5: 如何快速对比多个实验？

**A**: 使用 `analyze_tta_results.py` 脚本可以自动扫描 `outputs/` 目录，生成包含所有实验对比的 Markdown 报告。

---

## 九、参考资料

- **TTA 调参指南**: `docs/TTA_Hyperparameter_Tuning_Guide.md`
- **对话记录**: `docs/cursor_tta-251204.md`
- **TTA 实现**: `multimodal/src/autogluon/multimodal/utils/tta_utils.py`
- **实验脚本**: `examples/automm/Conv-LoRA/experiments_tta_tuning.sh`
- **结果分析**: `examples/automm/Conv-LoRA/analyze_tta_results.py`

---

## 十、后续改进方向

1. **自适应 TTA**: 根据预测置信度动态选择是否应用 TTA
2. **TTA_prob 参数**: 支持随机应用增强的概率控制（0.5~1.0）
3. **GPU 内存优化**: 进一步减少 TTA 推理时的显存占用
4. **训练时 Consistency Loss**: 利用 multi-scale consistency loss 提升训练效果

---

**文档维护者**: AutoGluon Conv-LoRA Team  
**最后更新**: 2024-12-10
