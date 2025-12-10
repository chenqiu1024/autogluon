# TTA 超参数调节实验指南

## 文档说明

本文档基于 `docs/cursor_tta-251204.md` 的讨论记录和截图中的超参建议，提供了针对 Test-Time Augmentation (TTA) 的系统化调参方案。

## 一、TTA 超参数概览

根据 Consistency Loss (TTA / multi-scale) 的调参建议，主要超参数分为两个方向：

### 1. TTA 方向超参

| 参数 | 说明 | 推荐范围 | 命令行参数 |
|------|------|----------|------------|
| **TTA_augment_types** | 翻转、旋转、镜像等增强类型 | horizontal_flip, vertical_flip, rotate ±10° | `--tta_flips`, `--tta_rotations` |
| **TTA_steps / num_aug** | 每个样本生成增强版本数量 | 2~4 个组合 | 通过 scales×flips×rotations 控制 |
| **TTA_fusion** | 融合方法 | mean / weighted_mean | `--tta_fusion` |

### 2. Multi-scale 方向超参

| 参数 | 说明 | 推荐范围 | 命令行参数 |
|------|------|----------|------------|
| **scale_ratios** | 输入图片缩放倍率 | [0.75, 1.0, 1.25] 或 [0.8, 1.0, 1.2] | `--tta_scales` |
| **resize_method** | 缩放方法 | bilinear / bicubic | `--tta_resize_method` |

### 3. 后处理超参

| 参数 | 说明 | 推荐范围 | 命令行参数 |
|------|------|----------|------------|
| **threshold** | 二值化阈值 | 0.4~0.6 | `--tta_threshold` |
| **min_area_ratio** | 小连通域去除阈值 | 0.0005~0.002 | `--tta_min_area` |
| **use_morphology** | 是否使用形态学闭运算 | True / False | `--tta_morphology` |

---

## 二、实验设计原则

### 2.1 基线配置（Baseline）

首先建立不使用 TTA 的基线性能：

```bash
# Baseline: 不使用 TTA
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --output_dir outputs/baseline
```

### 2.2 调参策略

采用**逐步优化**策略，避免组合爆炸：

1. **阶段 1**：优化 Multi-scale 配置（scales + resize_method）
2. **阶段 2**：优化 TTA augmentation 配置（flips + rotations）
3. **阶段 3**：优化后处理参数（threshold + morphology）
4. **阶段 4**：最优组合验证

---

## 三、实验步骤与命令行

### 阶段 1：Multi-scale 配置优化

**目标**：找到最佳的缩放倍率和插值方法

#### 实验 1.1：基础 Multi-scale（bilinear）

```bash
# 实验 1.1.1: [0.75, 1.0, 1.25] + bilinear
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_1.1.1_scales_075_100_125_bilinear

# 实验 1.1.2: [0.8, 1.0, 1.2] + bilinear
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.8 1.0 1.2 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_1.1.2_scales_080_100_120_bilinear

# 实验 1.1.3: [0.9, 1.0, 1.1] + bilinear (更保守的缩放)
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.9 1.0 1.1 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_1.1.3_scales_090_100_110_bilinear
```

#### 实验 1.2：测试 bicubic 插值方法

```bash
# 实验 1.2.1: [0.75, 1.0, 1.25] + bicubic
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bicubic \
  --tta_fusion mean \
  --output_dir outputs/exp_1.2.1_scales_075_100_125_bicubic

# 实验 1.2.2: [0.8, 1.0, 1.2] + bicubic
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.8 1.0 1.2 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bicubic \
  --tta_fusion mean \
  --output_dir outputs/exp_1.2.2_scales_080_100_120_bicubic
```

**预期结果**：
- bicubic 通常比 bilinear 略好，但速度稍慢
- 选择性能最好的 scale_ratios + resize_method 组合进入下一阶段

---

### 阶段 2：TTA Augmentation 配置优化

**假设**：阶段 1 中 `[0.75, 1.0, 1.25] + bilinear` 效果最好

#### 实验 2.1：测试不同的翻转策略

```bash
# 实验 2.1.1: 仅 horizontal flip
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_2.1.1_flip_h

# 实验 2.1.2: horizontal + vertical flips
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal vertical \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_2.1.2_flip_hv
```

#### 实验 2.2：测试旋转增强

```bash
# 实验 2.2.1: 添加 ±10° 旋转
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations -10 0 10 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_2.2.1_rot_10

# 实验 2.2.2: 添加 ±5° 旋转（更保守）
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations -5 0 5 \
  --tta_resize_method bilinear \
  --tta_fusion mean \
  --output_dir outputs/exp_2.2.2_rot_5
```

**注意事项**：
- 旋转会显著增加推理时间（scales=3, flips=2, rotations=3 → 18 次推理）
- 对于皮肤病变分割（ISIC2017），旋转增强可能收益有限

---

### 阶段 3：后处理参数优化

**假设**：阶段 2 中最佳配置为 `scales=[0.75,1.0,1.25], flips=[none,horizontal], rotations=[0]`

#### 实验 3.1：阈值调节

```bash
# 实验 3.1.1: threshold = 0.4
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.4 \
  --tta_fusion mean \
  --output_dir outputs/exp_3.1.1_thresh_04

# 实验 3.1.2: threshold = 0.45
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.45 \
  --tta_fusion mean \
  --output_dir outputs/exp_3.1.2_thresh_045

# 实验 3.1.3: threshold = 0.5 (default)
# (使用阶段 2 的最佳结果)

# 实验 3.1.4: threshold = 0.55
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.55 \
  --tta_fusion mean \
  --output_dir outputs/exp_3.1.4_thresh_055
```

#### 实验 3.2：形态学后处理

```bash
# 实验 3.2.1: 添加形态学闭运算
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.5 \
  --tta_morphology \
  --tta_fusion mean \
  --output_dir outputs/exp_3.2.1_morphology

# 实验 3.2.2: 形态学 + 最佳阈值（假设为 0.45）
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.45 \
  --tta_morphology \
  --tta_fusion mean \
  --output_dir outputs/exp_3.2.2_morphology_thresh045
```

#### 实验 3.3：小区域去除

```bash
# 实验 3.3.1: min_area_ratio = 0.0005
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.5 \
  --tta_min_area 0.0005 \
  --tta_fusion mean \
  --output_dir outputs/exp_3.3.1_minarea_0005

# 实验 3.3.2: min_area_ratio = 0.002
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.5 \
  --tta_min_area 0.002 \
  --tta_fusion mean \
  --output_dir outputs/exp_3.3.2_minarea_002
```

---

### 阶段 4：最优组合验证

基于前三阶段的结果，组合最优超参数进行最终验证：

```bash
# 实验 4.1: 最优组合（示例）
# 假设最优配置为：
#   - scales: [0.75, 1.0, 1.25]
#   - resize_method: bilinear
#   - flips: [none, horizontal]
#   - rotations: [0]
#   - threshold: 0.45
#   - morphology: True
#   - fusion: mean

python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.45 \
  --tta_morphology \
  --tta_fusion mean \
  --output_dir outputs/exp_4.1_best_config

# 实验 4.2: 最优组合 + weighted_mean fusion
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.45 \
  --tta_morphology \
  --tta_fusion weighted_mean \
  --output_dir outputs/exp_4.2_best_config_weighted
```

---

## 四、快速验证模式

在正式实验前，建议使用 `--quick_test` 参数快速验证配置：

```bash
# 快速测试：仅处理前 50 张图片
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --quick_test 50 \
  --output_dir outputs/quick_test
```

---

## 五、结果分析与对比

### 5.1 指标记录表

建议创建 Excel 表格记录所有实验结果：

| Exp ID | Scales | Resize | Flips | Rotations | Threshold | Morphology | Fusion | IoU | Dice | Time(s/img) |
|--------|--------|--------|-------|-----------|-----------|------------|--------|-----|------|-------------|
| baseline | - | - | - | - | - | - | - | 0.7500 | 0.8500 | 0.5 |
| 1.1.1 | [0.75,1.0,1.25] | bilinear | [n,h] | [0] | 0.5 | False | mean | ? | ? | ? |
| 1.1.2 | [0.8,1.0,1.2] | bilinear | [n,h] | [0] | 0.5 | False | mean | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 5.2 结果可视化

```bash
# 使用 analyze_results.py 分析实验结果
python analyze_results.py --results_dir outputs/
```

### 5.3 关键观察点

1. **性能提升幅度**：
   - 预期 TTA 可带来 +0.3~1.0% Dice/IoU 提升
   - 如果提升 < 0.2%，可能不值得增加推理时间

2. **推理时间**：
   - baseline: ~0.5s/image
   - TTA (6 augmentations): ~3-5s/image
   - TTA (18 augmentations): ~9-15s/image

3. **收益递减**：
   - Multi-scale 通常是最有效的增强
   - Flips 次之
   - Rotations 收益可能有限，但时间成本高

---

## 六、推荐的快速配置

如果时间有限，直接尝试以下几个高性价比配置：

### 配置 A：轻量级 TTA（推荐）

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal \
  --tta_rotations 0 \
  --tta_resize_method bilinear \
  --tta_threshold 0.5 \
  --tta_fusion mean
```

**特点**：6 次推理，速度快，效果稳定

### 配置 B：中等 TTA

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
  --tta_morphology \
  --tta_fusion mean
```

**特点**：6 次推理 + 更好的插值 + 形态学后处理

### 配置 C：完整 TTA（最高精度）

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251208_054753 \
  --tta_enable \
  --tta_scales 0.75 1.0 1.25 \
  --tta_flips none horizontal vertical \
  --tta_rotations -10 0 10 \
  --tta_resize_method bicubic \
  --tta_threshold 0.5 \
  --tta_morphology \
  --tta_fusion weighted_mean
```

**特点**：27 次推理，最高精度，但速度最慢

---

## 七、实验执行脚本

为方便批量执行，可以创建 shell 脚本：

```bash
# experiments/run_tta_experiments.sh

#!/bin/bash

CKPT="AutogluonModels/ag-20251208_054753"
TASK="isic2017"

# Baseline
echo "Running baseline..."
python run_semantic_segmentation.py \
  --task $TASK --eval --ckpt_path $CKPT \
  --output_dir outputs/baseline

# Experiments for Stage 1
echo "Running Stage 1 experiments..."
for scales in "0.75 1.0 1.25" "0.8 1.0 1.2" "0.9 1.0 1.1"; do
  for resize in "bilinear" "bicubic"; do
    exp_name="exp_scales_${scales// /_}_${resize}"
    echo "  - $exp_name"
    python run_semantic_segmentation.py \
      --task $TASK --eval --ckpt_path $CKPT \
      --tta_enable \
      --tta_scales $scales \
      --tta_flips none horizontal \
      --tta_rotations 0 \
      --tta_resize_method $resize \
      --tta_fusion mean \
      --output_dir "outputs/$exp_name"
  done
done

echo "All experiments completed!"
```

---

## 八、总结

### 8.1 调参优先级

1. **Multi-scale 配置** (scales + resize_method) - **最重要**
2. **Flips 配置** - 成本低，收益稳定
3. **后处理参数** (threshold + morphology) - 微调用
4. **Rotations** - 收益/成本比低，谨慎使用

### 8.2 预期性能提升

- **轻量级 TTA (配置 A)**：+0.3~0.5% Dice/IoU
- **中等 TTA (配置 B)**：+0.5~0.8% Dice/IoU
- **完整 TTA (配置 C)**：+0.8~1.2% Dice/IoU

### 8.3 注意事项

1. TTA 对模型本身质量要求较高，弱模型 TTA 收益有限
2. 不同数据集最优配置可能不同，需要针对性调参
3. 推理时间是实际应用的重要考量因素
4. 使用 `--quick_test` 快速验证配置可行性

---

## 参考资料

- 对话记录：`docs/cursor_tta-251204.md`
- TTA 实现：`multimodal/src/autogluon/multimodal/utils/tta_utils.py`
- 训练脚本：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

---

**文档创建时间**：2024-12-10  
**适用任务**：Semantic Segmentation (ISIC2017, Polyp, CAMO, etc.)  
**模型**：Conv-LoRA SAM
