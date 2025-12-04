# Test-Time Augmentation (TTA) 使用指南

本文档介绍如何在 Conv-LoRA + GSPO + Adapter 语义分割实验中使用测试时增强（TTA）功能。

---

## 概述

TTA（Test-Time Augmentation）通过对同一测试样本应用多种数据增强，分别预测，再融合结果，从而提升模型的鲁棒性和精度。

### 主要特性

- ✅ **多尺度测试**：支持 0.5x ~ 1.5x 等多种缩放比例
- ✅ **翻转增强**：水平/垂直翻转
- ✅ **旋转增强**：小角度旋转（如 ±10°）
- ✅ **多种融合策略**：平均融合、加权平均融合
- ✅ **后处理优化**：阈值调整、小连通域去除、形态学平滑

### 预期收益

根据医学图像分割的经验数据，TTA 通常能带来：
- **Dice 提升**：0.3% ~ 1.0%
- **IoU 提升**：0.2% ~ 0.8%

---

## 快速开始

### 1. 基础版 TTA（6 次推理/图像）

最简单的用法，使用默认的多尺度 + 水平翻转：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-XXXXXXXX_XXXXXX \
    --tta_enable \
    --output_dir outputs/eval_tta_basic
```

**配置详情**：
- Scales: `[0.75, 1.0, 1.25]`
- Flips: `["none", "horizontal"]`
- Rotations: `[0]`
- **总推理次数**：3 scales × 2 flips × 1 rotation = **6 次**

### 2. 进阶版 TTA（18 次推理/图像）

加入小角度旋转，效果更好但更慢：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-XXXXXXXX_XXXXXX \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --output_dir outputs/eval_tta_advanced
```

**配置详情**：
- Scales: `[0.75, 1.0, 1.25]`
- Flips: `["none", "horizontal"]`
- Rotations: `[-10, 0, 10]`
- Fusion: `weighted_mean`（scale=1.0 权重更高）
- **总推理次数**：3 scales × 2 flips × 3 rotations = **18 次**

### 3. 极致版 TTA（40 次推理/图像）

适用于对精度要求极高的场景（如比赛、论文提交）：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-XXXXXXXX_XXXXXX \
    --tta_enable \
    --tta_scales 0.5 0.75 1.0 1.25 1.5 \
    --tta_flips none horizontal vertical \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --output_dir outputs/eval_tta_extreme
```

**配置详情**：
- Scales: `[0.5, 0.75, 1.0, 1.25, 1.5]`
- Flips: `["none", "horizontal", "vertical"]`（注意：垂直翻转仅适用于部分任务）
- Rotations: `[-10, 0, 10]`
- Fusion: `weighted_mean`
- 形态学后处理：开启
- **总推理次数**：5 scales × 3 flips × 3 rotations = **45 次**

---

## 参数详解

### TTA 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tta_enable` | flag | False | 启用 TTA |
| `--tta_scales` | float[] | [0.75, 1.0, 1.25] | 多尺度缩放因子 |
| `--tta_flips` | str[] | ["none", "horizontal"] | 翻转类型：none, horizontal, vertical |
| `--tta_rotations` | float[] | [0] | 旋转角度（度数） |
| `--tta_fusion` | str | "mean" | 融合方法：mean 或 weighted_mean |
| `--tta_threshold` | float | 0.5 | 二值化阈值 |
| `--tta_min_area` | float | 0.001 | 去除小于此比例的连通域 |
| `--tta_morphology` | flag | False | 启用形态学闭运算 |

### 参数选择建议

#### 1. **Scales（尺度）**

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 快速评估 | `[1.0]` | 不做多尺度，最快 |
| 标准配置 | `[0.75, 1.0, 1.25]` | 平衡速度和效果 |
| 高精度 | `[0.5, 0.75, 1.0, 1.25, 1.5]` | 覆盖更广的尺度范围 |

**注意**：
- 过小的尺度（如 0.5）可能导致细节丢失
- 过大的尺度（如 1.5）可能导致边界不准确

#### 2. **Flips（翻转）**

| 翻转类型 | 适用场景 |
|---------|----------|
| `none` | 基线，无翻转 |
| `horizontal` | **推荐**，适用于大多数医学/自然图像 |
| `vertical` | 仅适用于上下对称的任务（慎用！） |

**注意**：
- 对于有明确方向性的任务（如胸部 X 光），垂直翻转可能破坏语义
- 皮肤病变、细胞图像等通常可以安全使用水平翻转

#### 3. **Rotations（旋转）**

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 快速评估 | `[0]` | 不旋转 |
| 标准配置 | `[-10, 0, 10]` | 小角度旋转，稳定 |
| 高鲁棒性 | `[-15, -10, 0, 10, 15]` | 更多角度，更慢 |

**注意**：
- 医学图像不建议大角度旋转（如 ±45°），可能破坏解剖学方位
- 自然场景（如道路分割）可能可以尝试更大角度

#### 4. **Fusion（融合方法）**

| 方法 | 优点 | 缺点 |
|------|------|------|
| `mean` | 简单，稳定 | 所有增强权重相同 |
| `weighted_mean` | scale=1.0 权重更高，更合理 | 略微复杂 |

**推荐**：使用 `weighted_mean`，因为原始分辨率的预测通常更可靠。

---

## 后处理选项

### 1. 阈值调整（`--tta_threshold`）

- **默认值**：0.5
- **建议**：在验证集上搜索最佳阈值（0.3 ~ 0.7）
- **影响**：直接影响精度，建议优先调优此参数

**如何调整**：
```bash
# 尝试不同阈值
for threshold in 0.3 0.4 0.5 0.6 0.7; do
    python3 run_semantic_segmentation.py \
        --task isic2017 --eval \
        --ckpt_path AutogluonModels/ag-XXX \
        --tta_enable \
        --tta_threshold $threshold \
        --output_dir outputs/eval_tta_threshold_$threshold
done
```

### 2. 小连通域去除（`--tta_min_area`）

- **默认值**：0.001（图像面积的 0.1%）
- **作用**：去除噪声、伪阳性
- **建议**：
  - 如果分割目标很小（如小肿瘤），设为 0.0001 或更小
  - 如果分割目标较大（如器官），可以设为 0.005 甚至 0.01

### 3. 形态学处理（`--tta_morphology`）

- **作用**：填补小空洞、平滑边界
- **适用场景**：
  - ✅ 器官分割（需要平滑边界）
  - ✅ 病灶分割（去除内部空洞）
  - ❌ 精细结构分割（可能过度平滑）

---

## 实战案例

### 案例 1：ISIC2017 皮肤病变分割

**任务特点**：
- 病灶大小变化大
- 边界不规则
- 需要高精度边界定位

**推荐配置**：
```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251129_083619 \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_threshold 0.5 \
    --tta_min_area 0.0005 \
    --tta_morphology \
    --output_dir outputs/eval_tta_isic2017
```

**预期结果**：
- 基线（无 TTA）：IoU ~77.5%, Dice ~85.5%
- 使用 TTA：IoU ~78.0%, Dice ~86.0%（+0.5%）

### 案例 2：SBU-Shadow 阴影检测

**任务特点**：
- 强方向性（光照方向）
- 大面积区域
- 边界相对平滑

**推荐配置**：
```bash
python3 run_semantic_segmentation.py \
    --task SBU-shadow \
    --eval \
    --ckpt_path AutogluonModels/ag-XXX \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_fusion mean \
    --tta_min_area 0.01 \
    --output_dir outputs/eval_tta_sbu_shadow
```

**注意**：阴影检测不适合旋转增强，因为光照方向很重要。

---

## 性能对比

### 推理时间对比（单张图像）

| 配置 | 推理次数 | 相对时间 | 预期提升 |
|------|---------|---------|---------|
| 无 TTA | 1 | 1.0x | 基线 |
| 基础 TTA | 6 | 6.0x | +0.3~0.5% |
| 进阶 TTA | 18 | 18.0x | +0.5~0.8% |
| 极致 TTA | 45 | 45.0x | +0.8~1.0% |

**说明**：
- TTA 的推理时间与增强次数成正比
- 可以通过 GPU 并行处理部分加速（未实现）

### 显存占用

TTA 不会显著增加显存占用（因为是串行推理），但会增加总推理时间。

---

## 与训练集成

### 完整训练 + TTA 评估流程

```bash
#!/bin/bash

# 1. 训练模型（Conv-LoRA + GSPO + Adapter）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --output_dir outputs/train_hybrid

# 2. 标准评估（无 TTA）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251129_083619 \
    --output_dir outputs/eval_baseline

# 3. TTA 评估（基础版）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251129_083619 \
    --tta_enable \
    --output_dir outputs/eval_tta_basic

# 4. TTA 评估（进阶版）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251129_083619 \
    --tta_enable \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --output_dir outputs/eval_tta_advanced

# 5. 对比结果
echo "=== Results Comparison ==="
echo "Baseline:"
cat outputs/eval_baseline/metrics.txt
echo ""
echo "TTA Basic:"
cat outputs/eval_tta_basic/metrics.txt
echo ""
echo "TTA Advanced:"
cat outputs/eval_tta_advanced/metrics.txt
```

---

## 常见问题

### Q1: TTA 会增加训练时间吗？

**A**：不会。TTA 仅在**推理/评估**阶段使用，不影响训练。

### Q2: TTA 一定有效吗？

**A**：通常有效，但效果取决于：
- 模型质量（过拟合的模型可能提升有限）
- 任务特性（对称性强的任务效果更好）
- 增强选择（不合适的增强可能无效甚至有害）

### Q3: 如何选择最佳配置？

**A**：建议在验证集上做消融实验：
1. 先测试不同的 scale 组合
2. 然后加入 flips
3. 最后尝试 rotations（如果有帮助）
4. 调整 threshold 和后处理参数

### Q4: 可以在 Python 代码中使用 TTA 吗？

**A**：可以！示例：

```python
from autogluon.multimodal import MultiModalPredictor
import pandas as pd

# 1. 加载模型
predictor = MultiModalPredictor.load("AutogluonModels/ag-XXX")

# 2. 启用 TTA
predictor.enable_tta(
    scales=[0.75, 1.0, 1.25],
    flips=["none", "horizontal"],
    rotations=[-10, 0, 10],
    fusion_method="weighted_mean",
)

# 3. 评估
test_df = pd.read_csv("test.csv")
results = predictor.evaluate(test_df, metrics=["iou", "dice"])
print(results)

# 4. 关闭 TTA（如果需要）
predictor.disable_tta()
```

### Q5: TTA 支持多类分割吗？

**A**：目前实现主要针对二值分割（如 ISIC2017）。多类分割的 TTA 需要额外适配。

---

## 技术实现细节

### TTA 流程

```
对每张测试图像:
    1. 生成 N 个增强版本（scale, flip, rotate 的组合）
    2. 对每个增强版本:
        a. 前向推理得到概率图
        b. 应用逆变换将概率图对齐到原图坐标
    3. 融合所有概率图（平均 or 加权平均）
    4. 应用阈值得到二值 mask
    5. 后处理（去小连通域、形态学平滑）
```

### 关键文件

- **TTA 工具模块**：`multimodal/src/autogluon/multimodal/utils/tta_utils.py`
- **Learner 集成**：`multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`
- **训练脚本**：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

---

## 参考文献

1. **Multi-scale Testing**：标准做法，见于大多数语义分割比赛（Kaggle, MICCAI）
2. **Test-Time Augmentation**：
   - Matsunaga, K., et al. "Image Classification with Test-Time Augmentation"
   - 医学图像分割中的常见做法
3. **医学图像分割竞赛经验**：
   - ISIC Challenge 获奖方案
   - Medical Decathlon 前排方案

---

## 总结

- ✅ **TTA 是提升分割精度的有效手段**，尤其适合评测阶段
- ✅ **推荐从基础配置开始**，逐步增加复杂度
- ✅ **在验证集上调优参数**，再应用到测试集
- ⚠️ **注意推理时间成本**，选择适合场景的配置

**快速推荐**：
- 日常评估：基础 TTA（6 次推理）
- 论文提交：进阶 TTA（18 次推理）
- 比赛冲榜：极致 TTA（45 次推理）

祝实验顺利！🚀

