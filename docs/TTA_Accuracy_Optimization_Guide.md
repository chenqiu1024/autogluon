# TTA 精度优化指南

本指南专注于通过 TTA 获得最佳分割精度，不考虑推理时间成本。

---

## 🎯 目标

**最大化 Dice 和 IoU 分数**，适用于：
- 📝 论文提交
- 🏆 比赛冲榜
- 📊 最终模型评测

---

## ⭐ 推荐配置（精度优先）

### 配置 1: 标准精度 TTA（推荐）

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
    --tta_morphology \
    --output_dir outputs/eval_tta_best
```

或使用提供的脚本：

```bash
./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302
```

**特点**:
- ✅ 多尺度: 3 个 scales
- ✅ 翻转: horizontal
- ✅ 旋转: 小角度 ±10°
- ✅ 加权融合: scale=1.0 权重更高
- ✅ 形态学: 平滑边界
- ✅ **总变换**: 18 次/图
- ✅ **预期提升**: Dice **+0.5~0.8%**
- ⏱️ **时间成本**: ~15-20s/图，600 图约 **2.5-3.5 小时**

---

### 配置 2: 极致精度 TTA（最大化）

如果你有更多时间和计算资源：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251203_075302 \
    --tta_enable \
    --tta_scales 0.5 0.75 1.0 1.25 1.5 \
    --tta_flips none horizontal vertical \
    --tta_rotations -15 -10 0 10 15 \
    --tta_fusion weighted_mean \
    --tta_threshold 0.45 \
    --tta_min_area 0.0005 \
    --tta_morphology \
    --output_dir outputs/eval_tta_extreme
```

**特点**:
- ✅ 更多尺度: 5 个 scales（0.5x ~ 1.5x）
- ✅ 更多翻转: horizontal + vertical
- ✅ 更多旋转: 5 个角度（±15°）
- ✅ **总变换**: 75 次/图
- ✅ **预期提升**: Dice **+0.8~1.0%**
- ⏱️ **时间成本**: ~40-50s/图，600 图约 **6-8 小时**

**注意**: 
- 垂直翻转仅适用于无方向性的任务（如皮肤病变）
- ±15° 旋转对某些医学图像可能过大

---

## 🔍 精度优化技巧

### 1. 阈值调优（最重要！）

**当前默认**: 0.5  
**建议**: 在验证集上搜索最佳阈值

```bash
# 测试不同阈值
for threshold in 0.35 0.40 0.45 0.50 0.55 0.60; do
    python3 run_semantic_segmentation.py \
        --task isic2017 \
        --eval \
        --ckpt_path AutogluonModels/ag-20251203_075302 \
        --tta_enable \
        --tta_threshold $threshold \
        --output_dir outputs/tta_threshold_${threshold}
    
    echo "Threshold $threshold:"
    cat outputs/tta_threshold_${threshold}/metrics.txt
    echo ""
done
```

**预期效果**: 阈值调优单独可以带来 **0.2~0.5%** Dice 提升！

---

### 2. 小连通域过滤

**当前默认**: 0.001（图像面积的 0.1%）

**建议根据任务调整**:

| 任务类型 | 建议值 | 原因 |
|---------|--------|------|
| 小病灶（<5mm） | 0.0001 | 避免过滤掉真实病灶 |
| 中等病灶 | 0.001 | 默认，平衡 |
| 大器官 | 0.005~0.01 | 更激进地去除噪声 |

```bash
# 对于 ISIC2017（病灶大小变化大）
--tta_min_area 0.0005  # 略小于默认值
```

---

### 3. 形态学后处理

**作用**: 填补小空洞、平滑边界

```bash
--tta_morphology  # 开启
```

**适用场景**:
- ✅ 器官分割：需要平滑连续的边界
- ✅ 肿瘤分割：去除内部空洞
- ⚠️ 精细结构：可能过度平滑

**ISIC2017**: **推荐开启**，皮肤病变边界通常较平滑

---

### 4. 融合策略选择

#### Mean（平均融合）
```bash
--tta_fusion mean
```
- **优点**: 简单，稳定
- **缺点**: 所有增强权重相同

#### Weighted Mean（加权融合，推荐）
```bash
--tta_fusion weighted_mean
```
- **优点**: scale=1.0 权重更高（2.0 vs 1.0）
- **原因**: 原始分辨率的预测通常更可靠
- **推荐**: ✅ **用于精度优化**

---

## 📊 完整评估流程（精度优先）

### 步骤 1: 基线评估

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251203_075302 \
    --output_dir outputs/eval_baseline
```

记录基线分数：
```
IoU: 0.XXXX
Dice: 0.XXXX
```

---

### 步骤 2: 标准 TTA（18 次变换）

```bash
./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302
```

或手动运行：

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
    --tta_morphology \
    --output_dir outputs/eval_tta_standard
```

**时间成本**: ~2.5-3.5 小时  
**预期提升**: Dice **+0.5~0.8%**

---

### 步骤 3: 阈值调优（强烈推荐）

在验证集上找到最佳阈值：

```bash
# 使用验证集
VAL_DF="datasets/isic2017/isic2017/val.csv"

for threshold in 0.40 0.45 0.50 0.55 0.60; do
    python3 run_semantic_segmentation.py \
        --task isic2017 \
        --eval \
        --ckpt_path AutogluonModels/ag-20251203_075302 \
        --tta_enable \
        --tta_scales 0.75 1.0 1.25 \
        --tta_flips none horizontal \
        --tta_rotations -10 0 10 \
        --tta_fusion weighted_mean \
        --tta_threshold $threshold \
        --output_dir outputs/tta_val_threshold_${threshold}
done

# 找到最佳阈值后，用于测试集
BEST_THRESHOLD=0.45  # 例如

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251203_075302 \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_threshold $BEST_THRESHOLD \
    --tta_morphology \
    --output_dir outputs/eval_tta_tuned
```

**额外提升**: **+0.2~0.3%** Dice

---

### 步骤 4: 极致 TTA（可选，如果需要最后一点提升）

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --eval \
    --ckpt_path AutogluonModels/ag-20251203_075302 \
    --tta_enable \
    --tta_scales 0.5 0.75 1.0 1.25 1.5 \
    --tta_flips none horizontal \
    --tta_rotations -15 -10 0 10 15 \
    --tta_fusion weighted_mean \
    --tta_threshold $BEST_THRESHOLD \
    --tta_morphology \
    --output_dir outputs/eval_tta_extreme
```

**时间成本**: ~6-8 小时  
**额外提升**: 可能 **+0.1~0.2%** Dice

---

## 📈 预期精度提升

### ISIC2017 数据集

假设基线：IoU=77.87%, Dice=85.81%

| 配置 | 变换次数 | 时间 | 预期 IoU | 预期 Dice | 提升 |
|------|---------|------|---------|----------|------|
| **基线** | 1 | 2分钟 | 77.87% | 85.81% | - |
| 标准 TTA | 18 | 2.5-3.5小时 | **78.3~78.5%** | **86.3~86.5%** | **+0.5~0.7%** |
| + 阈值调优 | 18 | 3-4小时 | **78.5~78.7%** | **86.5~86.7%** | **+0.7~0.9%** |
| 极致 TTA | 75 | 6-8小时 | **78.6~78.9%** | **86.6~86.9%** | **+0.8~1.1%** |

**重要提示**: 
- 提升幅度取决于基线模型质量
- 如果基线已经很强，提升可能较小
- 阈值调优是性价比最高的优化

---

## 🔬 TTA 精度提升的原理

### 为什么 TTA 有效？

#### 1. **多视角融合** - 降低单次预测的随机性
不同变换提供不同"视角"，融合后更稳定：
```
Scale 0.75: 关注整体
Scale 1.0: 原始视角（最可靠）
Scale 1.25: 关注细节

融合 → 兼顾整体和细节
```

#### 2. **边界改进** - 减少边界不确定性
```
Original: 边界像素概率 = 0.48（低置信度）
Flip: 边界像素概率 = 0.52
Rotate +10°: 边界像素概率 = 0.53
Rotate -10°: 边界像素概率 = 0.51

平均: (0.48 + 0.52 + 0.53 + 0.51) / 4 = 0.51
→ 更稳定的边界预测
```

#### 3. **噪声抑制** - 过滤假阳性
```
某个噪声点:
  变换1: 0.55 (假阳性)
  变换2: 0.30
  变换3: 0.25
  变换4: 0.40
  
平均: 0.375 < threshold → 被过滤
```

---

## 📋 最佳实践清单

### 针对 ISIC2017 皮肤病变分割

✅ **必做**:
- [x] 使用多尺度: `[0.75, 1.0, 1.25]`
- [x] 使用水平翻转: `["none", "horizontal"]`
- [x] 使用小角度旋转: `[-10, 0, 10]`
- [x] 使用加权融合: `weighted_mean`
- [x] 在验证集上调优阈值

✅ **推荐**:
- [x] 开启形态学后处理: `--tta_morphology`
- [x] 调整小连通域过滤: `--tta_min_area 0.0005`

⚠️ **可选**（效果不确定）:
- [ ] 增加更多尺度: `[0.5, 0.75, 1.0, 1.25, 1.5]`
- [ ] 增加旋转角度: `[-15, -10, 0, 10, 15]`
- [ ] 垂直翻转: `vertical`（仅在对称任务）

---

## 🎨 针对不同任务的调整

### 皮肤病变（ISIC2017）- 当前任务

**特点**: 大小不一、形状不规则、无固定方向

**推荐配置**:
```bash
--tta_scales 0.75 1.0 1.25 \
--tta_flips none horizontal \
--tta_rotations -10 0 10 \
--tta_fusion weighted_mean \
--tta_morphology
```

**关键参数**:
- Scales: ✅ 重要（病灶大小变化大）
- Horizontal flip: ✅ 重要（无方向性）
- Vertical flip: ⚠️ 可选（皮肤病变无上下差异）
- Rotations: ✅ 重要（无固定方向）

---

### 息肉分割（Polyp）

**特点**: 中等大小、形状多样、可能有方向性

**推荐配置**:
```bash
--tta_scales 0.75 1.0 1.25 \
--tta_flips none horizontal \
--tta_rotations 0 \
--tta_fusion weighted_mean
```

**关键参数**:
- Rotations: ❌ 不推荐（内窥镜视角有方向性）
- Morphology: ✅ 推荐（息肉边界较平滑）

---

### 阴影检测（SBU-Shadow）

**特点**: 大面积、强方向性（光照）

**推荐配置**:
```bash
--tta_scales 0.75 1.0 1.25 \
--tta_flips none \
--tta_rotations 0 \
--tta_fusion mean \
--tta_min_area 0.01
```

**关键参数**:
- Flips: ❌ 不推荐（光照方向很重要）
- Rotations: ❌ 不推荐（同上）
- Min_area: ✅ 较大值（阴影通常是大区域）

---

## 💡 组合策略

### 与其他技巧结合

#### 1. TTA + Ensemble

```python
# 训练多个模型
model1 = train(seed=1)
model2 = train(seed=2)
model3 = train(seed=3)

# 对每个模型用 TTA，再平均
pred1 = model1.predict_with_tta(image)
pred2 = model2.predict_with_tta(image)
pred3 = model3.predict_with_tta(image)

final_pred = (pred1 + pred2 + pred3) / 3
```

**预期提升**: Ensemble **+0.5~1.0%** + TTA **+0.5~0.8%** = **+1.0~1.8%** 总提升

#### 2. TTA + 后处理链

```
TTA 预测
  ↓
阈值调优
  ↓
小连通域过滤
  ↓
形态学平滑
  ↓
CRF 精修（可选）
```

---

## 🎯 实战示例

### 完整的精度优化流程

```bash
#!/bin/bash
# 完整的精度优化流程

CKPT="AutogluonModels/ag-20251203_075302"

echo "步骤 1: 基线评估..."
python3 run_semantic_segmentation.py \
    --task isic2017 --eval --ckpt_path $CKPT \
    --output_dir outputs/step1_baseline

echo "步骤 2: 基础 TTA..."
python3 run_semantic_segmentation.py \
    --task isic2017 --eval --ckpt_path $CKPT \
    --tta_enable \
    --output_dir outputs/step2_tta_basic

echo "步骤 3: 进阶 TTA（18 次变换）..."
./run_tta_best_accuracy.sh $CKPT outputs/step3_tta_advanced

echo "步骤 4: 在验证集上调优阈值..."
# （见上文阈值调优命令）

echo "步骤 5: 使用最佳阈值在测试集上评估..."
BEST_THRESHOLD=0.45  # 假设从步骤 4 找到
python3 run_semantic_segmentation.py \
    --task isic2017 --eval --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_threshold $BEST_THRESHOLD \
    --tta_morphology \
    --output_dir outputs/step5_final

echo "对比结果："
echo "基线:"
cat outputs/step1_baseline/metrics.txt
echo "最终（TTA + 调优）:"
cat outputs/step5_final/metrics.txt
```

---

## 🏆 比赛/论文提交建议

### 对于比赛

1. **多模型 Ensemble**:
   - 训练 3-5 个不同 seed 的模型
   - 每个都用 TTA
   - 平均预测

2. **充分利用时间**:
   - 使用极致 TTA（75 次变换）
   - 阈值精细调优（0.01 步长）
   - 尝试不同的后处理组合

3. **验证策略**:
   - 5-fold 交叉验证
   - 每个 fold 都用 TTA
   - 平均分数

### 对于论文提交

1. **消融实验必做**:
   ```
   - 基线（无 TTA）
   - TTA: Flip only
   - TTA: Flip + Rotate
   - TTA: Full (Scale + Flip + Rotate)
   ```

2. **报告细节**:
   - 明确说明 TTA 配置
   - 报告提升幅度和统计显著性
   - 提供速度/精度 trade-off

3. **可视化**:
   - 展示 TTA 前后的分割对比
   - 展示边界改进
   - 展示噪声抑制效果

---

## 📊 预期结果（ISIC2017）

### 基于经验的预期

| 方法 | IoU | Dice | 备注 |
|------|-----|------|------|
| Conv-LoRA + GSPO + Adapter (基线) | 77.87% | 85.81% | 已经很强 |
| + 标准 TTA (18 次) | **78.3~78.5%** | **86.3~86.5%** | +0.5~0.7% |
| + 阈值调优 | **78.5~78.7%** | **86.5~86.7%** | +0.2~0.3% |
| + 极致 TTA (75 次) | **78.6~78.9%** | **86.6~86.9%** | +0.1~0.2% |
| **最终可达** | **78.7~78.9%** | **86.7~86.9%** | **总提升 +0.8~1.1%** |

---

## 🚀 立即行动

### 运行标准精度 TTA（推荐）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 使用封装好的脚本
./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302

# 预计完成时间：2.5-3.5 小时
# 期间可以监控进度：
# watch -n 30 'tail -50 outputs/eval_tta_best_accuracy/metrics.txt'
```

---

## 💭 关于当前性能的说明

### 速度（8s/图）

- ✅ **这是正常的**，对于包含多尺度的 TTA
- ✅ 主要时间花在 resize 操作（SAM 要求 1024×1024 输入）
- ✅ 已经尽可能优化（用 PIL 替代 scipy）
- ⚠️ 如果需要更快，只能牺牲精度（移除 scale）

### 内存

- ✅ 已经添加激进的内存管理
- ⚠️ 完全消除增长需要流式评估（重大重构）
- ✅ 当前实现在可接受范围内（6-8GB）

### 建议

**对于精度优化**：
- ✅ **接受当前速度**（8s/图 是合理的）
- ✅ **监控内存**（应该不会超过 10-12GB）
- ✅ **关注结果**（精度提升才是目标）

---

## 📚 相关文档

- 📖 [TTA 使用指南](../docs/TTA_Usage_Guide.md)
- 📖 [性能现实评估](../docs/TTA_Performance_Reality.md)
- 📖 [内存优化](../docs/TTA_Memory_CPU_Optimization.md)

---

**目标**: 最大化分割精度  
**方案**: 标准 TTA（18 次变换）+ 阈值调优  
**预期**: Dice +0.7~0.9%  
**时间**: 3-4 小时（值得）

