# Box 扰动策略改进指南

## 🎯 问题背景

通过可视化发现，原始的 box 扰动策略存在问题：
- 使用高斯噪声进行正负随机抖动
- 导致 noisy box 经常**小于** GT mask 的最小包围框
- **不符合实际使用场景**：人类专家手绘框通常会比 GT 更大（外扩），而不会更小（内缩）

## ✅ 改进方案

### 新的扰动模式

1. **`none`** - 无扰动（精确 bbox）
   - 使用 GT mask 的精确最小包围框
   - 适合作为**对比基线**

2. **`expand_only`** - 只外扩（推荐）
   - 四个边界独立向外扩张
   - 符合实际人类标注行为
   - **推荐用于主要实验**

3. **`mixed`** - 主要外扩 + 少量内缩
   - 大概率外扩，小概率内缩
   - 可配置内缩概率和幅度
   - 适合模拟多样化的标注质量

### 度量单位

1. **`bbox_relative`** - 相对于 bbox 尺寸（默认，推荐）
   - 扰动量与 bbox 大小成正比
   - 大目标和小目标的扰动比例一致

2. **`image_relative`** - 相对于图片尺寸
   - 扰动量与图片大小成正比
   - 所有目标的绝对扰动量相近

3. **`absolute_pixels`** - 绝对像素值
   - 固定的像素扰动量
   - 适合已知标注精度的场景

## 🚀 使用方法

### 1. 基线实验（无扰动）

使用精确的 GT bbox 作为基线：

```bash
python prepare_semi_supervised_data_v2.py \
  --task isic2017 \
  --data_dir datasets/isic2017/isic2017 \
  --labeled_ratio 0.1 \
  --box_jitter_mode none
```

### 2. 推荐配置（只外扩）

符合实际使用场景的配置：

```bash
python prepare_semi_supervised_data_v2.py \
  --task isic2017 \
  --data_dir datasets/isic2017/isic2017 \
  --labeled_ratio 0.1 \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.15 \
  --box_jitter_unit bbox_relative
```

参数说明：
- `box_expand_ratio=0.15`：外扩比例为 bbox 尺寸的 15%
- 例如：100x100 的 bbox 会外扩约 15 像素

### 3. 混合模式（主要外扩 + 少量内缩）

模拟多样化的标注质量：

```bash
python prepare_semi_supervised_data_v2.py \
  --task isic2017 \
  --data_dir datasets/isic2017/isic2017 \
  --labeled_ratio 0.1 \
  --box_jitter_mode mixed \
  --box_expand_ratio 0.15 \
  --box_contract_ratio 0.05 \
  --box_contract_prob 0.1
```

参数说明：
- `box_expand_ratio=0.15`：外扩比例 15%
- `box_contract_ratio=0.05`：内缩比例 5%
- `box_contract_prob=0.1`：每个边界有 10% 概率内缩

### 4. 其他扰动程度

#### 小扰动（高质量标注）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.05  # 仅外扩 5%
```

#### 中等扰动（默认）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.15  # 外扩 15%
```

#### 大扰动（低质量标注）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.30  # 外扩 30%
```

## 📊 对比实验建议

为了评估 box 质量对训练效果的影响，建议进行以下对比实验：

### 实验 1: 精确 bbox（基线）
```bash
# 准备数据
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode none \
  --box_seed 123

# 训练
python run_semi_supervised_train_v3_fixed.py \
  --output_dir outputs/baseline_exact_bbox
```

### 实验 2: 小扰动（5%）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.05 \
  --box_seed 123

python run_semi_supervised_train_v3_fixed.py \
  --output_dir outputs/expand_5pct
```

### 实验 3: 中等扰动（15%，推荐）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.15 \
  --box_seed 123

python run_semi_supervised_train_v3_fixed.py \
  --output_dir outputs/expand_15pct
```

### 实验 4: 大扰动（30%）
```bash
python prepare_semi_supervised_data_v2.py \
  --box_jitter_mode expand_only \
  --box_expand_ratio 0.30 \
  --box_seed 123

python run_semi_supervised_train_v3_fixed.py \
  --output_dir outputs/expand_30pct
```

### 实验 5: 原始随机扰动（对比）
```bash
# 使用旧版脚本
python prepare_semi_supervised_data.py \
  --box_noise_std 0.10 \
  --box_seed 123

python run_semi_supervised_train_v3_fixed.py \
  --output_dir outputs/old_random_noise
```

## 📈 统计信息

脚本运行后会输出 box 扰动的统计信息：

```
Box 扰动统计:
  平均宽度扩张: 0.147 (±0.086)
  平均高度扩张: 0.152 (±0.091)
  内缩样本数: 0 / 1800
```

解读：
- **平均扩张**: 正值表示外扩，负值表示内缩
- **标准差**: 表示扰动的随机性
- **内缩样本数**: 应该为 0（expand_only 模式）或很少（mixed 模式）

## 🔍 可视化验证

准备好新数据后，使用可视化脚本检查 box 质量：

```bash
# 可视化前 50 个样本
python visualize_semi_supervised_data.py --max_samples 50

# 查看结果
ls datasets/isic2017/visualizations/weak_samples/*.png | head -10
```

**预期观察**：
- ✅ Noisy box（红框）应该**大于或等于** GT mask 的可见区域
- ✅ Box-GT IoU 应该在 0.6-0.9 之间（取决于扰动程度）
- ❌ 如果 box 明显小于 GT mask，说明配置有问题

## 📝 完整参数列表

```bash
python prepare_semi_supervised_data_v2.py --help
```

关键参数：
- `--box_jitter_mode`: 扰动模式（none/expand_only/mixed）
- `--box_expand_ratio`: 外扩比例（默认 0.15）
- `--box_contract_ratio`: 内缩比例（默认 0.05，仅 mixed 模式）
- `--box_contract_prob`: 内缩概率（默认 0.1，仅 mixed 模式）
- `--box_jitter_unit`: 度量单位（bbox_relative/image_relative/absolute_pixels）
- `--box_seed`: 随机种子（保证可重复性）

## 🎓 理论依据

根据实际使用场景：
1. 人类专家使用交互工具（如点击或拖拽）标注时
2. 出于"宁可多框一些，不要漏掉"的心理
3. 通常会给出比目标稍大的框
4. 很少会出现框比目标小的情况

因此，**只外扩（expand_only）** 的策略更符合真实的弱监督场景。

---

**创建日期**: 2025-12-31  
**版本**: 2.0
