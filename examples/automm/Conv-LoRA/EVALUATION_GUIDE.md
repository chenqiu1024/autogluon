# RLOO 模型评估指南

本指南说明如何评估 RLOO 训练后的模型在测试集上的性能。

## 📁 文件说明

- **`evaluate_rloo_lightning.py`** - 单个 checkpoint 评估脚本
- **`compare_rloo_models.py`** - 批量对比脚本（推荐）

## 🚀 快速开始

### 方法 1：批量对比（推荐）

一次性评估所有模型并生成对比报告：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 确保在正确的环境中
conda activate conv-lora

# 运行批量对比
python compare_rloo_models.py \
  --rloo_output_dir outputs/rloo/isic2017/20251127 \
  --base_checkpoint_path AutogluonModels/ag-20251126_062717 \
  --task isic2017
```

**输出示例**：
```
================================================================================
RLOO 模型性能对比
================================================================================
找到 4 个 checkpoint 待评估:
  1. Baseline (Before RLOO)
  2. RLOO - epoch=00 (rloo_mean_reward=1.8524)
  3. RLOO - epoch=01 (rloo_mean_reward=1.8479)
  4. RLOO - epoch=02 (rloo_mean_reward=1.8514)

运行评估: model.ckpt
[评估进度条...]

================================================================================
评估结果对比
================================================================================
模型                                      IoU          Dice         样本数
--------------------------------------------------------------------------------
Baseline (Before RLOO)                   0.8523±0.1234  0.9145±0.0876     600
RLOO - epoch=00 (reward=1.8524)          0.8645±0.1156  0.9234±0.0798     600
RLOO - epoch=01 (reward=1.8479)          0.8612±0.1189  0.9201±0.0823     600
RLOO - epoch=02 (reward=1.8514)          0.8631±0.1171  0.9218±0.0811     600
================================================================================

RLOO 训练改进:
--------------------------------------------------------------------------------
RLOO - epoch=00 (reward=1.8524):
  IoU:  +0.0122 (+1.43%)
  Dice: +0.0089 (+0.97%)

RLOO - epoch=01 (reward=1.8479):
  IoU:  +0.0089 (+1.04%)
  Dice: +0.0056 (+0.61%)

RLOO - epoch=02 (reward=1.8514):
  IoU:  +0.0108 (+1.27%)
  Dice: +0.0073 (+0.80%)
================================================================================

✓ 完整报告已保存到: outputs/rloo/isic2017/20251127/evaluation_results/comparison_report_20251127_123456.json
```

### 方法 2：评估单个 Checkpoint

评估特定的 checkpoint：

```bash
# 评估最佳 RLOO 模型（Epoch 0）
python evaluate_rloo_lightning.py \
  --checkpoint_path outputs/rloo/isic2017/20251127/checkpoints/rloo-epoch=00-rloo_mean_reward=1.8524.ckpt \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --output_file results_rloo_epoch0.json

# 评估基线模型（RLOO 训练前）
python evaluate_rloo_lightning.py \
  --checkpoint_path AutogluonModels/ag-20251126_062717/model.ckpt \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --output_file results_baseline.json
```

## 📊 评估指标

脚本会计算以下指标：

- **IoU (Intersection over Union)** - 交并比
- **Dice Coefficient** - Dice 系数
- **标准差** - 各样本间的性能波动

## 🎯 典型工作流

### 1. 训练完成后立即评估

```bash
# Step 1: 批量对比所有 checkpoint
python compare_rloo_models.py \
  --rloo_output_dir outputs/rloo/isic2017/20251127 \
  --base_checkpoint_path AutogluonModels/ag-20251126_062717 \
  --task isic2017

# Step 2: 查看结果
cat outputs/rloo/isic2017/20251127/evaluation_results/comparison_report_*.json
```

### 2. 选择最佳 Checkpoint

根据评估结果，选择性能最好的 checkpoint：

```bash
# 假设 epoch=00 表现最好
BEST_CKPT="outputs/rloo/isic2017/20251127/checkpoints/rloo-epoch=00-rloo_mean_reward=1.8524.ckpt"

# 详细评估最佳模型
python evaluate_rloo_lightning.py \
  --checkpoint_path $BEST_CKPT \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --batch_size 4 \
  --output_file best_model_results.json
```

### 3. 在其他测试集上评估

如果有多个测试集（如 COCO 不同子集）：

```bash
# ISIC2017 官方测试集
python evaluate_rloo_lightning.py \
  --checkpoint_path $BEST_CKPT \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --test_csv test.csv \
  --output_file results_official_test.json

# 如果有验证集也想测试
python evaluate_rloo_lightning.py \
  --checkpoint_path $BEST_CKPT \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --test_csv val.csv \
  --output_file results_val.json
```

## 📈 结果分析

### 查看 JSON 结果

```bash
# 使用 jq 查看格式化的 JSON
cat results_rloo_epoch0.json | jq

# 或者使用 Python
python -c "
import json
with open('results_rloo_epoch0.json') as f:
    data = json.load(f)
    print(f'IoU: {data[\"mean_iou\"]:.4f} ± {data[\"std_iou\"]:.4f}')
    print(f'Dice: {data[\"mean_dice\"]:.4f} ± {data[\"std_dice\"]:.4f}')
"
```

### 对比多个实验

如果你运行了多次 RLOO 训练（不同超参数）：

```bash
# 实验 1: beta=0.05
python compare_rloo_models.py \
  --rloo_output_dir outputs/rloo/isic2017/exp1_beta0.05 \
  --base_checkpoint_path AutogluonModels/ag-20251126_062717 \
  --task isic2017

# 实验 2: beta=0.1
python compare_rloo_models.py \
  --rloo_output_dir outputs/rloo/isic2017/exp2_beta0.1 \
  --base_checkpoint_path AutogluonModels/ag-20251126_062717 \
  --task isic2017

# 对比两个实验的报告
```

## ⚙️ 命令行参数

### `evaluate_rloo_lightning.py`

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint_path` | ✅ | - | Lightning checkpoint 路径 (.ckpt) |
| `--base_predictor_path` | ✅ | - | 基础模型路径（用于加载配置） |
| `--task` | ❌ | `isic2017` | 任务名称 |
| `--test_csv` | ❌ | `test.csv` | 测试数据文件名 |
| `--batch_size` | ❌ | `1` | 评估 batch size |
| `--num_workers` | ❌ | `4` | DataLoader workers |
| `--output_file` | ❌ | `None` | 结果保存路径（JSON） |

### `compare_rloo_models.py`

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--rloo_output_dir` | ✅ | - | RLOO 训练输出目录 |
| `--base_checkpoint_path` | ✅ | - | 基础模型路径 |
| `--task` | ❌ | `isic2017` | 任务名称 |

## 🐛 故障排查

### 问题 1: CUDA out of memory

**解决方法**：减小 batch_size

```bash
python evaluate_rloo_lightning.py \
  --checkpoint_path ... \
  --batch_size 1 \
  --num_workers 2
```

### 问题 2: 找不到测试数据

**检查**：
```bash
# 确认测试数据存在
ls datasets/isic2017/isic2017/test.csv

# 检查数据格式
head datasets/isic2017/isic2017/test.csv
```

### 问题 3: Checkpoint 加载失败

**可能原因**：
- Checkpoint 文件损坏
- 模型架构不匹配

**解决方法**：
```bash
# 使用 last.ckpt 而不是特定 epoch 的 checkpoint
python evaluate_rloo_lightning.py \
  --checkpoint_path outputs/rloo/isic2017/20251127/checkpoints/last.ckpt \
  ...
```

## 📝 输出文件

### 单个评估结果 (JSON)

```json
{
  "checkpoint_path": "outputs/rloo/.../rloo-epoch=00-....ckpt",
  "task": "isic2017",
  "mean_iou": 0.8645,
  "mean_dice": 0.9234,
  "std_iou": 0.1156,
  "std_dice": 0.0798,
  "num_samples": 600
}
```

### 对比报告 (JSON)

```json
{
  "task": "isic2017",
  "base_checkpoint": "AutogluonModels/ag-20251126_062717",
  "rloo_output_dir": "outputs/rloo/isic2017/20251127",
  "timestamp": "2025-11-27T12:34:56",
  "results": [
    {
      "name": "Baseline (Before RLOO)",
      "type": "baseline",
      "mean_iou": 0.8523,
      "mean_dice": 0.9145,
      ...
    },
    {
      "name": "RLOO - epoch=00 (reward=1.8524)",
      "type": "rloo",
      "mean_iou": 0.8645,
      "mean_dice": 0.9234,
      ...
    }
  ]
}
```

## 🎓 最佳实践

1. **始终对比基线** - 评估 RLOO 前后的模型
2. **评估多个 checkpoint** - 不同 epoch 可能有不同表现
3. **记录超参数** - 在报告中注明训练配置
4. **重复实验** - 多次运行以确认稳定性
5. **保存结果** - 使用 `--output_file` 保存 JSON 结果

## 📚 相关文档

- RLOO 训练指南：`RLOO_REAL_TRAINING_GUIDE_zh.md`
- Bug 修复记录：`BUGFIX_RLOO_TRAINING.md`
- 完整记录：`docs/cursor_rloo_conv_lora-251126.md`

---

**祝评估顺利！** 📊

