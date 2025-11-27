# RLOO 训练实现 - 快速开始

本目录包含 Conv-LoRA SAM 的 RLOO（Reinforcement Learning with Leave-One-Out baseline）训练实现。

## 📁 文件结构

### 核心实现

| 文件 | 说明 |
|------|------|
| `rloo_utils.py` | RLOO 算法工具函数（reward、KL、loss 等） |
| `lit_semantic_seg_rloo.py` | 扩展的 LitModule，支持 RLOO 训练 |
| `run_semantic_segmentation_rloo_real.py` | **真实训练脚本**（带梯度更新） |
| `run_semantic_segmentation_rloo.py` | 概念验证脚本（算法演示） |

### 文档

| 文件 | 说明 |
|------|------|
| `RLOO_REAL_TRAINING_GUIDE_zh.md` | 真实训练详细指南 |
| `RLOO_ConvLoRA_SAM_实验指南_zh.md` | 概念验证实验指南 |
| `RLOO_README.md` | 本文件 |

### 辅助文件

| 文件 | 说明 |
|------|------|
| `test_rloo_simple.py` | 功能测试脚本 |
| `sam_conv_lora_wrapper.py` | SAM Conv-LoRA 模型封装 |

## 🚀 快速开始

### 1. 准备阶段一模型

首先使用监督学习训练 Conv-LoRA SAM：

```bash
python run_semantic_segmentation.py \
  --task leaf_disease_segmentation \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --output_dir outputs/leaf_supervised
```

假设得到 checkpoint：`AutogluonModels/ag-20251126_062717/`

### 2. 运行 RLOO 训练

#### 方式 A：真实训练（推荐）

带梯度更新的完整训练：

```bash
python run_semantic_segmentation_rloo_real.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo_real/leaf \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2
```

#### 方式 B：概念验证

算法演示和指标监控（不更新参数）：

```bash
python run_semantic_segmentation_rloo.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo/leaf \
  --num_generations 4 \
  --beta 0.05 \
  --epochs 3 \
  --batch_size 2
```

## 🔑 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_generations` | 4 | 每个样本生成的候选数 G |
| `--beta` | 0.05 | KL 正则系数 |
| `--reward_type` | combo | iou / dice / combo |
| `--learning_rate` | 1e-5 | 学习率（仅真实训练） |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 2 | Batch 大小 |

## 📊 两种实现对比

| 特性 | 概念验证 | 真实训练 |
|------|---------|---------|
| **梯度更新** | ❌ 否 | ✅ 是 |
| **参数更新** | ❌ 否 | ✅ 是 |
| **用途** | 算法验证 | 实际训练 |
| **脚本** | `run_semantic_segmentation_rloo.py` | `run_semantic_segmentation_rloo_real.py` |

## 📖 详细文档

- **真实训练**：查看 `RLOO_REAL_TRAINING_GUIDE_zh.md`
- **概念验证**：查看 `RLOO_ConvLoRA_SAM_实验指南_zh.md`
- **完整记录**：查看 `docs/cursor_rloo_conv_lora-251126.md`

## 🎯 典型工作流

```bash
# 1. 阶段一：监督训练
python run_semantic_segmentation.py --task leaf_disease_segmentation ...

# 2. 阶段二：RLOO 微调
python run_semantic_segmentation_rloo_real.py \
  --ckpt_path AutogluonModels/ag-xxx \
  --num_generations 4 \
  --beta 0.05 \
  --epochs 3

# 3. 评估对比
# 比较阶段一和阶段二模型的 IoU/Dice
```

## ⚙️ 系统要求

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- AutoGluon Multimodal
- CUDA（推荐）

## 🐛 常见问题

### Q: 内存不足？

```bash
--batch_size 1 --num_generations 2
```

### Q: 如何选择 beta？

- 小值（0.01）：更多探索，可能不稳定
- 中值（0.05）：平衡（推荐）
- 大值（0.1）：保守，接近监督学习

### Q: 训练不收敛？

- 降低学习率：`--learning_rate 5e-6`
- 调整 beta：尝试 `0.01` 或 `0.1`
- 增加候选数：`--num_generations 8`

## 📝 引用

如果使用本实现，请引用：

```
Conv-LoRA SAM with RLOO Fine-tuning
AutoGluon Multimodal Framework
2025
```

## 🤝 贡献

- RLOO 算法实现：`rloo_utils.py`
- LitModule 扩展：`lit_semantic_seg_rloo.py`
- 训练框架：`run_semantic_segmentation_rloo_real.py`

---

**祝训练顺利！** 🚀

