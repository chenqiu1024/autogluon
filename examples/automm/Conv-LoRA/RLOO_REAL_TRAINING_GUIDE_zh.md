# RLOO 真实训练指南

本指南介绍如何使用新实现的**真实的 RLOO 训练**功能，该功能实现了完整的梯度更新和参数优化。

## 目录

1. [概述](#概述)
2. [实现架构](#实现架构)
3. [使用方法](#使用方法)
4. [与概念验证版本的对比](#与概念验证版本的对比)
5. [常见问题](#常见问题)

---

## 概述

### 什么是真实的 RLOO 训练？

**概念验证版本** (`run_semantic_segmentation_rloo.py`):
- 展示 RLOO 算法的完整计算流程
- 通过 `predict_per_run` 获取输出（在 `no_grad` 环境中）
- **不进行真实的梯度更新**
- 用于算法验证和指标监控

**真实训练版本** (`run_semantic_segmentation_rloo_real.py`):
- ✅ **完整的梯度计算和反向传播**
- ✅ **真实的参数更新**（只更新 Conv-LoRA 参数）
- ✅ 基于 PyTorch Lightning Trainer
- ✅ 直接访问模型的前向传播（带梯度）
- ✅ 支持 checkpointing、日志记录等训练功能

---

## 实现架构

### 核心组件

#### 1. `lit_semantic_seg_rloo.py`

扩展 AutoGluon 的 `SemanticSegmentationLitModule` 以支持 RLOO 训练模式。

**主要功能**：
```python
class RLOOSemanticSegmentationLitModule(SemanticSegmentationLitModule):
    def __init__(self, enable_rloo=True, num_generations=4, beta=0.05, ...):
        # 支持 RLOO 和标准监督训练两种模式
        
    def _rloo_training_step(self, batch):
        # 1. 前向传播获取 logits（保留梯度）
        # 2. 通过采样生成 G 个候选 mask
        # 3. 计算每个候选的 reward（IoU/Dice）
        # 4. 应用 KL 正则
        # 5. 计算 RLOO loss
        # 6. 返回 loss（会自动进行反向传播）
        
    def training_step(self, batch, batch_idx):
        # 根据 enable_rloo 选择训练模式
```

**关键改进**：
- **只进行一次前向传播**：避免多次调用 `run_model`，提高效率
- **保留梯度**：不使用 `no_grad`，确保梯度能够反向传播
- **分离 reward 计算**：reward 计算在 `no_grad` 中进行（不需要梯度）
- **log_prob 保留梯度**：通过 log_prob 将梯度传递给模型参数

#### 2. `run_semantic_segmentation_rloo_real.py`

完整的训练脚本，管理整个 RLOO 训练流程。

**主要类**：
```python
class RLOOTrainer:
    def __init__(self, ckpt_path, output_dir, num_generations, beta, ...):
        # 初始化训练器
        
    def setup_model_and_datamodule(self, predictor):
        # 从 AutoGluon Predictor 提取模型和 DataModule
        
    def freeze_non_conv_lora_params(self, model):
        # 冻结非 Conv-LoRA 参数
        
    def train(self, predictor):
        # 使用 PyTorch Lightning Trainer 进行训练
```

**训练流程**：
1. 加载 AutoGluon Predictor（阶段一的 checkpoint）
2. 提取内部模型和 DataModule
3. 冻结非 Conv-LoRA 参数
4. 创建 `RLOOSemanticSegmentationLitModule`
5. 使用 PyTorch Lightning Trainer 执行训练
6. 保存训练结果和 checkpoints

---

## 使用方法

### 前提条件

1. **完成阶段一训练**：使用 `run_semantic_segmentation.py` 训练 Conv-LoRA SAM 模型
2. **有可用的 checkpoint**：例如 `AutogluonModels/ag-20251126_062717/`

### 基本命令

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 激活环境
conda activate conv-lora

# 运行 RLOO 训练
python run_semantic_segmentation_rloo_real.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo_real/leaf-test \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `leaf_disease_segmentation` | 数据集名称 |
| `--ckpt_path` | **必需** | 阶段一 checkpoint 目录 |
| `--output_dir` | `outputs_rloo_real` | 输出目录 |
| `--num_generations` | `4` | 每个样本生成的候选数量 G |
| `--beta` | `0.05` | KL 正则系数 |
| `--reward_type` | `combo` | reward 类型（iou/dice/combo） |
| `--learning_rate` | `1e-5` | 学习率 |
| `--epochs` | `3` | 训练 epoch 数 |
| `--batch_size` | `2` | Batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `42` | 随机种子 |

### 快速测试配置

```bash
# 最小配置 - 快速验证
python run_semantic_segmentation_rloo_real.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo_real/quick_test \
  --num_generations 2 \
  --epochs 1 \
  --batch_size 1
```

### 完整训练配置

```bash
# 完整配置 - 实际实验
python run_semantic_segmentation_rloo_real.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo_real/leaf-full \
  --num_generations 8 \
  --beta 0.1 \
  --reward_type combo \
  --learning_rate 5e-6 \
  --epochs 5 \
  --batch_size 2
```

### 后台运行

```bash
nohup python run_semantic_segmentation_rloo_real.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo_real/leaf-bg \
  --num_generations 4 \
  --beta 0.05 \
  --epochs 3 \
  --batch_size 2 > rloo_real_train.log 2>&1 &

# 查看日志
tail -f rloo_real_train.log
```

---

## 与概念验证版本的对比

| 特性 | 概念验证版本 | 真实训练版本 |
|------|-------------|-------------|
| **脚本名称** | `run_semantic_segmentation_rloo.py` | `run_semantic_segmentation_rloo_real.py` |
| **梯度更新** | ❌ 否（使用 `predict_per_run`） | ✅ 是（直接访问模型） |
| **参数更新** | ❌ 否 | ✅ 是（只更新 Conv-LoRA） |
| **训练框架** | 自定义循环 | PyTorch Lightning Trainer |
| **Checkpointing** | ❌ 不支持 | ✅ 自动保存 |
| **日志记录** | 打印输出 | TensorBoard + 文件 |
| **前向传播次数** | G 次/batch | 1 次/batch |
| **内存效率** | 较低 | 较高 |
| **适用场景** | 算法验证、指标监控 | 实际训练、模型优化 |

### 迁移指南

如果你之前使用概念验证版本，迁移到真实训练版本只需：

1. **更换脚本名称**：
   ```bash
   # 从
   python run_semantic_segmentation_rloo.py ...
   # 改为
   python run_semantic_segmentation_rloo_real.py ...
   ```

2. **参数基本相同**，可能需要调整：
   - `--learning_rate`：建议从 `1e-5` 开始
   - `--batch_size`：根据内存调整
   - `--epochs`：真实训练可能需要更多 epochs

3. **查看输出**：
   - 概念版本：标准输出
   - 真实版本：`output_dir/` 下的 checkpoints 和日志

---

## 训练输出

训练过程中会输出：

```
================================================================================
开始真实的 RLOO 训练
================================================================================
配置:
  Checkpoint: AutogluonModels/ag-20251126_062717
  输出目录: outputs_rloo_real/leaf-test
  候选数量 (G): 4
  KL 系数 (β): 0.05
  Reward 类型: combo
  学习率: 1e-05
  Epochs: 3
  Batch size: 2
================================================================================

参数统计:
  可训练参数: 514,560
  冻结参数: 637,951,488
  总参数: 638,466,048

开始训练循环...

Epoch 1/3
  Batch 1: rloo_mean_reward=0.xxxx, rloo_mean_iou=0.xxxx, ...
  Batch 10: rloo_mean_reward=0.xxxx, rloo_mean_iou=0.xxxx, ...
  ...

Epoch 2/3
  ...

================================================================================
RLOO 训练完成！
================================================================================

模型已保存到: outputs_rloo_real/leaf-test
训练统计:
  总 epochs: 3
  最终 checkpoint: outputs_rloo_real/leaf-test/checkpoints/epoch=2-step=xxx.ckpt
================================================================================
```

### 监控指标

训练过程中会记录以下指标：

- `train_loss`：RLOO loss
- `rloo_mean_reward`：平均 reward（IoU/Dice - KL）
- `rloo_mean_iou`：平均 IoU
- `rloo_mean_dice`：平均 Dice
- `rloo_mean_kl`：平均 KL 散度

---

## 常见问题

### Q1: 内存不足怎么办？

```bash
# 减小 batch_size 和 num_generations
--batch_size 1 --num_generations 2
```

### Q2: 训练很慢怎么办？

可能原因：
- **网络连接问题**：Hugging Face 模型下载
  - 解决：使用离线模式或镜像
- **前向传播慢**：SAM 模型较大
  - 解决：减小 `num_generations`

### Q3: 如何验证训练效果？

训练后的模型保存在 `output_dir/checkpoints/` 下。可以：

1. **加载最佳 checkpoint**：
   ```python
   from pytorch_lightning import Trainer
   lit_module = RLOOSemanticSegmentationLitModule.load_from_checkpoint(
       "outputs_rloo_real/leaf-test/checkpoints/epoch=2-step=xxx.ckpt"
   )
   ```

2. **评估性能**：使用原始的评估脚本评估 IoU/Dice

3. **可视化预测**：生成分割 mask 并与 GT 对比

### Q4: loss 不下降怎么办？

可能需要调整：
- **学习率**：尝试 `5e-6` 或 `2e-5`
- **beta**：降低 KL 约束，如 `0.01`
- **reward_type**：尝试不同的 reward 类型
- **num_generations**：增加候选数量以改善 baseline

### Q5: 如何保存最终模型为 AutoGluon Predictor？

目前真实训练版本保存的是 PyTorch Lightning checkpoint。要转换为 AutoGluon Predictor 格式，需要额外的转换步骤（待实现）。

---

## 技术细节

### RLOO 算法流程

在 `_rloo_training_step` 中：

1. **前向传播**（1 次）：
   ```python
   output = run_model(self.model, batch)
   pred_logits = output[self.model.prefix][LOGITS]  # 保留梯度
   ```

2. **生成候选**（G 次）：
   ```python
   for g in range(G):
       if g == 0:
           sampled_mask = (torch.sigmoid(pred_logits) > 0.5).float()
       else:
           noisy_logits = pred_logits + noise
           sampled_mask = torch.bernoulli(torch.sigmoid(noisy_logits))
   ```

3. **计算 reward**（无梯度）：
   ```python
   with torch.no_grad():
       rewards_metric = compute_segmentation_reward(...)
       kl_values = bernoulli_kl(pred_logits, ref_logits)
       rewards_total = rewards_metric - beta * kl_values
   ```

4. **计算 log_prob**（有梯度）：
   ```python
   log_p = mask_log_prob_from_logits(sampled_mask.detach(), pred_logits)
   ```

5. **RLOO loss**：
   ```python
   loss = rloo_loss(log_probs, rewards, normalize_advantage=True)
   # loss.backward() 会自动在 training_step 返回后执行
   ```

### 梯度流

```
pred_logits (requires_grad=True)
    ↓
mask_log_prob_from_logits()
    ↓
log_probs (requires_grad=True)
    ↓
rloo_loss() = -(advantages * log_probs).mean()
    ↓
loss.backward()
    ↓
更新 Conv-LoRA 参数
```

---

## 下一步

1. **运行快速测试**：使用最小配置验证功能
2. **完整训练**：使用推荐配置训练完整模型
3. **评估对比**：对比阶段一（监督）和阶段二（RLOO）的性能
4. **超参数调优**：尝试不同的 `beta`、`num_generations`、`reward_type`
5. **记录实验**：在 `RLOO_ConvLoRA_SAM_实验记录_zh.md` 中记录结果

---

## 参考资料

- 概念验证版本：`RLOO_ConvLoRA_SAM_实验指南_zh.md`
- RLOO 工具函数：`rloo_utils.py`
- SAM Conv-LoRA 封装：`sam_conv_lora_wrapper.py`
- AutoGluon 文档：https://auto.gluon.ai/

---

**祝训练顺利！** 🚀

