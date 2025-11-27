### 概述

本实验指南介绍如何在 AutoGluon 的 `Conv-LoRA` 示例基础上，增加一个基于 **RLOO** 的强化学习微调阶段，
以进一步提升 Segment Anything Model（SAM）在下游语义/医学分割任务上的性能。

整体流程分为两阶段：

- **阶段一：监督学习（已有脚本）**  
  使用 `run_semantic_segmentation.py` 进行常规 Conv-LoRA + SAM 监督微调，得到一个性能较好的基线模型。

- **阶段二：RLOO 强化学习微调（新增脚本）**  
  使用 `run_semantic_segmentation_rloo.py` 读取阶段一 checkpoint，构造基于 GT 指标（IoU/Dice 等）的 reward，
  通过 RLOO 思想对 Conv-LoRA 参数进行小步优化。

> 注意：当前示例代码主要提供完整的算法结构与接口设计，便于后续在 AutoGluon 内部进一步集成。
> 由于 AutoGluon 推理链路默认在 `no_grad` 环境下执行，要实现真正端到端的 RL 训练，
> 需要在 Learner / DataModule / LitModule 层面做更深入的改造。

---

### 环境准备

1. 按照 `README.md` 中的说明安装 AutoGluon Multimodal 与 Conv-LoRA 示例依赖：

```bash
conda create -n conv-lora python=3.10
conda activate conv-lora
pip install -U pip
pip install -U setuptools wheel
git clone https://github.com/autogluon/autogluon
cd autogluon && pip install -e multimodal/[tests]
```

2. 下载语义分割数据集（如 leaf_disease_segmentation、polyp 等）：

```bash
cd examples/automm/Conv-LoRA
python prepare_semantic_segmentation_datasets.py
```

---

### 阶段一：Conv-LoRA + SAM 监督微调

使用已有脚本 `run_semantic_segmentation.py` 进行训练，例如：

```bash
python run_semantic_segmentation.py \
  --task leaf_disease_segmentation \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --output_dir outputs/leaf_supervised
```

训练完成后，会在对应 `output_dir` 下生成 AutoGluon 的 checkpoint 目录（例如 `AutogluonModels/ag-YYYYMMDD_xxxxxx/`），
我们将在下一阶段把它作为 RLOO 微调的起点和参考策略。

---

### 阶段二：基于 RLOO 的 RL 微调

新脚本：`run_semantic_segmentation_rloo.py`，核心步骤如下：

1. **数据准备**  
   - 继续使用 `datasets/{task}/{task}` 下的 `train.csv`，并通过 `expand_path` 补全图像与标签路径。
   - 构造一个简易的 `DataLoader`，每个 batch 返回若干行 DataFrame（后续交给 AutoGluon 的 Learner 处理）。

2. **加载 Conv-LoRA-SAM 模型**  
   - 使用 `MultiModalPredictor.load(ckpt_path)` 加载阶段一模型；
   - 用 `SAMConvLoRAWrapper` 包装内部的 SAM + Conv-LoRA 模型，冻结非 LoRA 参数，只保留 Conv-LoRA 部分可训练。

3. **RLOO 步骤（示意实现）**  
   对每个 batch：
   - 调用 `learner.predict_per_run` 得到 logits 与 GT 掩码；
   - 对每张图像生成 `G = num_generations` 个候选 mask：
     - 第 1 个候选：直接对 logits 阈值化（>0.5）得到模型的主要预测
     - 后续候选：在 logits 上添加小的高斯噪声后再阈值化，引入多样性
   - 基于 GT mask 计算每个候选的 IoU / Dice，组合成 reward；
   - 通过 `rloo_utils` 中的 `bernoulli_kl` 计算当前策略与参考策略之间的 KL 惩罚，形成总 reward；
   - 使用 `mask_log_prob_from_logits` 计算每个候选的 log π(M | logits)；
   - 调用 `rloo_loss` 计算 leave-one-out advantage 下的 REINFORCE 损失。

4. **指标监控与验证**  
   - 记录每个 epoch 的平均 loss、reward、IoU、Dice、KL 等指标；
   - 验证 RLOO 算法的各个组件是否正确工作。

### ⚠️ 重要说明：当前实现状态

**当前版本是一个概念验证（Proof of Concept）实现**，主要目的是：

✅ **已实现并验证**：
- 完整的 RLOO 算法逻辑（多候选采样、reward 计算、KL 正则、leave-one-out advantage）
- Conv-LoRA 参数识别与管理
- 与 AutoGluon 数据流的集成
- 详细的指标监控和日志输出

❌ **当前限制**：
- **不进行真实的梯度更新**：由于 AutoGluon 的 `predict_per_run` 在 `no_grad` 环境中运行，当前实现无法进行端到端的反向传播
- 模型参数实际上未被更新

### 🔧 如何实现真实的 RL 训练

要实现完整的 RLOO 训练，需要深入 AutoGluon 内部：

1. **在 `SemanticSegmentationLitModule` 中添加 RLOO 训练模式**
   - 扩展 `training_step` 方法，支持 RLOO 风格的 loss 计算
   - 直接使用 `self.model(batch)` 进行带梯度的前向传播

2. **修改 DataModule 以支持多次采样**
   - 每个 batch 需要重复前向 G 次以生成多个候选

3. **集成 reward 计算与 KL 正则**
   - 将 `rloo_utils.py` 中的函数集成到训练循环中

当前实现的所有核心组件都可以直接迁移到上述深度集成中。

---

### 关键命令示例

假设阶段一的 checkpoint 路径为 `AutogluonModels/ag-20251126_062717`（**注意：传入目录而非 `.ckpt` 文件**）：

```bash
python run_semantic_segmentation_rloo.py \
  --task leaf_disease_segmentation \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo/leaf \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --epochs 3 \
  --batch_size 2
```

**重要提示**：
- `--ckpt_path` 应该传入 **AutoGluon 模型目录**（包含 `config.yaml`、`model.ckpt` 等文件的目录），而不是直接传入 `.ckpt` 文件路径。
- 脚本会自动处理：如果你不小心传入了 `.ckpt` 文件路径，会自动取其父目录。
- 如果从其他目录运行，建议使用绝对路径或从 `examples/automm/Conv-LoRA/` 目录下执行。

运行结束后，脚本会在 `output_dir` 下保存一个新的 AutoGluon Predictor 目录（示意为 `rloo_finetuned`），
后续可以用 `MultiModalPredictor.load` 进行评估。

---

### 超参数与调参建议

- **`num_generations` (G)**  
  - 每张图像生成的候选 mask 数量；G 越大，RLOO baseline 越稳定，但计算开销也越大。  
  - 建议起始值：4。

- **`beta`（KL 系数）**  
  - 控制当前策略偏离参考策略（监督基线）的程度；过小可能导致模型“跑偏”，过大则学习过于保守。  
  - 建议从 0.01–0.1 之间网格搜索。

- **`reward_type`**  
  - `iou`：仅使用 IoU 作为 reward；  
  - `dice`：仅使用 Dice 系数；  
  - `combo`：IoU + Dice 的加权组合（当前实现默认 α=β=1）。  
  - 对前景/背景极不平衡的任务，`combo` 往往更稳定。

- **`lambda_supervised`**  
  - 预留用于混合监督 loss 的权重，目前示例脚本尚未加入具体监督 term，后续可根据需要扩展。

---

### 实验与对比建议

1. **基线对比**  
   - 模型 A：仅阶段一 Conv-LoRA 监督训练；  
   - 模型 B：在 A 的基础上增加阶段二 RLOO 微调。  
   - 对比 IoU / Dice 等指标，观察是否有稳定提升。

2. **消融实验**  
   - 改变 `num_generations`：2、4、8；  
   - 改变 `beta`：0、0.01、0.05、0.1；  
   - 尝试不同 `reward_type` 组合。  

3. **实验记录**  
   - 建议新建 `RLOO_ConvLoRA_SAM_实验记录_zh.md`，记录每次实验的：数据集、超参数设置、指标结果与简要分析，
   方便后续撰写论文或技术报告时复用。


