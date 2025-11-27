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
   - 对每张图像重复采样 `G = num_generations` 个候选 mask，形成 `{M_i}`；
   - 基于 GT mask 计算每个候选的 IoU / Dice，组合成 reward；
   - 通过 `rloo_utils` 中的 `bernoulli_kl` 计算当前策略与参考策略之间的 KL 惩罚，形成总 reward；
   - 使用 `mask_log_prob_from_logits` 计算每个候选的 log π(M | logits)；
   - 调用 `rloo_loss` 计算 leave-one-out advantage 下的 REINFORCE 损失。

4. **参数更新（结构示例）**  
   - 构造优化器：`AdamW(仅 Conv-LoRA 参数, lr=1e-5)`；
   - 对每个 batch：前向 → 计算 RLOO loss → `loss.backward()` → `optimizer.step()`。

> 当前示例代码出于安全起见，仍在 `no_grad` 环境下调用 AutoGluon 推理，
> 主要目的是演示 RLOO 的 **外层结构和 reward/advantage 计算逻辑**。
> 真正要启用梯度，需要参考 AutoGluon 的 `SemanticSegmentationLitModule` / `BaseDataModule` 等模块，
> 在模型和 DataModule 层实现自定义训练循环。

---

### 关键命令示例

假设阶段一的 checkpoint 路径为 `outputs/leaf_supervised/AutogluonModels/ag-20251126_062717`：

```bash
python run_semantic_segmentation_rloo.py \
  --task leaf_disease_segmentation \
  --ckpt_path outputs/leaf_supervised/AutogluonModels/ag-20251126_062717 \
  --output_dir outputs_rloo/leaf \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --epochs 3 \
  --batch_size 2
```

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


