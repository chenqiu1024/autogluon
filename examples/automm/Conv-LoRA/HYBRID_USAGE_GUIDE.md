# Conv-LoRA + GSPO + Adapters 混合架构使用指南

本文档提供了使用混合架构（Conv-LoRA + GSPO + 标准Adapters）对SAM模型进行医学图像分割的完整训练和评估流程。

## 架构概述

该混合架构结合了三种参数高效微调（PEFT）技术：

1. **Conv-LoRA**：在注意力层的 Q/K/V 投影中使用卷积专家的混合专家（MoE）机制，提供空间自适应能力
2. **GSPO (Group Sequence Policy Optimization)**：优化Conv-LoRA的专家选择策略，通过组级对比学习提高性能
3. **Adapters**：在每个Transformer层的MLP模块后并行插入标准瓶颈适配器，提供额外的领域适应能力

## 环境准备

### 1. 安装依赖

```bash
# 进入 autogluon 目录
cd /Users/domqiu/Works/autogluon

# 安装 Python 依赖
python3 -m pip install transformers torch torchvision
python3 -m pip install pandas numpy scipy scikit-learn
python3 -m pip install timm omegaconf pyyaml tqdm

# 安装 autogluon multimodal (开发模式)
cd multimodal
pip install -e .
```

### 2. 准备数据集

数据集应按以下结构组织：

```
datasets/
└── {dataset_name}/
    └── {dataset_name}/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── images/
```

CSV 文件格式：
```csv
image,label
images/train_001.jpg,masks/train_001.png
images/train_002.jpg,masks/train_002.png
...
```

支持的数据集：
- `leaf_disease_segmentation`
- `polyp`
- `camo_sem_seg`
- `isic2017`
- `road_segmentation`
- `SBU-shadow`

## 训练流程

### 基础训练（仅Conv-LoRA）

```bash
cd examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4 \
    --output_dir outputs/conv_lora_baseline \
    --seed 42686693
```

### 启用 GSPO 训练

```bash
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4 \
    --output_dir outputs/conv_lora_gspo \
    --gspo_enable \
    --gspo_group_size 4 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --gspo_quality_momentum 0.9 \
    --seed 42686693
```

### 完整混合架构训练（Conv-LoRA + GSPO + Adapters）

```bash
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4 \
    --output_dir outputs/hybrid_full \
    --gspo_enable \
    --gspo_group_size 4 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --adapter_enable \
    --adapter_dim 64 \
    --seed 42686693
```

### 参数说明

#### 基础参数
- `--task`: 数据集名称
- `--rank`: LoRA 秩 r（默认：3）
- `--expert_num`: Conv-LoRA 专家数量 M（默认：8）
- `--num_gpus`: 使用的GPU数量
- `--per_gpu_batch_size`: 每个GPU的批次大小
- `--batch_size`: 有效批次大小（如果 > per_gpu_batch_size * num_gpus，将使用梯度累积）
- `--output_dir`: 输出目录
- `--seed`: 随机种子

#### GSPO 参数
- `--gspo_enable`: 启用 GSPO 训练
- `--gspo_group_size`: 每组中的预测数量（默认：4）
- `--gspo_warmup_epochs`: 启用 GSPO 前的热身轮数（默认：5）
- `--gspo_contrastive_weight`: 对比损失权重（默认：0.1）
- `--gspo_quality_momentum`: 专家质量历史的动量系数（默认：0.9）

#### Adapter 参数
- `--adapter_enable`: 启用标准 Adapter 模块
- `--adapter_dim`: Adapter 瓶颈维度（默认：64，推荐范围：32-128）

## 评估流程

### 评估已训练模型

```bash
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --eval \
    --ckpt_path outputs/hybrid_full \
    --output_dir outputs/hybrid_full
```

评估结果将保存在 `outputs/hybrid_full/metrics.txt` 文件中。

### 评估指标

根据不同数据集，评估指标包括：
- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **BER (Balanced Error Rate)** - 用于 SBU-shadow 数据集
- **S-measure, F-measure, E-measure, MAE** - 用于 polyp 和 camo_sem_seg 数据集

## 实验对比

### 建议的实验设置

为了验证混合架构的有效性，建议运行以下对比实验：

1. **基线（仅Conv-LoRA）**
   ```bash
   python3 run_semantic_segmentation.py --task leaf_disease_segmentation \
       --rank 3 --expert_num 8 --output_dir outputs/baseline
   ```

2. **Conv-LoRA + GSPO**
   ```bash
   python3 run_semantic_segmentation.py --task leaf_disease_segmentation \
       --rank 3 --expert_num 8 --gspo_enable --output_dir outputs/gspo
   ```

3. **Conv-LoRA + Adapters**
   ```bash
   python3 run_semantic_segmentation.py --task leaf_disease_segmentation \
       --rank 3 --expert_num 8 --adapter_enable --adapter_dim 64 \
       --output_dir outputs/adapters
   ```

4. **完整混合架构（Conv-LoRA + GSPO + Adapters）**
   ```bash
   python3 run_semantic_segmentation.py --task leaf_disease_segmentation \
       --rank 3 --expert_num 8 --gspo_enable --adapter_enable \
       --adapter_dim 64 --output_dir outputs/hybrid_full
   ```

### 结果分析

每个实验完成后，检查 `outputs/{experiment_name}/metrics.txt` 文件，比较不同配置下的性能指标。

## 多数据集验证

建议在以下医学图像分割数据集上进行验证：

```bash
# 叶片疾病分割
python3 run_semantic_segmentation.py --task leaf_disease_segmentation \
    --gspo_enable --adapter_enable --output_dir outputs/leaf_hybrid

# 息肉分割
python3 run_semantic_segmentation.py --task polyp \
    --gspo_enable --adapter_enable --output_dir outputs/polyp_hybrid

# 皮肤病变分割 (ISIC2017)
python3 run_semantic_segmentation.py --task isic2017 \
    --gspo_enable --adapter_enable --output_dir outputs/isic_hybrid

# 伪装对象分割
python3 run_semantic_segmentation.py --task camo_sem_seg \
    --gspo_enable --adapter_enable --output_dir outputs/camo_hybrid
```

## 常见问题

### 1. 内存不足

如果遇到 GPU 内存不足，尝试：
- 减小 `--per_gpu_batch_size`
- 减小 `--adapter_dim`（如 32）
- 减少 `--expert_num`（如 4）

### 2. 训练速度慢

- 确保使用 GPU：`--num_gpus 1`
- 增大 `--per_gpu_batch_size`（如果内存允许）
- 减少 `--gspo_group_size`

### 3. 性能不理想

- 增加训练轮数（修改 `get_default_training_setting` 中的 `max_epoch`）
- 调整学习率
- 尝试不同的 `--adapter_dim` 值（32, 64, 128）
- 调整 `--gspo_contrastive_weight`

## 技术细节

### 可训练参数统计

- **Conv-LoRA**: ~0.5-2% 的 SAM 总参数（取决于 rank 和 expert_num）
- **Adapters**: ~0.1-0.5% 的 SAM 总参数（取决于 adapter_dim）
- **总计**: 约 1-3% 的参数微调（相比完全微调的 100%）

### 模型架构概览

```
SAM Vision Encoder (ViT-H/B/L)
├── PatchEmbed
└── Transformer Blocks [×32]
    ├── LayerNorm1
    ├── SamVisionAttention
    │   ├── Q/K/V Projections → Conv-LoRA (MoE-Conv)
    │   └── GSPO Expert Selection
    ├── LayerNorm2
    ├── MLP Block
    └── Adapter (并行) ← 新增
```

## 引用

如果您使用此混合架构，请引用以下论文：

```bibtex
@inproceedings{zhong2024convlora,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  booktitle={ICLR},
  year={2024}
}

@article{guo2024gspo,
  title={Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs},
  author={Guo, Arash and others},
  journal={arXiv preprint arXiv:2402.14740},
  year={2024}
}
```
