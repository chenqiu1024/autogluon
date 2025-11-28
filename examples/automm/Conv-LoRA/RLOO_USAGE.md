# RLOO Conv-LoRA SAM 训练与测试命令示范

## 1. 数据准备

首先下载并准备数据集：

```bash
cd examples/automm/Conv-LoRA
python3 prepare_semantic_segmentation_datasets.py
```

## 2. 训练命令

### 2.1 使用原始 Structure Loss 训练（baseline）

```bash
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4 \
    --output_dir outputs_baseline \
    --seed 42686693
```

### 2.2 使用 RLOO Loss 训练（我们的新方法）

```bash
python3 run_rloo_sam.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4 \
    --output_dir outputs_rloo \
    --seed 42686693
```

**注意：** RLOO 的超参数（k=4, rloo_weight=1.0, structure_weight=1.0）已经在配置文件中设置好了。如果需要调整，可以修改 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml` 文件中的 `rloo` 部分。

**参数说明：**
- `--task`: 数据集名称，可选 `polyp`, `leaf_disease_segmentation`, `camo_sem_seg`, `isic2017`, `road_segmentation`, `SBU-shadow`
- `--rank`: LoRA 秩 r（论文中的超参数）
- `--expert_num`: MoE-Conv 专家数量 M（论文中的超参数）
- `--num_gpus`: 使用的 GPU 数量
- `--batch_size`: 有效批大小（会自动使用梯度累积）
- `--per_gpu_batch_size`: 每个 GPU 的批大小

**如需自定义 RLOO 超参数：**
编辑 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`:
```yaml
rloo:
  k: 4  # 每个图像采样的 mask 数量
  weight: 1.0  # RLOO loss 权重
  structure_weight: 1.0  # Structure loss 权重
```

### 2.3 其他数据集训练示例

**Polyp 分割:**
```bash
python3 run_rloo_sam.py \
    --task polyp \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs_polyp_rloo
```

**Road Segmentation:**
```bash
python3 run_rloo_sam.py \
    --task road_segmentation \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs_road_rloo
```

## 3. 仅评估命令（使用已训练模型）

### 3.1 评估 Baseline 模型

```bash
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --eval \
    --ckpt_path outputs_baseline \
    --output_dir outputs_baseline
```

### 3.2 评估 RLOO 模型

```bash
python3 run_rloo_sam.py \
    --task leaf_disease_segmentation \
    --eval \
    --ckpt_path outputs_rloo \
    --output_dir outputs_rloo
```

## 4. 多 GPU 训练

```bash
python3 run_rloo_sam.py \
    --task leaf_disease_segmentation \
    --rank 3 \
    --expert_num 8 \
    --num_gpus 4 \
    --per_gpu_batch_size 2 \
    --batch_size 8 \
    --output_dir outputs_rloo_multigpu \
    --rloo_k 4
```

## 5. 超参数调优建议

### RLOO 采样数 k 的选择：
- `k=2-4`: 快速训练，适合初步实验
- `k=6-8`: 平衡性能和速度
- `k=10+`: 更低方差，但计算成本高

### 权重调整：
- **平衡模式**: `--rloo_weight 1.0 --structure_weight 1.0`
- **更多监督**: `--rloo_weight 0.5 --structure_weight 1.5`
- **纯强化学习**: `--rloo_weight 1.0 --structure_weight 0.0`（不推荐，可能不稳定）

## 6. 结果查看

训练完成后，结果保存在：
- 模型: `{output_dir}/` 目录
- 评估指标: `{output_dir}/metrics.txt`

```bash
# 查看评估结果
cat outputs_rloo/metrics.txt
```

## 7. 对比实验示例

```bash
# 实验 1: Baseline (Structure Loss)
python3 run_semantic_segmentation.py \
    --task leaf_disease_segmentation \
    --output_dir exp1_baseline

# 实验 2: RLOO (k=4)
python3 run_rloo_sam.py \
    --task leaf_disease_segmentation \
    --output_dir exp2_rloo_k4 \
    --rloo_k 4

# 实验 3: RLOO (k=8)
python3 run_rloo_sam.py \
    --task leaf_disease_segmentation \
    --output_dir exp3_rloo_k8 \
    --rloo_k 8

# 对比结果
cat exp1_baseline/metrics.txt
cat exp2_rloo_k4/metrics.txt
cat exp3_rloo_k8/metrics.txt
```
