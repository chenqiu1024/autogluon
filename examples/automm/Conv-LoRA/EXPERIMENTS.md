# GSPO-ConvLoRA 实验记录

## 实验目标

验证GSPO（Group Sequence Policy Optimization）方法能够提升Conv-LoRA在医学图像分割任务上的性能。

**核心假设**：通过组级策略优化和质量感知的专家选择，GSPO-ConvLoRA能够在ISIC2017皮肤病变分割数据集上显著优于baseline Conv-LoRA。

## 实验配置

### 数据集
- **主数据集**：ISIC2017 (International Skin Imaging Collaboration 2017)
  - 训练集：2000张皮肤病变图像
  - 验证集：150张
  - 测试集：600张
  - 任务：二分类分割（病变 vs 背景）

### 硬件环境
- GPU：[待填写]
- CUDA版本：[待填写]
- PyTorch版本：[待填写]

### 基础配置
| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank (r) | 3 | 低秩分解维度 |
| Expert数量 (M) | 8 | MoE专家数 |
| Batch size | 4 | 有效批次大小 |
| Per-GPU batch size | 1 | 单GPU批次大小 |
| Learning rate | 1e-4 | 学习率 |
| Max epochs | 30 | 最大训练轮数 |
| Loss function | structure_loss | 分割损失 |
| Validation metric | IoU | 验证指标 |
| Seed | 42686693 | 随机种子 |

### GSPO配置

| 参数 | 值 | 说明 |
|------|-----|------|
| gspo_enabled | True/False | 是否启用GSPO |
| gspo_group_size | 3, 4, 6 | 组大小（TopK） |
| gspo_warmup_epochs | 5 | GSPO预热轮数 |
| gspo_quality_momentum | 0.9 | 质量历史动量系数 |
| gspo_contrastive_weight | 0.1 | 对比损失权重 |

## 实验设计

### 主实验：GSPO vs Baseline

**实验1：Baseline Conv-LoRA**
```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs/baseline_convlora
```

**实验2：GSPO-ConvLoRA (group_size=4)**
```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --gspo_group_size 4 \
    --output_dir outputs/gspo_convlora_g4
```

### 消融实验：Group Size影响

**实验3：GSPO (group_size=3)**
```bash
python run_semantic_segmentation.py --task isic2017 --gspo_enable --gspo_group_size 3 --output_dir outputs/gspo_convlora_g3
```

**实验4：GSPO (group_size=6)**
```bash
python run_semantic_segmentation.py --task isic2017 --gspo_enable --gspo_group_size 6 --output_dir outputs/gspo_convlora_g6
```

## 实验结果

### 主实验结果

| Method | IoU | DICE | Train Time | Improvement |
|--------|-----|------|------------|-------------|
| **Baseline Conv-LoRA** | TBD | TBD | TBD | - |
| **GSPO-ConvLoRA (g=4)** | TBD | TBD | TBD | TBD |

> **注**：运行 `bash run_gspo_experiments.sh` 后，结果将自动填充到 `outputs/results_summary.txt`

### 消融实验：Group Size

| Group Size | IoU | DICE | Notes |
|------------|-----|------|-------|
| 3 | TBD | TBD | 较小的组，计算开销低 |
| 4 | TBD | TBD | 推荐配置 |
| 6 | TBD | TBD | 较大的组，可能过度平滑 |

### 训练曲线分析

[运行实验后生成训练曲线图]

### 专家使用分析

[待实验完成后分析专家选择分布]

## 分析与讨论

### 性能改进分析

**预期改进来源**：
1. **质量感知门控**：根据历史表现动态调整专家选择
2. **组级优化**：减少单样本噪声，提供更稳定的梯度信号
3. **对比学习**：强化高质量预测模式，抑制低质量模式
4. **多专家协作**：TopK>1允许专家互补，增强鲁棒性

### 训练效率分析

**时间开销**：
- GSPO增加了前向传播次数（×G倍）
- 但通过更好的收敛可能减少总训练时间
- 推理时无额外开销（可回退到TopK=1）

**内存开销**：
- 组内预测可复用中间特征
- 对比损失仅需特征向量，内存占用小

### 泛化能力

[待在其他数据集（Polyp等）上验证]

## 实验检查清单

运行实验前确认：
- [ ] 数据集已下载到 `datasets/isic2017/`
- [ ] 环境已安装：`pip install -e multimodal/[tests]`
- [ ] GPU可用且内存充足
- [ ] 输出目录有足够磁盘空间

运行实验：
```bash
cd examples/automm/Conv-LoRA
bash run_gspo_experiments.sh
```

分析结果：
```bash
python analyze_results.py
```

## 故障排除

### 常见问题

**Q1: OOM (Out of Memory)**
- 减小 `batch_size` 或 `gspo_group_size`
- 使用梯度累积：增大 `accumulate_grad_batches`

**Q2: 训练不稳定**
- 增加 `gspo_warmup_epochs`
- 降低 `gspo_contrastive_weight`

**Q3: GSPO没有改进**
- 检查 `gspo_enabled` 是否正确设置
- 确认已过warmup阶段
- 调整 `gspo_group_size` 和 `gspo_quality_momentum`

## 后续实验计划

### 扩展到其他数据集
- [ ] Polyp分割
- [ ] Leaf Disease分割
- [ ] Road分割

### 超参数优化
- [ ] Grid search: group_size ∈ {2, 3, 4, 5, 6}
- [ ] Ablation: contrastive_weight ∈ {0, 0.05, 0.1, 0.2}
- [ ] Ablation: quality_momentum ∈ {0.7, 0.8, 0.9, 0.95}

### 与其他方法对比
- [ ] vs 标准LoRA
- [ ] vs 全参数微调
- [ ] vs 其他PEFT方法

## 实验日志

### [日期] 实验运行记录
- 实验ID：
- 运行命令：
- 结果：
- 备注：

---

**更新日志**：
- [日期] 创建实验记录模板
- [日期] 完成Baseline实验
- [日期] 完成GSPO主实验
- [日期] 完成消融实验

