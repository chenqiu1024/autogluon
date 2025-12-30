# 质量感知的半监督分割方案

## 快速开始

### 1. 准备数据

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 分层抽样 + 生成 noisy box
python prepare_semi_supervised_data.py \
    --task isic2017 \
    --data_dir datasets/isic2017 \
    --labeled_ratio 0.1 \
    --box_noise_std 0.10 \
    --random_seed 42 \
    --box_seed 123
```

**输出**：
- `train_labeled_10pct.csv`：200 张 + full mask
- `train_weak_90pct.csv`：1800 张 + noisy box

### 2. 运行训练

```bash
python run_semi_supervised_train.py \
    --task isic2017 \
    --data_dir datasets/isic2017 \
    --output_dir outputs/semi_supervised_isic \
    --labeled_ratio 0.1 \
    --quality_k_samples 5 \
    --quality_min_threshold 0.6 \
    --ema_momentum 0.999 \
    --gspo_enable \
    --gspo_group_size 4 \
    --gspo_warmup_epochs 5 \
    --pseudo_lambda_warmup_epochs 5 \
    --consistency_lambda 0.1 \
    --max_epochs 30 \
    --batch_size 4 \
    --rank 4 \
    --expert_num 4 \
    --adapter_enable \
    --adapter_dim 64
```

### 3. 查看结果

训练日志和模型将保存在 `outputs/semi_supervised_isic/`

## 文档导航

- **[design_overview_zh.md](design_overview_zh.md)** - 方案原理、公式、创新点
- **[implementation_details_zh.md](implementation_details_zh.md)** - 实现细节、代码结构
- **[experiment_protocol_zh.md](experiment_protocol_zh.md)** - 实验协议、数据拆分、消融实验
- **[formulas_reference.md](formulas_reference.md)** - 公式速查表

## 核心模块

```
examples/automm/Conv-LoRA/semi_supervised/
├── __init__.py                 # 模块导出
├── ema_teacher.py              # EMA Teacher（公式 2.1）
├── quality_estimator.py        # 质量评估器（公式 2.2）
├── pseudo_label_gen.py         # 伪标签生成器（公式 2.3）
├── data_module.py              # 数据加载模块
└── gspo_semi_trainer.py        # GSPO 扩展（公式 2.5）
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quality_k_samples` | 5 | Teacher K 次采样数量 |
| `quality_min_threshold` | 0.6 | 伪标签质量过滤阈值 |
| `ema_momentum` | 0.999 | EMA 更新动量 |
| `pseudo_lambda_warmup_epochs` | 5 | 伪监督权重 warmup |
| `consistency_lambda` | 0.1 | Box jitter 一致性权重 |

## 预期性能

在 ISIC2017 数据集上（10% mask + 90% noisy box）：

| 方法 | DICE | 提升 |
|------|------|------|
| Baseline (仅 10% mask) | ~78% | - |
| + Teacher-Student | ~80% | +2% |
| + 质量加权 | ~82% | +2% |
| + GSPO 质量联动 | ~84% | +2% |
| + Box Jitter（完整方案） | ~86% | +2% |

**总提升**：~8% DICE

## 常见问题

### Q: 显存不足怎么办？
A: 减小 `quality_k_samples` 至 3，或减小 `batch_size`

### Q: 伪标签使用率过低？
A: 降低 `quality_min_threshold` 至 0.5，或减小 `box_noise_std`

### Q: 训练不稳定？
A: 增加 `pseudo_lambda_warmup_epochs` 至 10，提高 `ema_momentum` 至 0.9995




