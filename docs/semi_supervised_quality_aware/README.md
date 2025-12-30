# 质量感知半监督分割方案

## 快速开始

### 1. 准备数据
\`\`\`bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
python prepare_semi_supervised_data.py --task isic2017 --data_dir datasets/isic2017
\`\`\`

### 2. 运行训练
\`\`\`bash
python run_semi_supervised_train.py --task isic2017 --data_dir datasets/isic2017 --gspo_enable --adapter_enable
\`\`\`

### 3. 或使用一键脚本
\`\`\`bash
bash run_semi_supervised_example.sh
\`\`\`

## 核心模块

- **ema_teacher.py** - EMA Teacher（公式 2.1）
- **quality_estimator.py** - 质量评估器（公式 2.2）
- **pseudo_label_gen.py** - 伪标签生成器（公式 2.3）
- **data_module.py** - 数据加载模块
- **gspo_semi_trainer.py** - GSPO 扩展（公式 2.5）

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| quality_k_samples | 5 | K次采样数量 |
| quality_min_threshold | 0.6 | 质量过滤阈值 |
| ema_momentum | 0.999 | EMA动量 |
| box_noise_std | 0.10 | Box噪声强度 |

## 预期性能

在 ISIC2017（10% mask + 90% noisy box）：
- Baseline: ~78% DICE
- 完整方案: ~86% DICE
- **提升: +8% DICE**
