# 实验协议文档

## 数据拆分

### 源数据
- 数据集：ISIC2017 训练集（2000张）
- 标注：完整二值mask

### 拆分策略
- 10% (200张)：保留 full mask
- 90% (1800张)：仅保留 noisy box
- 种子：random_seed=42, box_seed=123

### Box噪声
\`\`\`
box_noisy = box_gt * (1 + σ * randn(4))
σ = 0.10
\`\`\`

## 对比实验

| 实验 | 标注形式 | 方法 |
|------|---------|------|
| Exp-A | 100% mask | 全监督baseline |
| Exp-B | 10% mask | 仅监督学习 |
| Exp-C | 10%+90% box | Teacher-Student |
| Exp-D | 10%+90% box | +质量加权 |
| Exp-E | 10%+90% box | +GSPO(q) |
| Exp-F | 10%+90% box | 完整方案 |

## 消融实验

1. 质量proxy对比：置信度 vs 一致性 vs 混合
2. 采样次数K：3 / 5 / 7
3. Box噪声强度：0.05 / 0.10 / 0.15
4. 质量阈值：0.5 / 0.6 / 0.7
